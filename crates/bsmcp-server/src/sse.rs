use std::collections::HashMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use reqwest::Client;
use serde_json::Value;
use subtle::ConstantTimeEq;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use zeroize::Zeroize;

use bsmcp_common::bookstack::BookStackClient;
use bsmcp_common::db::DbBackend;
use bsmcp_common::time::TimezoneConfig;

use crate::mcp;
use crate::oauth::{AuthCode, AUTH_CODE_TTL};
use crate::semantic::SemanticState;

const MAX_SESSIONS_PER_TOKEN: usize = 5;
const MAX_TOTAL_SESSIONS: usize = 1000;
const SESSION_TTL: Duration = Duration::from_secs(24 * 60 * 60); // 24 hours
const MAX_REQUESTS_PER_MINUTE: u32 = 100;

#[derive(Clone)]
pub struct AppState {
    pub bookstack_url: String,
    pub http_client: Client,
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    pub auth_codes: Arc<RwLock<HashMap<String, AuthCode>>>,
    pub db: Arc<dyn DbBackend>,
    pub known_urls: Vec<String>,
    pub authorize_rate_limit: Arc<Mutex<RateLimit>>,
    pub register_rate_limit: Arc<Mutex<RateLimit>>,
    streamable_rate_limits: Arc<RwLock<HashMap<String, Arc<Mutex<RateLimit>>>>>,
    streamable_sessions: Arc<RwLock<HashMap<String, Instant>>>,
    backup_interval_hours: Option<u64>,
    backup_path: PathBuf,
    pub semantic: Option<Arc<SemanticState>>,
    pub staging: crate::staging::StagingStore,
    pub settings_sessions: crate::settings_ui::SettingsSessionStore,
    /// Index db is always present (sqlite has full impl, postgres returns
    /// stub errors per #36). Webhook handler enqueues page:{id} index jobs
    /// here when BookStack page events arrive.
    pub index_db: Arc<dyn bsmcp_common::db::IndexDb>,
    /// Resolved at startup (`BSMCP_DEFAULT_TIMEZONE` → `TZ` → UTC). Used
    /// to build `_meta.time` on every tools/call response (issue #67).
    pub timezone: Arc<TimezoneConfig>,
}

pub(crate) struct RateLimit {
    count: u32,
    max: u32,
    window_start: Instant,
}

impl RateLimit {
    pub(crate) fn new(max: u32) -> Self {
        Self {
            count: 0,
            max,
            window_start: Instant::now(),
        }
    }

    pub(crate) fn check(&mut self) -> Result<(), ()> {
        if self.window_start.elapsed() > Duration::from_secs(60) {
            self.count = 0;
            self.window_start = Instant::now();
        }
        if self.count >= self.max {
            return Err(());
        }
        self.count += 1;
        Ok(())
    }
}

struct Session {
    tx: mpsc::Sender<Result<Event, Infallible>>,
    client: BookStackClient,
    token_id: String,
    token_secret: String,
    created_at: Instant,
    rate_limit: Arc<Mutex<RateLimit>>,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.token_id.zeroize();
        self.token_secret.zeroize();
    }
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bookstack_url: String,
        db: Arc<dyn DbBackend>,
        index_db: Arc<dyn bsmcp_common::db::IndexDb>,
        known_urls: Vec<String>,
        backup_interval_hours: Option<u64>,
        backup_path: PathBuf,
        semantic: Option<Arc<SemanticState>>,
        timezone: Arc<TimezoneConfig>,
    ) -> Self {
        let http_client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            bookstack_url: bookstack_url.trim_end_matches('/').to_string(),
            http_client,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            auth_codes: Arc::new(RwLock::new(HashMap::new())),
            db,
            known_urls,
            authorize_rate_limit: Arc::new(Mutex::new(RateLimit::new(20))),
            register_rate_limit: Arc::new(Mutex::new(RateLimit::new(10))),
            streamable_rate_limits: Arc::new(RwLock::new(HashMap::new())),
            streamable_sessions: Arc::new(RwLock::new(HashMap::new())),
            backup_interval_hours,
            backup_path,
            semantic,
            staging: crate::staging::new_staging_store(),
            settings_sessions: crate::settings_ui::new_settings_store(),
            index_db,
            timezone,
        }
    }

    pub fn spawn_cleanup(&self) {
        let sessions = self.sessions.clone();
        let auth_codes = self.auth_codes.clone();
        let db = self.db.clone();
        let streamable_rate_limits = self.streamable_rate_limits.clone();
        let streamable_sessions = self.streamable_sessions.clone();
        let staging_clone = self.staging.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                {
                    let mut sessions = sessions.write().await;
                    sessions.retain(|sid, s| {
                        let expired = s.tx.is_closed() || s.created_at.elapsed() > SESSION_TTL;
                        if expired {
                            eprintln!("Session {sid} cleaned up");
                        }
                        !expired
                    });
                }
                {
                    let mut codes = auth_codes.write().await;
                    codes.retain(|_, c| c.created_at.elapsed() < AUTH_CODE_TTL);
                }
                {
                    let mut srl = streamable_rate_limits.write().await;
                    srl.retain(|_, rl| {
                        let rl = rl.try_lock();
                        match rl {
                            Ok(rl) => rl.window_start.elapsed() < Duration::from_secs(120),
                            Err(_) => true,
                        }
                    });
                }
                {
                    let mut ss = streamable_sessions.write().await;
                    ss.retain(|_, created| created.elapsed() < SESSION_TTL);
                }
                {
                    crate::staging::cleanup_expired_sync(&staging_clone);
                }
                if let Err(e) = db.cleanup_expired_tokens().await {
                    eprintln!("Token cleanup error: {e}");
                }
            }
        });
    }

    pub fn spawn_backup(&self) {
        let Some(interval_hours) = self.backup_interval_hours else {
            return;
        };
        let db = self.db.clone();
        let backup_path = self.backup_path.clone();
        let interval = Duration::from_secs(interval_hours * 3600);
        eprintln!(
            "Backup: enabled every {interval_hours}h to {}",
            backup_path.display()
        );
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                match db.backup(&backup_path).await {
                    Ok(()) => eprintln!("Backup: completed successfully"),
                    Err(e) => eprintln!("Backup: failed — {e}"),
                }
            }
        });
    }
}

/// Constant-time string comparison to prevent timing side-channel attacks.
pub(crate) fn constant_time_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let len = a.len().max(b.len());
    let mut a_padded = vec![0xAAu8; len];
    let mut b_padded = vec![0xBBu8; len];
    a_padded[..a.len()].copy_from_slice(a);
    b_padded[..b.len()].copy_from_slice(b);
    let result: bool = a_padded.ct_eq(&b_padded).into();
    result && a.len() == b.len()
}

fn unauthorized(hint: &str, headers: &HeaderMap, known_urls: &[String]) -> Response {
    let base = crate::oauth::derive_base_url(headers, known_urls);
    let body = serde_json::json!({"error": "unauthorized", "hint": hint});
    let mut resp = (StatusCode::UNAUTHORIZED, Json(body)).into_response();
    let www_auth =
        format!("Bearer resource_metadata=\"{base}/.well-known/oauth-protected-resource\"");
    resp.headers_mut()
        .insert("WWW-Authenticate", www_auth.parse().unwrap());
    resp
}

pub async fn resolve_credentials(
    headers: &HeaderMap,
    db: &dyn DbBackend,
    known_urls: &[String],
) -> Result<(String, String), Response> {
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if !auth.starts_with("Bearer ") {
        eprintln!("Auth: no Bearer token in request");
        return Err(unauthorized("Bearer token required", headers, known_urls));
    }

    let token = auth.trim_start_matches("Bearer ").trim();

    // Legacy format: token_id:token_secret
    if let Some((id, secret)) = token.split_once(':') {
        eprintln!("Auth: legacy token format (token_id:secret)");
        return Ok((id.to_string(), secret.to_string()));
    }

    // OAuth access token (from database)
    match db.get_access_token(token).await {
        Ok(Some((token_id, token_secret))) => {
            eprintln!("Auth: OAuth token resolved");
            return Ok((token_id, token_secret));
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("Auth: token lookup error: {e}");
        }
    }

    eprintln!("Auth: token not recognized");
    Err(unauthorized(
        "Invalid or expired token",
        headers,
        known_urls,
    ))
}

pub async fn handle_sse(State(state): State<AppState>, headers: HeaderMap) -> Response {
    eprintln!("GET /mcp/sse — SSE connection attempt");
    let (token_id, token_secret) =
        match resolve_credentials(&headers, state.db.as_ref(), &state.known_urls).await {
            Ok(creds) => creds,
            Err(resp) => return resp,
        };

    let client = BookStackClient::new(
        &state.bookstack_url,
        &token_id,
        &token_secret,
        state.http_client.clone(),
    );

    if let Err(e) = client.validate().await {
        eprintln!("Credential validation failed: {e}");
        return unauthorized(
            "BookStack credentials are invalid or expired — please re-authenticate",
            &headers,
            &state.known_urls,
        );
    }

    let session_id = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);

    {
        let mut sessions = state.sessions.write().await;

        if sessions.len() >= MAX_TOTAL_SESSIONS {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "Server at session capacity"})),
            )
                .into_response();
        }

        let count = sessions.values().filter(|s| s.token_id == token_id).count();
        if count >= MAX_SESSIONS_PER_TOKEN {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "Too many sessions for this token"})),
            )
                .into_response();
        }
        sessions.insert(
            session_id.clone(),
            Session {
                tx: tx.clone(),
                client,
                token_id: token_id.clone(),
                token_secret: token_secret.clone(),
                created_at: Instant::now(),
                rate_limit: Arc::new(Mutex::new(RateLimit::new(MAX_REQUESTS_PER_MINUTE))),
            },
        );
    }

    eprintln!("SSE session {session_id} created");

    let endpoint_url = format!("/mcp/messages/?sessionId={session_id}");
    let _ = tx
        .send(Ok(Event::default().event("endpoint").data(endpoint_url)))
        .await;

    let stream = ReceiverStream::new(rx);
    let mut resp = Sse::new(stream)
        .keep_alive(KeepAlive::default().interval(Duration::from_secs(15)))
        .into_response();

    let hdrs = resp.headers_mut();
    hdrs.insert(header::CACHE_CONTROL, "no-cache, no-store".parse().unwrap());
    hdrs.insert("X-Accel-Buffering", "no".parse().unwrap());

    resp
}

pub async fn handle_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
    body: String,
) -> Response {
    eprintln!("POST /mcp/messages/ — message request");
    let (token_id, token_secret) =
        match resolve_credentials(&headers, state.db.as_ref(), &state.known_urls).await {
            Ok(creds) => creds,
            Err(resp) => return resp,
        };

    let session_id = match params.get("sessionId") {
        Some(id) => id,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "missing sessionId"})),
            )
                .into_response()
        }
    };

    let (tx, client, rate_limit) = {
        let sessions = state.sessions.read().await;
        let session = match sessions.get(session_id) {
            Some(s) => s,
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({"error": "unknown session"})),
                )
                    .into_response()
            }
        };

        if !constant_time_eq(&session.token_id, &token_id)
            || !constant_time_eq(&session.token_secret, &token_secret)
        {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({"error": "token does not match session"})),
            )
                .into_response();
        }

        if session.created_at.elapsed() > SESSION_TTL {
            return (
                StatusCode::GONE,
                Json(serde_json::json!({"error": "session expired"})),
            )
                .into_response();
        }

        (
            session.tx.clone(),
            session.client.clone(),
            session.rate_limit.clone(),
        )
    };

    {
        let mut rl = rate_limit.lock().await;
        if rl.check().is_err() {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "Rate limit exceeded"})),
            )
                .into_response();
        }
    }

    let request: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid JSON"})),
            )
                .into_response()
        }
    };

    let semantic = state.semantic.as_deref();
    let response = mcp::handle_request(
        &request,
        &client,
        semantic,
        state.index_db.as_ref(),
        &state.staging,
        &state.timezone,
    )
    .await;

    if let Some(response) = response {
        let data = serde_json::to_string(&response).unwrap_or_default();
        if let Err(e) = tx.try_send(Ok(Event::default().event("message").data(data))) {
            eprintln!("SSE send failed for session {session_id}: {e}");
        }
    }

    StatusCode::ACCEPTED.into_response()
}

pub async fn handle_streamable(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    eprintln!("POST /mcp/sse — Streamable HTTP request");
    let (token_id, token_secret) =
        match resolve_credentials(&headers, state.db.as_ref(), &state.known_urls).await {
            Ok(creds) => creds,
            Err(resp) => return resp,
        };

    {
        let rate_limits = state.streamable_rate_limits.read().await;
        if let Some(rl) = rate_limits.get(&token_id) {
            let mut rl = rl.lock().await;
            if rl.check().is_err() {
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(serde_json::json!({"error": "Rate limit exceeded"})),
                )
                    .into_response();
            }
        } else {
            drop(rate_limits);
            let mut rate_limits = state.streamable_rate_limits.write().await;
            let rl = rate_limits
                .entry(token_id.clone())
                .or_insert_with(|| Arc::new(Mutex::new(RateLimit::new(MAX_REQUESTS_PER_MINUTE))));
            let mut rl = rl.lock().await;
            let _ = rl.check();
        }
    }

    let client = BookStackClient::new(
        &state.bookstack_url,
        &token_id,
        &token_secret,
        state.http_client.clone(),
    );

    let request: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid JSON"})),
            )
                .into_response()
        }
    };

    let method = request["method"].as_str().unwrap_or("");

    if method == "initialize" {
        if let Err(e) = client.validate().await {
            eprintln!("Streamable: credential validation failed: {e}");
            return unauthorized(
                "BookStack credentials are invalid or expired — please re-authenticate",
                &headers,
                &state.known_urls,
            );
        }
    }

    if request.get("id").is_none() {
        return StatusCode::ACCEPTED.into_response();
    }

    let semantic = state.semantic.as_deref();

    // Streamable HTTP 2025-03-26: the `Mcp-Session-Id` header is minted on
    // initialize (below) and echoed by the client on every subsequent
    // request. We surface the incoming id back on the response so clients
    // that resume across requests stay associated with their session row.
    let incoming_session_id = headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty());

    let response = mcp::handle_request(
        &request,
        &client,
        semantic,
        state.index_db.as_ref(),
        &state.staging,
        &state.timezone,
    )
    .await;

    match response {
        Some(resp) => {
            let mut http_resp = Json(resp).into_response();

            if method == "initialize" {
                let new_session_id = uuid::Uuid::new_v4().to_string();
                eprintln!("Streamable session {new_session_id} created");
                {
                    let mut ss = state.streamable_sessions.write().await;
                    ss.insert(new_session_id.clone(), Instant::now());
                }
                http_resp
                    .headers_mut()
                    .insert("Mcp-Session-Id", new_session_id.parse().unwrap());
            } else if let Some(ref sid) = incoming_session_id {
                let ss = state.streamable_sessions.read().await;
                if ss.contains_key(sid) {
                    http_resp
                        .headers_mut()
                        .insert("Mcp-Session-Id", sid.parse().unwrap());
                }
            }

            http_resp
        }
        None => StatusCode::ACCEPTED.into_response(),
    }
}
