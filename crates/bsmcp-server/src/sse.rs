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

use bsmcp_common::bookstack::{BookStackClient, CredentialCheck};
use bsmcp_common::db::DbBackend;
use bsmcp_common::time::TimezoneConfig;

use crate::mcp;
use crate::oauth::{AuthCode, AUTH_CODE_TTL};
use crate::semantic::SemanticState;

/// Default per-token cap on concurrent legacy SSE sessions. Overridable at
/// startup via `BSMCP_MAX_SESSIONS_PER_TOKEN` (issue #138) — a workstation
/// running two MCP clients against one BookStack API token outgrows 5.
const DEFAULT_MAX_SESSIONS_PER_TOKEN: usize = 5;
const MAX_TOTAL_SESSIONS: usize = 1000;
const SESSION_TTL: Duration = Duration::from_secs(24 * 60 * 60); // 24 hours
const MAX_REQUESTS_PER_MINUTE: u32 = 100;
/// `Retry-After` on a 503 when an upstream is unreachable. Long enough to
/// cover a container restart, short enough that clients don't stall.
pub(crate) const RETRY_AFTER_SECS: &str = "5";

/// Resolve the per-token legacy-SSE session cap from
/// `BSMCP_MAX_SESSIONS_PER_TOKEN`, falling back to
/// [`DEFAULT_MAX_SESSIONS_PER_TOKEN`] when the value is absent, unparseable,
/// or zero. A typo in the env should not lock every client out of the server.
pub fn max_sessions_per_token_from_env() -> usize {
    parse_max_sessions_per_token(
        std::env::var("BSMCP_MAX_SESSIONS_PER_TOKEN")
            .ok()
            .as_deref(),
    )
}

/// Split out from the env read so the fallback rules stay unit-testable
/// without mutating process-global state mid-test-run.
fn parse_max_sessions_per_token(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_SESSIONS_PER_TOKEN)
}

/// Streamable HTTP (MCP 2025-03-26) clients open the server→client
/// notification stream with a GET on the *same* endpoint they POST to,
/// carrying the `Mcp-Session-Id` they were issued on `initialize`. A legacy
/// SSE client (MCP 2024-11-05) never sends that header — it learns its
/// session id from the `endpoint` event the handshake pushes back. The
/// header is therefore a reliable discriminator between the two transports
/// sharing the `/mcp/sse` route.
fn is_streamable_notification_stream(headers: &HeaderMap) -> bool {
    headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|s| !s.trim().is_empty())
}

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
    /// Per-token cap on concurrent *legacy* SSE sessions, resolved once at
    /// startup from `BSMCP_MAX_SESSIONS_PER_TOKEN` (issue #138). Streamable
    /// notification streams are deliberately not counted against it.
    max_sessions_per_token: usize,
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
        max_sessions_per_token: usize,
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
            max_sessions_per_token,
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
                            tracing::debug!(session_id = %sid, "session_cleaned_up");
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
                    tracing::error!(error = %e, "token_cleanup_error");
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
        tracing::info!(
            interval_hours,
            path = %backup_path.display(),
            "backup_enabled"
        );
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                match db.backup(&backup_path).await {
                    Ok(()) => tracing::info!("backup_completed"),
                    Err(e) => tracing::error!(error = %e, "backup_failed"),
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

/// BookStack couldn't answer. Deliberately *not* a 401: no
/// `WWW-Authenticate`, so clients hold onto their token and retry instead of
/// tearing down and re-running OAuth (#139).
fn upstream_unavailable(hint: &str) -> Response {
    let body = serde_json::json!({"error": "upstream_unavailable", "hint": hint});
    let mut resp = (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
    resp.headers_mut()
        .insert(header::RETRY_AFTER, RETRY_AFTER_SECS.parse().unwrap());
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
        tracing::debug!("auth_no_bearer");
        return Err(unauthorized("Bearer token required", headers, known_urls));
    }

    let token = auth.trim_start_matches("Bearer ").trim();

    // Legacy format: token_id:token_secret
    if let Some((id, secret)) = token.split_once(':') {
        tracing::debug!(scheme = "legacy", "auth_token_resolved");
        return Ok((id.to_string(), secret.to_string()));
    }

    // OAuth access token (from database)
    match db.get_access_token(token).await {
        Ok(Some((token_id, token_secret))) => {
            tracing::debug!(scheme = "oauth", "auth_token_resolved");
            return Ok((token_id, token_secret));
        }
        Ok(None) => {}
        Err(e) => {
            // Same trap as #139, one layer down: a DB blip says nothing
            // about the token. Falling through to the 401 below would tell
            // every connected client at once that its token is invalid and
            // stampede them all into re-auth.
            tracing::error!(error = %e, "auth_token_lookup_error");
            return Err(upstream_unavailable(
                "Token store is unavailable — retry shortly",
            ));
        }
    }

    tracing::debug!("auth_token_not_recognized");
    Err(unauthorized(
        "Invalid or expired token",
        headers,
        known_urls,
    ))
}

pub async fn handle_sse(State(state): State<AppState>, headers: HeaderMap) -> Response {
    tracing::debug!(method = "GET", route = "/mcp/sse", "sse_connect_attempt");
    let (token_id, token_secret) =
        match resolve_credentials(&headers, state.db.as_ref(), &state.known_urls).await {
            Ok(creds) => creds,
            Err(resp) => return resp,
        };

    // Issue #138: a streamable-HTTP client's notification GET is not a legacy
    // handshake. Allocating a `Session` for it burned a slot in the per-token
    // cap, so two MCP clients sharing one BookStack token exhausted all five
    // and every later GET got a 429 — cosmetic in the client log, but it also
    // meant a wasted `validate()` round-trip against BookStack per reconnect.
    //
    // Serve an inert stream instead. The spec permits answering this GET with
    // 405 as well, but rmcp's handling of that has drifted across versions and
    // a 405 can push a client into legacy-SSE *fallback* — allocating exactly
    // the session we are declining here. This server pushes no server-initiated
    // messages, so the stream carries keepalive comments and nothing else, and
    // it ends when the client goes away.
    //
    // The incoming id is deliberately not checked against `streamable_sessions`:
    // that map is empty after a restart while clients still hold ids, and
    // rejecting them would trade the 429 for a 404. `resolve_credentials` above
    // is the real gate; the header is only a transport discriminator.
    if is_streamable_notification_stream(&headers) {
        tracing::debug!("streamable_notification_stream_opened");
        return notification_stream_response();
    }

    let client = BookStackClient::new(
        &state.bookstack_url,
        &token_id,
        &token_secret,
        state.http_client.clone(),
    );

    match client.validate().await {
        CredentialCheck::Valid => {}
        CredentialCheck::Rejected(e) => {
            tracing::warn!(error = %e, "credential_validation_rejected");
            return unauthorized(
                "BookStack credentials are invalid or expired — please re-authenticate",
                &headers,
                &state.known_urls,
            );
        }
        CredentialCheck::Unavailable(e) => {
            tracing::warn!(error = %e, "credential_validation_unavailable");
            return upstream_unavailable("BookStack is unreachable — retry shortly");
        }
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
        if count >= state.max_sessions_per_token {
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

    tracing::info!(session_id = %session_id, "sse_session_created");

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

/// Inert `text/event-stream` for a streamable-HTTP client's notification
/// channel. Allocates no [`Session`] and counts against no cap: the server has
/// no server→client messages to push, so the stream carries only the keepalive
/// comments `KeepAlive` injects. `futures::stream::pending()` never yields, so
/// the response ends when the client disconnects and the keepalive write fails.
fn notification_stream_response() -> Response {
    let stream = futures::stream::pending::<Result<Event, Infallible>>();
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
    tracing::debug!(
        method = "POST",
        route = "/mcp/messages/",
        "sse_message_request"
    );
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

    let semantic = state.semantic.as_ref();
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
            tracing::warn!(session_id = %session_id, error = %e, "sse_send_failed");
        }
    }

    StatusCode::ACCEPTED.into_response()
}

pub async fn handle_streamable(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    tracing::debug!(method = "POST", route = "/mcp/sse", "streamable_request");
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
        match client.validate().await {
            CredentialCheck::Valid => {}
            CredentialCheck::Rejected(e) => {
                tracing::warn!(error = %e, "streamable_credential_validation_rejected");
                return unauthorized(
                    "BookStack credentials are invalid or expired — please re-authenticate",
                    &headers,
                    &state.known_urls,
                );
            }
            CredentialCheck::Unavailable(e) => {
                tracing::warn!(error = %e, "streamable_credential_validation_unavailable");
                return upstream_unavailable("BookStack is unreachable — retry shortly");
            }
        }
    }

    if request.get("id").is_none() {
        return StatusCode::ACCEPTED.into_response();
    }

    let semantic = state.semantic.as_ref();

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
                tracing::info!(session_id = %new_session_id, "streamable_session_created");
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

#[cfg(test)]
mod tests {
    use super::*;

    fn headers_with_session(value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("mcp-session-id", value.parse().unwrap());
        h
    }

    // Issue #138: this predicate is what keeps a streamable client's
    // notification GET from consuming a legacy per-token session slot.
    #[test]
    fn streamable_get_is_detected_by_session_header() {
        let h = headers_with_session("8acb2c83-0d1e-4f4a-9d21-1b2c3d4e5f60");
        assert!(is_streamable_notification_stream(&h));
    }

    #[test]
    fn legacy_sse_handshake_sends_no_session_header() {
        assert!(!is_streamable_notification_stream(&HeaderMap::new()));
    }

    #[test]
    fn blank_session_header_falls_through_to_legacy() {
        // An empty or whitespace-only header is not a resumable streamable
        // session, so it must take the legacy path rather than silently
        // opening an inert stream a legacy client would wait on forever.
        assert!(!is_streamable_notification_stream(&headers_with_session(
            ""
        )));
        assert!(!is_streamable_notification_stream(&headers_with_session(
            "   "
        )));
    }

    #[test]
    fn session_cap_defaults_when_unset() {
        assert_eq!(
            parse_max_sessions_per_token(None),
            DEFAULT_MAX_SESSIONS_PER_TOKEN
        );
    }

    #[test]
    fn session_cap_reads_a_valid_override() {
        assert_eq!(parse_max_sessions_per_token(Some("25")), 25);
        assert_eq!(parse_max_sessions_per_token(Some("  12  ")), 12);
    }

    #[test]
    fn session_cap_rejects_zero_and_garbage() {
        // A bad value must not lock every client out of the server.
        for raw in ["0", "-1", "many", ""] {
            assert_eq!(
                parse_max_sessions_per_token(Some(raw)),
                DEFAULT_MAX_SESSIONS_PER_TOKEN,
                "raw = {raw:?}"
            );
        }
    }
}
