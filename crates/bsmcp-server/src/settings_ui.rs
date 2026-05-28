//! Browser-based admin settings UI.
//!
//! v0.10.0: collapsed to admin-only global server config. Per-user
//! settings + briefing-only fields were stripped in #78.
//!
//! Auth flow:
//! 1. User visits GET /settings → no cookie → redirect to /authorize?return_to=/settings
//! 2. /authorize validates the BookStack token and (if return_to is /settings)
//!    creates a settings session, sets a cookie, redirects to /settings.
//! 3. /settings reads the cookie, looks up credentials, renders the form.
//! 4. POST /settings parses the form, persists, redirects.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::Form;
use serde::Deserialize;
use tokio::sync::RwLock;
use zeroize::Zeroize;

use bsmcp_common::settings::{hash_token_id, GlobalSettings};

use crate::sse::AppState;

pub const SETTINGS_COOKIE_NAME: &str = "bsmcp_settings_session";
pub const SETTINGS_SESSION_TTL: Duration = Duration::from_secs(8 * 60 * 60);

pub struct SettingsSession {
    pub token_id: String,
    pub token_secret: String,
    pub created_at: Instant,
}

impl Drop for SettingsSession {
    fn drop(&mut self) {
        self.token_id.zeroize();
        self.token_secret.zeroize();
    }
}

pub type SettingsSessionStore = Arc<RwLock<HashMap<String, SettingsSession>>>;

pub fn new_settings_store() -> SettingsSessionStore {
    Arc::new(RwLock::new(HashMap::new()))
}

pub fn spawn_settings_cleanup(store: SettingsSessionStore) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
            let mut sessions = store.write().await;
            sessions.retain(|_, s| s.created_at.elapsed() < SETTINGS_SESSION_TTL);
        }
    });
}

pub async fn issue_settings_session(
    store: &SettingsSessionStore,
    token_id: &str,
    token_secret: &str,
) -> String {
    let session_id = uuid::Uuid::new_v4().to_string();
    let mut sessions = store.write().await;
    if sessions.len() >= 1000 {
        sessions.retain(|_, s| s.created_at.elapsed() < SETTINGS_SESSION_TTL);
    }
    sessions.insert(
        session_id.clone(),
        SettingsSession {
            token_id: token_id.to_string(),
            token_secret: token_secret.to_string(),
            created_at: Instant::now(),
        },
    );
    session_id
}

pub fn build_session_cookie(session_id: &str) -> String {
    format!(
        "{name}={id}; Path=/settings; HttpOnly; SameSite=Lax; Max-Age={ttl}; Secure",
        name = SETTINGS_COOKIE_NAME,
        id = session_id,
        ttl = SETTINGS_SESSION_TTL.as_secs(),
    )
}

async fn resolve_session(
    headers: &HeaderMap,
    store: &SettingsSessionStore,
) -> Option<(String, String)> {
    let cookie_header = headers.get(header::COOKIE).and_then(|v| v.to_str().ok())?;
    let session_id = cookie_header
        .split(';')
        .map(str::trim)
        .find_map(|kv| kv.strip_prefix(&format!("{}=", SETTINGS_COOKIE_NAME)))?;

    let sessions = store.read().await;
    let session = sessions.get(session_id)?;
    if session.created_at.elapsed() >= SETTINGS_SESSION_TTL {
        return None;
    }
    Some((session.token_id.clone(), session.token_secret.clone()))
}

pub async fn has_valid_session(headers: &HeaderMap, store: &SettingsSessionStore) -> bool {
    resolve_session(headers, store).await.is_some()
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

// --- Handlers ---

pub async fn handle_settings_get(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if resolve_session(&headers, &state.settings_sessions)
        .await
        .is_none()
    {
        return Redirect::to("/authorize?response_type=code&client_id=settings-ui&redirect_uri=/settings&code_challenge=&code_challenge_method=&return_to=/settings").into_response();
    }
    let globals = state.db.get_global_settings().await.unwrap_or_default();
    Html(render_settings_page(&globals)).into_response()
}

#[derive(Deserialize, Default)]
pub struct SettingsForm {
    #[serde(default)]
    pub hive_shelf_id: Option<String>,
    #[serde(default)]
    pub user_journals_shelf_id: Option<String>,
}

pub async fn handle_settings_post(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<SettingsForm>,
) -> Response {
    let (token_id, token_secret) = match resolve_session(&headers, &state.settings_sessions).await {
        Some(creds) => creds,
        None => return Redirect::to("/authorize?response_type=code&client_id=settings-ui&redirect_uri=/settings&code_challenge=&code_challenge_method=&return_to=/settings").into_response(),
    };

    let token_id_hash = hash_token_id(&token_id);

    // Globals: only persist when the calling token is admin.
    let bs_client = bsmcp_common::bookstack::BookStackClient::new(
        &state.bookstack_url,
        &token_id,
        &token_secret,
        state.http_client.clone(),
    );
    let is_admin = bs_client.is_admin().await.unwrap_or(false);
    if is_admin {
        let mut globals = state.db.get_global_settings().await.unwrap_or_default();
        globals.hive_shelf_id = parse_optional_i64(&form.hive_shelf_id);
        globals.user_journals_shelf_id = parse_optional_i64(&form.user_journals_shelf_id);
        if let Err(e) = state
            .db
            .save_global_settings(&globals, &token_id_hash)
            .await
        {
            return error_response(format!("Failed to save global settings: {e}"));
        }
    }

    Redirect::to("/settings").into_response()
}

/// Probe handler retired in v0.8.0 — kept as a 410 stub so old links don't
/// 404 silently.
pub async fn handle_settings_probe_get(
    State(_state): State<AppState>,
    _headers: HeaderMap,
) -> Response {
    (
        StatusCode::GONE,
        Html("<p>Probe disabled. Use <a href=\"/settings\">/settings</a> directly.</p>"),
    )
        .into_response()
}

pub async fn handle_settings_probe_post(
    State(_state): State<AppState>,
    _headers: HeaderMap,
) -> Response {
    (StatusCode::GONE, "probe disabled").into_response()
}

// --- Form parsing helpers ---

fn parse_optional_i64(v: &Option<String>) -> Option<i64> {
    v.as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse().ok())
}

fn error_response(msg: String) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, Html(html_escape(&msg))).into_response()
}

// --- HTML rendering ---

fn render_settings_page(g: &GlobalSettings) -> String {
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BookStack MCP Settings</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1 {{ margin-bottom: 0.25em; }}
h2 {{ margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
.note {{ color: #666; font-size: 0.9em; }}
label {{ display: block; margin: 0.8em 0 0.3em; font-weight: 600; }}
input[type=number] {{ padding: 0.5em; box-sizing: border-box; font-family: inherit; width: 12em; }}
.help {{ font-size: 0.85em; color: #555; margin-top: 0.2em; }}
button {{ margin-top: 1.5em; padding: 0.6em 1.2em; font-size: 1em; cursor: pointer; }}
</style>
</head>
<body>
<h1>BookStack MCP Settings</h1>
<p class="note">v0.10.0 — admin-only global server config. Per-user and briefing fields were stripped in #78.</p>
<form method="post" action="/settings">
  <h2>Org-wide (admin-only)</h2>
  <p class="help">Non-admin saves silently drop every field below.</p>
  <label>hive_shelf_id <input type="number" name="hive_shelf_id" value="{hive_shelf_id}"></label>
  <p class="help">Identity-shelf id consumed by the index worker's full walk.</p>
  <label>user_journals_shelf_id <input type="number" name="user_journals_shelf_id" value="{user_journals_shelf_id}"></label>
  <p class="help">User-journals-shelf id consumed by the index worker's full walk.</p>
  <button type="submit">Save</button>
</form>
</body>
</html>"#,
        hive_shelf_id = g.hive_shelf_id.map(|i| i.to_string()).unwrap_or_default(),
        user_journals_shelf_id = g
            .user_journals_shelf_id
            .map(|i| i.to_string())
            .unwrap_or_default(),
    )
}
