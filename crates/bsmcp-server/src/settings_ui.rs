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

use bsmcp_common::settings::{hash_token_id, CascadeMultipliers, GlobalSettings, ScopeDef};

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
    // Path=/ so the cookie is sent on every endpoint that admits a
    // settings-session cookie auth path. Originally scoped to `/settings`
    // which silently broke browser-session auth on `/status` (page never
    // renders) and `/cancel/...` redirects. The session id is opaque,
    // HttpOnly, and cap'd at SETTINGS_SESSION_TTL so the widened scope
    // doesn't change the threat model — same secret, more routes.
    format!(
        "{name}={id}; Path=/; HttpOnly; SameSite=Lax; Max-Age={ttl}; Secure",
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
    // Issue #80 — cascade multipliers + named scopes. Multipliers come in
    // as four separate number inputs; kb_scopes is a single JSON textarea
    // because the shape is open-ended (HashMap<String, ScopeDef>).
    #[serde(default)]
    pub cascade_stage1: Option<String>,
    #[serde(default)]
    pub cascade_stage2: Option<String>,
    #[serde(default)]
    pub cascade_stage3: Option<String>,
    #[serde(default)]
    pub cascade_stage4: Option<String>,
    #[serde(default)]
    pub kb_scopes_json: Option<String>,
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

        // Issue #80 — cascade multipliers. An empty input keeps the existing
        // value (so partial form posts don't accidentally reset the cascade
        // to defaults). Validation happens at the lib level via
        // `CascadeMultipliers::validated`.
        let mut new_cascade = globals.cascade_multipliers;
        if let Some(v) = parse_optional_u32(&form.cascade_stage1) {
            new_cascade.stage1 = v;
        }
        if let Some(v) = parse_optional_u32(&form.cascade_stage2) {
            new_cascade.stage2 = v;
        }
        if let Some(v) = parse_optional_u32(&form.cascade_stage3) {
            new_cascade.stage3 = v;
        }
        if let Some(v) = parse_optional_u32(&form.cascade_stage4) {
            new_cascade.stage4 = v;
        }
        globals.cascade_multipliers = new_cascade.validated();

        // Named scopes. Empty textarea = clear the map; otherwise the
        // textarea must parse as `HashMap<String, ScopeDef>` JSON. Reject
        // malformed input so a save can't silently drop the saved scopes.
        if let Some(raw) = form.kb_scopes_json.as_deref().map(str::trim) {
            if raw.is_empty() {
                globals.kb_scopes = std::collections::HashMap::new();
            } else {
                match serde_json::from_str::<std::collections::HashMap<String, ScopeDef>>(raw) {
                    Ok(map) => globals.kb_scopes = map,
                    Err(e) => {
                        return error_response(format!("Failed to parse kb_scopes JSON: {e}"));
                    }
                }
            }
        }

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

fn parse_optional_u32(v: &Option<String>) -> Option<u32> {
    v.as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|n| *n > 0)
}

fn error_response(msg: String) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, Html(html_escape(&msg))).into_response()
}

// --- HTML rendering ---

fn render_settings_page(g: &GlobalSettings) -> String {
    let defaults = CascadeMultipliers::default();
    let kb_scopes_json = if g.kb_scopes.is_empty() {
        String::new()
    } else {
        serde_json::to_string_pretty(&g.kb_scopes).unwrap_or_default()
    };
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BookStack MCP Settings</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 760px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1 {{ margin-bottom: 0.25em; }}
h2 {{ margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
.note {{ color: #666; font-size: 0.9em; }}
label {{ display: block; margin: 0.8em 0 0.3em; font-weight: 600; }}
input[type=number] {{ padding: 0.5em; box-sizing: border-box; font-family: inherit; width: 8em; }}
textarea {{ width: 100%; min-height: 8em; box-sizing: border-box; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.9em; padding: 0.5em; }}
.help {{ font-size: 0.85em; color: #555; margin-top: 0.2em; }}
.row {{ display: flex; gap: 1em; flex-wrap: wrap; }}
.row > div {{ flex: 1 1 8em; }}
button {{ margin-top: 1.5em; padding: 0.6em 1.2em; font-size: 1em; cursor: pointer; }}
code {{ background: #f3f3f3; padding: 0.1em 0.3em; border-radius: 3px; }}
</style>
</head>
<body>
<h1>BookStack MCP Settings</h1>
<p class="note">Admin-only global server config. Non-admin saves silently drop every field below.</p>
<form method="post" action="/settings">
  <h2>Index worker</h2>
  <label>hive_shelf_id <input type="number" name="hive_shelf_id" value="{hive_shelf_id}"></label>
  <p class="help">Identity-shelf id consumed by the index worker's full walk.</p>
  <label>user_journals_shelf_id <input type="number" name="user_journals_shelf_id" value="{user_journals_shelf_id}"></label>
  <p class="help">User-journals-shelf id consumed by the index worker's full walk.</p>

  <h2>Semantic search — precision cascade (issue #80)</h2>
  <p class="help">
    Per-stage pool-size multipliers for the precision-mode cascade. The caller's
    <code>limit</code> (<em>N</em>) is multiplied at each input boundary: stage 1 sees
    <em>N × stage1</em> candidates, narrowed to <em>N × stage2</em> by the keyword stage,
    then <em>N × stage3</em> by the blanket + ACL stage, then <em>N × stage4</em> (= N when
    stage4 is 1) by the cross-encoder. Must form a non-increasing cascade
    (stage1 ≥ stage2 ≥ stage3 ≥ stage4 ≥ 1) — invalid configs fall back to defaults.
    Boot-time env overrides: <code>BSMCP_CASCADE_STAGE{{1,2,3,4}}_MULT</code>.
  </p>
  <div class="row">
    <div><label>stage1 <input type="number" name="cascade_stage1" min="1" value="{cascade_stage1}" placeholder="{cascade_default_stage1}"></label></div>
    <div><label>stage2 <input type="number" name="cascade_stage2" min="1" value="{cascade_stage2}" placeholder="{cascade_default_stage2}"></label></div>
    <div><label>stage3 <input type="number" name="cascade_stage3" min="1" value="{cascade_stage3}" placeholder="{cascade_default_stage3}"></label></div>
    <div><label>stage4 <input type="number" name="cascade_stage4" min="1" value="{cascade_stage4}" placeholder="{cascade_default_stage4}"></label></div>
  </div>
  <p class="help">Defaults: {cascade_default_stage1} / {cascade_default_stage2} / {cascade_default_stage3} / {cascade_default_stage4}. Blank input keeps the current value.</p>

  <h2>Semantic search — named scopes (kb_scopes)</h2>
  <p class="help">
    Named scope cuts that callers can address by name via
    <code>semantic_search</code>'s <code>scopes</code> array. JSON map of
    <code>name → {{ shelf_ids?, book_ids?, chapter_ids?, page_ids? }}</code>.
    Empty textarea clears the map. Example:
    <code>{{"policies": {{"book_ids": [12, 34]}}, "sops": {{"shelf_ids": [5]}}}}</code>
  </p>
  <label>kb_scopes JSON</label>
  <textarea name="kb_scopes_json" placeholder='{{"policies": {{"book_ids": [12, 34]}}}}'>{kb_scopes_json}</textarea>

  <button type="submit">Save</button>
</form>
</body>
</html>"#,
        hive_shelf_id = g.hive_shelf_id.map(|i| i.to_string()).unwrap_or_default(),
        user_journals_shelf_id = g
            .user_journals_shelf_id
            .map(|i| i.to_string())
            .unwrap_or_default(),
        cascade_stage1 = g.cascade_multipliers.stage1,
        cascade_stage2 = g.cascade_multipliers.stage2,
        cascade_stage3 = g.cascade_multipliers.stage3,
        cascade_stage4 = g.cascade_multipliers.stage4,
        cascade_default_stage1 = defaults.stage1,
        cascade_default_stage2 = defaults.stage2,
        cascade_default_stage3 = defaults.stage3,
        cascade_default_stage4 = defaults.stage4,
        kb_scopes_json = html_escape(&kb_scopes_json),
    )
}
