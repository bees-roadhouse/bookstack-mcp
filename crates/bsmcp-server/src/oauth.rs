use std::env;
use std::time::{Duration, Instant};

use axum::extract::{Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::Form;
use axum::Json;
use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Value};
use zeroize::Zeroize;

use bsmcp_common::bookstack::BookStackClient;
use bsmcp_common::config::access_token_ttl;

use crate::sse::AppState;

pub const AUTH_CODE_TTL: Duration = Duration::from_secs(300); // 5 minutes

pub struct AuthCode {
    pub code_challenge: Option<String>,
    pub code_challenge_method: Option<String>,
    pub redirect_uri: String,
    pub client_id: String,
    pub token_id: Option<String>,
    pub token_secret: Option<String>,
    pub created_at: Instant,
}

impl Drop for AuthCode {
    fn drop(&mut self) {
        self.client_id.zeroize();
        if let Some(ref mut t) = self.token_id {
            t.zeroize();
        }
        if let Some(ref mut t) = self.token_secret {
            t.zeroize();
        }
    }
}

#[derive(Deserialize)]
pub struct AuthorizeParams {
    response_type: String,
    client_id: String,
    redirect_uri: String,
    state: Option<String>,
    code_challenge: Option<String>,
    code_challenge_method: Option<String>,
    /// When set to "/settings", short-circuits the OAuth code flow: after validating
    /// the BookStack token, the server issues a settings-session cookie and redirects
    /// the browser directly to /settings. Used by the in-server settings UI.
    return_to: Option<String>,
}

#[derive(Deserialize)]
pub struct AuthorizeFormSubmit {
    token_id: String,
    token_secret: String,
    response_type: String,
    client_id: String,
    redirect_uri: String,
    state: Option<String>,
    code_challenge: Option<String>,
    code_challenge_method: Option<String>,
    return_to: Option<String>,
}

impl Drop for AuthorizeFormSubmit {
    fn drop(&mut self) {
        self.token_id.zeroize();
        self.token_secret.zeroize();
    }
}

#[derive(Deserialize)]
pub struct TokenForm {
    grant_type: String,
    code: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    code_verifier: Option<String>,
    redirect_uri: Option<String>,
    refresh_token: Option<String>,
}

impl Drop for TokenForm {
    fn drop(&mut self) {
        if let Some(ref mut s) = self.client_secret {
            s.zeroize();
        }
        if let Some(ref mut v) = self.code_verifier {
            v.zeroize();
        }
        if let Some(ref mut r) = self.refresh_token {
            r.zeroize();
        }
    }
}

pub fn derive_base_url(headers: &HeaderMap, known_urls: &[String]) -> String {
    if !known_urls.is_empty() {
        let incoming_host = headers
            .get("host")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        for url in known_urls {
            if let Some(url_host) = extract_host_from_url(url) {
                if url_host == incoming_host {
                    return url.clone();
                }
            }
        }

        return known_urls[0].clone();
    }

    let scheme = headers
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .filter(|s| *s == "http" || *s == "https")
        .unwrap_or("https");
    let host = headers
        .get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("localhost");
    format!("{scheme}://{host}")
}

fn extract_host_from_url(url: &str) -> Option<&str> {
    let after_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    Some(after_scheme.split('/').next().unwrap_or(after_scheme))
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Returns true if the given path is a safe in-server redirect (no scheme/host).
fn is_safe_internal_path(s: &str) -> bool {
    s.starts_with('/') && !s.starts_with("//") && !s.contains(':')
}

fn render_login_form(params: &AuthorizeParams, bookstack_url: &str, error: Option<&str>) -> String {
    let instance_name = env::var("BSMCP_INSTANCE_NAME").unwrap_or_default();
    let title = if instance_name.is_empty() {
        "BookStack MCP".to_string()
    } else {
        html_escape(&instance_name)
    };

    let error_html = if let Some(msg) = error {
        format!(r#"<div class="error">{}</div>"#, html_escape(msg))
    } else {
        String::new()
    };

    let hidden = |name: &str, value: &str| -> String {
        format!(
            r#"<input type="hidden" name="{}" value="{}">"#,
            html_escape(name),
            html_escape(value)
        )
    };

    let mut hidden_fields = vec![
        hidden("response_type", &params.response_type),
        hidden("client_id", &params.client_id),
        hidden("redirect_uri", &params.redirect_uri),
    ];
    if let Some(ref s) = params.state {
        hidden_fields.push(hidden("state", s));
    }
    if let Some(ref c) = params.code_challenge {
        hidden_fields.push(hidden("code_challenge", c));
    }
    if let Some(ref m) = params.code_challenge_method {
        hidden_fields.push(hidden("code_challenge_method", m));
    }
    if let Some(ref r) = params.return_to {
        if is_safe_internal_path(r) {
            hidden_fields.push(hidden("return_to", r));
        }
    }

    let bs_url = html_escape(bookstack_url.trim_end_matches('/'));

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Connect — {title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }}
.card {{ background: #16213e; border-radius: 12px; padding: 2.5rem; width: 100%; max-width: 420px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }}
h1 {{ font-size: 1.4rem; margin-bottom: 0.3rem; color: #fff; }}
.subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }}
.error {{ background: #3d1f1f; border: 1px solid #c0392b; color: #e74c3c; padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem; font-size: 0.9rem; }}
label {{ display: block; font-size: 0.85rem; color: #aaa; margin-bottom: 0.3rem; }}
input[type="text"], input[type="password"] {{ width: 100%; padding: 0.7rem; border: 1px solid #2a3a5c; border-radius: 6px; background: #0f1a30; color: #e0e0e0; font-size: 0.95rem; margin-bottom: 1rem; }}
input:focus {{ outline: none; border-color: #3498db; }}
button {{ width: 100%; padding: 0.75rem; background: #2980b9; color: #fff; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; }}
button:hover {{ background: #3498db; }}
.steps {{ margin-top: 1.2rem; padding: 1rem; background: #0f1a30; border-radius: 8px; font-size: 0.82rem; color: #999; line-height: 1.6; }}
.steps ol {{ padding-left: 1.2rem; }}
.steps a {{ color: #3498db; text-decoration: none; }}
.steps a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="card">
  <h1>{title}</h1>
  <p class="subtitle">Enter your BookStack API token to connect Claude.</p>
  {error_html}
  <form method="POST" action="/authorize">
    {hidden_fields}
    <label for="token_id">Token ID</label>
    <input type="text" id="token_id" name="token_id" required autocomplete="off" placeholder="e.g. abc123...">
    <label for="token_secret">Token Secret</label>
    <input type="password" id="token_secret" name="token_secret" required autocomplete="off" placeholder="e.g. xyz789...">
    <button type="submit">Connect</button>
  </form>
  <div class="steps">
    <strong>How to create an API token:</strong>
    <ol>
      <li>Click your profile avatar (top-right) and select <a href="{bs_url}/my-account" target="_blank"><strong>My Account</strong></a></li>
      <li>Click <strong>Access &amp; Security</strong> in the left sidebar</li>
      <li>Scroll down to <strong>API Tokens</strong> and click <strong>Create Token</strong></li>
      <li>Give it a name (e.g. &ldquo;Claude&rdquo;) and save</li>
      <li><strong>Save the Token ID and Token Secret in your password manager</strong> &mdash; BookStack only shows the secret once</li>
      <li>Paste them into the fields above</li>
    </ol>
  </div>
</div>
</body>
</html>"##,
        title = title,
        error_html = error_html,
        bs_url = bs_url,
        hidden_fields = hidden_fields.join("\n    "),
    )
}

pub async fn handle_metadata(State(state): State<AppState>, headers: HeaderMap) -> Json<Value> {
    let base = derive_base_url(&headers, &state.known_urls);
    Json(json!({
        "issuer": base,
        "authorization_endpoint": format!("{base}/authorize"),
        "token_endpoint": format!("{base}/token"),
        "registration_endpoint": format!("{base}/register"),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post", "client_secret_basic"],
    }))
}

pub async fn handle_resource_metadata(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<Value> {
    let base = derive_base_url(&headers, &state.known_urls);
    Json(json!({
        "resource": base,
        "authorization_servers": [base],
        "bearer_methods_supported": ["header"],
    }))
}

pub async fn handle_authorize(
    State(state): State<AppState>,
    Query(params): Query<AuthorizeParams>,
) -> Response {
    // Browser settings flow bypasses the OAuth code dance entirely — no PKCE required,
    // no response_type check. The form renders, the user enters their token, and on
    // success the server sets a session cookie and redirects to the return_to path.
    let is_settings_flow = params
        .return_to
        .as_deref()
        .map(|p| p == "/settings")
        .unwrap_or(false);

    if !is_settings_flow {
        if params.response_type != "code" {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "unsupported_response_type",
                Some("Only response_type=code is supported"),
            );
        }
        if params.code_challenge.is_none() {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                Some("code_challenge is required (PKCE)"),
            );
        }
    }
    Html(render_login_form(&params, &state.bookstack_url, None)).into_response()
}

pub async fn handle_authorize_submit(
    State(state): State<AppState>,
    Form(form): Form<AuthorizeFormSubmit>,
) -> Response {
    {
        let mut rl = state.authorize_rate_limit.lock().await;
        if rl.check().is_err() {
            return oauth_error(
                StatusCode::TOO_MANY_REQUESTS,
                "invalid_request",
                Some("Too many authorization attempts, try again later"),
            );
        }
    }

    let is_settings_flow = form
        .return_to
        .as_deref()
        .map(|p| p == "/settings")
        .unwrap_or(false);

    if !is_settings_flow && form.response_type != "code" {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            "unsupported_response_type",
            Some("Only response_type=code is supported"),
        );
    }

    let bs_client = BookStackClient::new(
        &state.bookstack_url,
        &form.token_id,
        &form.token_secret,
        state.http_client.clone(),
    );
    if let Err(e) = bs_client.validate().await {
        tracing::warn!(error = %e, "oauth_credential_validation_failed");
        let params = AuthorizeParams {
            response_type: form.response_type.clone(),
            client_id: form.client_id.clone(),
            redirect_uri: form.redirect_uri.clone(),
            state: form.state.clone(),
            code_challenge: form.code_challenge.clone(),
            code_challenge_method: form.code_challenge_method.clone(),
            return_to: form.return_to.clone(),
        };
        return Html(render_login_form(
            &params,
            &state.bookstack_url,
            Some("Invalid API token. Check your Token ID and Secret."),
        ))
        .into_response();
    }

    // Browser settings flow: skip the OAuth code dance entirely. Issue a settings
    // session, set the cookie, and redirect to the return path.
    if is_settings_flow {
        let session_id = crate::settings_ui::issue_settings_session(
            &state.settings_sessions,
            &form.token_id,
            &form.token_secret,
        )
        .await;
        let cookie = crate::settings_ui::build_session_cookie(&session_id);
        tracing::info!("oauth_settings_session_issued");
        return (
            StatusCode::FOUND,
            [
                (header::SET_COOKIE, cookie),
                (header::LOCATION, "/settings".to_string()),
            ],
        )
            .into_response();
    }

    let code = uuid::Uuid::new_v4().to_string();
    let redirect_uri = form.redirect_uri.clone();
    let form_state = form.state.clone();

    {
        let mut codes = state.auth_codes.write().await;
        if codes.len() >= 100 {
            codes.retain(|_, c| c.created_at.elapsed() < AUTH_CODE_TTL);
        }
        codes.insert(
            code.clone(),
            AuthCode {
                client_id: form.client_id.clone(),
                code_challenge: form.code_challenge.clone(),
                code_challenge_method: form.code_challenge_method.clone(),
                redirect_uri: redirect_uri.clone(),
                token_id: Some(form.token_id.clone()),
                token_secret: Some(form.token_secret.clone()),
                created_at: Instant::now(),
            },
        );
    }

    let code_encoded = urlencoding::encode(&code);
    let mut redirect_url = if redirect_uri.contains('?') {
        format!("{}&code={code_encoded}", redirect_uri)
    } else {
        format!("{}?code={code_encoded}", redirect_uri)
    };
    if let Some(ref state_param) = form_state {
        let state_encoded = urlencoding::encode(state_param);
        redirect_url.push_str(&format!("&state={state_encoded}"));
    }

    tracing::info!("oauth_auth_code_issued");
    (StatusCode::FOUND, [(header::LOCATION, redirect_url)]).into_response()
}

pub async fn handle_token(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<TokenForm>,
) -> Response {
    match form.grant_type.as_str() {
        "authorization_code" => handle_token_authorization_code(state, headers, form).await,
        "refresh_token" => handle_token_refresh(state, form).await,
        _ => oauth_error(
            StatusCode::BAD_REQUEST,
            "unsupported_grant_type",
            Some("Supported grant types: authorization_code, refresh_token"),
        ),
    }
}

async fn handle_token_authorization_code(
    state: AppState,
    headers: HeaderMap,
    form: TokenForm,
) -> Response {
    let code = match &form.code {
        Some(c) => c.clone(),
        None => {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                Some("Missing code"),
            )
        }
    };

    let auth_code = {
        let mut codes = state.auth_codes.write().await;
        codes.remove(&code)
    };

    let auth_code = match auth_code {
        Some(c) if c.created_at.elapsed() < AUTH_CODE_TTL => c,
        _ => {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                Some("Invalid or expired authorization code"),
            )
        }
    };

    match &form.redirect_uri {
        Some(redirect_uri) if *redirect_uri == auth_code.redirect_uri => {}
        Some(_) => {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                Some("Redirect URI mismatch"),
            );
        }
        None => {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                Some("redirect_uri is required"),
            );
        }
    }

    if auth_code.code_challenge.is_none() {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            Some("Authorization was not issued with PKCE"),
        );
    }

    if let Some(ref challenge) = auth_code.code_challenge {
        let verifier = match &form.code_verifier {
            Some(v) => v,
            None => {
                return oauth_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    Some("Missing code_verifier"),
                )
            }
        };

        let method = auth_code.code_challenge_method.as_deref().unwrap_or("S256");
        if method != "S256" {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                Some("Unsupported code_challenge_method"),
            );
        }

        if compute_s256_challenge(verifier) != *challenge {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                Some("PKCE verification failed"),
            );
        }
    }

    let (token_id, token_secret) = if let (Some(tid), Some(tsec)) =
        (auth_code.token_id.clone(), auth_code.token_secret.clone())
    {
        tracing::debug!(flow = "form", "oauth_using_credentials");
        (tid, tsec)
    } else {
        let (client_id, client_secret) = match extract_client_credentials(&headers, &form) {
            Some(creds) => creds,
            None => {
                return oauth_error(
                    StatusCode::UNAUTHORIZED,
                    "invalid_client",
                    Some("No credentials available"),
                )
            }
        };

        if auth_code.client_id != client_id {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                Some("Client ID mismatch"),
            );
        }

        let bs_client = BookStackClient::new(
            &state.bookstack_url,
            &client_id,
            &client_secret,
            state.http_client.clone(),
        );
        if let Err(e) = bs_client.validate().await {
            tracing::warn!(error = %e, "oauth_bookstack_credential_validation_failed");
            return oauth_error(
                StatusCode::UNAUTHORIZED,
                "invalid_client",
                Some("Invalid BookStack credentials"),
            );
        }

        tracing::debug!(
            flow = "legacy_client_credentials",
            "oauth_using_credentials"
        );
        (client_id, client_secret)
    };

    issue_tokens(&state, &token_id, &token_secret).await
}

async fn handle_token_refresh(state: AppState, form: TokenForm) -> Response {
    let old_refresh = match &form.refresh_token {
        Some(t) => t.clone(),
        None => {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                Some("Missing refresh_token"),
            )
        }
    };

    // Look up the refresh token to get the stored BookStack credentials
    let (token_id, token_secret) = match state.db.get_refresh_token(&old_refresh).await {
        Ok(Some(creds)) => creds,
        Ok(None) => {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                Some("Invalid or expired refresh token. Please re-authenticate with your BookStack API credentials."),
            )
        }
        Err(e) => {
            tracing::error!(error = %e, "oauth_refresh_token_lookup_failed");
            return oauth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                Some("Failed to validate refresh token"),
            );
        }
    };

    // Validate the stored BookStack credentials are still valid
    let bs_client = BookStackClient::new(
        &state.bookstack_url,
        &token_id,
        &token_secret,
        state.http_client.clone(),
    );
    if let Err(e) = bs_client.validate().await {
        tracing::warn!(error = %e, "oauth_stored_credentials_invalid");
        // Consume the invalid refresh token
        state.db.delete_refresh_token(&old_refresh).await.ok();
        return oauth_error(
            StatusCode::UNAUTHORIZED,
            "invalid_grant",
            Some("BookStack API credentials are no longer valid. Please re-authenticate with new credentials."),
        );
    }

    // Consume the old refresh token (rotation)
    if let Err(e) = state.db.delete_refresh_token(&old_refresh).await {
        tracing::error!(error = %e, "oauth_delete_old_refresh_token_failed");
    }

    tracing::info!("oauth_token_refresh");
    issue_tokens(&state, &token_id, &token_secret).await
}

/// Issue a new access token + refresh token pair for the given BookStack credentials.
async fn issue_tokens(state: &AppState, token_id: &str, token_secret: &str) -> Response {
    let access_token = uuid::Uuid::new_v4().to_string();
    let refresh_token = uuid::Uuid::new_v4().to_string();
    let expires_in = access_token_ttl().as_secs();

    if let Err(e) = state
        .db
        .insert_access_token(&access_token, token_id, token_secret)
        .await
    {
        tracing::error!(error = %e, "oauth_persist_access_token_failed");
        return oauth_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            Some("Failed to persist access token"),
        );
    }

    if let Err(e) = state
        .db
        .insert_refresh_token(&refresh_token, token_id, token_secret)
        .await
    {
        tracing::error!(error = %e, "oauth_persist_refresh_token_failed");
        return oauth_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            Some("Failed to persist refresh token"),
        );
    }

    tracing::info!("oauth_tokens_issued");

    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(json!({
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": expires_in,
            "refresh_token": refresh_token,
        })),
    )
        .into_response()
}

fn extract_client_credentials(headers: &HeaderMap, form: &TokenForm) -> Option<(String, String)> {
    if let (Some(id), Some(secret)) = (&form.client_id, &form.client_secret) {
        if !id.is_empty() && !secret.is_empty() {
            return Some((id.clone(), secret.clone()));
        }
    }

    if let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        if let Some(basic) = auth.strip_prefix("Basic ") {
            if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(basic.trim()) {
                if let Ok(decoded_str) = String::from_utf8(decoded) {
                    if let Some((id, secret)) = decoded_str.split_once(':') {
                        return Some((id.to_string(), secret.to_string()));
                    }
                }
            }
        }
    }

    None
}

fn compute_s256_challenge(verifier: &str) -> String {
    use sha2::Digest;
    let hash = sha2::Sha256::digest(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hash)
}

pub async fn handle_register(State(state): State<AppState>, body: String) -> Response {
    {
        let mut rl = state.register_rate_limit.lock().await;
        if rl.check().is_err() {
            return oauth_error(
                StatusCode::TOO_MANY_REQUESTS,
                "invalid_request",
                Some("Too many registration attempts, try again later"),
            );
        }
    }

    let request: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_client_metadata",
                Some("Invalid JSON"),
            )
        }
    };

    let client_name = request
        .get("client_name")
        .and_then(|v| v.as_str())
        .unwrap_or("<unnamed>");
    tracing::info!(client_name = %client_name, "oauth_register_request");

    let client_id = uuid::Uuid::new_v4().to_string();

    let mut response = json!({
        "client_id": client_id,
        "client_id_issued_at": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    });

    if let Some(uris) = request.get("redirect_uris") {
        response["redirect_uris"] = uris.clone();
    }
    if let Some(name) = request.get("client_name") {
        response["client_name"] = name.clone();
    }
    if let Some(scope) = request.get("scope") {
        response["scope"] = scope.clone();
    }

    tracing::info!(client_id = %client_id, "oauth_client_registered");

    (
        StatusCode::CREATED,
        [(header::CONTENT_TYPE, "application/json")],
        Json(response),
    )
        .into_response()
}

fn oauth_error(status: StatusCode, error: &str, description: Option<&str>) -> Response {
    let mut body = json!({"error": error});
    if let Some(desc) = description {
        body["error_description"] = json!(desc);
    }
    (status, Json(body)).into_response()
}
