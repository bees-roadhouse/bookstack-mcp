use reqwest::Client;
use serde_json::Value;
use std::net::IpAddr;
use std::time::Duration;
use url::Url;
use zeroize::Zeroize;

use crate::rate_limit::{self, RateLimiter};

/// Maximum size for file content fetched from URLs (50MB).
const MAX_FILE_CONTENT_SIZE: usize = 50 * 1024 * 1024;

/// Check if an IP address is in a private, loopback, or link-local range.
fn is_restricted_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()             // 127.0.0.0/8
            || v4.is_private()           // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
            || v4.is_link_local()        // 169.254.0.0/16 (AWS IMDS, etc.)
            || v4.is_broadcast()         // 255.255.255.255
            || v4.is_unspecified()       // 0.0.0.0
            || v4.octets()[0] == 100 && (v4.octets()[1] & 0xC0) == 64 // 100.64.0.0/10 (CGN)
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()             // ::1
            || v6.is_unspecified()       // ::
            || (v6.segments()[0] & 0xffc0) == 0xfe80  // fe80::/10 link-local
            || (v6.segments()[0] & 0xfe00) == 0xfc00 // fc00::/7 ULA
        }
    }
}

/// Resolve file content from either a local file path or a URL.
/// Exactly one of file_path or url must be provided.
/// Returns (bytes, filename).
pub async fn resolve_file_content(
    file_path: Option<&str>,
    url: Option<&str>,
) -> Result<(Vec<u8>, String), String> {
    match (file_path, url) {
        (Some(path), None) => {
            let bytes = tokio::fs::read(path)
                .await
                .map_err(|e| format!("Failed to read file '{}': {}", path, e))?;
            let filename = std::path::Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("file")
                .to_string();
            Ok((bytes, filename))
        }
        (None, Some(url)) => {
            let parsed = Url::parse(url).map_err(|e| format!("Invalid URL '{}': {}", url, e))?;

            // Only http and https schemes are permitted.
            match parsed.scheme() {
                "http" | "https" => {}
                scheme => {
                    return Err(format!(
                        "URL scheme '{}' is not allowed; only http and https are permitted",
                        scheme
                    ))
                }
            }

            // Resolve hostname, reject private/loopback/link-local IPs, then pin the
            // validated addresses into the reqwest client to prevent DNS rebinding.
            let host = parsed
                .host_str()
                .ok_or_else(|| format!("URL '{}' has no host", url))?;
            let port = parsed.port_or_known_default().unwrap_or(80);
            let addrs: Vec<std::net::SocketAddr> =
                tokio::net::lookup_host(format!("{}:{}", host, port))
                    .await
                    .map_err(|e| format!("Failed to resolve host '{}': {}", host, e))?
                    .collect();
            if addrs.is_empty() {
                return Err(format!("Host '{}' resolved to no addresses", host));
            }
            for addr in &addrs {
                if is_restricted_ip(&addr.ip()) {
                    return Err(format!("URL host '{}' resolves to restricted IP address {}; private, loopback, and link-local addresses are not allowed", host, addr.ip()));
                }
            }

            // Pin validated addresses into the client so reqwest uses them directly
            // instead of re-resolving DNS (prevents DNS rebinding attacks).
            let mut client_builder = reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .timeout(std::time::Duration::from_secs(120));
            for addr in &addrs {
                client_builder = client_builder.resolve(host, *addr);
            }
            let client = client_builder
                .build()
                .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

            let resp = client
                .get(url)
                .send()
                .await
                .map_err(|e| format!("Failed to fetch URL '{}': {}", url, e))?;
            if !resp.status().is_success() {
                return Err(format!("URL returned status {}", resp.status()));
            }

            // Fast-reject via Content-Length before downloading the body.
            if let Some(len) = resp.content_length() {
                if len as usize > MAX_FILE_CONTENT_SIZE {
                    return Err(format!(
                        "Remote file too large: {} bytes (limit {})",
                        len, MAX_FILE_CONTENT_SIZE
                    ));
                }
            }

            let bytes = resp
                .bytes()
                .await
                .map_err(|e| format!("Failed to read URL response: {}", e))?;

            if bytes.len() > MAX_FILE_CONTENT_SIZE {
                return Err(format!(
                    "Remote file too large: {} bytes (limit {})",
                    bytes.len(),
                    MAX_FILE_CONTENT_SIZE
                ));
            }

            let filename = url
                .rsplit('/')
                .next()
                .and_then(|s| s.split('?').next())
                .filter(|s| !s.is_empty())
                .unwrap_or("download")
                .to_string();
            Ok((bytes.to_vec(), filename))
        }
        (Some(_), Some(_)) => Err("Provide either file_path or url, not both".to_string()),
        (None, None) => Err("Either file_path or url is required".to_string()),
    }
}

// --- Type-safe enums for URL path parameters (defense-in-depth) ---

pub enum ExportFormat {
    Markdown,
    Plaintext,
    Html,
}

impl ExportFormat {
    pub fn parse_str(s: &str) -> Result<Self, String> {
        match s {
            "markdown" => Ok(Self::Markdown),
            "plaintext" => Ok(Self::Plaintext),
            "html" => Ok(Self::Html),
            _ => Err(format!(
                "Invalid export format: '{s}'. Must be one of: markdown, plaintext, html"
            )),
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Markdown => "markdown",
            Self::Plaintext => "plaintext",
            Self::Html => "html",
        }
    }
}

pub enum ContentType {
    Page,
    Chapter,
    Book,
    Shelf,
}

impl ContentType {
    pub fn parse_str(s: &str) -> Result<Self, String> {
        match s {
            "page" => Ok(Self::Page),
            "chapter" => Ok(Self::Chapter),
            "book" => Ok(Self::Book),
            "shelf" => Ok(Self::Shelf),
            _ => Err(format!(
                "Invalid content type: '{s}'. Must be one of: page, chapter, book, shelf"
            )),
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Page => "page",
            Self::Chapter => "chapter",
            Self::Book => "book",
            Self::Shelf => "shelf",
        }
    }
}

const MAX_RESPONSE_SIZE: usize = 50 * 1024 * 1024; // 50MB
const MAX_ERROR_BODY_SIZE: usize = 4096; // 4KB for error messages

/// Note: Zeroize on Drop clears the current String allocation. Intermediate copies
/// (e.g. from Clone, format!, auth_header()) and reqwest HeaderValue copies may remain
/// in freed memory until overwritten by the allocator.
/// This is a best-effort defense-in-depth measure, not a guarantee against memory forensics.
/// What `BookStackClient::whoami()` returns when it can identify the
/// authenticated user. `email` is `None` only when BookStack returns a user
/// row without one (rare — typically only seeded service accounts).
#[derive(Clone, Debug)]
pub struct UserIdentity {
    pub bookstack_user_id: i64,
    pub email: Option<String>,
    pub name: String,
}

/// Outcome of `BookStackClient::validate()`. See that method for why the
/// two failure arms must stay distinct.
#[derive(Clone, Debug)]
pub enum CredentialCheck {
    /// BookStack accepted the credentials.
    Valid,
    /// BookStack answered 401/403 — the credentials are genuinely bad.
    /// Re-authenticating is the fix.
    Rejected(String),
    /// BookStack could not be reached or could not answer. Says nothing
    /// about the credentials; the caller should retry, not re-authenticate.
    Unavailable(String),
}

impl CredentialCheck {
    /// Classify a non-success status from BookStack.
    ///
    /// Only 401/403 — BookStack answering "no" — is a credential problem.
    /// Everything else means we didn't get a usable answer: 5xx from a
    /// restarting instance or the proxy in front of it, 429, or a 404 from a
    /// misconfigured `BSMCP_BOOKSTACK_URL`. Re-authenticating fixes none of
    /// those, so they must not be reported as auth failures (#139).
    pub fn from_error_status(status: reqwest::StatusCode) -> Self {
        match status {
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                CredentialCheck::Rejected(format!("BookStack rejected the credentials: {status}"))
            }
            _ => CredentialCheck::Unavailable(format!("BookStack API error: {status}")),
        }
    }
}

#[derive(Clone)]
pub struct BookStackClient {
    client: Client,
    base_url: String,
    token_id: String,
    token_secret: String,
    rate_limiter: RateLimiter,
}

impl Drop for BookStackClient {
    fn drop(&mut self) {
        self.token_id.zeroize();
        self.token_secret.zeroize();
    }
}

impl BookStackClient {
    pub fn new(base_url: &str, token_id: &str, token_secret: &str, client: Client) -> Self {
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            token_id: token_id.to_string(),
            token_secret: token_secret.to_string(),
            rate_limiter: rate_limit::shared(),
        }
    }

    /// Get the base URL of the BookStack instance.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get the token ID (for use as a cache key, not a secret).
    pub fn token_id(&self) -> &str {
        &self.token_id
    }

    fn auth_header(&self) -> String {
        format!("Token {}:{}", self.token_id, self.token_secret)
    }

    /// Fast-reject via Content-Length header before downloading the body.
    fn check_content_length(resp: &reqwest::Response, limit: usize) -> Result<(), String> {
        if let Some(len) = resp.content_length() {
            if len as usize > limit {
                return Err(format!("Response too large: {len} bytes"));
            }
        }
        Ok(())
    }

    /// Read response as JSON, enforcing size limit even for chunked responses.
    async fn read_json(resp: reqwest::Response) -> Result<Value, String> {
        Self::check_content_length(&resp, MAX_RESPONSE_SIZE)?;
        let bytes = resp.bytes().await.map_err(|e| {
            tracing::error!(error = %e, "bookstack_response_read_error");
            "Failed to read response".to_string()
        })?;
        if bytes.len() > MAX_RESPONSE_SIZE {
            return Err(format!("Response too large: {} bytes", bytes.len()));
        }
        serde_json::from_slice(&bytes).map_err(|e| {
            tracing::error!(error = %e, "bookstack_json_parse_error");
            "Invalid response from BookStack".to_string()
        })
    }

    /// Read response as text, enforcing size limit even for chunked responses.
    async fn read_text(resp: reqwest::Response) -> Result<String, String> {
        Self::check_content_length(&resp, MAX_RESPONSE_SIZE)?;
        let bytes = resp.bytes().await.map_err(|e| {
            tracing::error!(error = %e, "bookstack_response_read_error");
            "Failed to read response".to_string()
        })?;
        if bytes.len() > MAX_RESPONSE_SIZE {
            return Err(format!("Response too large: {} bytes", bytes.len()));
        }
        String::from_utf8(bytes.to_vec()).map_err(|e| {
            tracing::error!(error = %e, "bookstack_utf8_decode_error");
            "Invalid response encoding".to_string()
        })
    }

    /// Read error body with a size limit to prevent memory exhaustion from error responses.
    /// Streams chunks to avoid buffering arbitrarily large error responses.
    async fn read_error_body(mut resp: reqwest::Response) -> String {
        // Fast-reject if Content-Length exceeds limit
        if resp
            .content_length()
            .is_some_and(|len| len as usize > MAX_ERROR_BODY_SIZE)
        {
            return "[error body too large]".to_string();
        }
        let mut buf = Vec::with_capacity(MAX_ERROR_BODY_SIZE.min(4096));
        while buf.len() < MAX_ERROR_BODY_SIZE {
            match resp.chunk().await {
                Ok(Some(chunk)) => {
                    let remaining = MAX_ERROR_BODY_SIZE - buf.len();
                    buf.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
                }
                _ => break,
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    /// Cap on 429 retries before surfacing the error to the caller.
    const RETRY_429_MAX_ATTEMPTS: u32 = 4;

    async fn send_with_retry(
        &self,
        builder: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response, String> {
        for attempt in 0..Self::RETRY_429_MAX_ATTEMPTS {
            self.rate_limiter.acquire().await;
            let req = builder
                .try_clone()
                .ok_or_else(|| "non-cloneable request".to_string())?;
            let resp = req.send().await.map_err(|e| {
                tracing::error!(error = %e, "bookstack_request_error");
                "Request failed".to_string()
            })?;
            if resp.status().as_u16() == 429 {
                // Jitter both the parsed Retry-After and the exponential
                // fallback. When the embedder + worker share a token they
                // would otherwise wake on the same millisecond after a
                // synchronized Retry-After=N and stampede BookStack right
                // back into 429. 0–500ms uniform jitter is enough to
                // desync them without meaningfully extending the retry.
                let base = resp
                    .headers()
                    .get("Retry-After")
                    .and_then(|v| v.to_str().ok())
                    .and_then(rate_limit::parse_retry_after)
                    .unwrap_or_else(|| Duration::from_millis(500 * 2u64.pow(attempt)));
                let delay = base + rate_limit::jitter(500);
                if attempt + 1 == Self::RETRY_429_MAX_ATTEMPTS {
                    let status = resp.status();
                    let body = Self::read_error_body(resp).await;
                    tracing::error!(
                        body = %body,
                        "bookstack_429_retry_exhausted"
                    );
                    return Err(format!("BookStack API error: {status}"));
                }
                tracing::warn!(
                    attempt = attempt + 1,
                    max_attempts = Self::RETRY_429_MAX_ATTEMPTS - 1,
                    delay_ms = delay.as_millis() as u64,
                    "bookstack_429_retry"
                );
                tokio::time::sleep(delay).await;
                continue;
            }
            self.rate_limiter.observe_limit(resp.headers());
            return Ok(resp);
        }
        Err("BookStack API error: 429".to_string())
    }

    async fn get(&self, path: &str, query: &[(&str, &str)]) -> Result<Value, String> {
        let builder = self
            .client
            .get(format!("{}/api/{}", self.base_url, path))
            .header("Authorization", self.auth_header())
            .query(query);
        let resp = self.send_with_retry(builder).await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = Self::read_error_body(resp).await;
            tracing::error!(%status, body = %body, "bookstack_api_error");
            return Err(format!("BookStack API error: {status}"));
        }
        Self::read_json(resp).await
    }

    async fn post(&self, path: &str, body: &Value) -> Result<Value, String> {
        let builder = self
            .client
            .post(format!("{}/api/{}", self.base_url, path))
            .header("Authorization", self.auth_header())
            .json(body);
        let resp = self.send_with_retry(builder).await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = Self::read_error_body(resp).await;
            tracing::error!(%status, body = %body, "bookstack_api_error");
            return Err(format!("BookStack API error: {status}"));
        }
        Self::read_json(resp).await
    }

    async fn post_multipart(
        &self,
        path: &str,
        form: reqwest::multipart::Form,
    ) -> Result<Value, String> {
        // Multipart bodies stream and aren't `try_clone`-safe; skip retry.
        self.rate_limiter.acquire().await;
        let resp = self
            .client
            .post(format!("{}/api/{}", self.base_url, path))
            .header("Authorization", self.auth_header())
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "bookstack_request_error");
                "Request failed".to_string()
            })?;
        self.rate_limiter.observe_limit(resp.headers());

        if !resp.status().is_success() {
            let status = resp.status();
            let body = Self::read_error_body(resp).await;
            tracing::error!(%status, body = %body, "bookstack_api_error");
            return Err(format!("BookStack API error: {status}"));
        }

        Self::read_json(resp).await
    }

    async fn put(&self, path: &str, body: &Value) -> Result<Value, String> {
        let builder = self
            .client
            .put(format!("{}/api/{}", self.base_url, path))
            .header("Authorization", self.auth_header())
            .json(body);
        let resp = self.send_with_retry(builder).await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = Self::read_error_body(resp).await;
            tracing::error!(%status, body = %body, "bookstack_api_error");
            return Err(format!("BookStack API error: {status}"));
        }
        Self::read_json(resp).await
    }

    async fn get_text(&self, path: &str) -> Result<String, String> {
        let builder = self
            .client
            .get(format!("{}/api/{}", self.base_url, path))
            .header("Authorization", self.auth_header());
        let resp = self.send_with_retry(builder).await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = Self::read_error_body(resp).await;
            tracing::error!(%status, body = %body, "bookstack_api_error");
            return Err(format!("BookStack API error: {status}"));
        }
        Self::read_text(resp).await
    }

    async fn delete(&self, path: &str) -> Result<(), String> {
        let builder = self
            .client
            .delete(format!("{}/api/{}", self.base_url, path))
            .header("Authorization", self.auth_header());
        let resp = self.send_with_retry(builder).await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = Self::read_error_body(resp).await;
            tracing::error!(%status, body = %body, "bookstack_api_error");
            return Err(format!("BookStack API error: {status}"));
        }
        Ok(())
    }

    // --- Validation ---

    /// Check the configured credentials against BookStack.
    ///
    /// The Rejected/Unavailable split is load-bearing, not cosmetic. Only
    /// `Rejected` means BookStack answered and said no — the sole case where
    /// a client should be told to re-authenticate. `Unavailable` means we
    /// never got an answer: BookStack restarting, a proxy 5xx, a timeout, a
    /// misconfigured base URL. Collapsing the two into one error (as this
    /// did before #139) makes every BookStack blip look like an expired
    /// token, and the callers that acted on it issued 401 +
    /// `WWW-Authenticate` or deleted a still-valid refresh token.
    pub async fn validate(&self) -> CredentialCheck {
        let builder = self
            .client
            .get(format!("{}/api/books", self.base_url))
            .header("Authorization", self.auth_header())
            .query(&[("count", "1")]);

        let resp = match self.send_with_retry(builder).await {
            Ok(resp) => resp,
            // Transport failure, or 429 with the retry budget spent. Never a
            // statement about the credentials.
            Err(e) => return CredentialCheck::Unavailable(e),
        };

        if resp.status().is_success() {
            // A 2xx alone isn't proof we reached BookStack. An auth portal or
            // misconfigured proxy in front of it happily answers 200 with an
            // HTML login page; treating that as Valid would admit a session
            // whose every tool call then fails. Requiring parseable JSON
            // keeps the old `get()` strictness — but classifies the failure
            // as Unavailable, since an interceptor says nothing about
            // whether the credentials are good.
            return match Self::read_json(resp).await {
                Ok(_) => CredentialCheck::Valid,
                Err(e) => {
                    tracing::warn!(error = %e, "bookstack_non_json_success_response");
                    CredentialCheck::Unavailable(
                        "BookStack returned a non-JSON response — check for a proxy or auth portal in front of the API".to_string(),
                    )
                }
            };
        }

        let status = resp.status();
        let body = Self::read_error_body(resp).await;
        let check = CredentialCheck::from_error_status(status);
        match check {
            CredentialCheck::Rejected(_) => {
                tracing::warn!(%status, body = %body, "bookstack_credentials_rejected")
            }
            _ => tracing::warn!(%status, body = %body, "bookstack_unavailable"),
        }
        check
    }

    /// Heuristic admin check: BookStack returns 403 from `/api/users` for
    /// non-admins. We don't need the user list, just the success/failure.
    /// Returns Ok(true) on success, Ok(false) on a 403, Err on other failures
    /// (so callers can distinguish "not admin" from "couldn't reach BookStack").
    pub async fn is_admin(&self) -> Result<bool, String> {
        match self.list_users(1, 0).await {
            Ok(_) => Ok(true),
            Err(e) if e.contains("403") || e.to_lowercase().contains("forbidden") => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Look up BookStack's "Admin" role ID. Used to lock auto-created Hive
    /// content to admin-only edit. Matches `display_name` case-insensitively
    /// against "admin"; returns the first match. Errors if no matching role
    /// is found (caller should treat as a non-fatal warning).
    pub async fn find_admin_role_id(&self) -> Result<i64, String> {
        let resp = self.list_roles(50, 0).await?;
        let data = resp
            .get("data")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        for role in data {
            let name = role
                .get("display_name")
                .and_then(|n| n.as_str())
                .unwrap_or("");
            if name.eq_ignore_ascii_case("admin") {
                if let Some(id) = role.get("id").and_then(|i| i.as_i64()) {
                    return Ok(id);
                }
            }
        }
        Err(
            "No role named \"Admin\" found in BookStack — cannot apply admin-only permission lock"
                .to_string(),
        )
    }

    // --- Permission check ---

    /// Check if the user can access a specific page.
    /// Uses GET /api/pages/{id} which correctly evaluates entity permissions.
    /// Returns true on 200, false on 403/404 or any error.
    pub async fn can_access_page(&self, page_id: i64) -> bool {
        let builder = self
            .client
            .get(format!("{}/api/pages/{page_id}", self.base_url))
            .header("Authorization", self.auth_header());
        match self.send_with_retry(builder).await {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }

    /// Resolve the calling user's role IDs (issue #58 lever a.5).
    ///
    /// BookStack has no `/api/users/me` endpoint, so we reuse the same
    /// search-by-`{created_by:me}` probe `whoami` uses to discover the
    /// caller's user id, then fetch `/api/users/{id}` and extract the
    /// `roles` array. Reading your own user row works for any
    /// authenticated user; the public API contract documents `roles` as
    /// part of the user-read response.
    ///
    /// Returns `Ok(None)` when the user has no created content yet
    /// (brand-new accounts) — the caller should treat that as
    /// "prefilter disabled, fall back to HTTP fan-out". Returns `Err`
    /// only when BookStack is unreachable or rejects the call.
    pub async fn list_my_roles(&self) -> Result<Option<Vec<i64>>, String> {
        let ident = match self.whoami().await? {
            Some(i) => i,
            None => return Ok(None),
        };
        let user = self.get_user(ident.bookstack_user_id).await?;
        let roles = match user.get("roles").and_then(|v| v.as_array()) {
            Some(r) => r,
            None => return Ok(Some(Vec::new())),
        };
        let mut ids: Vec<i64> = Vec::with_capacity(roles.len());
        for role in roles {
            if let Some(id) = role.get("id").and_then(|v| v.as_i64()) {
                ids.push(id);
            }
        }
        Ok(Some(ids))
    }

    // --- Shelves ---

    pub async fn list_shelves(&self, count: i64, offset: i64) -> Result<Value, String> {
        self.get(
            "shelves",
            &[
                ("count", &count.to_string()),
                ("offset", &offset.to_string()),
            ],
        )
        .await
    }

    /// Paginated enumeration of every shelf the token can see. Returns the
    /// flat list of shelf ids — used by the index worker's full walk on
    /// every run (issue #122, the unconditional walk-all path; replaces the
    /// briefly-shipped #119/#120 "walk-all when `indexed_shelves` is empty"
    /// branching). BookStack's `/api/shelves` caps at 500 per page; we page
    /// until `data` comes back empty.
    pub async fn list_all_shelves(&self) -> Result<Vec<i64>, String> {
        const PAGE_SIZE: i64 = 500;
        let mut ids = Vec::new();
        let mut offset = 0i64;
        loop {
            let page = self.list_shelves(PAGE_SIZE, offset).await?;
            let arr = page
                .get("data")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            if arr.is_empty() {
                break;
            }
            let page_len = arr.len() as i64;
            for shelf in arr {
                if let Some(id) = shelf.get("id").and_then(|v| v.as_i64()) {
                    ids.push(id);
                }
            }
            if page_len < PAGE_SIZE {
                break;
            }
            offset += PAGE_SIZE;
        }
        Ok(ids)
    }

    pub async fn get_shelf(&self, id: i64) -> Result<Value, String> {
        self.get(&format!("shelves/{id}"), &[]).await
    }

    pub async fn create_shelf(&self, name: &str, description: &str) -> Result<Value, String> {
        self.post(
            "shelves",
            &serde_json::json!({
                "name": name, "description": description,
            }),
        )
        .await
    }

    pub async fn update_shelf(&self, id: i64, data: &Value) -> Result<Value, String> {
        self.put(&format!("shelves/{id}"), data).await
    }

    pub async fn delete_shelf(&self, id: i64) -> Result<(), String> {
        self.delete(&format!("shelves/{id}")).await
    }

    // --- Books ---

    pub async fn list_books(&self, count: i64, offset: i64) -> Result<Value, String> {
        self.get(
            "books",
            &[
                ("count", &count.to_string()),
                ("offset", &offset.to_string()),
            ],
        )
        .await
    }

    /// Every book id the token can see, paginated. Mirrors
    /// `list_all_shelves`. The index worker's full walk is shelf-rooted, so
    /// this is how it reaches books that sit on no shelf at all (issue #147).
    pub async fn list_all_books(&self) -> Result<Vec<i64>, String> {
        const PAGE_SIZE: i64 = 500;
        let mut ids = Vec::new();
        let mut offset = 0i64;
        loop {
            let page = self.list_books(PAGE_SIZE, offset).await?;
            let arr = page
                .get("data")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            if arr.is_empty() {
                break;
            }
            let page_len = arr.len() as i64;
            for book in arr {
                if let Some(id) = book.get("id").and_then(|v| v.as_i64()) {
                    ids.push(id);
                }
            }
            if page_len < PAGE_SIZE {
                break;
            }
            offset += PAGE_SIZE;
        }
        Ok(ids)
    }

    pub async fn get_book(&self, id: i64) -> Result<Value, String> {
        self.get(&format!("books/{id}"), &[]).await
    }

    pub async fn create_book(&self, name: &str, description: &str) -> Result<Value, String> {
        self.post(
            "books",
            &serde_json::json!({
                "name": name, "description": description,
            }),
        )
        .await
    }

    pub async fn update_book(&self, id: i64, data: &Value) -> Result<Value, String> {
        self.put(&format!("books/{id}"), data).await
    }

    pub async fn delete_book(&self, id: i64) -> Result<(), String> {
        self.delete(&format!("books/{id}")).await
    }

    // --- Chapters ---

    pub async fn list_chapters(&self, count: i64, offset: i64) -> Result<Value, String> {
        self.get(
            "chapters",
            &[
                ("count", &count.to_string()),
                ("offset", &offset.to_string()),
            ],
        )
        .await
    }

    pub async fn get_chapter(&self, id: i64) -> Result<Value, String> {
        self.get(&format!("chapters/{id}"), &[]).await
    }

    pub async fn create_chapter(
        &self,
        book_id: i64,
        name: &str,
        description: &str,
    ) -> Result<Value, String> {
        self.post(
            "chapters",
            &serde_json::json!({
                "book_id": book_id, "name": name, "description": description,
            }),
        )
        .await
    }

    pub async fn update_chapter(&self, id: i64, data: &Value) -> Result<Value, String> {
        self.put(&format!("chapters/{id}"), data).await
    }

    pub async fn delete_chapter(&self, id: i64) -> Result<(), String> {
        self.delete(&format!("chapters/{id}")).await
    }

    // --- Pages ---

    pub async fn list_pages(&self, count: i64, offset: i64) -> Result<Value, String> {
        self.get(
            "pages",
            &[
                ("count", &count.to_string()),
                ("offset", &offset.to_string()),
            ],
        )
        .await
    }

    /// List pages whose `updated_at` is strictly greater than the given
    /// ISO 8601 timestamp, sorted oldest-first so the index reconciler
    /// can advance `last_delta_walk_at` to the newest page seen on each
    /// pass without losing entries to "process out of order then crash"
    /// races. Used by the v1.0.0 reconciliation worker's periodic delta
    /// walk (Phase 4c).
    pub async fn list_pages_updated_since(
        &self,
        since_iso_utc: &str,
        count: i64,
    ) -> Result<Vec<Value>, String> {
        let resp = self
            .get(
                "pages",
                &[
                    ("count", &count.to_string()),
                    ("sort", "+updated_at"),
                    ("filter[updated_at:gt]", since_iso_utc),
                ],
            )
            .await?;
        Ok(resp
            .get("data")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default())
    }

    pub async fn get_page(&self, id: i64) -> Result<Value, String> {
        self.get(&format!("pages/{id}"), &[]).await
    }

    pub async fn create_page(&self, data: &Value) -> Result<Value, String> {
        self.post("pages", data).await
    }

    pub async fn update_page(&self, id: i64, data: &Value) -> Result<Value, String> {
        self.put(&format!("pages/{id}"), data).await
    }

    pub async fn delete_page(&self, id: i64) -> Result<(), String> {
        self.delete(&format!("pages/{id}")).await
    }

    // --- Book traversal helpers ---
    //
    // These exist because BookStack's search API silently ignores
    // `{in_book:N}` / `{name:foo}` filters when the query has no positive
    // keyword term — `{type:page} {in_book:986}` parses fine but returns
    // system-wide matches, not book-scoped ones. Filter-only listings must
    // go through `get_book` (page row metadata) instead. Callers get
    // `updated_at` from the database row, never parsed from page content.

    /// Returns the most-recently-updated pages within a book, sorted by
    /// `updated_at` descending, capped at `limit`. Page rows include
    /// `id`, `name`, `slug`, `book_id`, `chapter_id`, `updated_at`, `url`.
    pub async fn list_book_pages_by_updated(
        &self,
        book_id: i64,
        limit: usize,
    ) -> Result<Vec<Value>, String> {
        let book = self.get_book(book_id).await?;
        let mut pages = flatten_book_pages(&book);
        pages.sort_by(|a, b| {
            let a_t = a.get("updated_at").and_then(|t| t.as_str()).unwrap_or("");
            let b_t = b.get("updated_at").and_then(|t| t.as_str()).unwrap_or("");
            b_t.cmp(a_t)
        });
        pages.truncate(limit);
        Ok(pages)
    }

    /// Find a page in a book by exact (case-insensitive) name. Returns the
    /// page row if found, or `None`. One `get_book` call.
    pub async fn find_page_in_book(
        &self,
        book_id: i64,
        name: &str,
    ) -> Result<Option<Value>, String> {
        let book = self.get_book(book_id).await?;
        let pages = flatten_book_pages(&book);
        Ok(pages.into_iter().find(|p| {
            p.get("name")
                .and_then(|n| n.as_str())
                .map(|n| n.eq_ignore_ascii_case(name))
                .unwrap_or(false)
        }))
    }

    /// Find a page inside a chapter by exact (case-insensitive) name.
    /// Returns the page row if found, or `None`. One `get_chapter` call.
    /// Used by chapter-scoped collection resources (Phase 6 journal).
    pub async fn find_page_in_chapter(
        &self,
        chapter_id: i64,
        name: &str,
    ) -> Result<Option<Value>, String> {
        let chapter = self.get_chapter(chapter_id).await?;
        let pages = chapter
            .get("pages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        Ok(pages.into_iter().find(|p| {
            p.get("name")
                .and_then(|n| n.as_str())
                .map(|n| n.eq_ignore_ascii_case(name))
                .unwrap_or(false)
        }))
    }

    /// List pages inside a chapter, ordered by `updated_at` descending.
    /// Returns up to `limit` pages. Used by chapter-scoped collections to
    /// list recent entries.
    pub async fn list_chapter_pages_by_updated(
        &self,
        chapter_id: i64,
        limit: usize,
    ) -> Result<Vec<Value>, String> {
        let chapter = self.get_chapter(chapter_id).await?;
        let mut pages = chapter
            .get("pages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        // Sort by updated_at descending (lexicographic on ISO-8601 == time order)
        pages.sort_by(|a, b| {
            let av = a.get("updated_at").and_then(|v| v.as_str()).unwrap_or("");
            let bv = b.get("updated_at").and_then(|v| v.as_str()).unwrap_or("");
            bv.cmp(av)
        });
        pages.truncate(limit);
        Ok(pages)
    }

    /// Find a chapter in a book by exact (case-insensitive) name. Returns
    /// the chapter row if found, or `None`. One `get_book` call.
    pub async fn find_chapter_in_book(
        &self,
        book_id: i64,
        name: &str,
    ) -> Result<Option<Value>, String> {
        let book = self.get_book(book_id).await?;
        let contents = book
            .get("contents")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        Ok(contents.into_iter().find(|item| {
            item.get("type").and_then(|t| t.as_str()) == Some("chapter")
                && item
                    .get("name")
                    .and_then(|n| n.as_str())
                    .map(|n| n.eq_ignore_ascii_case(name))
                    .unwrap_or(false)
        }))
    }

    // --- Search ---

    pub async fn search(&self, query: &str, page: i64, count: i64) -> Result<Value, String> {
        self.get(
            "search",
            &[
                ("query", query),
                ("page", &page.to_string()),
                ("count", &count.to_string()),
            ],
        )
        .await
    }

    // --- Attachments ---

    pub async fn list_attachments(&self) -> Result<Value, String> {
        self.get("attachments", &[]).await
    }

    pub async fn get_attachment(&self, id: i64) -> Result<Value, String> {
        self.get(&format!("attachments/{id}"), &[]).await
    }

    pub async fn create_attachment(&self, data: &Value) -> Result<Value, String> {
        self.post("attachments", data).await
    }

    pub async fn update_attachment(&self, id: i64, data: &Value) -> Result<Value, String> {
        self.put(&format!("attachments/{id}"), data).await
    }

    pub async fn delete_attachment(&self, id: i64) -> Result<(), String> {
        self.delete(&format!("attachments/{id}")).await
    }

    // --- Exports ---

    pub async fn export_page(&self, id: i64, format: ExportFormat) -> Result<String, String> {
        let fmt = format.as_str();
        self.get_text(&format!("pages/{id}/export/{fmt}")).await
    }

    pub async fn export_chapter(&self, id: i64, format: ExportFormat) -> Result<String, String> {
        let fmt = format.as_str();
        self.get_text(&format!("chapters/{id}/export/{fmt}")).await
    }

    pub async fn export_book(&self, id: i64, format: ExportFormat) -> Result<String, String> {
        let fmt = format.as_str();
        self.get_text(&format!("books/{id}/export/{fmt}")).await
    }

    // --- Comments ---

    pub async fn list_comments(&self, query: &[(&str, &str)]) -> Result<Value, String> {
        self.get("comments", query).await
    }

    pub async fn get_comment(&self, id: i64) -> Result<Value, String> {
        self.get(&format!("comments/{id}"), &[]).await
    }

    pub async fn create_comment(&self, data: &Value) -> Result<Value, String> {
        self.post("comments", data).await
    }

    pub async fn update_comment(&self, id: i64, data: &Value) -> Result<Value, String> {
        self.put(&format!("comments/{id}"), data).await
    }

    pub async fn delete_comment(&self, id: i64) -> Result<(), String> {
        self.delete(&format!("comments/{id}")).await
    }

    // --- Recycle Bin ---

    pub async fn list_recycle_bin(&self, count: i64, offset: i64) -> Result<Value, String> {
        self.get(
            "recycle-bin",
            &[
                ("count", &count.to_string()),
                ("offset", &offset.to_string()),
            ],
        )
        .await
    }

    pub async fn restore_recycle_bin_item(&self, id: i64) -> Result<Value, String> {
        self.put(&format!("recycle-bin/{id}"), &serde_json::json!({}))
            .await
    }

    pub async fn destroy_recycle_bin_item(&self, id: i64) -> Result<(), String> {
        self.delete(&format!("recycle-bin/{id}")).await
    }

    // --- Users ---

    pub async fn list_users(&self, count: i64, offset: i64) -> Result<Value, String> {
        self.get(
            "users",
            &[
                ("count", &count.to_string()),
                ("offset", &offset.to_string()),
            ],
        )
        .await
    }

    pub async fn get_user(&self, id: i64) -> Result<Value, String> {
        self.get(&format!("users/{id}"), &[]).await
    }

    /// Discover the authenticated user's BookStack identity (id + email + name)
    /// without requiring user configuration.
    ///
    /// BookStack has no `/api/users/me` endpoint, but its search API resolves
    /// `{created_by:me}` server-side. We probe by searching for any single
    /// page the user has created, extract `created_by.id` from the result,
    /// then fetch `/api/users/{id}` to get email + name (the search response
    /// only carries id/name/slug — email lives on the user record).
    ///
    /// Returns `Ok(None)` when the user has no content yet (brand-new accounts)
    /// — the caller should retry on first write or fall back to manual config.
    /// Returns `Err` only when BookStack is unreachable or rejects the call
    /// for non-empty-result reasons.
    pub async fn whoami(&self) -> Result<Option<UserIdentity>, String> {
        // Probe via search. Single-page results, page-type only, created-by-self.
        let resp = self.search("{type:page} {created_by:me}", 1, 1).await?;
        let candidates = resp.get("data").and_then(|v| v.as_array());
        let Some(items) = candidates else {
            return Ok(None);
        };
        for item in items {
            // Each result has a `preview_html` block plus the underlying entity
            // shape. The created_by ref is at the top level on page rows.
            let created_by = match item.get("created_by") {
                Some(v) => v,
                None => continue,
            };
            let user_id = match created_by.get("id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => continue,
            };
            // Fetch the user record for email — search responses don't carry it.
            // Reading your own user row works for any authenticated user; admin
            // is only required to read OTHER users.
            let user = match self.get_user(user_id).await {
                Ok(u) => u,
                Err(e) => return Err(format!("whoami: get_user({user_id}) failed: {e}")),
            };
            let email = user
                .get("email")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let name = user
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_default();
            return Ok(Some(UserIdentity {
                bookstack_user_id: user_id,
                email,
                name,
            }));
        }
        Ok(None)
    }

    // --- Audit Log ---

    pub async fn list_audit_log(&self, count: i64, offset: i64) -> Result<Value, String> {
        self.get(
            "audit-log",
            &[
                ("count", &count.to_string()),
                ("offset", &offset.to_string()),
            ],
        )
        .await
    }

    // --- System ---

    pub async fn get_system_info(&self) -> Result<Value, String> {
        self.get("system", &[]).await
    }

    // --- Image Gallery ---

    pub async fn list_images(
        &self,
        count: i64,
        offset: i64,
        filter: &[(&str, &str)],
    ) -> Result<Value, String> {
        let mut query: Vec<(&str, &str)> = vec![];
        let count_str = count.to_string();
        let offset_str = offset.to_string();
        query.push(("count", &count_str));
        query.push(("offset", &offset_str));
        query.extend_from_slice(filter);
        self.get("image-gallery", &query).await
    }

    pub async fn get_image(&self, id: i64) -> Result<Value, String> {
        self.get(&format!("image-gallery/{id}"), &[]).await
    }

    pub async fn update_image(&self, id: i64, data: &Value) -> Result<Value, String> {
        self.put(&format!("image-gallery/{id}"), data).await
    }

    pub async fn delete_image(&self, id: i64) -> Result<(), String> {
        self.delete(&format!("image-gallery/{id}")).await
    }

    pub async fn upload_image(
        &self,
        name: &str,
        image_type: &str,
        uploaded_to: i64,
        filename: &str,
        bytes: Vec<u8>,
        mime_type: &str,
    ) -> Result<Value, String> {
        let file_part = reqwest::multipart::Part::bytes(bytes)
            .file_name(filename.to_string())
            .mime_str(mime_type)
            .map_err(|e| {
                tracing::error!(error = %e, "bookstack_multipart_error");
                "Invalid mime type".to_string()
            })?;
        let form = reqwest::multipart::Form::new()
            .text("name", name.to_string())
            .text("type", image_type.to_string())
            .text("uploaded_to", uploaded_to.to_string())
            .part("image", file_part);
        self.post_multipart("image-gallery", form).await
    }

    // --- File Attachments ---

    pub async fn create_file_attachment(
        &self,
        name: &str,
        uploaded_to: i64,
        filename: &str,
        bytes: Vec<u8>,
        mime_type: &str,
    ) -> Result<Value, String> {
        let file_part = reqwest::multipart::Part::bytes(bytes)
            .file_name(filename.to_string())
            .mime_str(mime_type)
            .map_err(|e| {
                tracing::error!(error = %e, "bookstack_multipart_error");
                "Invalid mime type".to_string()
            })?;
        let form = reqwest::multipart::Form::new()
            .text("name", name.to_string())
            .text("uploaded_to", uploaded_to.to_string())
            .part("file", file_part);
        self.post_multipart("attachments", form).await
    }

    // --- Content Permissions ---

    pub async fn get_content_permissions(
        &self,
        content_type: ContentType,
        content_id: i64,
    ) -> Result<Value, String> {
        let ct = content_type.as_str();
        self.get(&format!("content-permissions/{ct}/{content_id}"), &[])
            .await
    }

    pub async fn update_content_permissions(
        &self,
        content_type: ContentType,
        content_id: i64,
        data: &Value,
    ) -> Result<Value, String> {
        let ct = content_type.as_str();
        self.put(&format!("content-permissions/{ct}/{content_id}"), data)
            .await
    }

    // --- Roles ---

    pub async fn list_roles(&self, count: i64, offset: i64) -> Result<Value, String> {
        self.get(
            "roles",
            &[
                ("count", &count.to_string()),
                ("offset", &offset.to_string()),
            ],
        )
        .await
    }

    pub async fn get_role(&self, id: i64) -> Result<Value, String> {
        self.get(&format!("roles/{id}"), &[]).await
    }
}

/// Flatten a `get_book` response into a single list of page rows —
/// top-level pages plus every chapter's nested pages. Returns an empty
/// vec if `contents` is missing or malformed.
fn flatten_book_pages(book: &Value) -> Vec<Value> {
    let Some(contents) = book.get("contents").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let mut pages = Vec::new();
    for item in contents {
        match item.get("type").and_then(|t| t.as_str()) {
            Some("page") => pages.push(item.clone()),
            Some("chapter") => {
                if let Some(ch_pages) = item.get("pages").and_then(|p| p.as_array()) {
                    for p in ch_pages {
                        pages.push(p.clone());
                    }
                }
            }
            _ => {}
        }
    }
    pages
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;
    use serde_json::json;

    fn fixture_book() -> Value {
        // Mimics the shape of `GET /api/books/{id}` — top-level pages
        // mixed with chapters that have their own nested pages.
        json!({
            "id": 986,
            "name": "Pia's Journal",
            "contents": [
                {
                    "type": "page",
                    "id": 1003,
                    "name": "Archive Daily Log",
                    "updated_at": "2026-03-02T20:07:50Z"
                },
                {
                    "type": "chapter",
                    "id": 989,
                    "name": "2026-02",
                    "pages": [
                        { "id": 990, "name": "2026-02-22", "updated_at": "2026-03-03T20:32:51Z" },
                        { "id": 991, "name": "2026-02-19", "updated_at": "2026-03-03T20:32:53Z" },
                    ]
                },
                {
                    "type": "chapter",
                    "id": 1869,
                    "name": "2026-04",
                    "pages": [
                        { "id": 2025, "name": "2026-04-26", "updated_at": "2026-04-26T06:10:24Z" },
                        { "id": 2006, "name": "2026-04-25", "updated_at": "2026-04-25T22:29:51Z" },
                    ]
                },
            ]
        })
    }

    #[test]
    fn flatten_collects_top_level_and_chapter_pages() {
        let pages = flatten_book_pages(&fixture_book());
        let ids: Vec<i64> = pages
            .iter()
            .map(|p| p.get("id").and_then(|v| v.as_i64()).unwrap_or(0))
            .collect();
        // 5 pages total: 1 top-level + 2 in 2026-02 + 2 in 2026-04
        assert_eq!(ids.len(), 5);
        assert!(ids.contains(&1003));
        assert!(ids.contains(&2025));
        assert!(ids.contains(&990));
    }

    #[test]
    fn flatten_handles_missing_contents() {
        let book = json!({ "id": 1, "name": "Empty" });
        assert!(flatten_book_pages(&book).is_empty());
    }

    #[test]
    fn flatten_handles_malformed_chapter() {
        let book = json!({
            "contents": [
                { "type": "chapter", "id": 1, "name": "no pages array" }
            ]
        });
        assert!(flatten_book_pages(&book).is_empty());
    }

    // #139: only BookStack answering 401/403 may be reported as a credential
    // problem. Anything else must stay Unavailable — callers turn Rejected
    // into a 401 + WWW-Authenticate and delete refresh tokens on it.
    #[test]
    fn only_401_and_403_are_credential_rejections() {
        for status in [StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN] {
            assert!(
                matches!(
                    CredentialCheck::from_error_status(status),
                    CredentialCheck::Rejected(_)
                ),
                "{status} should be Rejected"
            );
        }
    }

    #[test]
    fn upstream_failures_are_not_credential_rejections() {
        let transient = [
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::GATEWAY_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
            // A misconfigured BSMCP_BOOKSTACK_URL, not a bad token.
            StatusCode::NOT_FOUND,
        ];
        for status in transient {
            assert!(
                matches!(
                    CredentialCheck::from_error_status(status),
                    CredentialCheck::Unavailable(_)
                ),
                "{status} must not be reported as a credential failure"
            );
        }
    }
}
