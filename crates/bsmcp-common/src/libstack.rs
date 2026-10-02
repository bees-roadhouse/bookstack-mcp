//! `LibStackClient` — the LibStack implementation of [`crate::backend::Backend`].
//!
//! LibStack (bees-roadhouse/libstack) is the household's BookStack
//! replacement: a JSON API under `/api`, bearer-token auth, org-scoped
//! nested collections, markdown pages rendered server-side. This client
//! speaks that API and answers in BookStack's shapes so the 59 tools, the
//! semantic index and the reconciliation worker keep working unchanged
//! (issue #156, design: https://kb.beesroadhouse.com/link/2507).
//!
//! # The mapping
//!
//! | BookStack | LibStack |
//! |---|---|
//! | shelf | root collection (depth 0) |
//! | book | depth-1 collection (child of a shelf) |
//! | chapter | depth-2 collection; deeper collections are chapters of the same book whose name is the path from depth 2 joined by ` / ` |
//! | page | page; `markdown` is the body, `html` carries `rendered_html`, `editor` is always `markdown` |
//! | `move_page` / `move_chapter` / `move_book_to_shelf` | `collection_id` / `parent_id` change |
//! | tags | page tags (`name`/`value`) |
//! | attachment | `/api/pages/{id}/attachments`; bytes served at `/blobs/{id}` |
//! | image | an attachment whose MIME type is `image/*`; `upload_image` stores a blob and returns its `/blobs/{id}` URL |
//! | comment | `/api/pages/{id}/comments` |
//! | user | `/api/auth/me` only — `list_users` returns the caller |
//! | audit log | `/api/activity` |
//! | system info | `/api/health` plus the public URL |
//! | search | `/api/search` hits mapped onto BookStack's `data[]` rows |
//!
//! A page that sits directly in a root collection reports that collection as
//! its `book_id` (there is no book above it); `get_book` therefore accepts
//! any collection id, and `list_all_books` includes root collections that
//! hold pages directly so the index worker's orphan sweep reaches them.
//!
//! Not available on LibStack, every one of them answering
//! [`crate::backend::not_available`]: content permissions, roles, the
//! recycle bin (`delete_page` is a soft delete, but LibStack has no bin
//! listing yet), exports, and link attachments. The three extra tools
//! `list_collections` / `get_collection` / `create_collection` expose the
//! real tree.
//!
//! # Integer ids over UUIDs
//!
//! LibStack ids are UUIDs; the whole tool surface, the semantic index and
//! the structural index are keyed by `i64`. The bridge is [`ids`]: every
//! UUID maps to the stable integer made of its top 63 bits, and a
//! process-wide table remembers UUIDs seen in any response so the integer
//! resolves back. An id that was never listed in this process is looked up
//! by re-listing collections and pages once; if it is still unknown the
//! call fails with a message saying so. Integer ids are therefore stable
//! across restarts (they are a pure function of the UUID) and never
//! collide in practice (63 random bits).
//!
//! # Credentials
//!
//! LibStack has one bearer token per user (`/api/auth/tokens`). The MCP's
//! credential path carries a `(token_id, token_secret)` pair everywhere —
//! OAuth form, encrypted token store, legacy `Bearer id:secret` header — so
//! [`LibStackClient::bearer_from_pair`] joins the two: the id alone when the
//! secret is blank, `id:secret` otherwise.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use zeroize::Zeroize;

use crate::backend::{not_available, Backend, BackendKind};
use crate::bookstack::{ContentType, CredentialCheck, ExportFormat, UserIdentity};
use crate::rate_limit::{self, RateLimiter};

const MAX_RESPONSE_SIZE: usize = 50 * 1024 * 1024;
const MAX_ERROR_BODY_SIZE: usize = 4096;
const RETRY_429_MAX_ATTEMPTS: u32 = 4;

// ---------------------------------------------------------------------------
// Integer ↔ UUID bridge
// ---------------------------------------------------------------------------

/// See the module docs, "Integer ids over UUIDs".
pub mod ids {
    use super::*;

    /// Stable integer for a UUID: its top 63 bits, so it is never negative
    /// and survives a round trip through JSON numbers and SQLite INTEGER.
    pub fn to_i64(u: &Uuid) -> i64 {
        (u.as_u128() >> 65) as i64
    }

    fn table() -> &'static RwLock<HashMap<i64, Uuid>> {
        static TABLE: OnceLock<RwLock<HashMap<i64, Uuid>>> = OnceLock::new();
        TABLE.get_or_init(Default::default)
    }

    /// Record a UUID and return its integer.
    pub fn remember(u: &Uuid) -> i64 {
        let id = to_i64(u);
        if let Ok(mut t) = table().write() {
            t.insert(id, *u);
        }
        id
    }

    /// Integer → UUID, if this process has seen the UUID.
    pub fn lookup(id: i64) -> Option<Uuid> {
        table().read().ok().and_then(|t| t.get(&id).copied())
    }

    /// Parse a UUID out of a JSON string field.
    pub fn parse_field(v: &Value, key: &str) -> Option<Uuid> {
        v.get(key)
            .and_then(|s| s.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
    }

    /// Remember the `id` of a JSON object and return its integer.
    pub fn remember_field(v: &Value, key: &str) -> Option<i64> {
        parse_field(v, key).map(|u| remember(&u))
    }
}

// ---------------------------------------------------------------------------
// Error translation
// ---------------------------------------------------------------------------

/// Turn a LibStack error response into the one-line string the dispatcher
/// surfaces. The envelope is `{"error":{"code","message","validation"?}}`;
/// validation errors list each field. A body that is not that envelope
/// still yields a line carrying the status so callers that match on
/// `"403"` / `"404"` keep working.
pub fn describe_error(status: u16, body: &str) -> String {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let err = parsed.as_ref().and_then(|v| v.get("error"));
    let message = err
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .map(|m| m.trim())
        .filter(|m| !m.is_empty());
    let mut out = match message {
        Some(m) => format!("LibStack API error: {status} {m}"),
        None => format!("LibStack API error: {status}"),
    };
    if let Some(fields) = err
        .and_then(|e| e.get("validation"))
        .and_then(|v| v.as_object())
    {
        let mut parts: Vec<String> = fields
            .iter()
            .map(|(field, msgs)| {
                let msgs: Vec<&str> = msgs
                    .as_array()
                    .map(|a| a.iter().filter_map(|m| m.as_str()).collect())
                    .unwrap_or_default();
                format!("{field}: {}", msgs.join(", "))
            })
            .collect();
        parts.sort();
        if !parts.is_empty() {
            out.push_str(" — ");
            out.push_str(&parts.join("; "));
        }
    }
    if status == 409 {
        out.push_str(" (the page changed since you read it — re-read it and retry)");
    }
    out
}

// ---------------------------------------------------------------------------
// Collection tree
// ---------------------------------------------------------------------------

/// One collection row as LibStack returns it, plus its parsed parent.
#[derive(Clone, Debug)]
pub struct Coll {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub name: String,
    pub description: String,
    pub created_at: String,
    pub updated_at: String,
}

/// What a collection looks like through the BookStack surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Shelf,
    Book,
    Chapter,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::Shelf => "shelf",
            Role::Book => "book",
            Role::Chapter => "chapter",
        }
    }
}

/// The org's collections, indexed for the depth questions the mapping asks.
/// Pure data — every method here is unit-tested without HTTP.
#[derive(Clone, Debug, Default)]
pub struct Tree {
    colls: HashMap<Uuid, Coll>,
    children: HashMap<Option<Uuid>, Vec<Uuid>>,
    /// Listing order as the API returned it (sort_order, name).
    order: Vec<Uuid>,
}

/// Parent chains longer than this are treated as cycles (a corrupt
/// `parent_id` must not hang the server).
const MAX_DEPTH: usize = 64;

impl Tree {
    /// Accepts the bare array `GET /api/collections` returns today, or a
    /// `{"data": [...]}` envelope should it grow one. A `parent_id` that
    /// is missing or null makes a root; `parent_id` is the nested-collections
    /// field (libstack issue #12).
    pub fn from_json(v: &Value) -> Tree {
        let rows = v
            .as_array()
            .or_else(|| v.get("data").and_then(|d| d.as_array()))
            .cloned()
            .unwrap_or_default();
        let mut tree = Tree::default();
        for row in &rows {
            let Some(id) = ids::parse_field(row, "id") else {
                continue;
            };
            ids::remember(&id);
            let parent_id = ids::parse_field(row, "parent_id");
            tree.colls.insert(
                id,
                Coll {
                    id,
                    parent_id,
                    name: str_field(row, "name"),
                    description: str_field(row, "description_md"),
                    created_at: str_field(row, "created_at"),
                    updated_at: str_field(row, "updated_at"),
                },
            );
            tree.order.push(id);
        }
        // A parent that is not in the listing (deleted, or out of org) makes
        // its children roots — the same thing the UI would show.
        for id in tree.order.clone() {
            let parent = tree.colls[&id]
                .parent_id
                .filter(|p| tree.colls.contains_key(p) && *p != id);
            if let Some(c) = tree.colls.get_mut(&id) {
                c.parent_id = parent;
            }
            tree.children.entry(parent).or_default().push(id);
        }
        tree
    }

    pub fn get(&self, id: &Uuid) -> Option<&Coll> {
        self.colls.get(id)
    }

    pub fn len(&self) -> usize {
        self.colls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.colls.is_empty()
    }

    /// All collections in listing order.
    pub fn all(&self) -> impl Iterator<Item = &Coll> {
        self.order.iter().filter_map(|id| self.colls.get(id))
    }

    /// Root first, excluding `id` itself. Empty for a root.
    pub fn ancestors(&self, id: &Uuid) -> Vec<Uuid> {
        let mut chain = Vec::new();
        let mut cur = self.colls.get(id).and_then(|c| c.parent_id);
        while let Some(p) = cur {
            if chain.len() >= MAX_DEPTH || chain.contains(&p) || p == *id {
                break;
            }
            chain.push(p);
            cur = self.colls.get(&p).and_then(|c| c.parent_id);
        }
        chain.reverse();
        chain
    }

    /// Root = 0.
    pub fn depth(&self, id: &Uuid) -> usize {
        self.ancestors(id).len()
    }

    /// Depth 0 → shelf, 1 → book, 2 and deeper → chapter.
    pub fn role(&self, id: &Uuid) -> Role {
        match self.depth(id) {
            0 => Role::Shelf,
            1 => Role::Book,
            _ => Role::Chapter,
        }
    }

    /// The name shown through the BookStack surface. A collection deeper
    /// than a chapter shows its path from the chapter level down, joined by
    /// ` / `, so `Shelf > Book > Ops > Runbooks > Backups` is the chapter
    /// `Ops / Runbooks / Backups` of book `Book`.
    pub fn path_name(&self, id: &Uuid) -> String {
        let own = self
            .colls
            .get(id)
            .map(|c| c.name.clone())
            .unwrap_or_default();
        let anc = self.ancestors(id);
        if anc.len() < 3 {
            return own;
        }
        let mut parts: Vec<String> = anc[2..]
            .iter()
            .filter_map(|a| self.colls.get(a).map(|c| c.name.clone()))
            .collect();
        parts.push(own);
        parts.join(" / ")
    }

    /// The shelf above `id` (or `id` itself when it is a root).
    pub fn shelf_of(&self, id: &Uuid) -> Option<Uuid> {
        let anc = self.ancestors(id);
        if anc.is_empty() {
            self.colls.contains_key(id).then_some(*id)
        } else {
            Some(anc[0])
        }
    }

    /// The book above `id`: the depth-1 ancestor, or `id` itself when it
    /// is at depth 1. `None` for a root.
    pub fn book_of(&self, id: &Uuid) -> Option<Uuid> {
        let anc = self.ancestors(id);
        match anc.len() {
            0 => None,
            1 => Some(*id),
            _ => Some(anc[1]),
        }
    }

    /// `id` itself when it is at chapter depth, else `None`.
    pub fn chapter_of(&self, id: &Uuid) -> Option<Uuid> {
        (self.depth(id) >= 2).then_some(*id)
    }

    /// Which collection a page in `coll` reports as its book: the book
    /// above it, or the root collection itself for a page that lives
    /// directly in a shelf.
    pub fn book_for_page(&self, coll: &Uuid) -> Uuid {
        self.book_of(coll).unwrap_or(*coll)
    }

    pub fn children(&self, parent: Option<Uuid>) -> &[Uuid] {
        self.children
            .get(&parent)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Every collection under `id`, depth-first in listing order.
    pub fn descendants(&self, id: &Uuid) -> Vec<Uuid> {
        let mut out = Vec::new();
        let mut stack: Vec<Uuid> = self.children(Some(*id)).iter().rev().copied().collect();
        while let Some(c) = stack.pop() {
            if out.contains(&c) || out.len() > self.colls.len() {
                continue;
            }
            out.push(c);
            stack.extend(self.children(Some(c)).iter().rev().copied());
        }
        out
    }

    /// Collections playing `role`, in listing order.
    pub fn with_role(&self, role: Role) -> Vec<Uuid> {
        self.order
            .iter()
            .copied()
            .filter(|id| self.role(id) == role)
            .collect()
    }
}

fn str_field(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(|s| s.as_str())
        .unwrap_or_default()
        .to_string()
}

/// Everything the mapping needs for one tool call: the tree and the org's
/// page metadata (no bodies). Two requests.
struct Snapshot {
    tree: Tree,
    pages: Vec<Value>,
}

impl Snapshot {
    fn pages_in(&self, coll: &Uuid) -> Vec<&Value> {
        self.pages
            .iter()
            .filter(|p| ids::parse_field(p, "collection_id").as_ref() == Some(coll))
            .collect()
    }

    fn has_pages(&self, coll: &Uuid) -> bool {
        !self.pages_in(coll).is_empty()
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

pub struct LibStackClient {
    client: Client,
    base_url: String,
    public_url: String,
    bearer: String,
    /// Non-secret label for the credential (sha256 prefix), the cache key
    /// the rest of the server expects from `token_id()`.
    token_label: String,
    rate_limiter: RateLimiter,
}

impl Drop for LibStackClient {
    fn drop(&mut self) {
        self.bearer.zeroize();
    }
}

impl LibStackClient {
    /// `token_id` / `token_secret` are the pair the MCP credential path
    /// carries; see [`Self::bearer_from_pair`].
    pub fn new(base_url: &str, token_id: &str, token_secret: &str, client: Client) -> Self {
        let base_url = base_url.trim().trim_end_matches('/').to_string();
        let bearer = Self::bearer_from_pair(token_id, token_secret);
        let token_label = {
            let digest = Sha256::digest(bearer.as_bytes());
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            format!("ls-{}", &hex[..16])
        };
        Self {
            client,
            public_url: base_url.clone(),
            base_url,
            bearer,
            token_label,
            rate_limiter: rate_limit::shared(),
        }
    }

    /// One LibStack bearer token from the MCP's id/secret pair: the id alone
    /// when the secret is blank (paste the LibStack token as the Token ID),
    /// otherwise `id:secret` so a deployment issuing two-part tokens works
    /// without a change here.
    pub fn bearer_from_pair(token_id: &str, token_secret: &str) -> String {
        let id = token_id.trim();
        let secret = token_secret.trim();
        if secret.is_empty() {
            id.to_string()
        } else {
            format!("{id}:{secret}")
        }
    }

    pub fn with_public_url(mut self, url: &str) -> Self {
        let trimmed = url.trim().trim_end_matches('/');
        if !trimmed.is_empty() {
            self.public_url = trimmed.to_string();
        }
        self
    }

    fn unavailable(&self, feature: &str) -> String {
        not_available(BackendKind::LibStack, feature)
    }

    // --- HTTP ---

    fn url(&self, path: &str) -> String {
        format!("{}/api/{}", self.base_url, path.trim_start_matches('/'))
    }

    fn authed(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder.header("Authorization", format!("Bearer {}", self.bearer))
    }

    async fn read_error_body(mut resp: reqwest::Response) -> String {
        if resp
            .content_length()
            .is_some_and(|len| len as usize > MAX_ERROR_BODY_SIZE)
        {
            return String::new();
        }
        let mut buf = Vec::with_capacity(1024);
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

    async fn read_json(resp: reqwest::Response) -> Result<Value, String> {
        if resp
            .content_length()
            .is_some_and(|len| len as usize > MAX_RESPONSE_SIZE)
        {
            return Err("Response too large".to_string());
        }
        if resp.status() == StatusCode::NO_CONTENT {
            return Ok(Value::Null);
        }
        let bytes = resp.bytes().await.map_err(|e| {
            tracing::error!(error = %e, "libstack_response_read_error");
            "Failed to read response".to_string()
        })?;
        if bytes.len() > MAX_RESPONSE_SIZE {
            return Err(format!("Response too large: {} bytes", bytes.len()));
        }
        if bytes.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&bytes).map_err(|e| {
            tracing::error!(error = %e, "libstack_json_parse_error");
            "Invalid response from LibStack".to_string()
        })
    }

    /// Send with the shared rate limiter and the same 429 back-off the
    /// BookStack client uses.
    async fn send(&self, builder: reqwest::RequestBuilder) -> Result<reqwest::Response, String> {
        for attempt in 0..RETRY_429_MAX_ATTEMPTS {
            self.rate_limiter.acquire().await;
            let req = builder
                .try_clone()
                .ok_or_else(|| "non-cloneable request".to_string())?;
            let resp = req.send().await.map_err(|e| {
                tracing::error!(error = %e, "libstack_request_error");
                "Request failed".to_string()
            })?;
            if resp.status().as_u16() == 429 {
                let base = resp
                    .headers()
                    .get("Retry-After")
                    .and_then(|v| v.to_str().ok())
                    .and_then(rate_limit::parse_retry_after)
                    .unwrap_or_else(|| Duration::from_millis(500 * 2u64.pow(attempt)));
                let delay = base + rate_limit::jitter(500);
                if attempt + 1 == RETRY_429_MAX_ATTEMPTS {
                    let status = resp.status();
                    let body = Self::read_error_body(resp).await;
                    tracing::error!(body = %body, "libstack_429_retry_exhausted");
                    return Err(describe_error(status.as_u16(), &body));
                }
                tokio::time::sleep(delay).await;
                continue;
            }
            self.rate_limiter.observe_limit(resp.headers());
            return Ok(resp);
        }
        Err("LibStack API error: 429".to_string())
    }

    async fn expect_ok(&self, builder: reqwest::RequestBuilder) -> Result<Value, String> {
        let resp = self.send(builder).await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = Self::read_error_body(resp).await;
            tracing::error!(%status, body = %body, "libstack_api_error");
            return Err(describe_error(status.as_u16(), &body));
        }
        Self::read_json(resp).await
    }

    async fn get(&self, path: &str, query: &[(&str, &str)]) -> Result<Value, String> {
        self.expect_ok(self.authed(self.client.get(self.url(path))).query(query))
            .await
    }

    async fn post(&self, path: &str, body: &Value) -> Result<Value, String> {
        self.expect_ok(self.authed(self.client.post(self.url(path))).json(body))
            .await
    }

    async fn put(&self, path: &str, body: &Value) -> Result<Value, String> {
        self.expect_ok(self.authed(self.client.put(self.url(path))).json(body))
            .await
    }

    async fn patch(&self, path: &str, body: &Value) -> Result<Value, String> {
        self.expect_ok(self.authed(self.client.patch(self.url(path))).json(body))
            .await
    }

    async fn delete(&self, path: &str) -> Result<(), String> {
        self.expect_ok(self.authed(self.client.delete(self.url(path))))
            .await
            .map(|_| ())
    }

    async fn post_multipart(
        &self,
        path: &str,
        form: reqwest::multipart::Form,
    ) -> Result<Value, String> {
        self.rate_limiter.acquire().await;
        let resp = self
            .authed(self.client.post(self.url(path)))
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "libstack_request_error");
                "Request failed".to_string()
            })?;
        self.rate_limiter.observe_limit(resp.headers());
        if !resp.status().is_success() {
            let status = resp.status();
            let body = Self::read_error_body(resp).await;
            tracing::error!(%status, body = %body, "libstack_api_error");
            return Err(describe_error(status.as_u16(), &body));
        }
        Self::read_json(resp).await
    }

    // --- Fetching the tree and page metadata ---

    async fn fetch_tree(&self) -> Result<Tree, String> {
        let v = self.get("collections", &[]).await?;
        Ok(Tree::from_json(&v))
    }

    async fn fetch_pages(&self, collection: Option<&Uuid>) -> Result<Vec<Value>, String> {
        let coll_str;
        let mut q: Vec<(&str, &str)> = Vec::new();
        if let Some(c) = collection {
            coll_str = c.to_string();
            q.push(("collection_id", &coll_str));
        }
        let v = self.get("pages", &q).await?;
        let rows = v
            .get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .or_else(|| v.as_array().cloned())
            .unwrap_or_default();
        for p in &rows {
            ids::remember_field(p, "id");
        }
        Ok(rows)
    }

    async fn snapshot(&self) -> Result<Snapshot, String> {
        let tree = self.fetch_tree().await?;
        let pages = self.fetch_pages(None).await?;
        Ok(Snapshot { tree, pages })
    }

    /// Integer → UUID, re-listing collections and pages once on a miss.
    async fn resolve(&self, id: i64, what: &str) -> Result<Uuid, String> {
        if let Some(u) = ids::lookup(id) {
            return Ok(u);
        }
        // Populate the table from the two listings and try again.
        let _ = self.snapshot().await;
        ids::lookup(id).ok_or_else(|| {
            format!(
                "{what} id {id} is not known to this LibStack: it has not appeared in any listing (call the matching list_* tool first)"
            )
        })
    }

    /// Resolve an id that may belong to a page's attachment or comment
    /// (never listed by `snapshot`): the table only, with a pointed error.
    fn resolve_seen(&self, id: i64, what: &str) -> Result<Uuid, String> {
        ids::lookup(id).ok_or_else(|| {
            format!("{what} id {id} is not known to this LibStack: list it under its page first")
        })
    }

    // --- JSON shapes ---

    fn coll_url(&self, id: &Uuid) -> String {
        format!("/c/{id}")
    }

    fn page_url(&self, id: &Uuid) -> String {
        format!("/p/{id}")
    }

    fn base_row(&self, c: &Coll) -> Value {
        json!({
            "id": ids::remember(&c.id),
            "name": c.name,
            "slug": c.id.to_string(),
            "description": c.description,
            "created_at": c.created_at,
            "updated_at": c.updated_at,
            "url": self.coll_url(&c.id),
            "libstack_id": c.id.to_string(),
            "collection_id": ids::to_i64(&c.id),
        })
    }

    fn shelf_row(&self, c: &Coll) -> Value {
        let mut v = self.base_row(c);
        v["type"] = json!("bookshelf");
        v
    }

    fn book_row(&self, c: &Coll, tree: &Tree) -> Value {
        let mut v = self.base_row(c);
        v["type"] = json!("book");
        v["shelf_id"] = tree
            .get(&c.id)
            .and_then(|c| c.parent_id)
            .map(|p| json!(ids::remember(&p)))
            .unwrap_or(Value::Null);
        v
    }

    fn chapter_row(&self, c: &Coll, tree: &Tree) -> Value {
        let mut v = self.base_row(c);
        v["type"] = json!("chapter");
        v["name"] = json!(tree.path_name(&c.id));
        v["collection_name"] = json!(c.name);
        let book = tree.book_for_page(&c.id);
        v["book_id"] = json!(ids::remember(&book));
        v["book_slug"] = json!(book.to_string());
        v
    }

    /// A page row in BookStack's shape. `p` may be a list row (metadata) or
    /// a full page (with `markdown` / `rendered_html` / `tags`).
    fn page_row(&self, p: &Value, tree: &Tree) -> Value {
        let id = ids::parse_field(p, "id");
        let coll = ids::parse_field(p, "collection_id");
        let (book_id, chapter_id, book_name, chapter_name) = match coll {
            Some(c) => {
                let book = tree.book_for_page(&c);
                let chapter = tree.chapter_of(&c);
                (
                    Some(ids::remember(&book)),
                    chapter.map(|ch| ids::remember(&ch)),
                    tree.get(&book).map(|b| b.name.clone()),
                    chapter.map(|ch| tree.path_name(&ch)),
                )
            }
            None => (None, None, None, None),
        };
        let mut v = json!({
            "id": id.map(|u| ids::remember(&u)),
            "name": p.get("title").cloned().unwrap_or(Value::Null),
            "slug": id.map(|u| u.to_string()),
            "book_id": book_id,
            "chapter_id": chapter_id.map(Value::from).unwrap_or(Value::Null),
            "collection_id": coll.map(|c| ids::to_i64(&c)),
            "libstack_id": id.map(|u| u.to_string()),
            "draft": p.get("is_draft").cloned().unwrap_or(json!(false)),
            "revision_count": p.get("revision_count").cloned().unwrap_or(json!(0)),
            "priority": p.get("sort_order").cloned().unwrap_or(json!(0)),
            "created_at": p.get("created_at").cloned().unwrap_or(Value::Null),
            "updated_at": p.get("updated_at").cloned().unwrap_or(Value::Null),
            "created_by": user_ref(p.get("created_by")),
            "updated_by": user_ref(p.get("updated_by")),
            "editor": "markdown",
            "url": id.map(|u| self.page_url(&u)),
            "book": book_id.map(|b| json!({ "id": b, "name": book_name, "slug": coll.map(|c| tree.book_for_page(&c).to_string()) })),
        });
        if let (Some(cid), Some(cname)) = (chapter_id, chapter_name) {
            v["chapter"] = json!({ "id": cid, "name": cname });
        }
        if let Some(md) = p.get("markdown") {
            v["markdown"] = md.clone();
            v["html"] = p.get("rendered_html").cloned().unwrap_or(json!(""));
            v["raw_html"] = v["html"].clone();
        }
        if let Some(tags) = p.get("tags") {
            v["tags"] = tags.clone();
        }
        v
    }

    fn user_row(&self, me: &Value) -> Value {
        let user = me.get("user").cloned().unwrap_or(Value::Null);
        let id = ids::parse_field(&user, "id");
        json!({
            "id": id.map(|u| ids::remember(&u)),
            "name": user.get("display_name").cloned().unwrap_or(Value::Null),
            "email": user.get("email").cloned().unwrap_or(Value::Null),
            "slug": id.map(|u| u.to_string()),
            "libstack_id": id.map(|u| u.to_string()),
            "kind": user.get("kind").cloned().unwrap_or(Value::Null),
            "active_org": me.get("active_org").cloned().unwrap_or(Value::Null),
            "orgs": me.get("orgs").cloned().unwrap_or(json!([])),
            "capabilities": me.get("capabilities").cloned().unwrap_or(json!([])),
            "roles": [],
        })
    }

    fn attachment_row(&self, a: &Value, page_uuid: Option<&Uuid>) -> Value {
        let id = ids::parse_field(a, "id");
        let blob = ids::parse_field(a, "blob_id").or(id);
        let page = ids::parse_field(a, "page_id").or(page_uuid.copied());
        let name = str_field(a, "name");
        let extension = name.rsplit('.').next().filter(|e| *e != name).unwrap_or("");
        let url = blob.map(|b| format!("/blobs/{b}"));
        let mime = str_field(a, "mime");
        json!({
            "id": id.map(|u| ids::remember(&u)),
            "name": name,
            "extension": extension,
            "uploaded_to": page.map(|p| ids::remember(&p)),
            "external": false,
            "order": a.get("sort_order").cloned().unwrap_or(json!(0)),
            "mime": mime,
            "size": a.get("size").cloned().unwrap_or(Value::Null),
            "created_at": a.get("created_at").cloned().unwrap_or(Value::Null),
            "updated_at": a.get("updated_at").cloned().unwrap_or(Value::Null),
            "created_by": user_ref(a.get("created_by")),
            "url": url,
            "links": { "html": url.as_ref().map(|u| format!("<a href=\"{u}\">{}</a>", str_field(a, "name"))), "markdown": url.as_ref().map(|u| format!("[{}]({u})", str_field(a, "name"))) },
            "libstack_id": id.map(|u| u.to_string()),
            "blob_id": blob.map(|b| b.to_string()),
        })
    }

    fn image_row(&self, a: &Value, page_uuid: Option<&Uuid>) -> Value {
        let mut v = self.attachment_row(a, page_uuid);
        v["type"] = json!("gallery");
        v["path"] = v["url"].clone();
        v["thumbs"] = json!({});
        v["content"] = json!({
            "html": v["url"].as_str().map(|u| format!("<img src=\"{u}\" alt=\"{}\">", str_field(a, "name"))),
            "markdown": v["url"].as_str().map(|u| format!("![{}]({u})", str_field(a, "name"))),
        });
        v
    }

    fn comment_row(&self, c: &Value, page_uuid: Option<&Uuid>) -> Value {
        let id = ids::parse_field(c, "id");
        let page = ids::parse_field(c, "page_id").or(page_uuid.copied());
        json!({
            "id": id.map(|u| ids::remember(&u)),
            "page_id": page.map(|p| ids::remember(&p)),
            "parent_id": ids::parse_field(c, "parent_id").map(|p| ids::remember(&p)),
            "markdown": c.get("markdown").cloned().unwrap_or(Value::Null),
            "html": c.get("rendered_html").cloned().unwrap_or(Value::Null),
            "created_by": user_ref(c.get("created_by")),
            "updated_by": user_ref(c.get("updated_by")),
            "created_at": c.get("created_at").cloned().unwrap_or(Value::Null),
            "updated_at": c.get("updated_at").cloned().unwrap_or(Value::Null),
            "libstack_id": id.map(|u| u.to_string()),
        })
    }

    fn activity_row(&self, a: &Value) -> Value {
        let id = ids::parse_field(a, "id");
        let subject = ids::parse_field(a, "subject_id");
        json!({
            "id": id.map(|u| ids::remember(&u)).map(Value::from).unwrap_or_else(|| a.get("id").cloned().unwrap_or(Value::Null)),
            "type": a.get("action").cloned().unwrap_or(Value::Null),
            "detail": a.get("label").cloned().unwrap_or(Value::Null),
            "entity_type": a.get("subject_type").cloned().unwrap_or(Value::Null),
            "entity_id": subject.map(|s| ids::remember(&s)),
            "user_id": ids::parse_field(a, "user_id").map(|u| ids::remember(&u)),
            "ip": a.get("ip").cloned().unwrap_or(Value::Null),
            "created_at": a.get("created_at").cloned().unwrap_or(Value::Null),
            "libstack_id": a.get("id").cloned().unwrap_or(Value::Null),
        })
    }

    // --- Shared pieces ---

    async fn me(&self) -> Result<Value, String> {
        self.get("auth/me", &[]).await
    }

    /// Partial update of a collection: BookStack's `name` / `description`
    /// keys become LibStack's `name` / `description_md`; `parent_id` (an
    /// integer from the BookStack surface) is resolved to the UUID.
    async fn update_collection(&self, id: i64, data: &Value, what: &str) -> Result<Coll, String> {
        let uuid = self.resolve(id, what).await?;
        let mut body = json!({});
        if let Some(n) = data.get("name").and_then(|v| v.as_str()) {
            body["name"] = json!(n);
        }
        if let Some(d) = data.get("description").and_then(|v| v.as_str()) {
            body["description_md"] = json!(d);
        }
        if let Some(p) = data.get("parent_id").and_then(|v| v.as_i64()) {
            let parent = self.resolve(p, "collection").await?;
            body["parent_id"] = json!(parent.to_string());
        }
        let v = self.put(&format!("collections/{uuid}"), &body).await?;
        coll_from_json(&v).ok_or_else(|| "LibStack returned a collection without an id".to_string())
    }

    async fn create_collection_raw(
        &self,
        parent: Option<Uuid>,
        name: &str,
        description: &str,
    ) -> Result<Coll, String> {
        let mut body = json!({ "name": name, "description_md": description });
        if let Some(p) = parent {
            body["parent_id"] = json!(p.to_string());
        }
        let v = self.post("collections", &body).await?;
        coll_from_json(&v).ok_or_else(|| "LibStack returned a collection without an id".to_string())
    }

    /// Full collection payload for the three `*_collection` tools.
    fn collection_row(&self, c: &Coll, tree: &Tree) -> Value {
        let mut v = self.base_row(c);
        v["parent_id"] = c
            .parent_id
            .map(|p| json!(ids::remember(&p)))
            .unwrap_or(Value::Null);
        v["parent_libstack_id"] = c
            .parent_id
            .map(|p| json!(p.to_string()))
            .unwrap_or(Value::Null);
        v["depth"] = json!(tree.depth(&c.id));
        v["appears_as"] = json!(tree.role(&c.id).as_str());
        v["path"] = json!(tree
            .ancestors(&c.id)
            .iter()
            .chain(std::iter::once(&c.id))
            .filter_map(|a| tree.get(a).map(|x| x.name.clone()))
            .collect::<Vec<_>>()
            .join(" / "));
        v["children"] = json!(tree
            .children(Some(c.id))
            .iter()
            .filter_map(|ch| tree.get(ch))
            .map(|ch| json!({ "id": ids::remember(&ch.id), "name": ch.name, "libstack_id": ch.id.to_string() }))
            .collect::<Vec<_>>());
        v
    }

    async fn page_attachments(&self, page: &Uuid) -> Result<Vec<Value>, String> {
        let v = self.get(&format!("pages/{page}/attachments"), &[]).await?;
        Ok(v.get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .or_else(|| v.as_array().cloned())
            .unwrap_or_default())
    }

    async fn upload_blob(
        &self,
        page: &Uuid,
        name: &str,
        filename: &str,
        bytes: Vec<u8>,
        mime_type: &str,
    ) -> Result<Value, String> {
        let part = reqwest::multipart::Part::bytes(bytes)
            .file_name(filename.to_string())
            .mime_str(mime_type)
            .map_err(|e| {
                tracing::error!(error = %e, "libstack_multipart_error");
                "Invalid mime type".to_string()
            })?;
        let form = reqwest::multipart::Form::new()
            .text("name", name.to_string())
            .part("file", part);
        self.post_multipart(&format!("pages/{page}/attachments"), form)
            .await
    }
}

fn coll_from_json(v: &Value) -> Option<Coll> {
    let id = ids::parse_field(v, "id")?;
    ids::remember(&id);
    Some(Coll {
        id,
        parent_id: ids::parse_field(v, "parent_id"),
        name: str_field(v, "name"),
        description: str_field(v, "description_md"),
        created_at: str_field(v, "created_at"),
        updated_at: str_field(v, "updated_at"),
    })
}

/// `created_by` / `updated_by` are user UUIDs (or null) on LibStack; the
/// BookStack shape is `{id, name, slug}`. Name and slug are unknown here.
fn user_ref(v: Option<&Value>) -> Value {
    match v
        .and_then(|s| s.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
    {
        Some(u) => json!({ "id": ids::remember(&u), "libstack_id": u.to_string() }),
        None => Value::Null,
    }
}

/// `{data, total}` page of `rows` in BookStack's listing shape.
fn page_slice(rows: Vec<Value>, count: i64, offset: i64) -> Value {
    let total = rows.len();
    let start = offset.max(0) as usize;
    let data: Vec<Value> = rows
        .into_iter()
        .skip(start.min(total))
        .take(count.max(0) as usize)
        .collect();
    json!({ "data": data, "total": total })
}

/// Drop BookStack's `{filter:value}` and `[tag=value]` operator tokens —
/// LibStack's search is `websearch_to_tsquery`, which has its own grammar
/// (quoted phrases, `-negation`, `OR`).
pub fn strip_bookstack_operators(query: &str) -> String {
    let mut out = String::with_capacity(query.len());
    let mut depth_brace = 0usize;
    let mut depth_bracket = 0usize;
    for ch in query.chars() {
        match ch {
            '{' => depth_brace += 1,
            '}' if depth_brace > 0 => depth_brace -= 1,
            '[' => depth_bracket += 1,
            ']' if depth_bracket > 0 => depth_bracket -= 1,
            _ if depth_brace == 0 && depth_bracket == 0 => out.push(ch),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `a > b` for two RFC 3339 timestamps, parsed when possible (LibStack
/// may emit fractional seconds, BookStack never does) and compared as
/// strings otherwise.
fn later_than(a: &str, b: &str) -> bool {
    use chrono::DateTime;
    match (
        DateTime::parse_from_rfc3339(a),
        DateTime::parse_from_rfc3339(b),
    ) {
        (Ok(x), Ok(y)) => x > y,
        _ => a > b,
    }
}

// ---------------------------------------------------------------------------
// Backend impl
// ---------------------------------------------------------------------------

#[async_trait]
impl Backend for LibStackClient {
    fn kind(&self) -> BackendKind {
        BackendKind::LibStack
    }
    fn base_url(&self) -> &str {
        &self.base_url
    }
    fn public_url(&self) -> &str {
        &self.public_url
    }
    fn token_id(&self) -> &str {
        &self.token_label
    }

    // --- Credentials and identity ---

    /// `GET /api/auth/me`: 2xx JSON is Valid, 401/403 Rejected, anything
    /// else (or a non-JSON 2xx from a proxy) Unavailable — the same split
    /// the BookStack client keeps for issue #139.
    async fn validate(&self) -> CredentialCheck {
        let builder = self.authed(self.client.get(self.url("auth/me")));
        let resp = match self.send(builder).await {
            Ok(r) => r,
            Err(e) => return CredentialCheck::Unavailable(e),
        };
        if resp.status().is_success() {
            return match Self::read_json(resp).await {
                Ok(v) if v.get("user").is_some() => CredentialCheck::Valid,
                Ok(_) => CredentialCheck::Unavailable(
                    "LibStack answered /api/auth/me without a user — check BSMCP_LIBSTACK_URL".to_string(),
                ),
                Err(_) => CredentialCheck::Unavailable(
                    "LibStack returned a non-JSON response — check for a proxy or auth portal in front of the API".to_string(),
                ),
            };
        }
        let status = resp.status();
        let body = Self::read_error_body(resp).await;
        match status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                tracing::warn!(%status, body = %body, "libstack_credentials_rejected");
                CredentialCheck::Rejected(format!("LibStack rejected the token: {status}"))
            }
            _ => {
                tracing::warn!(%status, body = %body, "libstack_unavailable");
                CredentialCheck::Unavailable(describe_error(status.as_u16(), &body))
            }
        }
    }

    /// Admin means the token holds `org.manage_members` in its active org.
    async fn is_admin(&self) -> Result<bool, String> {
        let me = self.me().await?;
        Ok(me
            .get("capabilities")
            .and_then(|c| c.as_array())
            .map(|caps| {
                caps.iter()
                    .any(|c| c.as_str() == Some("org.manage_members"))
            })
            .unwrap_or(false))
    }

    async fn can_access_page(&self, page_id: i64) -> bool {
        let Some(uuid) = ids::lookup(page_id) else {
            return false;
        };
        let builder = self.authed(self.client.get(self.url(&format!("pages/{uuid}"))));
        match self.send(builder).await {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }

    /// LibStack has capabilities, not roles: the ACL prefilter is disabled
    /// and every candidate page goes through `can_access_page`, which RLS
    /// answers correctly.
    async fn list_my_roles(&self) -> Result<Option<Vec<i64>>, String> {
        Ok(None)
    }

    async fn whoami(&self) -> Result<Option<UserIdentity>, String> {
        let me = self.me().await?;
        let user = me.get("user").cloned().unwrap_or(Value::Null);
        let Some(id) = ids::parse_field(&user, "id") else {
            return Ok(None);
        };
        Ok(Some(UserIdentity {
            bookstack_user_id: ids::remember(&id),
            email: user.get("email").and_then(|v| v.as_str()).map(String::from),
            name: str_field(&user, "display_name"),
        }))
    }

    // --- Shelves: root collections ---

    async fn list_shelves(&self, count: i64, offset: i64) -> Result<Value, String> {
        let tree = self.fetch_tree().await?;
        let rows: Vec<Value> = tree
            .with_role(Role::Shelf)
            .iter()
            .filter_map(|id| tree.get(id))
            .map(|c| self.shelf_row(c))
            .collect();
        Ok(page_slice(rows, count, offset))
    }

    async fn list_all_shelves(&self) -> Result<Vec<i64>, String> {
        let tree = self.fetch_tree().await?;
        Ok(tree
            .with_role(Role::Shelf)
            .iter()
            .map(ids::remember)
            .collect())
    }

    /// Any collection id is accepted; its children are the `books`.
    async fn get_shelf(&self, id: i64) -> Result<Value, String> {
        let uuid = self.resolve(id, "shelf").await?;
        let tree = self.fetch_tree().await?;
        let c = tree.get(&uuid).ok_or_else(|| {
            describe_error(404, r#"{"error":{"code":404,"message":"not found"}}"#)
        })?;
        let mut v = self.shelf_row(c);
        v["books"] = json!(tree
            .children(Some(uuid))
            .iter()
            .filter_map(|b| tree.get(b))
            .map(|b| self.book_row(b, &tree))
            .collect::<Vec<_>>());
        v["tags"] = json!([]);
        Ok(v)
    }

    async fn create_shelf(&self, name: &str, description: &str) -> Result<Value, String> {
        let c = self.create_collection_raw(None, name, description).await?;
        Ok(self.shelf_row(&c))
    }

    /// `name` / `description` update the collection. `books` re-parents
    /// each listed book under this shelf; books missing from the list are
    /// left where they are (a LibStack collection has exactly one parent,
    /// so there is no "remove from shelf" without choosing a new one).
    async fn update_shelf(&self, id: i64, data: &Value) -> Result<Value, String> {
        let c = self.update_collection(id, data, "shelf").await?;
        if let Some(books) = data.get("books").and_then(|v| v.as_array()) {
            for b in books.iter().filter_map(|v| v.as_i64()) {
                self.update_collection(b, &json!({ "parent_id": id }), "book")
                    .await?;
            }
        }
        let tree = self.fetch_tree().await?;
        let mut v = self.shelf_row(&c);
        v["books"] = json!(tree
            .children(Some(c.id))
            .iter()
            .filter_map(|b| tree.get(b))
            .map(|b| self.book_row(b, &tree))
            .collect::<Vec<_>>());
        Ok(v)
    }

    /// Soft-deletes the collection. Unlike BookStack, the books (child
    /// collections) go with it — LibStack's delete is by subtree.
    async fn delete_shelf(&self, id: i64) -> Result<(), String> {
        let uuid = self.resolve(id, "shelf").await?;
        self.delete(&format!("collections/{uuid}")).await
    }

    // --- Books: depth-1 collections ---

    async fn list_books(&self, count: i64, offset: i64) -> Result<Value, String> {
        let tree = self.fetch_tree().await?;
        let rows: Vec<Value> = tree
            .with_role(Role::Book)
            .iter()
            .filter_map(|id| tree.get(id))
            .map(|c| self.book_row(c, &tree))
            .collect();
        Ok(page_slice(rows, count, offset))
    }

    /// Depth-1 collections, plus root collections that hold pages directly
    /// (those pages report the root as their book, so the index worker's
    /// orphan sweep must walk it as one).
    async fn list_all_books(&self) -> Result<Vec<i64>, String> {
        let snap = self.snapshot().await?;
        let mut out: Vec<i64> = snap
            .tree
            .with_role(Role::Book)
            .iter()
            .map(ids::remember)
            .collect();
        for root in snap.tree.with_role(Role::Shelf) {
            if snap.has_pages(&root) {
                out.push(ids::remember(&root));
            }
        }
        Ok(out)
    }

    /// Any collection id is accepted. `contents` lists every descendant
    /// collection as a chapter (with its pages) followed by the pages that
    /// sit directly in this collection, in BookStack's mixed-array shape.
    async fn get_book(&self, id: i64) -> Result<Value, String> {
        let uuid = self.resolve(id, "book").await?;
        let snap = self.snapshot().await?;
        let tree = &snap.tree;
        let c = tree.get(&uuid).ok_or_else(|| {
            describe_error(404, r#"{"error":{"code":404,"message":"not found"}}"#)
        })?;
        let mut contents: Vec<Value> = Vec::new();
        for ch in tree.descendants(&uuid) {
            let Some(chc) = tree.get(&ch) else { continue };
            let mut row = self.chapter_row(chc, tree);
            row["book_id"] = json!(ids::remember(&uuid));
            row["pages"] = json!(snap
                .pages_in(&ch)
                .into_iter()
                .map(|p| self.page_row(p, tree))
                .collect::<Vec<_>>());
            contents.push(row);
        }
        for p in snap.pages_in(&uuid) {
            let mut row = self.page_row(p, tree);
            row["type"] = json!("page");
            contents.push(row);
        }
        let mut v = self.book_row(c, tree);
        v["contents"] = json!(contents);
        v["tags"] = json!([]);
        Ok(v)
    }

    async fn create_book(
        &self,
        name: &str,
        description: &str,
        shelf_id: Option<i64>,
    ) -> Result<Value, String> {
        let Some(shelf_id) = shelf_id else {
            return Err(
                "shelf_id is required on LibStack: a book is a collection inside a shelf (pass the shelf's id, or use create_collection with parent_id)"
                    .to_string(),
            );
        };
        let parent = self.resolve(shelf_id, "shelf").await?;
        let c = self
            .create_collection_raw(Some(parent), name, description)
            .await?;
        let tree = self.fetch_tree().await?;
        Ok(self.book_row(&c, &tree))
    }

    async fn update_book(&self, id: i64, data: &Value) -> Result<Value, String> {
        let c = self.update_collection(id, data, "book").await?;
        let tree = self.fetch_tree().await?;
        Ok(self.book_row(&c, &tree))
    }

    async fn delete_book(&self, id: i64) -> Result<(), String> {
        let uuid = self.resolve(id, "book").await?;
        self.delete(&format!("collections/{uuid}")).await
    }

    // --- Chapters: depth ≥ 2 collections ---

    async fn list_chapters(&self, count: i64, offset: i64) -> Result<Value, String> {
        let tree = self.fetch_tree().await?;
        let rows: Vec<Value> = tree
            .with_role(Role::Chapter)
            .iter()
            .filter_map(|id| tree.get(id))
            .map(|c| self.chapter_row(c, &tree))
            .collect();
        Ok(page_slice(rows, count, offset))
    }

    async fn get_chapter(&self, id: i64) -> Result<Value, String> {
        let uuid = self.resolve(id, "chapter").await?;
        let tree = self.fetch_tree().await?;
        let c = tree.get(&uuid).ok_or_else(|| {
            describe_error(404, r#"{"error":{"code":404,"message":"not found"}}"#)
        })?;
        let pages = self.fetch_pages(Some(&uuid)).await?;
        let mut v = self.chapter_row(c, &tree);
        v["pages"] = json!(pages
            .iter()
            .map(|p| self.page_row(p, &tree))
            .collect::<Vec<_>>());
        v["tags"] = json!([]);
        Ok(v)
    }

    async fn create_chapter(
        &self,
        book_id: i64,
        name: &str,
        description: &str,
    ) -> Result<Value, String> {
        let parent = self.resolve(book_id, "book").await?;
        let c = self
            .create_collection_raw(Some(parent), name, description)
            .await?;
        let tree = self.fetch_tree().await?;
        Ok(self.chapter_row(&c, &tree))
    }

    /// `book_id` moves the chapter (re-parents the collection).
    async fn update_chapter(&self, id: i64, data: &Value) -> Result<Value, String> {
        let mut data = data.clone();
        if let Some(b) = data.get("book_id").cloned() {
            data["parent_id"] = b;
        }
        let c = self.update_collection(id, &data, "chapter").await?;
        let tree = self.fetch_tree().await?;
        Ok(self.chapter_row(&c, &tree))
    }

    async fn delete_chapter(&self, id: i64) -> Result<(), String> {
        let uuid = self.resolve(id, "chapter").await?;
        self.delete(&format!("collections/{uuid}")).await
    }

    // --- Pages ---

    async fn list_pages(&self, count: i64, offset: i64) -> Result<Value, String> {
        let snap = self.snapshot().await?;
        let rows: Vec<Value> = snap
            .pages
            .iter()
            .map(|p| self.page_row(p, &snap.tree))
            .collect();
        Ok(page_slice(rows, count, offset))
    }

    /// `/api/pages` has no `updated_at` filter, so the cut is client-side:
    /// every page newer than `since`, oldest first, at most `count`.
    async fn list_pages_updated_since(
        &self,
        since_iso_utc: &str,
        count: i64,
    ) -> Result<Vec<Value>, String> {
        let snap = self.snapshot().await?;
        let mut rows: Vec<&Value> = snap
            .pages
            .iter()
            .filter(|p| {
                p.get("updated_at")
                    .and_then(|v| v.as_str())
                    .map(|t| later_than(t, since_iso_utc))
                    .unwrap_or(false)
            })
            .collect();
        rows.sort_by(|a, b| {
            let at = a.get("updated_at").and_then(|v| v.as_str()).unwrap_or("");
            let bt = b.get("updated_at").and_then(|v| v.as_str()).unwrap_or("");
            if later_than(at, bt) {
                std::cmp::Ordering::Greater
            } else if later_than(bt, at) {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        });
        Ok(rows
            .into_iter()
            .take(count.max(0) as usize)
            .map(|p| self.page_row(p, &snap.tree))
            .collect())
    }

    async fn get_page(&self, id: i64) -> Result<Value, String> {
        let uuid = self.resolve(id, "page").await?;
        let page = self.get(&format!("pages/{uuid}"), &[]).await?;
        let tree = self.fetch_tree().await?;
        Ok(self.page_row(&page, &tree))
    }

    /// `name` → `title`, `markdown` → `markdown`, `chapter_id` / `book_id`
    /// → `collection_id`. `html` is refused: LibStack pages are markdown
    /// and there is no HTML editor to fake.
    async fn create_page(&self, data: &Value) -> Result<Value, String> {
        if data
            .get("html")
            .and_then(|v| v.as_str())
            .is_some_and(|h| !h.is_empty())
        {
            return Err(self.unavailable(
                "creating a page from `html` (pass `markdown`; LibStack has no WYSIWYG editor)",
            ));
        }
        let target = data
            .get("chapter_id")
            .and_then(|v| v.as_i64())
            .or_else(|| data.get("book_id").and_then(|v| v.as_i64()))
            .ok_or("Either book_id or chapter_id is required")?;
        let coll = self.resolve(target, "collection").await?;
        let body = json!({
            "collection_id": coll.to_string(),
            "title": data.get("name").cloned().unwrap_or(json!("")),
            "markdown": data.get("markdown").cloned().unwrap_or(json!("")),
            "tags": data.get("tags").cloned().unwrap_or(json!([])),
        });
        let page = self.post("pages", &body).await?;
        let tree = self.fetch_tree().await?;
        Ok(self.page_row(&page, &tree))
    }

    async fn update_page(&self, id: i64, data: &Value) -> Result<Value, String> {
        if data
            .get("html")
            .and_then(|v| v.as_str())
            .is_some_and(|h| !h.is_empty())
        {
            return Err(self.unavailable(
                "updating a page from `html` (pass `markdown`; LibStack has no WYSIWYG editor)",
            ));
        }
        let uuid = self.resolve(id, "page").await?;
        let mut body = json!({});
        if let Some(n) = data.get("name") {
            body["title"] = n.clone();
        }
        if let Some(m) = data.get("markdown") {
            body["markdown"] = m.clone();
        }
        if let Some(t) = data.get("tags") {
            body["tags"] = t.clone();
        }
        if let Some(s) = data.get("summary") {
            body["summary"] = s.clone();
        }
        if let Some(r) = data.get("base_revision_number") {
            body["base_revision_number"] = r.clone();
        }
        let target = data
            .get("chapter_id")
            .and_then(|v| v.as_i64())
            .or_else(|| data.get("book_id").and_then(|v| v.as_i64()));
        if let Some(t) = target {
            let coll = self.resolve(t, "collection").await?;
            body["collection_id"] = json!(coll.to_string());
        }
        let page = self.patch(&format!("pages/{uuid}"), &body).await?;
        let tree = self.fetch_tree().await?;
        Ok(self.page_row(&page, &tree))
    }

    async fn delete_page(&self, id: i64) -> Result<(), String> {
        let uuid = self.resolve(id, "page").await?;
        self.delete(&format!("pages/{uuid}")).await
    }

    // --- Search ---

    /// `GET /api/search?q=` returns at most 50 ranked hits with no paging;
    /// `page` / `count` slice that list. BookStack operator tokens are
    /// stripped from the query.
    async fn search(&self, query: &str, page: i64, count: i64) -> Result<Value, String> {
        let q = strip_bookstack_operators(query);
        if q.is_empty() {
            return Ok(json!({ "data": [], "total": 0 }));
        }
        let query: [(&str, &str); 1] = [("q", &q)];
        let (resp, tree) = tokio::try_join!(self.get("search", &query), self.fetch_tree())?;
        let hits = resp
            .get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default();
        let total = resp
            .get("total")
            .and_then(|t| t.as_i64())
            .unwrap_or(hits.len() as i64);
        let rows: Vec<Value> = hits
            .iter()
            .map(|h| {
                let id = ids::parse_field(h, "id");
                let coll = ids::parse_field(h, "collection_id");
                let kind = h.get("type").and_then(|t| t.as_str()).unwrap_or("page");
                let (item_type, url, book_id, chapter_id) = if kind == "collection" {
                    let role = id.map(|u| tree.role(&u)).unwrap_or(Role::Shelf);
                    (
                        role.as_str().to_string(),
                        id.map(|u| self.coll_url(&u)),
                        id.and_then(|u| tree.book_of(&u)).map(|b| ids::remember(&b)),
                        id.and_then(|u| tree.chapter_of(&u)).map(|c| ids::remember(&c)),
                    )
                } else {
                    (
                        "page".to_string(),
                        id.map(|u| self.page_url(&u)),
                        coll.map(|c| ids::remember(&tree.book_for_page(&c))),
                        coll.and_then(|c| tree.chapter_of(&c)).map(|c| ids::remember(&c)),
                    )
                };
                let name = h.get("title").cloned().unwrap_or(Value::Null);
                json!({
                    "id": id.map(|u| ids::remember(&u)),
                    "name": name,
                    "slug": id.map(|u| u.to_string()),
                    "type": item_type,
                    "url": url,
                    "book_id": book_id,
                    "chapter_id": chapter_id,
                    "collection_id": coll.map(|c| ids::remember(&c)),
                    "libstack_id": id.map(|u| u.to_string()),
                    "rank": h.get("rank").cloned().unwrap_or(Value::Null),
                    "preview_html": { "name": name, "content": h.get("excerpt").cloned().unwrap_or(json!("")) },
                    "tags": [],
                })
            })
            .collect();
        let offset = (page.max(1) - 1) * count.max(0);
        let mut out = page_slice(rows, count, offset);
        out["total"] = json!(total);
        Ok(out)
    }

    // --- Attachments ---

    async fn list_attachments(&self, page_id: Option<i64>) -> Result<Value, String> {
        let Some(pid) = page_id else {
            return Err(
                "list_attachments on LibStack needs page_id: attachments are stored per page"
                    .to_string(),
            );
        };
        let page = self.resolve(pid, "page").await?;
        let rows: Vec<Value> = self
            .page_attachments(&page)
            .await?
            .iter()
            .map(|a| self.attachment_row(a, Some(&page)))
            .collect();
        let total = rows.len();
        Ok(json!({ "data": rows, "total": total }))
    }

    async fn get_attachment(&self, id: i64) -> Result<Value, String> {
        let uuid = self.resolve_seen(id, "attachment")?;
        let a = self.get(&format!("attachments/{uuid}"), &[]).await?;
        Ok(self.attachment_row(&a, None))
    }

    /// BookStack link attachments have no LibStack counterpart; file
    /// attachments go through `create_file_attachment` / `upload_attachment`.
    async fn create_attachment(&self, _data: &Value) -> Result<Value, String> {
        Err(self.unavailable("link attachments (use upload_attachment for a file)"))
    }

    async fn update_attachment(&self, id: i64, data: &Value) -> Result<Value, String> {
        let uuid = self.resolve_seen(id, "attachment")?;
        let mut body = json!({});
        if let Some(n) = data.get("name") {
            body["name"] = n.clone();
        }
        if data.get("link").is_some() {
            return Err(self.unavailable("link attachments"));
        }
        let a = self.patch(&format!("attachments/{uuid}"), &body).await?;
        Ok(self.attachment_row(&a, None))
    }

    async fn delete_attachment(&self, id: i64) -> Result<(), String> {
        let uuid = self.resolve_seen(id, "attachment")?;
        self.delete(&format!("attachments/{uuid}")).await
    }

    async fn create_file_attachment(
        &self,
        name: &str,
        uploaded_to: i64,
        filename: &str,
        bytes: Vec<u8>,
        mime_type: &str,
    ) -> Result<Value, String> {
        let page = self.resolve(uploaded_to, "page").await?;
        let a = self
            .upload_blob(&page, name, filename, bytes, mime_type)
            .await?;
        Ok(self.attachment_row(&a, Some(&page)))
    }

    // --- Exports ---

    async fn export_page(&self, _id: i64, _format: ExportFormat) -> Result<String, String> {
        Err(self.unavailable("exports (get_page returns the markdown and rendered html)"))
    }
    async fn export_chapter(&self, _id: i64, _format: ExportFormat) -> Result<String, String> {
        Err(self.unavailable("exports"))
    }
    async fn export_book(&self, _id: i64, _format: ExportFormat) -> Result<String, String> {
        Err(self.unavailable("exports"))
    }

    // --- Comments ---

    async fn list_comments(&self, query: &[(&str, &str)]) -> Result<Value, String> {
        let pid = query
            .iter()
            .find(|(k, _)| *k == "filter[page_id]" || *k == "page_id")
            .and_then(|(_, v)| v.parse::<i64>().ok())
            .ok_or("list_comments on LibStack needs page_id: comments are stored per page")?;
        let page = self.resolve(pid, "page").await?;
        let v = self.get(&format!("pages/{page}/comments"), &[]).await?;
        let rows: Vec<Value> = v
            .get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .or_else(|| v.as_array().cloned())
            .unwrap_or_default()
            .iter()
            .map(|c| self.comment_row(c, Some(&page)))
            .collect();
        let total = rows.len();
        Ok(json!({ "data": rows, "total": total }))
    }

    async fn get_comment(&self, id: i64) -> Result<Value, String> {
        let uuid = self.resolve_seen(id, "comment")?;
        let c = self.get(&format!("comments/{uuid}"), &[]).await?;
        Ok(self.comment_row(&c, None))
    }

    /// Posts the caller's `markdown`; when only `html` was supplied it is
    /// sent as the markdown body (LibStack's renderer sanitises inline HTML).
    async fn create_comment(&self, data: &Value) -> Result<Value, String> {
        let pid = data
            .get("page_id")
            .and_then(|v| v.as_i64())
            .ok_or("page_id is required")?;
        let page = self.resolve(pid, "page").await?;
        let markdown = data
            .get("markdown")
            .and_then(|v| v.as_str())
            .or_else(|| data.get("html").and_then(|v| v.as_str()))
            .unwrap_or("");
        let mut body = json!({ "markdown": markdown });
        if let Some(parent) = data.get("parent_id").and_then(|v| v.as_i64()) {
            let parent = self.resolve_seen(parent, "comment")?;
            body["parent_id"] = json!(parent.to_string());
        }
        let c = self.post(&format!("pages/{page}/comments"), &body).await?;
        Ok(self.comment_row(&c, Some(&page)))
    }

    async fn update_comment(&self, id: i64, data: &Value) -> Result<Value, String> {
        let uuid = self.resolve_seen(id, "comment")?;
        let markdown = data
            .get("markdown")
            .and_then(|v| v.as_str())
            .or_else(|| data.get("html").and_then(|v| v.as_str()))
            .unwrap_or("");
        let c = self
            .patch(
                &format!("comments/{uuid}"),
                &json!({ "markdown": markdown }),
            )
            .await?;
        Ok(self.comment_row(&c, None))
    }

    async fn delete_comment(&self, id: i64) -> Result<(), String> {
        let uuid = self.resolve_seen(id, "comment")?;
        self.delete(&format!("comments/{uuid}")).await
    }

    // --- Recycle bin ---

    async fn list_recycle_bin(&self, _count: i64, _offset: i64) -> Result<Value, String> {
        Err(self.unavailable("the recycle bin"))
    }
    async fn restore_recycle_bin_item(&self, _id: i64) -> Result<Value, String> {
        Err(self.unavailable("the recycle bin"))
    }
    async fn destroy_recycle_bin_item(&self, _id: i64) -> Result<(), String> {
        Err(self.unavailable("the recycle bin"))
    }

    // --- Users ---

    /// LibStack exposes only the caller (`/api/auth/me`).
    async fn list_users(&self, _count: i64, _offset: i64) -> Result<Value, String> {
        let me = self.me().await?;
        Ok(json!({ "data": [self.user_row(&me)], "total": 1 }))
    }

    async fn get_user(&self, id: i64) -> Result<Value, String> {
        let me = self.me().await?;
        let row = self.user_row(&me);
        if row.get("id").and_then(|v| v.as_i64()) == Some(id) {
            Ok(row)
        } else {
            Err(format!(
                "user {id} is not available on LibStack: only the calling user (id {}) can be read",
                row.get("id").and_then(|v| v.as_i64()).unwrap_or(0)
            ))
        }
    }

    // --- Audit log / system ---

    async fn list_audit_log(&self, count: i64, offset: i64) -> Result<Value, String> {
        let v = self.get("activity", &[]).await?;
        let rows: Vec<Value> = v
            .get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .or_else(|| v.as_array().cloned())
            .unwrap_or_default()
            .iter()
            .map(|a| self.activity_row(a))
            .collect();
        Ok(page_slice(rows, count, offset))
    }

    async fn get_system_info(&self) -> Result<Value, String> {
        let health = self.get("health", &[]).await?;
        Ok(json!({
            "app_name": "LibStack",
            "backend": "libstack",
            "base_url": self.public_url,
            "api_url": self.base_url,
            "version": health.get("version").cloned().unwrap_or(json!("unknown")),
            "health": health,
        }))
    }

    // --- Images: attachments with an image/* MIME type ---

    async fn list_images(
        &self,
        count: i64,
        offset: i64,
        filter: &[(&str, &str)],
    ) -> Result<Value, String> {
        let pid = filter
            .iter()
            .find(|(k, _)| *k == "filter[uploaded_to]" || *k == "uploaded_to")
            .and_then(|(_, v)| v.parse::<i64>().ok())
            .ok_or("list_images on LibStack needs uploaded_to (a page id): images are attachments of a page")?;
        let page = self.resolve(pid, "page").await?;
        let rows: Vec<Value> = self
            .page_attachments(&page)
            .await?
            .iter()
            .filter(|a| str_field(a, "mime").starts_with("image/"))
            .map(|a| self.image_row(a, Some(&page)))
            .collect();
        Ok(page_slice(rows, count, offset))
    }

    async fn get_image(&self, id: i64) -> Result<Value, String> {
        let uuid = self.resolve_seen(id, "image")?;
        let a = self.get(&format!("attachments/{uuid}"), &[]).await?;
        if !str_field(&a, "mime").starts_with("image/") {
            return Err(format!("attachment {id} is not an image"));
        }
        Ok(self.image_row(&a, None))
    }

    async fn update_image(&self, id: i64, data: &Value) -> Result<Value, String> {
        let uuid = self.resolve_seen(id, "image")?;
        let mut body = json!({});
        if let Some(n) = data.get("name") {
            body["name"] = n.clone();
        }
        let a = self.patch(&format!("attachments/{uuid}"), &body).await?;
        Ok(self.image_row(&a, None))
    }

    async fn delete_image(&self, id: i64) -> Result<(), String> {
        let uuid = self.resolve_seen(id, "image")?;
        self.delete(&format!("attachments/{uuid}")).await
    }

    /// Stores the bytes as a page attachment; the returned `url` is the
    /// blob's `/blobs/{id}` path, ready for `![name](url)`.
    async fn upload_image(
        &self,
        name: &str,
        _image_type: &str,
        uploaded_to: i64,
        filename: &str,
        bytes: Vec<u8>,
        mime_type: &str,
    ) -> Result<Value, String> {
        let page = self.resolve(uploaded_to, "page").await?;
        let a = self
            .upload_blob(&page, name, filename, bytes, mime_type)
            .await?;
        Ok(self.image_row(&a, Some(&page)))
    }

    // --- Content permissions / roles ---

    async fn get_content_permissions(
        &self,
        _content_type: ContentType,
        _content_id: i64,
    ) -> Result<Value, String> {
        Err(self.unavailable(
            "content permissions (LibStack scopes access by org capability, not per item)",
        ))
    }
    async fn update_content_permissions(
        &self,
        _content_type: ContentType,
        _content_id: i64,
        _data: &Value,
    ) -> Result<Value, String> {
        Err(self.unavailable(
            "content permissions (LibStack scopes access by org capability, not per item)",
        ))
    }
    async fn list_roles(&self, _count: i64, _offset: i64) -> Result<Value, String> {
        Err(self.unavailable("roles (see the `capabilities` on get_user)"))
    }
    async fn get_role(&self, _id: i64) -> Result<Value, String> {
        Err(self.unavailable("roles (see the `capabilities` on get_user)"))
    }

    // --- Collections: the real tree ---

    async fn list_collections(&self) -> Result<Value, String> {
        let tree = self.fetch_tree().await?;
        let rows: Vec<Value> = tree.all().map(|c| self.collection_row(c, &tree)).collect();
        let total = rows.len();
        Ok(json!({ "data": rows, "total": total }))
    }

    async fn get_collection(&self, id: i64) -> Result<Value, String> {
        let uuid = self.resolve(id, "collection").await?;
        let tree = self.fetch_tree().await?;
        let c = tree.get(&uuid).ok_or_else(|| {
            describe_error(404, r#"{"error":{"code":404,"message":"not found"}}"#)
        })?;
        let pages = self.fetch_pages(Some(&uuid)).await?;
        let mut v = self.collection_row(c, &tree);
        v["pages"] = json!(pages
            .iter()
            .map(|p| self.page_row(p, &tree))
            .collect::<Vec<_>>());
        Ok(v)
    }

    async fn create_collection(
        &self,
        parent_id: Option<i64>,
        name: &str,
        description: &str,
    ) -> Result<Value, String> {
        let parent = match parent_id {
            Some(p) => Some(self.resolve(p, "collection").await?),
            None => None,
        };
        let c = self
            .create_collection_raw(parent, name, description)
            .await?;
        let tree = self.fetch_tree().await?;
        Ok(self.collection_row(&c, &tree))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    fn coll(id: Uuid, parent: Option<Uuid>, name: &str) -> Value {
        json!({
            "id": id.to_string(),
            "parent_id": parent.map(|p| p.to_string()),
            "name": name,
            "description_md": format!("about {name}"),
            "created_at": "2026-10-01T00:00:00Z",
            "updated_at": "2026-10-01T00:00:00Z",
        })
    }

    /// Shelf A > Book B > Ops > Runbooks > Backups, plus a second root.
    fn fixture() -> (Tree, [Uuid; 6]) {
        let a = u(0x1000_0000_0000_0000_0000_0000_0000_0001);
        let b = u(0x2000_0000_0000_0000_0000_0000_0000_0002);
        let ops = u(0x3000_0000_0000_0000_0000_0000_0000_0003);
        let run = u(0x4000_0000_0000_0000_0000_0000_0000_0004);
        let bak = u(0x5000_0000_0000_0000_0000_0000_0000_0005);
        let z = u(0x6000_0000_0000_0000_0000_0000_0000_0006);
        let tree = Tree::from_json(&json!([
            coll(a, None, "Household"),
            coll(b, Some(a), "Infrastructure"),
            coll(ops, Some(b), "Ops"),
            coll(run, Some(ops), "Runbooks"),
            coll(bak, Some(run), "Backups"),
            coll(z, None, "Work"),
        ]));
        (tree, [a, b, ops, run, bak, z])
    }

    // --- hierarchy mapping ---

    #[test]
    fn depth_zero_one_two_map_to_shelf_book_chapter() {
        let (tree, [a, b, ops, run, bak, z]) = fixture();
        assert_eq!(tree.depth(&a), 0);
        assert_eq!(tree.role(&a), Role::Shelf);
        assert_eq!(tree.role(&z), Role::Shelf);
        assert_eq!(tree.depth(&b), 1);
        assert_eq!(tree.role(&b), Role::Book);
        assert_eq!(tree.depth(&ops), 2);
        assert_eq!(tree.role(&ops), Role::Chapter);
        // Deeper collections are still chapters of the same book.
        assert_eq!(tree.role(&run), Role::Chapter);
        assert_eq!(tree.depth(&bak), 4);
        assert_eq!(tree.role(&bak), Role::Chapter);
        assert_eq!(tree.book_of(&bak), Some(b));
        assert_eq!(tree.shelf_of(&bak), Some(a));
        assert_eq!(tree.chapter_of(&bak), Some(bak));
        assert_eq!(tree.chapter_of(&b), None);
    }

    #[test]
    fn deep_collection_name_is_the_path_from_chapter_level() {
        let (tree, [a, b, ops, run, bak, _]) = fixture();
        assert_eq!(tree.path_name(&a), "Household");
        assert_eq!(tree.path_name(&b), "Infrastructure");
        assert_eq!(tree.path_name(&ops), "Ops");
        assert_eq!(tree.path_name(&run), "Ops / Runbooks");
        assert_eq!(tree.path_name(&bak), "Ops / Runbooks / Backups");
    }

    #[test]
    fn listings_group_by_role_in_api_order() {
        let (tree, [a, b, ops, run, bak, z]) = fixture();
        assert_eq!(tree.with_role(Role::Shelf), vec![a, z]);
        assert_eq!(tree.with_role(Role::Book), vec![b]);
        assert_eq!(tree.with_role(Role::Chapter), vec![ops, run, bak]);
        assert_eq!(tree.children(None), &[a, z]);
        assert_eq!(tree.descendants(&b), vec![ops, run, bak]);
    }

    #[test]
    fn a_page_in_a_root_collection_reports_the_root_as_its_book() {
        let (tree, [a, b, ops, _, _, _]) = fixture();
        assert_eq!(tree.book_for_page(&a), a);
        assert_eq!(tree.book_for_page(&b), b);
        assert_eq!(tree.book_for_page(&ops), b);
    }

    #[test]
    fn missing_parent_makes_a_root_and_cycles_do_not_hang() {
        let x = u(7);
        let y = u(8);
        let ghost = u(9);
        let tree = Tree::from_json(&json!([
            coll(x, Some(ghost), "Orphan"),
            coll(y, Some(y), "Self-parent"),
        ]));
        assert_eq!(tree.role(&x), Role::Shelf);
        assert_eq!(tree.role(&y), Role::Shelf);
        assert!(tree.ancestors(&y).is_empty());
    }

    #[test]
    fn tree_accepts_a_data_envelope_too() {
        let x = u(11);
        let tree = Tree::from_json(&json!({ "data": [coll(x, None, "Only")] }));
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.role(&x), Role::Shelf);
    }

    // --- id bridge ---

    #[test]
    fn integer_ids_are_stable_non_negative_and_resolve_back() {
        let a = Uuid::parse_str("0193d3c0-5e5a-7f4e-9a3b-2f6c8d1e0b7a").unwrap();
        let i1 = ids::to_i64(&a);
        let i2 = ids::to_i64(&a);
        assert_eq!(i1, i2);
        assert!(i1 >= 0);
        assert_eq!(ids::to_i64(&Uuid::from_u128(u128::MAX)), i64::MAX);
        assert_eq!(ids::lookup(i1), None);
        assert_eq!(ids::remember(&a), i1);
        assert_eq!(ids::lookup(i1), Some(a));
    }

    #[test]
    fn distinct_uuids_get_distinct_integers() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        assert_ne!(ids::to_i64(&a), ids::to_i64(&b));
    }

    // --- error translation ---

    #[test]
    fn envelope_errors_become_one_line_with_status_and_message() {
        assert_eq!(
            describe_error(404, r#"{"error":{"code":404,"message":"not found"}}"#),
            "LibStack API error: 404 not found"
        );
        assert_eq!(
            describe_error(403, r#"{"error":{"code":403,"message":"forbidden"}}"#),
            "LibStack API error: 403 forbidden"
        );
    }

    #[test]
    fn validation_errors_list_every_field() {
        let body = r#"{"error":{"code":422,"message":"validation failed","validation":{"title":["required"],"markdown":["must be at most 256 KiB","no"]}}}"#;
        assert_eq!(
            describe_error(422, body),
            "LibStack API error: 422 validation failed — markdown: must be at most 256 KiB, no; title: required"
        );
    }

    #[test]
    fn conflicts_tell_the_caller_to_re_read() {
        let s = describe_error(
            409,
            r#"{"error":{"code":409,"message":"conflict: stale base_revision_number"}}"#,
        );
        assert!(s.starts_with("LibStack API error: 409 conflict: stale base_revision_number"));
        assert!(s.contains("re-read"));
    }

    #[test]
    fn non_envelope_bodies_still_carry_the_status() {
        assert_eq!(
            describe_error(502, "<html>bad gateway</html>"),
            "LibStack API error: 502"
        );
        assert_eq!(describe_error(500, ""), "LibStack API error: 500");
        // `is_admin` matches on "403" in the string.
        assert!(describe_error(403, "nope").contains("403"));
    }

    // --- small helpers ---

    #[test]
    fn bearer_is_the_id_alone_or_the_joined_pair() {
        assert_eq!(LibStackClient::bearer_from_pair(" tok ", ""), "tok");
        assert_eq!(
            LibStackClient::bearer_from_pair("id", "secret"),
            "id:secret"
        );
    }

    #[test]
    fn token_label_is_not_the_token() {
        let c = LibStackClient::new("http://ls", "supersecrettoken", "", Client::new());
        assert!(c.token_id().starts_with("ls-"));
        assert!(!c.token_id().contains("supersecret"));
        assert_eq!(c.base_url(), "http://ls");
        let c = c.with_public_url("https://libstack.example.com/");
        assert_eq!(c.public_url(), "https://libstack.example.com");
    }

    #[test]
    fn bookstack_operators_are_stripped_from_queries() {
        assert_eq!(
            strip_bookstack_operators("{type:page} backups [kind=sop]"),
            "backups"
        );
        assert_eq!(
            strip_bookstack_operators("\"exact phrase\" -not"),
            "\"exact phrase\" -not"
        );
        assert_eq!(strip_bookstack_operators("{created_by:me}"), "");
    }

    #[test]
    fn page_slice_paginates_and_reports_total() {
        let rows: Vec<Value> = (1..=5).map(|i| json!({ "id": i })).collect();
        let v = page_slice(rows.clone(), 2, 2);
        assert_eq!(v["total"], 5);
        assert_eq!(v["data"], json!([{ "id": 3 }, { "id": 4 }]));
        let v = page_slice(rows, 10, 7);
        assert_eq!(v["data"], json!([]));
    }

    #[test]
    fn later_than_handles_fractional_seconds() {
        assert!(later_than("2026-10-01T12:00:00.5Z", "2026-10-01T12:00:00Z"));
        assert!(!later_than("2026-10-01T12:00:00Z", "2026-10-01T12:00:00Z"));
        assert!(later_than("2026-10-02T00:00:00Z", "2026-10-01T23:59:59Z"));
    }
}
