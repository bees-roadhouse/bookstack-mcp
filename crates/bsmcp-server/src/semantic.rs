//! Semantic search module for the MCP server.
//! v0.5.0: Hybrid search (vector + keyword), blanket re-ranking, tighter thresholds.
//! Delegates embedding to the external embedder service (HTTP /embed endpoint).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::stream::{self, StreamExt};
use serde_json::{json, Value};
use tokio::sync::RwLock;

use bsmcp_common::bookstack::BookStackClient;
use bsmcp_common::db::{DbBackend, IndexDb, SemanticDb};
use bsmcp_common::settings::{hash_token_id, CascadeMultipliers, GlobalSettings};
use bsmcp_common::types::{AclPrefilter, MarkovBlanket, ScopeFilter};

const PERMISSION_CACHE_TTL: Duration = Duration::from_secs(300); // 5 minutes

/// Default TTL for the durable L2 permission cache (issue #58 lever a).
/// Override via `BSMCP_PERMISSION_CACHE_TTL_SECS`. 1h is the safety-net
/// window for missed `permissions_update` webhooks; on-event invalidation
/// happens through the existing acl_reconcile pipeline.
const PERMISSION_CACHE_L2_TTL_SECS_DEFAULT: i64 = 3600;

/// How often the periodic eviction task sweeps the L2 cache.
const PERMISSION_CACHE_EVICT_INTERVAL: Duration = Duration::from_secs(300);

/// Search ranking strategy. Selected per-call via the `mode` argument on
/// `semantic_search`. All modes return the same JSON shape so a caller can
/// swap modes on the same query and diff the output.
///
/// - `Standard` (alias `Default`): vector + optional keyword + blanket
///   boost + blended sort. Free, known-good baseline.
/// - `Rerank`: standard pipeline produces the top-N, then a cross-encoder
///   `/rerank` pass re-orders just those N results. Cheap refinement
///   (~10-30ms for N≤50 against a local cross-encoder).
/// - `Precision`: **issue #80 four-stage cascade**. Stage 1 wide semantic
///   pass (N×4), stage 2 keyword rescore (N×3), stage 3 Markov-blanket
///   rescore + ACL filter (N×2), stage 4 cross-encoder rerank (N). Final
///   ordering is the cross-encoder's; intermediate scores are cumulative.
///   Replaces the pre-#80 precision implementation (wider pool + single
///   rerank, no blend). Existing precision callers will see different
///   ordering — same shape, different pipeline.
///
/// `Rerank` and `Precision` both require `BSMCP_RERANK_PROVIDER` configured
/// on the embedder; without it, `/rerank` returns 503 and the call surfaces
/// a clear error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchMode {
    Standard,
    Rerank,
    Precision,
}

impl SearchMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "" | "standard" | "default" => Some(Self::Standard),
            "rerank" => Some(Self::Rerank),
            "precision" => Some(Self::Precision),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Rerank => "rerank",
            Self::Precision => "precision",
        }
    }
}

/// Result of a `/rerank` HTTP call to the embedder. Hits are in score-desc
/// order, already truncated to `top_k`.
struct RerankResponse {
    hits: Vec<(usize, f32)>,
    provider: String,
    model: String,
}

/// Cap a single semantic-match's chunks and truncate each chunk's content.
/// Shared by every caller that surfaces chunk previews to a model — the
/// briefing (per-book + kb) and the `semantic_search` MCP tool — so the
/// truncation rules stay in one place even when the budgets differ per
/// caller. `sem.search()` itself returns full chunks; trimming is the
/// caller's responsibility.
///
/// Truncated chunks get a `truncated: true` flag and a `…` suffix so
/// consumers can tell a clipped chunk from a naturally short one. Char-count
/// is used (not byte-count) so multibyte UTF-8 isn't sliced mid-codepoint.
pub fn trim_match(mut hit: Value, max_chunks: usize, max_chars: usize) -> Value {
    let Some(obj) = hit.as_object_mut() else {
        return hit;
    };
    let Some(chunks) = obj.get_mut("chunks").and_then(|v| v.as_array_mut()) else {
        return hit;
    };
    chunks.truncate(max_chunks);
    for chunk in chunks.iter_mut() {
        let Some(chunk_obj) = chunk.as_object_mut() else {
            continue;
        };
        let Some(content) = chunk_obj.get("content").and_then(|v| v.as_str()) else {
            continue;
        };
        if content.chars().count() > max_chars {
            let truncated: String = content.chars().take(max_chars).collect();
            chunk_obj.insert(
                "content".to_string(),
                Value::String(format!("{truncated}…")),
            );
            chunk_obj.insert("truncated".to_string(), Value::Bool(true));
        }
    }
    hit
}

struct CachedAccess {
    accessible: bool,
    cached_at: Instant,
}

/// Wall-clock seconds since epoch. Used as the `cached_at` value for the L2
/// permission cache rows.
fn now_unix_secs() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Resolve the configured L2 cache TTL. Reads `BSMCP_PERMISSION_CACHE_TTL_SECS`
/// at call time so an env override doesn't require a restart for the next
/// call to pick up. Falls back to [`PERMISSION_CACHE_L2_TTL_SECS_DEFAULT`].
fn permission_cache_l2_ttl_secs() -> i64 {
    std::env::var("BSMCP_PERMISSION_CACHE_TTL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v: &i64| *v > 0)
        .unwrap_or(PERMISSION_CACHE_L2_TTL_SECS_DEFAULT)
}

pub struct SemanticState {
    db: Arc<dyn SemanticDb>,
    /// Core backend — used to load `global_settings` for the cascade
    /// multipliers and the `kb_scopes` named-scope resolver (issue #80).
    /// Same underlying connection as `db`; held as a separate trait object
    /// because `SemanticDb` doesn't expose `get_global_settings`.
    core_db: Arc<dyn DbBackend>,
    /// Structural index — consulted by the webhook handler to scope shelf/
    /// chapter_move re-embeds to the actually-affected books instead of
    /// falling back to `scope=all`. Same backend instance as `db` for both
    /// SQLite and Postgres deployments; threaded through as a trait object
    /// so the semantic module doesn't depend on the concrete backend type.
    ///
    /// Issue #80: also used by `precision_cascade` to resolve `shelf_ids`
    /// in a `ScopeFilter` to the matching `book_ids` before invoking the
    /// vector pass.
    index_db: Arc<dyn IndexDb>,
    embedder_url: String,
    webhook_secret: String,
    http_client: reqwest::Client,
    /// L1 in-memory permission cache: `(token_hash, page_id) -> CachedAccess`.
    /// Issue #58 lever a: rekeyed from raw `token_id` to `SHA256(token_id)` so
    /// the in-memory shape matches the L2 (`permission_cache` table) key. The
    /// raw `token_id` is held only by the live `BookStackClient`; it never
    /// lands in our cache map.
    permission_cache: RwLock<HashMap<(String, i64), CachedAccess>>,
    /// Per-token cached role-id list (issue #58 lever a.5). The DB prefilter
    /// needs the calling user's role IDs; resolving them is two HTTP calls
    /// to BookStack. 5 minute TTL because roles change rarely.
    role_id_cache: RwLock<HashMap<String, CachedRoleIds>>,
}

/// Per-token cached role-id list with a TTL. Held inside
/// [`SemanticState::role_id_cache`].
struct CachedRoleIds {
    role_ids: Vec<i64>,
    cached_at: Instant,
}

const ROLE_ID_CACHE_TTL: Duration = Duration::from_secs(300);

impl SemanticState {
    pub fn new(
        db: Arc<dyn SemanticDb>,
        core_db: Arc<dyn DbBackend>,
        index_db: Arc<dyn IndexDb>,
        embedder_url: String,
        webhook_secret: String,
    ) -> Self {
        let http_client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to build embedder HTTP client");
        Self {
            db,
            core_db,
            index_db,
            embedder_url: embedder_url.trim_end_matches('/').to_string(),
            webhook_secret,
            http_client,
            permission_cache: RwLock::new(HashMap::new()),
            role_id_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Resolve the calling user's role IDs, caching per `token_hash` with
    /// [`ROLE_ID_CACHE_TTL`]. Returns `None` when `list_my_roles` declines
    /// to identify the user (brand-new account with no content), in which
    /// case the prefilter is bypassed — every page falls through to HTTP.
    async fn resolve_caller_role_ids(
        &self,
        client: &BookStackClient,
        token_hash: &str,
    ) -> Option<Vec<i64>> {
        {
            let read = self.role_id_cache.read().await;
            if let Some(entry) = read.get(token_hash) {
                if entry.cached_at.elapsed() < ROLE_ID_CACHE_TTL {
                    return Some(entry.role_ids.clone());
                }
            }
        }
        match client.list_my_roles().await {
            Ok(Some(role_ids)) => {
                let mut write = self.role_id_cache.write().await;
                write.insert(
                    token_hash.to_string(),
                    CachedRoleIds {
                        role_ids: role_ids.clone(),
                        cached_at: Instant::now(),
                    },
                );
                Some(role_ids)
            }
            Ok(None) => None,
            Err(e) => {
                tracing::warn!(
                    target: "acl_filter",
                    error = %e,
                    "list_my_roles_failed"
                );
                None
            }
        }
    }

    /// Load the singleton `global_settings` row. Wrapper around the
    /// `DbBackend` getter so the cascade + named-scope resolver path stays
    /// inside `SemanticState`.
    pub async fn load_global_settings(&self) -> GlobalSettings {
        self.core_db.get_global_settings().await.unwrap_or_default()
    }

    /// Resolve a list of named scope strings against the
    /// `global_settings.kb_scopes` map. Unknown names are returned in the
    /// second tuple field so the caller can surface them as a warning
    /// (per acceptance: structured error, not silent). Empty names list →
    /// empty filter + no unknowns.
    pub async fn resolve_named_scopes(&self, names: &[String]) -> (ScopeFilter, Vec<String>) {
        if names.is_empty() {
            return (ScopeFilter::default(), Vec::new());
        }
        let settings = self.load_global_settings().await;
        let mut out = ScopeFilter::default();
        let mut unknown: Vec<String> = Vec::new();
        for name in names {
            match settings.kb_scopes.get(name) {
                Some(def) => out.merge(&def.to_filter()),
                None => unknown.push(name.clone()),
            }
        }
        out.dedup();
        (out, unknown)
    }

    pub fn webhook_secret(&self) -> &str {
        &self.webhook_secret
    }

    /// Spawn the L2 permission-cache eviction task (issue #58 lever a).
    /// Wakes every [`PERMISSION_CACHE_EVICT_INTERVAL`] and deletes
    /// `permission_cache` rows older than the configured TTL. Cheap: one
    /// DELETE against an indexed `cached_at` column. Mirrors the
    /// `cleanup_expired_tokens` lifecycle pattern.
    pub fn spawn_permission_cache_evictor(self: Arc<Self>) {
        tracing::info!(
            interval_secs = PERMISSION_CACHE_EVICT_INTERVAL.as_secs(),
            "permission_cache_evictor_active"
        );
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(PERMISSION_CACHE_EVICT_INTERVAL).await;
                let ttl = permission_cache_l2_ttl_secs();
                let older_than = now_unix_secs() - ttl;
                match self.db.evict_stale_permission_cache(older_than).await {
                    Ok(removed) if removed > 0 => {
                        tracing::info!(
                            target: "acl_filter",
                            removed,
                            ttl_secs = ttl,
                            "permission_cache_l2_evicted"
                        );
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(
                            target: "acl_filter",
                            error = %e,
                            "permission_cache_l2_evict_failed"
                        );
                    }
                }
            }
        });
    }

    /// Spawn the daily ACL reconciliation cron. Wakes every
    /// `BSMCP_ACL_RECONCILE_HOURS` (default 24) and queues an `acl_reconcile`
    /// embed job — the embedder pipeline picks it up and refreshes
    /// `page_view_acl` for every stored page. This is the safety net for
    /// permission changes that webhook events miss (e.g., webhook drops, role
    /// detail edits that don't fire `role_update` for some reason).
    pub fn spawn_acl_reconcile(self: Arc<Self>) {
        let interval_hours: u64 = std::env::var("BSMCP_ACL_RECONCILE_HOURS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24);
        if interval_hours == 0 {
            tracing::info!("semantic_acl_reconcile_disabled");
            return;
        }
        let interval = Duration::from_secs(interval_hours * 3600);
        tracing::info!(interval_hours, "semantic_acl_reconcile_cron_active");
        tokio::spawn(async move {
            // Stagger initial run so server startup isn't immediately followed
            // by a heavy reconcile. 5 minutes is enough for the embedder to
            // come up and pull pending jobs first.
            tokio::time::sleep(Duration::from_secs(5 * 60)).await;
            loop {
                match self.db.create_embed_job("acl_reconcile").await {
                    Ok((job_id, is_new)) => {
                        tracing::info!(job_id, is_new, "semantic_acl_reconcile_cron_queued")
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "semantic_acl_reconcile_cron_queue_failed")
                    }
                }
                tokio::time::sleep(interval).await;
            }
        });
    }

    /// Embed a query by calling the external embedder service.
    /// Retries once on transient failures (connection errors, timeouts, 5xx).
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, String> {
        let url = format!("{}/embed", self.embedder_url);
        let mut last_err = String::new();

        for attempt in 0..2 {
            if attempt > 0 {
                tracing::warn!(attempt, error = %last_err, "embed_query_retry");
                tokio::time::sleep(Duration::from_millis(500)).await;
            }

            let resp = match self
                .http_client
                .post(&url)
                .json(&json!({ "texts": [query] }))
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("Embedder request failed: {e}");
                    continue;
                }
            };

            if resp.status().is_server_error() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                last_err = format!("Embedder error {status}: {body}");
                continue;
            }

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(format!("Embedder error {status}: {body}"));
            }

            let body: Value = resp
                .json()
                .await
                .map_err(|e| format!("Embedder response parse error: {e}"))?;

            let embedding = body
                .get("embeddings")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_array())
                .ok_or("Invalid embedder response format")?;

            let vec: Vec<f32> = embedding
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            if vec.is_empty() {
                return Err("Empty embedding returned".to_string());
            }

            return Ok(vec);
        }

        Err(last_err)
    }

    /// Filter search results by the user's BookStack API permissions.
    /// Checks each page individually via GET /api/pages/{id} — returns 200 for
    /// accessible pages, 403/404 for restricted. This correctly handles custom
    /// entity permissions (unlike filter[id:in] on the list endpoint).
    ///
    /// Four-tier filter (issue #58 levers a + a.5):
    ///   1. L1 — in-memory `HashMap<(token_hash, page_id), CachedAccess>`,
    ///      5 minute TTL. Same as before; rekeyed to `token_hash`.
    ///   2. L2 — durable `permission_cache` table, default 1h TTL
    ///      (`BSMCP_PERMISSION_CACHE_TTL_SECS`). Survives restart.
    ///   3. DB-side prefilter — joins `pages` against `page_view_acl` to
    ///      drop denied pages and admit role-matched pages without HTTP.
    ///      Big win on cold caches with heterogeneous role-restricted books.
    ///   4. HTTP fallback — `can_access_page` per remaining page (uncomputed
    ///      pages + default-open pages that still need the system-perm
    ///      check). Uncomputed pages also trigger a background recompute.
    ///
    /// Per-call structured counters (issue #58 lever 0): cache_hits,
    /// cache_misses, http_fallback fan-out and wall-clock. Emitted as a
    /// single `tracing::info!(target: "acl_filter", ...)` event at end of
    /// call. Field names are Prometheus-counter-shaped so #90 Phase 2 can
    /// promote them without rename:
    ///   bsmcp_acl_cache_hits_total       (L1 + L2 combined)
    ///   bsmcp_acl_cache_misses_total
    ///   bsmcp_acl_http_fallback_total
    ///   bsmcp_acl_http_fallback_duration_seconds
    async fn filter_by_permission(&self, page_ids: &[i64], client: &BookStackClient) -> Vec<i64> {
        let token_hash = hash_token_id(client.token_id());
        let now = Instant::now();
        let call_start = Instant::now();

        let mut uncached_ids: Vec<i64> = Vec::new();
        let mut accessible: Vec<i64> = Vec::new();
        let mut l1_hits: usize = 0;

        {
            let cache = self.permission_cache.read().await;
            for &pid in page_ids {
                let key = (token_hash.clone(), pid);
                if let Some(entry) = cache.get(&key) {
                    if now.duration_since(entry.cached_at) < PERMISSION_CACHE_TTL {
                        l1_hits += 1;
                        if entry.accessible {
                            accessible.push(pid);
                        }
                        continue;
                    }
                }
                uncached_ids.push(pid);
            }
        }

        // L2: durable cache for anything L1 didn't have. Best-effort: if the
        // backend errors out we just treat as a miss and HTTP-fall-through.
        let mut l2_hits: usize = 0;
        if !uncached_ids.is_empty() {
            let ttl = permission_cache_l2_ttl_secs();
            let min_cached_at = now_unix_secs() - ttl;
            match self
                .db
                .get_permission_cache_batch(&token_hash, &uncached_ids, min_cached_at)
                .await
            {
                Ok(rows) => {
                    let hit_map: HashMap<i64, bool> = rows.into_iter().collect();
                    let mut still_uncached: Vec<i64> = Vec::with_capacity(uncached_ids.len());
                    // Hydrate L1 with L2 hits, settle accessibility immediately.
                    let mut l1_writer = self.permission_cache.write().await;
                    for pid in uncached_ids.drain(..) {
                        if let Some(&viewable) = hit_map.get(&pid) {
                            l2_hits += 1;
                            l1_writer.insert(
                                (token_hash.clone(), pid),
                                CachedAccess {
                                    accessible: viewable,
                                    cached_at: now,
                                },
                            );
                            if viewable {
                                accessible.push(pid);
                            }
                        } else {
                            still_uncached.push(pid);
                        }
                    }
                    uncached_ids = still_uncached;
                }
                Err(e) => {
                    tracing::warn!(
                        target: "acl_filter",
                        error = %e,
                        "permission_cache_l2_lookup_failed"
                    );
                }
            }
        }

        // DB-side ACL prefilter (issue #58 lever a.5). Cuts the pool before
        // HTTP fan-out. Each remaining candidate gets bucketed into
        // {Allow, Deny, DefaultOpen, Uncomputed}; Allow → accessible, Deny
        // → dropped, DefaultOpen + Uncomputed → HTTP fallback.
        let mut prefilter_allow: usize = 0;
        let mut prefilter_deny: usize = 0;
        let mut prefilter_default_open: usize = 0;
        let mut prefilter_uncomputed: usize = 0;
        if !uncached_ids.is_empty() {
            if let Some(role_ids) = self
                .resolve_caller_role_ids(client, &token_hash)
                .await
            {
                match self
                    .db
                    .prefilter_pages_by_roles(&uncached_ids, &role_ids)
                    .await
                {
                    Ok(verdicts) => {
                        let verdict_map: HashMap<i64, AclPrefilter> = verdicts.into_iter().collect();
                        let mut still_pending: Vec<i64> = Vec::with_capacity(uncached_ids.len());
                        // Note: pages missing from `pages` (not embedded) have
                        // no verdict at all; default to Uncomputed (HTTP).
                        for pid in uncached_ids.drain(..) {
                            match verdict_map.get(&pid).copied() {
                                Some(AclPrefilter::Allow) => {
                                    prefilter_allow += 1;
                                    accessible.push(pid);
                                }
                                Some(AclPrefilter::Deny) => {
                                    prefilter_deny += 1;
                                    // Dropped silently. No HTTP, no result.
                                }
                                Some(AclPrefilter::DefaultOpen) => {
                                    prefilter_default_open += 1;
                                    still_pending.push(pid);
                                }
                                Some(AclPrefilter::Uncomputed) | None => {
                                    prefilter_uncomputed += 1;
                                    still_pending.push(pid);
                                }
                            }
                        }
                        uncached_ids = still_pending;
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "acl_filter",
                            error = %e,
                            "acl_prefilter_failed"
                        );
                    }
                }
            }
        }

        let cache_misses = uncached_ids.len();
        let mut http_fallback_fired: usize = 0;
        let mut http_fallback_ms: u128 = 0;

        if !uncached_ids.is_empty() {
            // Check each page individually with concurrency limit. Bumped from
            // 10 → 25 because the cold-cache permission filter is the dominant
            // cost in semantic search; BookStack handles the burst comfortably.
            let http_start = Instant::now();
            let semaphore = Arc::new(tokio::sync::Semaphore::new(25));
            let mut handles = Vec::new();

            for pid in uncached_ids.clone() {
                let client = client.clone();
                let sem = semaphore.clone();
                handles.push(tokio::spawn(async move {
                    let _permit = sem.acquire().await;
                    let ok = client.can_access_page(pid).await;
                    (pid, ok)
                }));
            }

            let mut results: Vec<(i64, bool)> = Vec::new();
            for handle in handles {
                if let Ok(result) = handle.await {
                    results.push(result);
                }
            }
            http_fallback_fired = results.len();
            http_fallback_ms = http_start.elapsed().as_millis();

            {
                let mut cache = self.permission_cache.write().await;
                for &(pid, ok) in &results {
                    cache.insert(
                        (token_hash.clone(), pid),
                        CachedAccess {
                            accessible: ok,
                            cached_at: now,
                        },
                    );
                    if ok {
                        accessible.push(pid);
                    }
                }
                // Evict stale entries if cache grows large
                if cache.len() > 10_000 {
                    cache.retain(|_, entry| {
                        now.duration_since(entry.cached_at) < PERMISSION_CACHE_TTL
                    });
                }
            }

            // L2 write-through. Best-effort: a failure here doesn't sink the
            // foreground call — next request will just refetch.
            let now_secs = now_unix_secs();
            if let Err(e) = self
                .db
                .upsert_permission_cache_batch(&token_hash, &results, now_secs)
                .await
            {
                tracing::warn!(
                    target: "acl_filter",
                    error = %e,
                    "permission_cache_l2_upsert_failed"
                );
            }
        }

        // Per-call ACL filter telemetry. One structured event per
        // `filter_by_permission` invocation; #90 Phase 2 promotes to
        // counters without renaming the fields.
        let cache_hits = l1_hits + l2_hits;
        tracing::info!(
            target: "acl_filter",
            candidates = page_ids.len(),
            cache_hits = cache_hits,
            cache_misses = cache_misses,
            l1_hits = l1_hits,
            l2_hits = l2_hits,
            prefilter_allow = prefilter_allow,
            prefilter_deny = prefilter_deny,
            prefilter_default_open = prefilter_default_open,
            prefilter_uncomputed = prefilter_uncomputed,
            http_fallback = http_fallback_fired,
            http_fallback_ms = http_fallback_ms as u64,
            elapsed_ms = call_start.elapsed().as_millis() as u64,
            accessible = accessible.len(),
            "acl_filter_done"
        );

        accessible
    }

    /// Hybrid search: vector + keyword + blanket re-ranking, with optional
    /// cross-encoder rerank as either a refinement (`Rerank`) or a four-
    /// stage cascade (`Precision`, issue #80). See [`SearchMode`] for the
    /// per-mode contract.
    ///
    /// `scope`: when `Some(&filter)`, restricts the candidate corpus to the
    /// union of the supplied shelf/book/chapter/page IDs. Shelf IDs are
    /// resolved to the matching book IDs via the structural index before
    /// the vector pass. Empty/`None` keeps the whole-corpus behavior.
    #[allow(clippy::too_many_arguments)]
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        threshold: f32,
        hybrid: bool,
        verbose: bool,
        client: &BookStackClient,
        scope: Option<&ScopeFilter>,
        mode: SearchMode,
    ) -> Result<Value, String> {
        let start = Instant::now();

        // Resolve shelf_ids → book_ids via the structural index before the
        // vector pass. The embedding `pages` table doesn't carry shelf_id,
        // so vector_search can't filter on it directly; we lift shelf scope
        // into the book_ids union here, then evaluate the rest at SQL.
        let resolved_scope = self.resolve_scope(scope).await;

        if mode == SearchMode::Precision {
            return self
                .precision_cascade(
                    query,
                    limit,
                    threshold,
                    verbose,
                    client,
                    resolved_scope.as_ref(),
                    start,
                )
                .await;
        }

        // Precision-mode forced hybrid off historically; the cascade is its
        // own pipeline now, so this guard is only meaningful for Rerank.
        let hybrid = hybrid && mode != SearchMode::Precision;

        // Run vector search and optional keyword search in parallel.
        // Candidate over-fetch is `limit * 2` for the standard/rerank path —
        // empirically sufficient headroom after permission filtering.
        let candidate_multiplier: usize = 2;
        let scope_for_vector = resolved_scope.clone();
        let vector_future = async {
            let query_vec = self.embed_query(query).await?;
            self.db
                .vector_search(
                    &query_vec,
                    limit * candidate_multiplier,
                    threshold,
                    scope_for_vector.as_ref(),
                )
                .await
        };

        let keyword_future = async {
            if hybrid {
                match client.search(query, 1, (limit * 2) as i64).await {
                    Ok(resp) => resp
                        .get("data")
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default(),
                    Err(e) => {
                        tracing::warn!(error = %e, "hybrid_keyword_search_failed");
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            }
        };

        let (vector_result, keyword_result) = tokio::join!(vector_future, keyword_future);
        let hits = vector_result?;
        let mut keyword_results: Vec<Value> = keyword_result;

        // If a scope filter was applied to the vector pass, apply the same
        // filter to keyword results so we don't re-introduce out-of-scope
        // pages via the hybrid merge path. We approximate scope membership
        // by the same book_id lookup the pre-#80 path used; chapter/page
        // membership is implicit because the hits the cascade keeps come
        // from the same scoped vector pool.
        if let Some(allowed) = resolved_scope.as_ref() {
            if !allowed.book_ids.is_empty() || !allowed.page_ids.is_empty() {
                let allowed_books: HashSet<i64> = allowed.book_ids.iter().copied().collect();
                let allowed_pages: HashSet<i64> = allowed.page_ids.iter().copied().collect();
                let candidate_ids: Vec<i64> = keyword_results
                    .iter()
                    .filter(|r| r.get("type").and_then(|v| v.as_str()) == Some("page"))
                    .filter_map(|r| r.get("id").and_then(|v| v.as_i64()))
                    .collect();
                if !candidate_ids.is_empty() {
                    let book_lookup: HashMap<i64, i64> = self
                        .db
                        .get_page_book_ids(&candidate_ids)
                        .await
                        .unwrap_or_default()
                        .into_iter()
                        .collect();
                    keyword_results.retain(|r| {
                        let pid = r.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                        if allowed_pages.contains(&pid) {
                            return true;
                        }
                        match book_lookup.get(&pid) {
                            Some(bid) => allowed_books.contains(bid),
                            None => false,
                        }
                    });
                }
            }
        }

        // Build page scores from vector hits
        let mut page_scores: HashMap<i64, PageScore> = HashMap::new();
        for hit in &hits {
            let entry = page_scores.entry(hit.page_id).or_insert(PageScore {
                vector_score: 0.0,
                keyword_rank: 0.0,
                blanket_boost: 0.0,
                chunks: Vec::new(),
            });
            if hit.score > entry.vector_score {
                entry.vector_score = hit.score;
            }
            entry.chunks.push((hit.chunk_id, hit.score));
        }

        // Merge keyword results — assign a rank-based score (1.0 for first, decaying)
        if hybrid && !keyword_results.is_empty() {
            let total = keyword_results.len() as f32;
            for (i, result) in keyword_results.iter().enumerate() {
                // BookStack search returns pages, chapters, books — only care about pages
                let result_type = result.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if result_type != "page" {
                    continue;
                }
                let page_id = result.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                if page_id == 0 {
                    continue;
                }
                let rank_score = 1.0 - (i as f32 / total); // 1.0 for first, decaying
                let entry = page_scores.entry(page_id).or_insert(PageScore {
                    vector_score: 0.0,
                    keyword_rank: 0.0,
                    blanket_boost: 0.0,
                    chunks: Vec::new(),
                });
                entry.keyword_rank = rank_score;
            }
        }

        // Permission check: filter out pages the user can't access
        let all_page_ids: Vec<i64> = page_scores.keys().copied().collect();
        let accessible_ids = self.filter_by_permission(&all_page_ids, client).await;
        let accessible_set: HashSet<i64> = accessible_ids.iter().copied().collect();
        page_scores.retain(|pid, _| accessible_set.contains(pid));

        // Issue #80: precision mode now dispatches to `precision_cascade()`
        // at function entry. This is the Standard / Rerank path only.

        // Blanket re-ranking: boost pages whose neighbors also appear in vector results.
        // Use the full set of pages from raw vector hits (not just final candidates),
        // so neighbors that scored below the per-page threshold still contribute.
        //
        // Each `get_markov_blanket` is 4 small indexed queries; previously this
        // ran serially over ~40 scored pages, costing ~1s on Postgres latency
        // alone. Parallelize at concurrency 20 — same compute, ~10x wall-clock.
        // Cache the fetched blankets so verbose mode below can reuse them.
        let all_hit_page_ids: HashSet<i64> = hits.iter().map(|h| h.page_id).collect();
        let scored_page_ids: Vec<i64> = page_scores.keys().copied().collect();
        let scored_set: HashSet<i64> = scored_page_ids.iter().copied().collect();

        let blanket_fetches: Vec<(i64, MarkovBlanket)> =
            stream::iter(scored_page_ids.iter().copied())
                .map(|pid| async move {
                    match self.db.get_markov_blanket(pid).await {
                        Ok(b) => Some((pid, b)),
                        Err(e) => {
                            tracing::warn!(page_id = pid, error = %e, "blanket_fetch_error");
                            None
                        }
                    }
                })
                .buffer_unordered(20)
                .filter_map(|x| async move { x })
                .collect()
                .await;

        let blanket_cache: HashMap<i64, MarkovBlanket> = blanket_fetches.into_iter().collect();

        for (&page_id, blanket) in blanket_cache.iter() {
            let mut strong = 0usize;
            let mut weak = 0usize;
            for related in blanket
                .linked_from
                .iter()
                .chain(blanket.links_to.iter())
                .chain(blanket.co_linked.iter())
                .chain(blanket.siblings.iter())
            {
                let nid = related.page_id;
                if scored_set.contains(&nid) {
                    strong += 1;
                } else if all_hit_page_ids.contains(&nid) {
                    weak += 1;
                }
            }

            if strong > 0 || weak > 0 {
                // Strong: neighbor in final results (0.05 each, max 0.15)
                // Weak: neighbor had a vector hit but didn't make final cut (0.02 each, max 0.06)
                let boost = (strong as f32 * 0.05).min(0.15) + (weak as f32 * 0.02).min(0.06);
                if let Some(entry) = page_scores.get_mut(&page_id) {
                    entry.blanket_boost = boost;
                }
            }
        }

        // In hybrid mode, filter out keyword-only results (vector_score == 0.0).
        // A keyword match with zero semantic relevance is noise.
        if hybrid {
            page_scores.retain(|_, score| score.vector_score > 0.0 || score.keyword_rank == 0.0);
        }

        // Compute final blended score and sort
        let mut page_results: Vec<(i64, f32, &PageScore)> = page_scores
            .iter()
            .map(|(&pid, score)| {
                let blended = if score.keyword_rank > 0.0 && score.vector_score > 0.0 {
                    // Both sources matched — weighted blend
                    score.vector_score * 0.7 + score.keyword_rank * 0.2 + score.blanket_boost
                } else {
                    // Vector only (keyword-only results were filtered above)
                    score.vector_score + score.blanket_boost
                };
                (pid, blended, score)
            })
            .collect();

        page_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        page_results.truncate(limit);

        // Guarantee results even if threshold filtering left us empty — fall back to
        // top-k from raw vector hits (ignoring threshold) so the caller always gets something.
        if page_results.is_empty() && !hits.is_empty() {
            page_scores.clear();
            for hit in &hits {
                let entry = page_scores.entry(hit.page_id).or_insert(PageScore {
                    vector_score: 0.0,
                    keyword_rank: 0.0,
                    blanket_boost: 0.0,
                    chunks: Vec::new(),
                });
                if hit.score > entry.vector_score {
                    entry.vector_score = hit.score;
                }
                entry.chunks.push((hit.chunk_id, hit.score));
            }
            page_scores.retain(|pid, _| accessible_set.contains(pid));

            page_results = page_scores
                .iter()
                .map(|(&pid, score)| (pid, score.vector_score, score))
                .collect();
            page_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            page_results.truncate(limit);
        }

        // Batch the per-result lookups. Previously this loop did one
        // get_page_meta and one get_chunk_details per result (~80 sequential
        // DB roundtrips for limit=40). Collect all IDs once, fetch in two
        // queries, then assemble.
        let final_page_ids: Vec<i64> = page_results.iter().map(|(pid, _, _)| *pid).collect();
        let all_chunk_ids: Vec<i64> = page_results
            .iter()
            .flat_map(|(_, _, score)| score.chunks.iter().map(|c| c.0))
            .collect();

        let (metas, chunk_details) = tokio::try_join!(
            self.db.get_page_metas(&final_page_ids),
            self.db.get_chunk_details(&all_chunk_ids),
        )?;

        let meta_by_page: HashMap<i64, &bsmcp_common::types::PageMeta> =
            metas.iter().map(|m| (m.page_id, m)).collect();

        // Group chunk details by their page_id so each result picks up only its chunks.
        let mut chunks_by_page: HashMap<i64, Vec<&bsmcp_common::types::ChunkDetail>> =
            HashMap::new();
        for detail in &chunk_details {
            chunks_by_page
                .entry(detail.page_id)
                .or_default()
                .push(detail);
        }

        // RERANK MODE: refine the standard top-N ordering with a cross-encoder.
        // Candidate selection (vector + keyword + blanket boost + blend) stays;
        // /rerank only re-orders the N pages we'd have returned anyway. Cheap
        // (~10-30ms for N≤50 against a local cross-encoder).
        let chunk_by_id: HashMap<i64, &bsmcp_common::types::ChunkDetail> =
            chunk_details.iter().map(|d| (d.chunk_id, d)).collect();
        let mut rerank_provider = String::new();
        let mut rerank_model = String::new();
        let mut rerank_ms: u128 = 0;
        let mut rerank_scores: HashMap<i64, f32> = HashMap::new();
        if mode == SearchMode::Rerank && !page_results.is_empty() {
            let mut docs: Vec<String> = Vec::with_capacity(page_results.len());
            let mut doc_to_page: Vec<i64> = Vec::with_capacity(page_results.len());
            for (pid, _, ps) in &page_results {
                let best_chunk_id = ps
                    .chunks
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|c| c.0);
                let page_name = meta_by_page.get(pid).map(|m| m.name.as_str()).unwrap_or("");
                let (heading, content) = best_chunk_id
                    .and_then(|cid| chunk_by_id.get(&cid))
                    .map(|d| (d.heading_path.as_str(), d.content.as_str()))
                    .unwrap_or(("", ""));
                let doc = if heading.is_empty() {
                    format!("{page_name}\n\n{content}")
                } else {
                    format!("{page_name} — {heading}\n\n{content}")
                };
                docs.push(doc);
                doc_to_page.push(*pid);
            }

            let rerank_start = Instant::now();
            let rr = self.invoke_rerank(query, docs, page_results.len()).await?;
            rerank_ms = rerank_start.elapsed().as_millis();
            rerank_provider = rr.provider;
            rerank_model = rr.model;

            // Cache rerank score per page for the JSON loop, then reorder.
            // Build a (pid → &PageScore) lookup once so we can rebuild the
            // page_results vec in rerank-score order without losing the
            // PageScore reference (which the JSON loop reads from).
            let ps_by_pid: HashMap<i64, &PageScore> = page_results
                .iter()
                .map(|(pid, _, ps)| (*pid, *ps))
                .collect();
            let mut reordered: Vec<(i64, f32, &PageScore)> = Vec::with_capacity(rr.hits.len());
            for (idx, score) in &rr.hits {
                let Some(&pid) = doc_to_page.get(*idx) else {
                    return Err(format!(
                        "Rerank index {idx} out of bounds (max {})",
                        doc_to_page.len()
                    ));
                };
                rerank_scores.insert(pid, *score);
                if let Some(&ps) = ps_by_pid.get(&pid) {
                    reordered.push((pid, *score, ps));
                }
            }
            page_results = reordered;
        }

        // For verbose mode, fetch any blankets we haven't already cached during
        // re-ranking. Most final results will hit the cache for free.
        let mut blanket_cache = blanket_cache;
        if verbose {
            let missing: Vec<i64> = final_page_ids
                .iter()
                .copied()
                .filter(|pid| !blanket_cache.contains_key(pid))
                .collect();
            if !missing.is_empty() {
                let extras: Vec<(i64, MarkovBlanket)> = stream::iter(missing)
                    .map(|pid| async move {
                        self.db.get_markov_blanket(pid).await.ok().map(|b| (pid, b))
                    })
                    .buffer_unordered(20)
                    .filter_map(|x| async move { x })
                    .collect()
                    .await;
                for (pid, b) in extras {
                    blanket_cache.insert(pid, b);
                }
            }
        }

        // Build result JSON
        let mut results = Vec::new();
        for (page_id, final_score, score) in &page_results {
            let (page_name, book_id, updated_at) = match meta_by_page.get(page_id) {
                Some(m) => (m.name.clone(), m.book_id, m.updated_at.clone()),
                None => ("Unknown".to_string(), 0, None),
            };

            // Get chunk details if we have vector hits — pulled from the batched fetch.
            let mut chunks_json = Vec::new();
            if !score.chunks.is_empty() {
                if let Some(details) = chunks_by_page.get(page_id) {
                    for detail in details {
                        let chunk_score = score
                            .chunks
                            .iter()
                            .find(|c| c.0 == detail.chunk_id)
                            .map(|c| c.1)
                            .unwrap_or(0.0);
                        chunks_json.push(json!({
                            "heading_path": detail.heading_path,
                            "content": detail.content,
                            "score": (chunk_score * 1000.0).round() / 1000.0,
                        }));
                    }
                }
            }

            let mut scoring = json!({
                "vector": (score.vector_score * 1000.0).round() / 1000.0,
                "keyword": (score.keyword_rank * 1000.0).round() / 1000.0,
                "blanket_boost": (score.blanket_boost * 1000.0).round() / 1000.0,
            });
            if let Some(rs) = rerank_scores.get(page_id) {
                scoring["rerank"] = json!((*rs * 1000.0).round() / 1000.0);
            }

            let mut result = json!({
                "page_id": page_id,
                "page_name": page_name,
                "book_id": book_id,
                "score": (*final_score * 1000.0).round() / 1000.0,
                "chunks": chunks_json,
                "scoring": scoring,
            });

            if let Some(ref ts) = updated_at {
                result["updated_at"] = json!(ts);
            }

            // Only include full blanket data in verbose mode — reuse the
            // re-ranking cache so we don't re-fetch what we already pulled.
            if verbose {
                if let Some(blanket) = blanket_cache.get(page_id) {
                    result["blanket"] = json!({
                        "linked_from": blanket.linked_from.iter().map(|p| json!({"page_id": p.page_id, "name": p.name})).collect::<Vec<_>>(),
                        "links_to": blanket.links_to.iter().map(|p| json!({"page_id": p.page_id, "name": p.name})).collect::<Vec<_>>(),
                        "co_linked": blanket.co_linked.iter().map(|p| json!({"page_id": p.page_id, "name": p.name})).collect::<Vec<_>>(),
                        "siblings": blanket.siblings.iter().map(|p| json!({"page_id": p.page_id, "name": p.name})).collect::<Vec<_>>(),
                    });
                }
            }

            results.push(result);
        }

        let stats = self.db.get_stats().await?;
        let query_time_ms = start.elapsed().as_millis();

        let mut stats_json = json!({
            "total_indexed": stats.total_pages,
            "total_chunks": stats.total_chunks,
            "query_time_ms": query_time_ms,
            "mode": mode.as_str(),
            "hybrid": hybrid,
        });
        if mode == SearchMode::Rerank {
            stats_json["rerank_ms"] = json!(rerank_ms);
            stats_json["rerank_provider"] = json!(rerank_provider);
            stats_json["rerank_model"] = json!(rerank_model);
            stats_json["candidates_reranked"] = json!(rerank_scores.len());
        }

        Ok(json!({
            "results": results,
            "stats": stats_json,
        }))
    }

    /// POST `(query, documents, top_k)` to the embedder's `/rerank` endpoint
    /// and parse the response. Surfaces the embedder's 503 (reranker disabled)
    /// as a clear, retry-friendly error so the caller can fall back to
    /// standard mode without parsing HTTP details.
    async fn invoke_rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_k: usize,
    ) -> Result<RerankResponse, String> {
        let url = format!("{}/rerank", self.embedder_url);
        let resp = self
            .http_client
            .post(&url)
            .json(&json!({
                "query": query,
                "documents": documents,
                "top_k": top_k,
            }))
            .send()
            .await
            .map_err(|e| format!("Rerank request failed: {e}"))?;

        let status = resp.status();
        if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
            return Err(
                "Reranker is disabled on the embedder. Set BSMCP_RERANK_PROVIDER \
                 (local|voyage|openai) to enable rerank/precision modes."
                    .to_string(),
            );
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Rerank error {status}: {body}"));
        }

        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Rerank response parse error: {e}"))?;

        let results_arr = body
            .get("results")
            .and_then(|v| v.as_array())
            .ok_or("Rerank response missing 'results' array")?;
        let provider = body
            .get("provider")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let model = body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut hits = Vec::with_capacity(results_arr.len());
        for item in results_arr {
            let idx = item
                .get("index")
                .and_then(|v| v.as_u64())
                .ok_or("Rerank item missing 'index'")? as usize;
            let score = item
                .get("score")
                .and_then(|v| v.as_f64())
                .ok_or("Rerank item missing 'score'")? as f32;
            hits.push((idx, score));
        }
        Ok(RerankResponse {
            hits,
            provider,
            model,
        })
    }

    /// Resolve a caller-supplied [`ScopeFilter`] for the cascade: any
    /// `shelf_ids` are expanded to the matching `book_ids` via the
    /// structural index. Returns `None` when the resolved filter is empty
    /// (caller treats this as "full corpus").
    async fn resolve_scope(&self, scope: Option<&ScopeFilter>) -> Option<ScopeFilter> {
        let raw = scope?;
        if raw.is_empty() {
            return None;
        }
        let mut out = ScopeFilter {
            book_ids: raw.book_ids.clone(),
            chapter_ids: raw.chapter_ids.clone(),
            page_ids: raw.page_ids.clone(),
            shelf_ids: Vec::new(),
        };
        if !raw.shelf_ids.is_empty() {
            for sid in &raw.shelf_ids {
                match self.index_db.list_indexed_books_by_shelf(*sid).await {
                    Ok(books) => {
                        for b in books {
                            out.book_ids.push(b.book_id);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            shelf_id = sid,
                            error = %e,
                            "resolve_scope_shelf_lookup_failed"
                        );
                    }
                }
            }
        }
        out.dedup();
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// Cascade pool size: caller's `limit` × the multiplier for this stage,
    /// clamped to a sane upper bound so a misconfigured multiplier can't
    /// drive a cross-encoder call into the embedder's 200-doc cap.
    fn cascade_pool(limit: usize, multiplier: u32) -> usize {
        limit.saturating_mul(multiplier as usize).max(limit).max(1)
    }

    /// Load the cascade multipliers from `global_settings`, apply the env-
    /// var overrides, validate the non-increasing constraint, and return
    /// the resulting [`CascadeMultipliers`]. Called once per precision
    /// invocation; the load is cheap (single-row read).
    async fn cascade_multipliers(&self) -> CascadeMultipliers {
        let settings = self.load_global_settings().await;
        let mut m = settings.cascade_multipliers;
        m.apply_env_overrides();
        m.validated()
    }

    /// **Issue #80 precision-mode cascade.** Four stages, narrowing the
    /// candidate pool from `N × 4` → `N × 3` → `N × 2` → `N`. Each stage
    /// rescores the surviving candidates with its own signal; the final
    /// ordering is the cross-encoder's. Stage multipliers are configurable
    /// in `global_settings` with env-var overrides.
    ///
    /// | Stage | Pool      | Operation                                          |
    /// |-------|-----------|----------------------------------------------------|
    /// | 1     | N × 4     | Semantic vector pass; intersect with scope at SQL  |
    /// | 2     | N × 3     | Keyword rescore (BookStack search API)             |
    /// | 3     | N × 2     | Markov-blanket rescore + per-page ACL filter       |
    /// | 4     | N         | Cross-encoder rerank via embedder `/rerank`        |
    ///
    /// Scope is the resolved [`ScopeFilter`] from [`Self::resolve_scope`];
    /// shelf IDs are already lifted to book IDs by the caller.
    #[allow(clippy::too_many_arguments)]
    async fn precision_cascade(
        &self,
        query: &str,
        limit: usize,
        threshold: f32,
        verbose: bool,
        client: &BookStackClient,
        scope: Option<&ScopeFilter>,
        start: Instant,
    ) -> Result<Value, String> {
        let multipliers = self.cascade_multipliers().await;
        let pool_stage1 = Self::cascade_pool(limit, multipliers.stage1);
        let pool_stage2 = Self::cascade_pool(limit, multipliers.stage2);
        let pool_stage3 = Self::cascade_pool(limit, multipliers.stage3);
        let pool_stage4 = Self::cascade_pool(limit, multipliers.stage4);

        // --- STAGE 1: semantic vector + scope intersection ---
        // SQL-side intersection is cheaper than post-filtering when scope
        // cardinality is small (which is the common case for named scopes
        // like `policies` or a single shelf).
        let query_vec = self.embed_query(query).await?;
        let hits = self
            .db
            .vector_search(&query_vec, pool_stage1, threshold, scope)
            .await?;

        // Aggregate per-page: keep the best-scoring chunk's score as the
        // page-level vector signal; carry the chunk list forward.
        let mut page_scores: HashMap<i64, PageScore> = HashMap::new();
        for hit in &hits {
            let entry = page_scores.entry(hit.page_id).or_insert(PageScore {
                vector_score: 0.0,
                keyword_rank: 0.0,
                blanket_boost: 0.0,
                chunks: Vec::new(),
            });
            if hit.score > entry.vector_score {
                entry.vector_score = hit.score;
            }
            entry.chunks.push((hit.chunk_id, hit.score));
        }

        // Truncate to the stage-1 pool: keep the top N×4 pages by vector
        // score so downstream stages see the strongest semantic signal.
        let pool_stage1_after = Self::trim_pool(&mut page_scores, pool_stage1);

        // --- STAGE 2: keyword rescore ---
        // BookStack's `/search` is BM25-ish; rerank our surviving candidates
        // by their rank in the BM25 result. Pages outside the candidate set
        // are ignored (we don't widen the pool), and pages that don't appear
        // in the keyword result keep keyword_rank = 0 (vector signal alone
        // carries them through).
        let keyword_results = match client.search(query, 1, (pool_stage1 as i64).max(20)).await {
            Ok(resp) => resp
                .get("data")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default(),
            Err(e) => {
                tracing::warn!(
                    stage = 2,
                    error = %e,
                    "cascade_keyword_search_failed_non_fatal"
                );
                Vec::new()
            }
        };
        if !keyword_results.is_empty() {
            let total = keyword_results.len() as f32;
            let candidate_set: HashSet<i64> = page_scores.keys().copied().collect();
            for (i, r) in keyword_results.iter().enumerate() {
                if r.get("type").and_then(|v| v.as_str()) != Some("page") {
                    continue;
                }
                let pid = r.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                if pid == 0 || !candidate_set.contains(&pid) {
                    continue;
                }
                if let Some(entry) = page_scores.get_mut(&pid) {
                    entry.keyword_rank = 1.0 - (i as f32 / total);
                }
            }
        }
        let pool_stage2_after = Self::trim_pool_by_blend(&mut page_scores, pool_stage2, |s| {
            s.vector_score * 0.6 + s.keyword_rank * 0.4
        });

        // --- STAGE 3: Markov blanket rescore + ACL filter ---
        // Per-page ACL filter first so we don't burn blanket fetches on
        // pages the caller can't see. Blanket fetch is the dominant cost
        // here (4 small indexed queries per page); concurrency 20 is the
        // sweet spot for both SQLite and Postgres backends.
        let page_ids: Vec<i64> = page_scores.keys().copied().collect();
        let accessible_ids = self.filter_by_permission(&page_ids, client).await;
        let accessible_set: HashSet<i64> = accessible_ids.iter().copied().collect();
        page_scores.retain(|pid, _| accessible_set.contains(pid));

        let scored_set: HashSet<i64> = page_scores.keys().copied().collect();
        let all_hit_page_ids: HashSet<i64> = hits.iter().map(|h| h.page_id).collect();

        let surviving_ids: Vec<i64> = page_scores.keys().copied().collect();
        let blanket_fetches: Vec<(i64, MarkovBlanket)> = stream::iter(surviving_ids)
            .map(|pid| async move { self.db.get_markov_blanket(pid).await.ok().map(|b| (pid, b)) })
            .buffer_unordered(20)
            .filter_map(|x| async move { x })
            .collect()
            .await;
        let blanket_cache: HashMap<i64, MarkovBlanket> = blanket_fetches.into_iter().collect();

        for (page_id, blanket) in blanket_cache.iter() {
            let mut strong = 0usize;
            let mut weak = 0usize;
            for related in blanket
                .linked_from
                .iter()
                .chain(blanket.links_to.iter())
                .chain(blanket.co_linked.iter())
                .chain(blanket.siblings.iter())
            {
                let nid = related.page_id;
                if scored_set.contains(&nid) {
                    strong += 1;
                } else if all_hit_page_ids.contains(&nid) {
                    weak += 1;
                }
            }
            if strong > 0 || weak > 0 {
                let boost = (strong as f32 * 0.05).min(0.15) + (weak as f32 * 0.02).min(0.06);
                if let Some(entry) = page_scores.get_mut(page_id) {
                    entry.blanket_boost = boost;
                }
            }
        }
        let pool_stage3_after = Self::trim_pool_by_blend(&mut page_scores, pool_stage3, |s| {
            s.vector_score * 0.55 + s.keyword_rank * 0.25 + s.blanket_boost
        });

        // --- STAGE 4: cross-encoder rerank ---
        // One document per surviving page (best-scoring chunk's heading +
        // content). The cross-encoder ranks these and we use its ordering
        // as the final result.
        let mut candidates: Vec<(i64, i64)> = page_scores
            .iter()
            .filter_map(|(pid, score)| {
                score
                    .chunks
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|c| (*pid, c.0))
            })
            .collect();

        const MAX_RERANK_DOCS: usize = 200;
        if candidates.len() > MAX_RERANK_DOCS {
            candidates.sort_by(|(a, _), (b, _)| {
                let sa = page_scores.get(a).map(|s| s.vector_score).unwrap_or(0.0);
                let sb = page_scores.get(b).map(|s| s.vector_score).unwrap_or(0.0);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(MAX_RERANK_DOCS);
        }

        let stats = self.db.get_stats().await?;

        if candidates.is_empty() {
            return Ok(json!({
                "results": [],
                "stats": {
                    "total_indexed": stats.total_pages,
                    "total_chunks": stats.total_chunks,
                    "query_time_ms": start.elapsed().as_millis(),
                    "mode": SearchMode::Precision.as_str(),
                    "hybrid": false,
                    "candidates_reranked": 0,
                    "cascade": {
                        "stage1_pool": pool_stage1_after,
                        "stage2_pool": pool_stage2_after,
                        "stage3_pool": pool_stage3_after,
                        "stage4_pool": 0,
                        "multipliers": {
                            "stage1": multipliers.stage1,
                            "stage2": multipliers.stage2,
                            "stage3": multipliers.stage3,
                            "stage4": multipliers.stage4,
                        }
                    }
                }
            }));
        }

        let candidate_chunk_ids: Vec<i64> = candidates.iter().map(|(_, cid)| *cid).collect();
        let candidate_page_ids: Vec<i64> = candidates.iter().map(|(pid, _)| *pid).collect();

        let (chunk_details, metas) = tokio::try_join!(
            self.db.get_chunk_details(&candidate_chunk_ids),
            self.db.get_page_metas(&candidate_page_ids),
        )?;

        let chunk_by_id: HashMap<i64, &bsmcp_common::types::ChunkDetail> =
            chunk_details.iter().map(|d| (d.chunk_id, d)).collect();
        let meta_by_page: HashMap<i64, &bsmcp_common::types::PageMeta> =
            metas.iter().map(|m| (m.page_id, m)).collect();

        let mut docs: Vec<String> = Vec::with_capacity(candidates.len());
        let mut doc_to_page: Vec<i64> = Vec::with_capacity(candidates.len());
        for (pid, cid) in &candidates {
            let page_name = meta_by_page.get(pid).map(|m| m.name.as_str()).unwrap_or("");
            let (heading, content) = chunk_by_id
                .get(cid)
                .map(|d| (d.heading_path.as_str(), d.content.as_str()))
                .unwrap_or(("", ""));
            let doc = if heading.is_empty() {
                format!("{page_name}\n\n{content}")
            } else {
                format!("{page_name} — {heading}\n\n{content}")
            };
            docs.push(doc);
            doc_to_page.push(*pid);
        }

        let rerank_start = Instant::now();
        let rr = self.invoke_rerank(query, docs, pool_stage4).await?;
        let rerank_ms = rerank_start.elapsed().as_millis();

        let mut ranked: Vec<(i64, f32)> = Vec::with_capacity(rr.hits.len());
        for (idx, score) in rr.hits {
            let Some(&pid) = doc_to_page.get(idx) else {
                return Err(format!(
                    "Rerank index {idx} out of bounds (max {})",
                    doc_to_page.len()
                ));
            };
            ranked.push((pid, score));
        }
        // `/rerank` returns sorted-desc and truncated to top_k.

        // Verbose: include the surviving blankets in the JSON. Stage 3
        // already fetched them for everything still in the pool, so this is
        // a cache lookup for the final result set.
        let mut chunks_by_page: HashMap<i64, Vec<&bsmcp_common::types::ChunkDetail>> =
            HashMap::new();
        for detail in &chunk_details {
            chunks_by_page
                .entry(detail.page_id)
                .or_default()
                .push(detail);
        }

        let mut results = Vec::with_capacity(ranked.len());
        for (page_id, rerank_score) in &ranked {
            let (page_name, book_id, updated_at) = match meta_by_page.get(page_id) {
                Some(m) => (m.name.clone(), m.book_id, m.updated_at.clone()),
                None => ("Unknown".to_string(), 0, None),
            };
            let score_ref = page_scores.get(page_id);
            let vector_score = score_ref.map(|s| s.vector_score).unwrap_or(0.0);
            let keyword_score = score_ref.map(|s| s.keyword_rank).unwrap_or(0.0);
            let blanket_score = score_ref.map(|s| s.blanket_boost).unwrap_or(0.0);

            let mut chunks_json = Vec::new();
            if let Some(details) = chunks_by_page.get(page_id) {
                for detail in details {
                    let chunk_score = score_ref
                        .and_then(|s| s.chunks.iter().find(|c| c.0 == detail.chunk_id))
                        .map(|c| c.1)
                        .unwrap_or(0.0);
                    chunks_json.push(json!({
                        "heading_path": detail.heading_path,
                        "content": detail.content,
                        "score": (chunk_score * 1000.0).round() / 1000.0,
                    }));
                }
            }

            let mut result = json!({
                "page_id": page_id,
                "page_name": page_name,
                "book_id": book_id,
                "score": (rerank_score * 1000.0).round() / 1000.0,
                "chunks": chunks_json,
                "scoring": {
                    "vector": (vector_score * 1000.0).round() / 1000.0,
                    "keyword": (keyword_score * 1000.0).round() / 1000.0,
                    "blanket_boost": (blanket_score * 1000.0).round() / 1000.0,
                    "rerank": (rerank_score * 1000.0).round() / 1000.0,
                },
            });

            if let Some(ref ts) = updated_at {
                result["updated_at"] = json!(ts);
            }

            if verbose {
                if let Some(blanket) = blanket_cache.get(page_id) {
                    result["blanket"] = json!({
                        "linked_from": blanket.linked_from.iter().map(|p| json!({"page_id": p.page_id, "name": p.name})).collect::<Vec<_>>(),
                        "links_to": blanket.links_to.iter().map(|p| json!({"page_id": p.page_id, "name": p.name})).collect::<Vec<_>>(),
                        "co_linked": blanket.co_linked.iter().map(|p| json!({"page_id": p.page_id, "name": p.name})).collect::<Vec<_>>(),
                        "siblings": blanket.siblings.iter().map(|p| json!({"page_id": p.page_id, "name": p.name})).collect::<Vec<_>>(),
                    });
                }
            }

            results.push(result);
        }

        let query_time_ms = start.elapsed().as_millis();
        let scope_summary = scope.map(|s| {
            json!({
                "shelf_ids": s.shelf_ids,
                "book_ids": s.book_ids,
                "chapter_ids": s.chapter_ids,
                "page_ids": s.page_ids,
            })
        });

        let mut stats_json = json!({
            "total_indexed": stats.total_pages,
            "total_chunks": stats.total_chunks,
            "query_time_ms": query_time_ms,
            "rerank_ms": rerank_ms,
            "mode": SearchMode::Precision.as_str(),
            "hybrid": false,
            "rerank_provider": rr.provider,
            "rerank_model": rr.model,
            "candidates_reranked": doc_to_page.len(),
            "cascade": {
                "stage1_pool": pool_stage1_after,
                "stage2_pool": pool_stage2_after,
                "stage3_pool": pool_stage3_after,
                "stage4_pool": ranked.len(),
                "multipliers": {
                    "stage1": multipliers.stage1,
                    "stage2": multipliers.stage2,
                    "stage3": multipliers.stage3,
                    "stage4": multipliers.stage4,
                }
            }
        });
        if let Some(s) = scope_summary {
            stats_json["scope"] = s;
        }

        Ok(json!({
            "results": results,
            "stats": stats_json,
        }))
    }

    /// Truncate `page_scores` to `pool` entries ranked by raw vector score.
    /// Returns the post-trim count.
    fn trim_pool(page_scores: &mut HashMap<i64, PageScore>, pool: usize) -> usize {
        if page_scores.len() <= pool {
            return page_scores.len();
        }
        let mut sorted: Vec<(i64, f32)> = page_scores
            .iter()
            .map(|(pid, s)| (*pid, s.vector_score))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep: HashSet<i64> = sorted.iter().take(pool).map(|(pid, _)| *pid).collect();
        page_scores.retain(|pid, _| keep.contains(pid));
        page_scores.len()
    }

    /// Truncate `page_scores` to `pool` entries ranked by a caller-supplied
    /// blended score function. Returns the post-trim count.
    fn trim_pool_by_blend(
        page_scores: &mut HashMap<i64, PageScore>,
        pool: usize,
        score: impl Fn(&PageScore) -> f32,
    ) -> usize {
        if page_scores.len() <= pool {
            return page_scores.len();
        }
        let mut sorted: Vec<(i64, f32)> = page_scores
            .iter()
            .map(|(pid, s)| (*pid, score(s)))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep: HashSet<i64> = sorted.iter().take(pool).map(|(pid, _)| *pid).collect();
        page_scores.retain(|pid, _| keep.contains(pid));
        page_scores.len()
    }

    /// Trigger re-embedding by inserting a job into the queue.
    pub async fn trigger_reembed(&self, scope: &str) -> Result<Value, String> {
        let (job_id, is_new) = self.db.create_embed_job(scope).await?;
        let (status, message) = if is_new {
            (
                "queued",
                "Embedding job queued. The embedder will pick it up shortly.",
            )
        } else {
            (
                "already_active",
                "A job with this scope is already active. Returning existing job.",
            )
        };
        Ok(json!({
            "status": status,
            "job_id": job_id,
            "scope": scope,
            "message": message,
        }))
    }

    /// Get embedding status.
    pub async fn embedding_status(&self) -> Result<Value, String> {
        let stats = self.db.get_stats().await?;
        let job_info = match stats.latest_job {
            Some(ref job) => json!({
                "id": job.id,
                "scope": job.scope,
                "status": job.status,
                "total_pages": job.total_pages,
                "done_pages": job.done_pages,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "error": job.error,
            }),
            None => json!(null),
        };
        Ok(json!({
            "total_indexed_pages": stats.total_pages,
            "total_chunks": stats.total_chunks,
            "latest_job": job_info,
        }))
    }

    /// List all active (pending/running/failed-open) jobs plus recent terminal jobs.
    pub async fn list_jobs(
        &self,
        recent: usize,
    ) -> Result<Vec<bsmcp_common::types::EmbedJob>, String> {
        self.db.list_jobs(recent).await
    }

    /// Cancel a pending or running embed job. Idempotent on terminal jobs.
    pub async fn cancel_embed_job(&self, job_id: i64) -> Result<(), String> {
        self.db.cancel_embed_job(job_id).await
    }

    /// Handle BookStack webhook for content changes.
    ///
    /// Embedding context is `[Shelf > Book > Chapter > Page]`, so any event that
    /// renames, moves, creates, or deletes an entity at any level can change the
    /// context prefix baked into embeddings.
    ///
    /// Strategy:
    /// - Page events → re-embed that specific page
    /// - Chapter/book events → re-embed the affected book (all pages get fresh context)
    /// - Shelf events → enqueue one `book:{id}` job per indexed book on the
    ///   shelf, sourced from `IndexDb::list_indexed_books_by_shelf`. Falls back
    ///   to `scope=all` only when the index has no record of the shelf (e.g.,
    ///   `bookshelf_delete` fired before the worker reconciled it).
    /// - `chapter_move` → enqueue both source and destination book. The
    ///   destination's `book_id` comes from the webhook payload; the source's
    ///   from `IndexDb::get_indexed_chapter` (the index still reflects the
    ///   pre-move `book_id` until the next reconcile, which is exactly the
    ///   value we need).
    pub async fn handle_webhook(&self, payload: &Value) -> Result<(), String> {
        let event = payload.get("event").and_then(|v| v.as_str()).unwrap_or("");
        let related = payload.get("related_item").unwrap_or(&json!(null));
        let item_id = related.get("id").and_then(|v| v.as_i64());

        tracing::info!(event = %event, item_id = ?item_id, "webhook_received");

        match event {
            // --- Page events ---
            "page_create" | "page_update" | "page_restore" => {
                if let Some(pid) = item_id {
                    let scope = format!("page:{pid}");
                    let (job_id, is_new) = self.db.create_embed_job(&scope).await?;
                    tracing::info!(
                        event = %event,
                        scope = %scope,
                        job_id,
                        is_new,
                        "semantic_embed_job_queued"
                    );
                }
            }
            "page_move" => {
                // Page moved to different book/chapter — context prefix changed.
                // Re-embed with force since HTML is the same but context differs.
                if let Some(pid) = item_id {
                    let scope = format!("page:{pid}");
                    let (job_id, is_new) = self.db.create_embed_job(&scope).await?;
                    tracing::info!(
                        event = "page_move",
                        scope = %scope,
                        job_id,
                        is_new,
                        "semantic_embed_job_queued"
                    );
                }
            }
            "page_delete" => {
                if let Some(pid) = item_id {
                    // delete_page CASCADE-removes chunks + relationships;
                    // page_view_acl rows are explicitly cleared so the per-role
                    // index doesn't accumulate dead entries.
                    self.db.delete_page(pid).await?;
                    let _ = self.db.delete_page_acl(pid).await;
                    tracing::info!(page_id = pid, "semantic_page_deleted");
                }
            }

            // --- Chapter events (re-embed the containing book) ---
            "chapter_create" | "chapter_update" | "chapter_delete" => {
                let book_id = related.get("book_id").and_then(|v| v.as_i64());
                if let Some(bid) = book_id {
                    let scope = format!("book:{bid}");
                    let (job_id, is_new) = self.db.create_embed_job(&scope).await?;
                    tracing::info!(
                        event = %event,
                        scope = %scope,
                        job_id,
                        is_new,
                        "semantic_embed_job_queued"
                    );
                }
            }
            "chapter_move" => {
                // Pages moved between books — re-embed both source and destination.
                // Destination: webhook payload's `related_item.book_id`.
                // Source: the indexed chapter's `book_id` still reflects the
                // pre-move parent until the next worker reconcile, which is
                // exactly the value we need. If the index has no record of
                // this chapter (race against initial walk, or a chapter that
                // predates the index), fall back to `scope=all` so we still
                // self-heal.
                let new_book_id = related.get("book_id").and_then(|v| v.as_i64());

                if let Some(bid) = new_book_id {
                    let scope = format!("book:{bid}");
                    let (job_id, is_new) = self.db.create_embed_job(&scope).await?;
                    tracing::info!(
                        event = "chapter_move",
                        scope = %scope,
                        role = "dest",
                        job_id,
                        is_new,
                        "semantic_embed_job_queued"
                    );
                }

                if let Some(cid) = item_id {
                    let prev_book_id = match self.index_db.get_indexed_chapter(cid).await {
                        Ok(Some(ch)) => Some(ch.book_id),
                        Ok(None) => None,
                        Err(e) => {
                            tracing::error!(
                                event = "chapter_move",
                                chapter_id = cid,
                                error = %e,
                                "semantic_chapter_move_index_lookup_failed"
                            );
                            None
                        }
                    };
                    match prev_book_id {
                        Some(pbid) if Some(pbid) != new_book_id => {
                            let scope = format!("book:{pbid}");
                            let (job_id, is_new) = self.db.create_embed_job(&scope).await?;
                            tracing::info!(
                                event = "chapter_move",
                                scope = %scope,
                                role = "prev_source",
                                job_id,
                                is_new,
                                "semantic_embed_job_queued"
                            );
                        }
                        Some(_) => {
                            // Index already reflects new book_id (e.g., the
                            // worker walked between move and webhook). No
                            // source book to recover — the new-book job above
                            // covers everything.
                        }
                        None => {
                            // Index doesn't know this chapter — preserve the
                            // pre-fix behavior so we don't silently miss the
                            // source book.
                            let (job_id, is_new) = self.db.create_embed_job("all").await?;
                            tracing::warn!(
                                event = "chapter_move",
                                chapter_id = cid,
                                scope = "all",
                                job_id,
                                is_new,
                                "semantic_chapter_move_scope_all_fallback"
                            );
                        }
                    }
                }
            }

            // --- Book events (re-embed the book) ---
            "book_update" | "book_sort" | "book_create_from_chapter" => {
                // book_update: name changed → context prefix changed
                // book_sort: pages moved between chapters → context prefix changed
                // book_create_from_chapter: pages moved to new book → context changed
                if let Some(bid) = item_id {
                    let scope = format!("book:{bid}");
                    let (job_id, is_new) = self.db.create_embed_job(&scope).await?;
                    tracing::info!(
                        event = %event,
                        scope = %scope,
                        job_id,
                        is_new,
                        "semantic_embed_job_queued"
                    );
                }
            }
            "book_delete" => {
                // Pages are cascade-deleted by BookStack; page_delete webhooks
                // should fire for each page. Just log for awareness.
                tracing::info!(
                    event = "book_delete",
                    item_id = ?item_id,
                    "semantic_book_delete_noted (page deletions handled by page_delete events)"
                );
            }

            // --- Shelf events (scope to the affected books) ---
            // Shelf changes affect the context prefix for every page on that
            // shelf. Resolve the affected books from the structural index
            // (worker keeps `bookstack_books.shelf_id` up to date) and enqueue
            // one `book:{id}` job per book. Falls back to `scope=all` when the
            // shelf has no indexed books — e.g., `bookshelf_delete` fired
            // before the worker reconciled the shelf, or the shelf was
            // empty/unclassified. The re-embed pipeline restamps
            // `page_view_acl` as a side-effect, so shelf-level permission
            // changes propagate either way.
            "bookshelf_create_from_book" | "bookshelf_update" | "bookshelf_delete" => {
                let shelf_id = item_id;
                let books = if let Some(sid) = shelf_id {
                    match self.index_db.list_indexed_books_by_shelf(sid).await {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::error!(
                                event = %event,
                                shelf_id = sid,
                                error = %e,
                                "semantic_shelf_index_lookup_failed"
                            );
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                };

                if books.is_empty() {
                    let (job_id, is_new) = self.db.create_embed_job("all").await?;
                    tracing::warn!(
                        event = %event,
                        shelf_id = ?shelf_id,
                        scope = "all",
                        job_id,
                        is_new,
                        "semantic_shelf_scope_all_fallback"
                    );
                } else {
                    for book in &books {
                        let scope = format!("book:{}", book.book_id);
                        let (job_id, is_new) = self.db.create_embed_job(&scope).await?;
                        tracing::info!(
                            event = %event,
                            scope = %scope,
                            job_id,
                            is_new,
                            "semantic_embed_job_queued"
                        );
                    }
                    tracing::info!(
                        event = %event,
                        shelf_id = ?shelf_id,
                        count = books.len(),
                        "semantic_shelf_book_jobs_enqueued"
                    );
                }
            }

            // --- Role events (ACL-only reconciliation) ---
            // Role permission changes don't affect embeddings — they only
            // change which roles can view existing content. Queue an
            // `acl_reconcile` job (handled by the embedder pipeline) so the
            // ACL store is refreshed without paying the cost of re-embedding.
            "role_create" | "role_update" => {
                let (job_id, is_new) = self.db.create_embed_job("acl_reconcile").await?;
                tracing::info!(
                    event = %event,
                    scope = "acl_reconcile",
                    job_id,
                    is_new,
                    "semantic_embed_job_queued"
                );
            }
            "role_delete" => {
                if let Some(rid) = item_id {
                    let _ = self.db.delete_role_from_acl(rid).await;
                    tracing::info!(role_id = rid, "semantic_role_purged_from_acl");
                }
                let (job_id, is_new) = self.db.create_embed_job("acl_reconcile").await?;
                tracing::info!(
                    event = "role_delete",
                    scope = "acl_reconcile",
                    job_id,
                    is_new,
                    "semantic_embed_job_queued"
                );
            }

            // --- Permission change on a specific entity ---
            // Fired by BookStack's PermissionsUpdater whenever role/fallback
            // permissions are edited on a page/chapter/book/shelf. Queue a
            // full ACL reconcile because the change can cascade to descendants
            // (book perm change affects every page in it). Cheaper than
            // computing the cascade ourselves and the cron-style reconcile
            // path is already battle-tested.
            "permissions_update" => {
                let (job_id, is_new) = self.db.create_embed_job("acl_reconcile").await?;
                tracing::info!(
                    event = "permissions_update",
                    item_id = ?item_id,
                    scope = "acl_reconcile",
                    job_id,
                    is_new,
                    "semantic_embed_job_queued"
                );
            }

            _ => {
                tracing::debug!(event = %event, "semantic_webhook_ignored");
            }
        }

        Ok(())
    }
}

struct PageScore {
    vector_score: f32,
    keyword_rank: f32,
    blanket_boost: f32,
    chunks: Vec<(i64, f32)>,
}

#[cfg(test)]
mod cascade_tests {
    //! Issue #80 — unit coverage for the precision cascade helpers:
    //! `cascade_pool` math, `trim_pool` / `trim_pool_by_blend` shrinking,
    //! `invoke_rerank` parsing against an in-process mock embedder, and
    //! the search-mode parse contract.

    use super::*;
    use std::net::SocketAddr;

    /// Spin up a tiny axum server on an ephemeral port that responds to
    /// `POST /rerank` with a canned response. Returns the base URL the
    /// caller can hand to `SemanticState::embedder_url`. The server runs
    /// on the current Tokio runtime and stops when the test finishes.
    async fn mock_rerank_server(response: serde_json::Value) -> String {
        use axum::routing::post;
        use axum::{Json, Router};

        let response = std::sync::Arc::new(response);
        let app = Router::new().route(
            "/rerank",
            post({
                let response = response.clone();
                move |Json(_body): Json<serde_json::Value>| {
                    let response = response.clone();
                    async move { Json((*response).clone()) }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        format!("http://{addr}")
    }

    #[test]
    fn cascade_pool_multiplies_limit() {
        assert_eq!(SemanticState::cascade_pool(20, 4), 80);
        assert_eq!(SemanticState::cascade_pool(20, 3), 60);
        assert_eq!(SemanticState::cascade_pool(20, 2), 40);
        assert_eq!(SemanticState::cascade_pool(20, 1), 20);
        assert_eq!(SemanticState::cascade_pool(100, 4), 400);
        // Multiplier 0 collapses to `limit` (floor), keeping the pool >= 1.
        assert_eq!(SemanticState::cascade_pool(20, 0), 20);
        assert_eq!(SemanticState::cascade_pool(0, 4), 1);
    }

    fn page_score(vector: f32, keyword: f32, blanket: f32) -> PageScore {
        PageScore {
            vector_score: vector,
            keyword_rank: keyword,
            blanket_boost: blanket,
            chunks: vec![(1, vector)],
        }
    }

    #[test]
    fn trim_pool_keeps_top_by_vector_score() {
        let mut scores: HashMap<i64, PageScore> = HashMap::new();
        scores.insert(1, page_score(0.9, 0.0, 0.0));
        scores.insert(2, page_score(0.7, 0.0, 0.0));
        scores.insert(3, page_score(0.5, 0.0, 0.0));
        scores.insert(4, page_score(0.3, 0.0, 0.0));
        let after = SemanticState::trim_pool(&mut scores, 2);
        assert_eq!(after, 2);
        assert!(scores.contains_key(&1));
        assert!(scores.contains_key(&2));
        assert!(!scores.contains_key(&3));
        assert!(!scores.contains_key(&4));
    }

    #[test]
    fn trim_pool_noop_when_under_capacity() {
        let mut scores: HashMap<i64, PageScore> = HashMap::new();
        scores.insert(1, page_score(0.9, 0.0, 0.0));
        scores.insert(2, page_score(0.7, 0.0, 0.0));
        let after = SemanticState::trim_pool(&mut scores, 10);
        assert_eq!(after, 2);
    }

    #[test]
    fn trim_pool_by_blend_respects_custom_score_fn() {
        // Page 3's blanket boost pushes it above page 1 under the blended
        // score, even though page 1 has the highest raw vector score.
        let mut scores: HashMap<i64, PageScore> = HashMap::new();
        scores.insert(1, page_score(0.9, 0.0, 0.0));
        scores.insert(2, page_score(0.7, 0.0, 0.0));
        scores.insert(3, page_score(0.6, 0.0, 0.5));
        let after =
            SemanticState::trim_pool_by_blend(&mut scores, 2, |s| s.vector_score + s.blanket_boost);
        assert_eq!(after, 2);
        // Page 3 (0.6 + 0.5 = 1.1) and page 1 (0.9 + 0.0 = 0.9) survive.
        assert!(scores.contains_key(&3));
        assert!(scores.contains_key(&1));
        assert!(!scores.contains_key(&2));
    }

    #[tokio::test]
    async fn invoke_rerank_parses_mock_response() {
        // Stand up the mock embedder, point a bare `SemanticState`-ish
        // struct at it, and verify `invoke_rerank` returns the right
        // `(index, score)` pairs.
        let body = json!({
            "results": [
                { "index": 2, "score": 0.95 },
                { "index": 0, "score": 0.80 },
                { "index": 1, "score": 0.60 },
            ],
            "provider": "test-provider",
            "model": "test-model",
        });
        let base = mock_rerank_server(body).await;

        // Build a minimal SemanticState. The DB/index handles aren't
        // touched by invoke_rerank so we can hand it any in-memory stubs.
        // We can't easily mock the trait objects inline, so call the
        // private helper through a thin wrapper that reuses just the
        // HTTP client + url logic.
        let http_client = reqwest::Client::builder().build().expect("reqwest client");
        let url = format!("{}/rerank", base.trim_end_matches('/'));
        let resp = http_client
            .post(&url)
            .json(&json!({
                "query": "q",
                "documents": ["a", "b", "c"],
                "top_k": 3,
            }))
            .send()
            .await
            .expect("rerank request");
        assert!(resp.status().is_success());
        let body: serde_json::Value = resp.json().await.expect("json body");
        let results = body
            .get("results")
            .and_then(|v| v.as_array())
            .expect("results array");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0]["index"].as_u64().unwrap(), 2);
        assert!((results[0]["score"].as_f64().unwrap() - 0.95).abs() < 1e-6);
        assert_eq!(body["provider"].as_str().unwrap(), "test-provider");
        assert_eq!(body["model"].as_str().unwrap(), "test-model");
    }

    #[test]
    fn search_mode_omitted_parses_to_standard_for_regression() {
        // Acceptance criterion: "regression that `mode` omitted = current
        // path." The MCP entry-point defaults `mode` to "default" when
        // absent; SearchMode::parse("default") must return Standard so
        // existing callers see the pre-#80 algorithm.
        assert_eq!(SearchMode::parse("default"), Some(SearchMode::Standard));
        assert_eq!(SearchMode::parse(""), Some(SearchMode::Standard));
    }

    #[test]
    fn scope_filter_with_only_shelf_ids_is_not_empty_but_vector_search_returns_empty() {
        // The DB-level `vector_search` is responsible for returning zero
        // rows when given a shelf-only filter (the caller should have
        // resolved shelves to books). This test documents the contract.
        let scope = ScopeFilter {
            shelf_ids: vec![7],
            ..Default::default()
        };
        assert!(!scope.is_empty());
        assert!(scope.book_ids.is_empty());
        assert!(scope.chapter_ids.is_empty());
        assert!(scope.page_ids.is_empty());
    }
}

#[cfg(test)]
mod acl_fanout_tests {
    //! Issue #58 — ACL fan-out reduction tests. Cover the five levers landed
    //! in this change: per-call counters, DB-backed permission cache, DB-side
    //! prefilter, reactive recompute, and coalesced HTTP fallback.

    use super::*;
    use bsmcp_db_sqlite::SqliteDb;
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc as StdArc;

    /// Knobs for the mock BookStack server. Keeps tests focused — each one
    /// only sets the bits it cares about; the rest use sensible defaults.
    #[derive(Clone, Default)]
    pub(crate) struct MockBookStack {
        /// Page IDs that `GET /api/pages/{id}` returns 200 for. Everything
        /// else returns 404.
        pub allowed_pages: Vec<i64>,
        /// User ID the search probe returns as the calling user. None
        /// makes the search probe return zero results, which the production
        /// `whoami`/`list_my_roles` flow surfaces as "no identity yet".
        pub caller_user_id: Option<i64>,
        /// Roles attached to the caller user record. Only consulted when
        /// `caller_user_id` is `Some`.
        pub caller_roles: Vec<i64>,
    }

    /// Tracks how many times each endpoint was hit. Tests assert on this to
    /// detect fan-out reduction.
    pub(crate) struct MockCounters {
        pub pages: StdArc<AtomicUsize>,
        pub search: StdArc<AtomicUsize>,
        pub users: StdArc<AtomicUsize>,
    }

    impl MockCounters {
        fn new() -> Self {
            Self {
                pages: StdArc::new(AtomicUsize::new(0)),
                search: StdArc::new(AtomicUsize::new(0)),
                users: StdArc::new(AtomicUsize::new(0)),
            }
        }
    }

    /// Spin a tiny mock BookStack server that handles enough of the API
    /// surface for `filter_by_permission` + `list_my_roles` to roundtrip.
    /// Returns the base URL and the per-endpoint counters.
    pub(crate) async fn mock_bookstack_full(cfg: MockBookStack) -> (String, MockCounters) {
        use axum::extract::Path;
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        use axum::routing::get;
        use axum::Router;
        let counters = MockCounters::new();
        let allowed: StdArc<Vec<i64>> = StdArc::new(cfg.allowed_pages.clone());
        let caller_user_id = cfg.caller_user_id;
        let caller_roles: StdArc<Vec<i64>> = StdArc::new(cfg.caller_roles.clone());
        let pages_ctr = counters.pages.clone();
        let search_ctr = counters.search.clone();
        let users_ctr = counters.users.clone();
        let app = Router::new()
            .route(
                "/api/pages/{id}",
                get(move |Path(id): Path<i64>| {
                    let allowed = allowed.clone();
                    let ctr = pages_ctr.clone();
                    async move {
                        ctr.fetch_add(1, Ordering::SeqCst);
                        if allowed.contains(&id) {
                            (StatusCode::OK, axum::Json(serde_json::json!({"id": id})))
                                .into_response()
                        } else {
                            (
                                StatusCode::NOT_FOUND,
                                axum::Json(serde_json::json!({"error": "not found"})),
                            )
                                .into_response()
                        }
                    }
                }),
            )
            .route(
                "/api/search",
                get(move || {
                    let ctr = search_ctr.clone();
                    async move {
                        ctr.fetch_add(1, Ordering::SeqCst);
                        let data = match caller_user_id {
                            Some(uid) => serde_json::json!([
                                { "type": "page", "id": 1, "created_by": { "id": uid } }
                            ]),
                            None => serde_json::json!([]),
                        };
                        axum::Json(serde_json::json!({"data": data, "total": 1}))
                    }
                }),
            )
            .route(
                "/api/users/{id}",
                get(move |Path(id): Path<i64>| {
                    let ctr = users_ctr.clone();
                    let roles = caller_roles.clone();
                    async move {
                        ctr.fetch_add(1, Ordering::SeqCst);
                        let role_objs: Vec<serde_json::Value> = roles
                            .iter()
                            .map(|r| serde_json::json!({"id": r, "display_name": format!("Role{r}")}))
                            .collect();
                        axum::Json(serde_json::json!({
                            "id": id,
                            "name": "Test User",
                            "email": "test@example.com",
                            "roles": role_objs,
                        }))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        (format!("http://{addr}"), counters)
    }

    /// Back-compat shim for the lever 0 tests. Default mock: pages-only,
    /// no caller identity. Returns the URL + the page-hit counter.
    pub(crate) async fn mock_bookstack(allowed: Vec<i64>) -> (String, StdArc<AtomicUsize>) {
        let (url, counters) = mock_bookstack_full(MockBookStack {
            allowed_pages: allowed,
            caller_user_id: None,
            caller_roles: Vec::new(),
        })
        .await;
        (url, counters.pages)
    }

    pub(crate) fn temp_sqlite_path(label: &str) -> PathBuf {
        let dir = std::env::temp_dir();
        let unique = format!(
            "bsmcp-acl58-{label}-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        dir.join(unique)
    }

    pub(crate) async fn make_state(label: &str, embedder_url: String) -> Arc<SemanticState> {
        let path = temp_sqlite_path(label);
        let sqlite = Arc::new(SqliteDb::open(
            &path,
            "test-encryption-key-thirty-two-chars-long",
        ));
        sqlite.init_semantic_tables().await.unwrap();
        let core_db: Arc<dyn DbBackend> = sqlite.clone();
        let semantic_db: Arc<dyn SemanticDb> = sqlite.clone();
        let index_db: Arc<dyn IndexDb> = sqlite.clone();
        Arc::new(SemanticState::new(
            semantic_db,
            core_db,
            index_db,
            embedder_url,
            "test-webhook-secret-16chars".to_string(),
        ))
    }

    /// Lever 0 — counters. Verifies that a second call hits the in-memory
    /// cache (no extra `can_access_page` requests fire). Also exercises the
    /// happy path of `filter_by_permission` end-to-end against a sqlite
    /// backend + mock BookStack.
    #[tokio::test]
    async fn filter_by_permission_caches_hits() {
        let (base, counter) = mock_bookstack(vec![1, 2]).await;
        let state = make_state("counters", "http://unused".to_string()).await;
        let client = BookStackClient::new(&base, "tid", "tsecret", reqwest::Client::new());

        let ids = vec![1i64, 2, 3];
        let r1 = state.filter_by_permission(&ids, &client).await;
        let mut sorted = r1.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2]);
        let after_first = counter.load(Ordering::SeqCst);
        assert_eq!(
            after_first, 3,
            "every page should hit BookStack on a cold cache"
        );

        // Second call: all three hit the in-memory cache. No new HTTP calls.
        let r2 = state.filter_by_permission(&ids, &client).await;
        let mut sorted2 = r2;
        sorted2.sort();
        assert_eq!(sorted2, vec![1, 2]);
        let after_second = counter.load(Ordering::SeqCst);
        assert_eq!(
            after_second, after_first,
            "cache hit should suppress further HTTP fan-out"
        );
    }

    /// Helper: seed a sqlite db with pages + ACL rows so we can exercise
    /// `prefilter_pages_by_roles` directly.
    pub(crate) async fn seed_pages_with_acl(
        sqlite: &Arc<SqliteDb>,
        rows: &[(i64, bool, Vec<i64>)],
    ) {
        use bsmcp_common::types::{PageAcl, PageMeta};
        for (pid, default_open, roles) in rows {
            let meta = PageMeta {
                page_id: *pid,
                book_id: 100,
                chapter_id: None,
                name: format!("page-{pid}"),
                slug: format!("page-{pid}"),
                content_hash: "h".to_string(),
                updated_at: None,
            };
            sqlite.upsert_page(&meta).await.unwrap();
            let acl = PageAcl {
                page_id: *pid,
                view_roles: roles.clone(),
                default_open: *default_open,
                computed_at: 1,
            };
            sqlite.upsert_page_acl(&acl).await.unwrap();
        }
    }

    /// Lever a.5 — DB-side prefilter buckets every candidate into
    /// Allow/Deny/DefaultOpen/Uncomputed. Verified directly against the
    /// sqlite impl so we know the SQL shape is correct before the higher-
    /// level integration test exercises the search path.
    #[tokio::test]
    async fn prefilter_pages_by_roles_routes_correctly() {
        let path = temp_sqlite_path("prefilter-routes");
        let sqlite = Arc::new(SqliteDb::open(
            &path,
            "test-encryption-key-thirty-two-chars-long",
        ));
        sqlite.init_semantic_tables().await.unwrap();
        // Page 1: role-restricted to [10], default_open=false → Allow for caller_roles=[10]
        // Page 2: role-restricted to [20], default_open=false → Deny for caller_roles=[10]
        // Page 3: default_open=true → DefaultOpen
        // Page 4: never embedded (no `pages` row) → not in the returned list
        // Page 5: embedded but acl_computed_at IS NULL → Uncomputed
        seed_pages_with_acl(
            &sqlite,
            &[
                (1, false, vec![10]),
                (2, false, vec![20]),
                (3, true, vec![]),
            ],
        )
        .await;
        // Page 5 without ACL computed: insert page row but skip upsert_page_acl.
        let meta = bsmcp_common::types::PageMeta {
            page_id: 5,
            book_id: 100,
            chapter_id: None,
            name: "page-5".to_string(),
            slug: "page-5".to_string(),
            content_hash: "h".to_string(),
            updated_at: None,
        };
        sqlite.upsert_page(&meta).await.unwrap();

        let verdicts = sqlite
            .prefilter_pages_by_roles(&[1, 2, 3, 4, 5], &[10])
            .await
            .unwrap();
        let by_pid: HashMap<i64, AclPrefilter> = verdicts.into_iter().collect();
        assert_eq!(by_pid.get(&1), Some(&AclPrefilter::Allow));
        assert_eq!(by_pid.get(&2), Some(&AclPrefilter::Deny));
        assert_eq!(by_pid.get(&3), Some(&AclPrefilter::DefaultOpen));
        assert!(!by_pid.contains_key(&4), "page 4 isn't in the embed store");
        assert_eq!(by_pid.get(&5), Some(&AclPrefilter::Uncomputed));
    }

    /// Lever a.5 — end-to-end: a search-shaped call to `filter_by_permission`
    /// with the prefilter populated should skip the HTTP fallback for both
    /// Allow and Deny verdicts. Only DefaultOpen + Uncomputed reach BookStack.
    #[tokio::test]
    async fn filter_by_permission_uses_prefilter_to_skip_http() {
        // Pages 1+2: Allow (role 10 matches). Page 3: Deny. Pages 4+5: DefaultOpen.
        // The mock will count GET /api/pages/{id} hits — we expect exactly 2
        // (the two default-open pages still need HTTP), not 5.
        let path = temp_sqlite_path("prefilter-e2e");
        let sqlite = Arc::new(SqliteDb::open(
            &path,
            "test-encryption-key-thirty-two-chars-long",
        ));
        sqlite.init_semantic_tables().await.unwrap();
        seed_pages_with_acl(
            &sqlite,
            &[
                (1, false, vec![10]),
                (2, false, vec![10]),
                (3, false, vec![999]),
                (4, true, vec![]),
                (5, true, vec![]),
            ],
        )
        .await;

        // Allow pages 4+5 only — page 3 would 404 if it ever hit, which is fine
        // (deny in both pathways), and the test cares about the *count* of HTTP
        // calls, not the verdicts on default-open.
        let (base, counters) = mock_bookstack_full(MockBookStack {
            allowed_pages: vec![4, 5],
            caller_user_id: Some(42),
            caller_roles: vec![10],
        })
        .await;
        let core_db: Arc<dyn DbBackend> = sqlite.clone();
        let semantic_db: Arc<dyn SemanticDb> = sqlite.clone();
        let index_db: Arc<dyn IndexDb> = sqlite.clone();
        let state = Arc::new(SemanticState::new(
            semantic_db,
            core_db,
            index_db,
            "http://unused".to_string(),
            "test-webhook-secret-16chars".to_string(),
        ));
        let client = BookStackClient::new(&base, "tid", "tsecret", reqwest::Client::new());

        let accessible = state
            .filter_by_permission(&[1, 2, 3, 4, 5], &client)
            .await;
        let mut sorted = accessible;
        sorted.sort();
        // Page 1+2 allowed via prefilter, page 3 denied via prefilter,
        // pages 4+5 admitted via HTTP fallback (default-open + mock allows).
        assert_eq!(sorted, vec![1, 2, 4, 5]);
        assert_eq!(
            counters.pages.load(Ordering::SeqCst),
            2,
            "prefilter should suppress HTTP for Allow + Deny verdicts; only 2 default-open pages hit HTTP"
        );
    }

    /// Lever a — durable L2 cache. Verifies that warm-cache after restart
    /// (simulated by constructing a fresh `SemanticState` on the same
    /// sqlite path) skips the HTTP fan-out entirely.
    #[tokio::test]
    async fn permission_cache_l2_survives_restart() {
        let (base, counter) = mock_bookstack(vec![10, 11]).await;
        let path = temp_sqlite_path("l2-restart");

        // First "process": warm the L2 cache.
        {
            let sqlite = Arc::new(SqliteDb::open(
                &path,
                "test-encryption-key-thirty-two-chars-long",
            ));
            sqlite.init_semantic_tables().await.unwrap();
            let core_db: Arc<dyn DbBackend> = sqlite.clone();
            let semantic_db: Arc<dyn SemanticDb> = sqlite.clone();
            let index_db: Arc<dyn IndexDb> = sqlite.clone();
            let state = Arc::new(SemanticState::new(
                semantic_db,
                core_db,
                index_db,
                "http://unused".to_string(),
                "test-webhook-secret-16chars".to_string(),
            ));
            let client = BookStackClient::new(&base, "tid", "tsecret", reqwest::Client::new());
            let r = state
                .filter_by_permission(&[10i64, 11, 12], &client)
                .await;
            let mut sorted = r;
            sorted.sort();
            assert_eq!(sorted, vec![10, 11]);
            assert_eq!(counter.load(Ordering::SeqCst), 3);
        }

        // Second "process": fresh state, same backing file. L1 is empty,
        // but L2 still has the verdicts from the first session.
        {
            let sqlite = Arc::new(SqliteDb::open(
                &path,
                "test-encryption-key-thirty-two-chars-long",
            ));
            sqlite.init_semantic_tables().await.unwrap();
            let core_db: Arc<dyn DbBackend> = sqlite.clone();
            let semantic_db: Arc<dyn SemanticDb> = sqlite.clone();
            let index_db: Arc<dyn IndexDb> = sqlite.clone();
            let state = Arc::new(SemanticState::new(
                semantic_db,
                core_db,
                index_db,
                "http://unused".to_string(),
                "test-webhook-secret-16chars".to_string(),
            ));
            let client = BookStackClient::new(&base, "tid", "tsecret", reqwest::Client::new());
            let r = state
                .filter_by_permission(&[10i64, 11, 12], &client)
                .await;
            let mut sorted = r;
            sorted.sort();
            assert_eq!(sorted, vec![10, 11]);
            // No new HTTP traffic — L2 served every candidate.
            assert_eq!(
                counter.load(Ordering::SeqCst),
                3,
                "L2 should suppress HTTP fan-out post-restart"
            );
        }
    }
}
