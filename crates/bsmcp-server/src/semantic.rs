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
use bsmcp_common::db::{IndexDb, SemanticDb};
use bsmcp_common::types::MarkovBlanket;

const PERMISSION_CACHE_TTL: Duration = Duration::from_secs(300); // 5 minutes

/// Search ranking strategy. Selected per-call via the `mode` argument on
/// `semantic_search`. All three modes return the same JSON shape so a
/// caller can swap modes on the same query and diff the output.
///
/// - `Standard`: vector + optional keyword + blanket boost + blended sort.
///   Free, known-good baseline. Default.
/// - `Rerank`: standard pipeline produces the top-N, then a cross-encoder
///   /rerank pass re-orders just those N results. Cheap refinement on top
///   of what works (~10-30ms for N≤50 against a local cross-encoder).
/// - `Precision`: wider initial vector pass (5× limit), permission filter,
///   then cross-encoder /rerank as the ranker of record (replaces the
///   blanket+blend). More expensive, more potential to rescue a hit the
///   blend would have missed. `hybrid` is forced false in this mode.
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
    let Some(obj) = hit.as_object_mut() else { return hit; };
    let Some(chunks) = obj.get_mut("chunks").and_then(|v| v.as_array_mut()) else { return hit; };
    chunks.truncate(max_chunks);
    for chunk in chunks.iter_mut() {
        let Some(chunk_obj) = chunk.as_object_mut() else { continue; };
        let Some(content) = chunk_obj.get("content").and_then(|v| v.as_str()) else { continue; };
        if content.chars().count() > max_chars {
            let truncated: String = content.chars().take(max_chars).collect();
            chunk_obj.insert("content".to_string(), Value::String(format!("{truncated}…")));
            chunk_obj.insert("truncated".to_string(), Value::Bool(true));
        }
    }
    hit
}

struct CachedAccess {
    accessible: bool,
    cached_at: Instant,
}

pub struct SemanticState {
    db: Arc<dyn SemanticDb>,
    /// Structural index — consulted by the webhook handler to scope shelf/
    /// chapter_move re-embeds to the actually-affected books instead of
    /// falling back to `scope=all`. Same backend instance as `db` for both
    /// SQLite and Postgres deployments; threaded through as a trait object
    /// so the semantic module doesn't depend on the concrete backend type.
    index_db: Arc<dyn IndexDb>,
    embedder_url: String,
    webhook_secret: String,
    http_client: reqwest::Client,
    /// Permission cache: (token_id, page_id) -> CachedAccess
    permission_cache: RwLock<HashMap<(String, i64), CachedAccess>>,
}

impl SemanticState {
    pub fn new(
        db: Arc<dyn SemanticDb>,
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
            index_db,
            embedder_url: embedder_url.trim_end_matches('/').to_string(),
            webhook_secret,
            http_client,
            permission_cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn webhook_secret(&self) -> &str {
        &self.webhook_secret
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
            eprintln!("Semantic: ACL reconciliation disabled (BSMCP_ACL_RECONCILE_HOURS=0)");
            return;
        }
        let interval = Duration::from_secs(interval_hours * 3600);
        eprintln!("Semantic: ACL reconcile cron active — every {interval_hours}h");
        tokio::spawn(async move {
            // Stagger initial run so server startup isn't immediately followed
            // by a heavy reconcile. 5 minutes is enough for the embedder to
            // come up and pull pending jobs first.
            tokio::time::sleep(Duration::from_secs(5 * 60)).await;
            loop {
                match self.db.create_embed_job("acl_reconcile").await {
                    Ok((job_id, is_new)) => eprintln!(
                        "Semantic: ACL reconcile cron — queued job {job_id} (new={is_new})"
                    ),
                    Err(e) => eprintln!("Semantic: ACL reconcile cron — queue failed: {e}"),
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
                eprintln!("embed_query: retry {attempt} after error: {last_err}");
                tokio::time::sleep(Duration::from_millis(500)).await;
            }

            let resp = match self.http_client
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

            let body: Value = resp.json().await
                .map_err(|e| format!("Embedder response parse error: {e}"))?;

            let embedding = body.get("embeddings")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_array())
                .ok_or("Invalid embedder response format")?;

            let vec: Vec<f32> = embedding.iter()
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
    /// Results are cached per (token_id, page_id) for 5 minutes.
    async fn filter_by_permission(
        &self,
        page_ids: &[i64],
        client: &BookStackClient,
    ) -> Vec<i64> {
        let token_id = client.token_id().to_string();
        let now = Instant::now();

        let mut uncached_ids: Vec<i64> = Vec::new();
        let mut accessible: Vec<i64> = Vec::new();

        {
            let cache = self.permission_cache.read().await;
            for &pid in page_ids {
                let key = (token_id.clone(), pid);
                if let Some(entry) = cache.get(&key) {
                    if now.duration_since(entry.cached_at) < PERMISSION_CACHE_TTL {
                        if entry.accessible {
                            accessible.push(pid);
                        }
                        continue;
                    }
                }
                uncached_ids.push(pid);
            }
        }

        if !uncached_ids.is_empty() {
            // Check each page individually with concurrency limit. Bumped from
            // 10 → 25 because the cold-cache permission filter is the dominant
            // cost in semantic search; BookStack handles the burst comfortably.
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

            {
                let mut cache = self.permission_cache.write().await;
                for &(pid, ok) in &results {
                    cache.insert((token_id.clone(), pid), CachedAccess {
                        accessible: ok,
                        cached_at: now,
                    });
                    if ok {
                        accessible.push(pid);
                    }
                }
                // Evict stale entries if cache grows large
                if cache.len() > 10_000 {
                    cache.retain(|_, entry| now.duration_since(entry.cached_at) < PERMISSION_CACHE_TTL);
                }
            }
        }

        accessible
    }

    /// Hybrid search: vector + keyword + blanket re-ranking, with optional
    /// cross-encoder rerank as either a refinement (`Rerank`) or a full
    /// replacement of the blend (`Precision`). See [`SearchMode`] for the
    /// per-mode contract.
    ///
    /// `book_filter`: when `Some(&[..])`, restricts the vector pass to chunks
    /// whose page lives in one of the supplied books. The keyword pass and
    /// permission/blanket steps are unaffected; the vector candidate pool is
    /// just smaller from the outset, which proportionally shrinks the
    /// permission filter and per-result fan-out. `None` keeps the old
    /// whole-corpus behavior.
    #[allow(clippy::too_many_arguments)]
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        threshold: f32,
        hybrid: bool,
        verbose: bool,
        client: &BookStackClient,
        book_filter: Option<&[i64]>,
        mode: SearchMode,
    ) -> Result<Value, String> {
        let start = Instant::now();

        // Precision mode forces hybrid off. The cross-encoder is the ranker
        // of record; mixing in keyword-rank scoring just dilutes its signal.
        // Rerank mode keeps hybrid intact — the rerank only re-orders the
        // final top-N from the standard pipeline.
        let hybrid = hybrid && mode != SearchMode::Precision;

        // Run vector search and optional keyword search in parallel.
        // Candidate over-fetch is `limit * 2` for the standard/rerank path —
        // empirically sufficient headroom after permission filtering. In
        // precision mode we cast a wider net (`limit * 5`) because the
        // cross-encoder benefits from seeing more candidates, and the rerank
        // step itself caps the embedder side at 200 documents.
        let candidate_multiplier: usize = if mode == SearchMode::Precision { 5 } else { 2 };
        let book_filter_owned: Option<Vec<i64>> = book_filter
            .filter(|s| !s.is_empty())
            .map(|s| s.to_vec());
        let vector_future = async {
            let query_vec = self.embed_query(query).await?;
            self.db
                .vector_search(
                    &query_vec,
                    limit * candidate_multiplier,
                    threshold,
                    book_filter_owned.as_deref(),
                )
                .await
        };

        let keyword_future = async {
            if hybrid {
                match client.search(query, 1, (limit * 2) as i64).await {
                    Ok(resp) => {
                        resp.get("data")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default()
                    }
                    Err(e) => {
                        eprintln!("Hybrid: keyword search failed (non-fatal): {e}");
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

        // If a book filter was applied to the vector pass, apply the same
        // filter to keyword results so we don't re-introduce out-of-scope
        // pages via the hybrid merge path.
        if let Some(allowed) = book_filter_owned.as_deref() {
            let allowed_set: HashSet<i64> = allowed.iter().copied().collect();
            // Keyword results don't carry book_id; fetch the book_id for each
            // candidate page in one batched DB call and drop anything outside
            // the allowed set.
            let candidate_ids: Vec<i64> = keyword_results.iter()
                .filter(|r| r.get("type").and_then(|v| v.as_str()) == Some("page"))
                .filter_map(|r| r.get("id").and_then(|v| v.as_i64()))
                .collect();
            if !candidate_ids.is_empty() {
                let book_lookup: HashMap<i64, i64> = self.db
                    .get_page_book_ids(&candidate_ids)
                    .await
                    .unwrap_or_default()
                    .into_iter()
                    .collect();
                keyword_results.retain(|r| {
                    let pid = r.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                    match book_lookup.get(&pid) {
                        Some(bid) => allowed_set.contains(bid),
                        // Page not in embedding store — drop it (out-of-scope by definition).
                        None => false,
                    }
                });
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

        // PRECISION MODE: replace blanket+blend with cross-encoder rerank.
        // The candidate set is `page_scores` post-ACL. We pick the best
        // chunk per page, send `(query, [doc_per_page])` to the embedder's
        // /rerank, and use the returned scores as the final ordering.
        // Skips the blanket boost and hybrid blend below.
        if mode == SearchMode::Precision {
            return self
                .precision_rerank(query, limit, &page_scores, verbose, start)
                .await;
        }

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

        let blanket_fetches: Vec<(i64, MarkovBlanket)> = stream::iter(scored_page_ids.iter().copied())
            .map(|pid| async move {
                match self.db.get_markov_blanket(pid).await {
                    Ok(b) => Some((pid, b)),
                    Err(e) => {
                        eprintln!("Blanket: error for page {pid}: {e}");
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
            for related in blanket.linked_from.iter()
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
        let mut page_results: Vec<(i64, f32, &PageScore)> = page_scores.iter()
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

            page_results = page_scores.iter()
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
        let all_chunk_ids: Vec<i64> = page_results.iter()
            .flat_map(|(_, _, score)| score.chunks.iter().map(|c| c.0))
            .collect();

        let (metas, chunk_details) = tokio::try_join!(
            self.db.get_page_metas(&final_page_ids),
            self.db.get_chunk_details(&all_chunk_ids),
        )?;

        let meta_by_page: HashMap<i64, &bsmcp_common::types::PageMeta> =
            metas.iter().map(|m| (m.page_id, m)).collect();

        // Group chunk details by their page_id so each result picks up only its chunks.
        let mut chunks_by_page: HashMap<i64, Vec<&bsmcp_common::types::ChunkDetail>> = HashMap::new();
        for detail in &chunk_details {
            chunks_by_page.entry(detail.page_id).or_default().push(detail);
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
                let page_name = meta_by_page
                    .get(pid)
                    .map(|m| m.name.as_str())
                    .unwrap_or("");
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
            let missing: Vec<i64> = final_page_ids.iter()
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
                        let chunk_score = score.chunks.iter().find(|c| c.0 == detail.chunk_id).map(|c| c.1).unwrap_or(0.0);
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
        Ok(RerankResponse { hits, provider, model })
    }

    /// Cross-encoder rerank step for **precision** mode. Replaces the blanket
    /// boost + hybrid blend that the standard search path applies after the
    /// permission filter. Picks one document per candidate page (the best-
    /// scoring chunk's heading + content), POSTs `(query, [doc])` to the
    /// embedder's `/rerank`, then renders the response in the same JSON
    /// shape as the standard search so callers don't need to branch.
    async fn precision_rerank(
        &self,
        query: &str,
        limit: usize,
        page_scores: &HashMap<i64, PageScore>,
        verbose: bool,
        start: Instant,
    ) -> Result<Value, String> {
        // One document per page = the highest-scoring chunk that contributed
        // to this page's match. Pages with no chunks (keyword-only matches)
        // are dropped — precision mode forces hybrid off, so this is rare,
        // but it keeps the rerank input well-formed if anything slipped past.
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

        // Embedder caps per-request docs at 200. If we ever exceed that, keep
        // the top-vector-scoring candidates so the cross-encoder still sees
        // the strongest signal.
        const MAX_RERANK_DOCS: usize = 200;
        if candidates.len() > MAX_RERANK_DOCS {
            candidates.sort_by(|(a, _), (b, _)| {
                let sa = page_scores.get(a).map(|s| s.vector_score).unwrap_or(0.0);
                let sb = page_scores.get(b).map(|s| s.vector_score).unwrap_or(0.0);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(MAX_RERANK_DOCS);
        }

        if candidates.is_empty() {
            let stats = self.db.get_stats().await?;
            return Ok(json!({
                "results": [],
                "stats": {
                    "total_indexed": stats.total_pages,
                    "total_chunks": stats.total_chunks,
                    "query_time_ms": start.elapsed().as_millis(),
                    "mode": SearchMode::Precision.as_str(),
                    "hybrid": false,
                    "candidates_reranked": 0,
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

        // Build documents and a parallel index → page_id map. Heading path +
        // chunk content gives the cross-encoder enough surface to score; the
        // page name is included so a query about a topic that appears only in
        // a heading doesn't get penalized for content-body keyword absence.
        let mut docs: Vec<String> = Vec::with_capacity(candidates.len());
        let mut doc_to_page: Vec<i64> = Vec::with_capacity(candidates.len());
        for (pid, cid) in &candidates {
            let page_name = meta_by_page
                .get(pid)
                .map(|m| m.name.as_str())
                .unwrap_or("");
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
        let rr = self.invoke_rerank(query, docs, limit).await?;
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
        // /rerank already sorted by score desc and truncated to top_k.

        // Verbose: fetch blankets for the final result set only.
        let mut blanket_cache: HashMap<i64, MarkovBlanket> = HashMap::new();
        if verbose {
            let final_pids: Vec<i64> = ranked.iter().map(|(pid, _)| *pid).collect();
            let extras: Vec<(i64, MarkovBlanket)> = stream::iter(final_pids)
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

        let mut chunks_by_page: HashMap<i64, Vec<&bsmcp_common::types::ChunkDetail>> =
            HashMap::new();
        for detail in &chunk_details {
            chunks_by_page.entry(detail.page_id).or_default().push(detail);
        }

        let mut results = Vec::with_capacity(ranked.len());
        for (page_id, rerank_score) in &ranked {
            let (page_name, book_id, updated_at) = match meta_by_page.get(page_id) {
                Some(m) => (m.name.clone(), m.book_id, m.updated_at.clone()),
                None => ("Unknown".to_string(), 0, None),
            };
            let score_ref = page_scores.get(page_id);
            let vector_score = score_ref.map(|s| s.vector_score).unwrap_or(0.0);

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

        let stats = self.db.get_stats().await?;
        let query_time_ms = start.elapsed().as_millis();

        Ok(json!({
            "results": results,
            "stats": {
                "total_indexed": stats.total_pages,
                "total_chunks": stats.total_chunks,
                "query_time_ms": query_time_ms,
                "rerank_ms": rerank_ms,
                "mode": SearchMode::Precision.as_str(),
                "hybrid": false,
                "rerank_provider": rr.provider,
                "rerank_model": rr.model,
                "candidates_reranked": doc_to_page.len(),
            }
        }))
    }

    /// Trigger re-embedding by inserting a job into the queue.
    pub async fn trigger_reembed(&self, scope: &str) -> Result<Value, String> {
        let (job_id, is_new) = self.db.create_embed_job(scope).await?;
        let (status, message) = if is_new {
            ("queued", "Embedding job queued. The embedder will pick it up shortly.")
        } else {
            ("already_active", "A job with this scope is already active. Returning existing job.")
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
    pub async fn list_jobs(&self, recent: usize) -> Result<Vec<bsmcp_common::types::EmbedJob>, String> {
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

        eprintln!("Semantic: webhook event={event} item_id={item_id:?}");

        match event {
            // --- Page events ---
            "page_create" | "page_update" | "page_restore" => {
                if let Some(pid) = item_id {
                    let scope = format!("page:{pid}");
                    let (job_id, is_new) = self.db.create_embed_job(&scope).await?;
                    eprintln!("Semantic: {event} — queued page:{pid} embed job {job_id} (new={is_new})");
                }
            }
            "page_move" => {
                // Page moved to different book/chapter — context prefix changed.
                // Re-embed with force since HTML is the same but context differs.
                if let Some(pid) = item_id {
                    let scope = format!("page:{pid}");
                    let (job_id, is_new) = self.db.create_embed_job(&scope).await?;
                    eprintln!("Semantic: page_move — queued page:{pid} embed job {job_id} (new={is_new})");
                }
            }
            "page_delete" => {
                if let Some(pid) = item_id {
                    // delete_page CASCADE-removes chunks + relationships;
                    // page_view_acl rows are explicitly cleared so the per-role
                    // index doesn't accumulate dead entries.
                    self.db.delete_page(pid).await?;
                    let _ = self.db.delete_page_acl(pid).await;
                    eprintln!("Semantic: deleted embeddings + ACL for page {pid}");
                }
            }

            // --- Chapter events (re-embed the containing book) ---
            "chapter_create" | "chapter_update" | "chapter_delete" => {
                let book_id = related.get("book_id").and_then(|v| v.as_i64());
                if let Some(bid) = book_id {
                    let scope = format!("book:{bid}");
                    let (job_id, is_new) = self.db.create_embed_job(&scope).await?;
                    eprintln!("Semantic: {event} — queued book:{bid} embed job {job_id} (new={is_new})");
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
                    eprintln!("Semantic: chapter_move — queued book:{bid} (dest) embed job {job_id} (new={is_new})");
                }

                if let Some(cid) = item_id {
                    let prev_book_id = match self.index_db.get_indexed_chapter(cid).await {
                        Ok(Some(ch)) => Some(ch.book_id),
                        Ok(None) => None,
                        Err(e) => {
                            eprintln!("Semantic: chapter_move — index lookup failed for chapter {cid}: {e}");
                            None
                        }
                    };
                    match prev_book_id {
                        Some(pbid) if Some(pbid) != new_book_id => {
                            let scope = format!("book:{pbid}");
                            let (job_id, is_new) = self.db.create_embed_job(&scope).await?;
                            eprintln!("Semantic: chapter_move — queued book:{pbid} (prev source) embed job {job_id} (new={is_new})");
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
                            eprintln!("Semantic: chapter_move — chapter {cid} not in index, fell back to scope=all embed job {job_id} (new={is_new})");
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
                    eprintln!("Semantic: {event} — queued book:{bid} embed job {job_id} (new={is_new})");
                }
            }
            "book_delete" => {
                // Pages are cascade-deleted by BookStack; page_delete webhooks
                // should fire for each page. Just log for awareness.
                eprintln!("Semantic: book_delete (id={item_id:?}) — page deletions handled by page_delete events");
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
                            eprintln!("Semantic: {event} — index lookup failed for shelf {sid}: {e}");
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                };

                if books.is_empty() {
                    let (job_id, is_new) = self.db.create_embed_job("all").await?;
                    eprintln!("Semantic: {event} — shelf {shelf_id:?} has no indexed books, fell back to scope=all embed job {job_id} (new={is_new})");
                } else {
                    for book in &books {
                        let scope = format!("book:{}", book.book_id);
                        let (job_id, is_new) = self.db.create_embed_job(&scope).await?;
                        eprintln!(
                            "Semantic: {event} — queued book:{} embed job {job_id} (new={is_new})",
                            book.book_id
                        );
                    }
                    eprintln!(
                        "Semantic: {event} — enqueued {} scoped book jobs for shelf {shelf_id:?}",
                        books.len()
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
                eprintln!("Semantic: {event} — queued ACL reconcile job {job_id} (new={is_new})");
            }
            "role_delete" => {
                if let Some(rid) = item_id {
                    let _ = self.db.delete_role_from_acl(rid).await;
                    eprintln!("Semantic: role_delete — purged role {rid} from page_view_acl");
                }
                let (job_id, is_new) = self.db.create_embed_job("acl_reconcile").await?;
                eprintln!("Semantic: role_delete — queued ACL reconcile job {job_id} (new={is_new})");
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
                eprintln!("Semantic: permissions_update (item={item_id:?}) — queued ACL reconcile job {job_id} (new={is_new})");
            }

            _ => {
                eprintln!("Semantic: ignoring webhook event {event}");
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
