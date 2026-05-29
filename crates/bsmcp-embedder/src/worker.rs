//! v1.0.0 reconciliation worker (folded into bsmcp-embedder in v0.13.0).
//!
//! Background tokio task that keeps `bookstack_*` index tables and
//! `page_cache` in sync with the live BookStack instance. Phase 4b ships
//! the worker scaffolding + the initial full walk + a single-page
//! reconcile helper (the building block that webhook + delta walk in
//! Phase 4c will reuse).
//!
//! As of v0.13.0 this lives inside the `bsmcp-embedder` crate — the
//! standalone `bsmcp-worker` binary was folded in (closes #56). The
//! embedder's `--role={embedder|worker|both}` flag selects which loops
//! run; the worker code itself is unchanged from the standalone binary.
//!
//! Triggers (this phase):
//!   - On startup: enqueue an `all` job if `index_meta.last_full_walk_at`
//!     is not yet set.
//!
//! Triggers added in Phase 4c:
//!   - Webhook: enqueue `page:{id}` jobs on BookStack page events.
//!   - Periodic delta walk: every BSMCP_INDEX_DELTA_INTERVAL_SECONDS,
//!     reconcile pages whose `updated_at > last_delta_walk_at`.
//!
//! Storage backend: writes through the IndexDb trait. SQLite has the real
//! impl; Postgres returns a clear error from each method (issue #36) — so
//! the worker is a no-op on Postgres deployments until that lands. A
//! BSMCP_INDEX_WORKER env flag gates the spawn entirely so operators can
//! opt out.
//!
//! Auth: uses the BSMCP_INDEX_TOKEN_* (falling back to BSMCP_EMBED_TOKEN_*)
//! BookStack API token, which must have read access to every shelf the
//! worker walks. Per-user content access for live MCP requests still uses
//! the user's own token; the worker is structural reconciliation only.

use std::env;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use bsmcp_common::bookstack::BookStackClient;
use bsmcp_common::db::{DbBackend, IndexDb, SemanticDb};
use bsmcp_common::index::*;

const DEFAULT_RECONCILE_SECS: u64 = 300;
const DEFAULT_TIMEOUT_SECS: i64 = 3600;
const DEFAULT_MAX_RETRY_CHAIN: usize = 5;
const DEFAULT_CLOSE_GRACE_SECS: i64 = 30;
const ARCHIVER_TICK_SECS: u64 = 10;
const TIMEOUT_TICK_SECS: u64 = 30;

/// How often the poll loop checks for a pending job when the queue is empty.
const POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Cap on the number of pages reconciled before we yield the runtime
/// briefly during a full walk. Prevents one giant walk from monopolising
/// the worker task.
const YIELD_EVERY: usize = 25;

pub(crate) struct IndexWorker {
    bs_client: BookStackClient,
    db: Arc<dyn DbBackend>,
    index_db: Arc<dyn IndexDb>,
}

impl IndexWorker {
    pub(crate) fn new(
        bs_client: BookStackClient,
        db: Arc<dyn DbBackend>,
        index_db: Arc<dyn IndexDb>,
    ) -> Self {
        Self {
            bs_client,
            db,
            index_db,
        }
    }

    /// Spawn the worker as a background tokio task. Returns the JoinHandle
    /// so the caller can hold a reference (or `forget()` it for fire-and-
    /// forget semantics, which matches the existing `spawn_acl_reconcile`
    /// pattern in semantic.rs).
    ///
    /// `delta_interval_secs` controls the periodic delta-walk cadence
    /// (0 disables it). Webhook-triggered jobs still arrive in real time;
    /// the periodic walk is a safety net for missed webhooks.
    pub(crate) fn spawn(
        self,
        delta_interval_secs: u64,
        semantic_db: Option<Arc<dyn SemanticDb>>,
    ) -> tokio::task::JoinHandle<()> {
        let worker = Arc::new(self);
        let worker_for_delta = worker.clone();
        let worker_for_lifecycle = worker.clone();
        let semantic_for_lifecycle = semantic_db.clone();

        // Job lifecycle housekeeping: timeout watcher + archiver + reconciler.
        // Single task multiplexed across the three loops; cadence + thresholds
        // come from BSMCP_JOB_* envs (see .env.example). Operates on both
        // index_jobs (always) and embed_jobs (when semantic_db is wired).
        tokio::spawn(async move {
            let timeout_secs: i64 = env::var("BSMCP_JOB_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_TIMEOUT_SECS);
            let reconcile_secs: u64 = env::var("BSMCP_JOB_RECONCILE_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_RECONCILE_SECS);
            let max_chain: usize = env::var("BSMCP_JOB_MAX_RETRY_CHAIN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_MAX_RETRY_CHAIN);
            let close_grace: i64 = env::var("BSMCP_JOB_CLOSE_GRACE_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_CLOSE_GRACE_SECS);

            tracing::info!(
                timeout_s = timeout_secs,
                reconcile_s = reconcile_secs,
                max_chain,
                close_grace_s = close_grace,
                "indexworker_lifecycle_housekeeper"
            );

            // Stagger 30s after startup so the initial walk gets a head start.
            tokio::time::sleep(Duration::from_secs(30)).await;
            let mut last_reconcile = std::time::Instant::now();
            loop {
                let now = current_unix();
                let cutoff = now - timeout_secs;

                // Timeout watcher — fail any running job whose started_at is
                // older than BSMCP_JOB_TIMEOUT_SECS.
                if let Ok(jobs) = worker_for_lifecycle
                    .index_db
                    .list_running_index_jobs_started_before(cutoff)
                    .await
                {
                    for j in jobs {
                        tracing::warn!(
                            job_id = j.id,
                            started_at = ?j.started_at,
                            "indexworker_index_job_timing_out"
                        );
                        if let Err(e) = worker_for_lifecycle
                            .index_db
                            .fail_index_job(j.id, "timeout")
                            .await
                        {
                            tracing::error!(
                                job_id = j.id,
                                error = %e,
                                "indexworker_fail_index_job_failed"
                            );
                        }
                    }
                }
                if let Some(sdb) = semantic_for_lifecycle.as_ref() {
                    if let Ok(jobs) = sdb.list_running_embed_jobs_started_before(cutoff).await {
                        for j in jobs {
                            tracing::warn!(
                                job_id = j.id,
                                started_at = ?j.started_at,
                                "indexworker_embed_job_timing_out"
                            );
                            if let Err(e) = sdb.fail_embed_job(j.id, "timeout").await {
                                tracing::error!(
                                    job_id = j.id,
                                    error = %e,
                                    "indexworker_fail_embed_job_failed"
                                );
                            }
                        }
                    }
                }

                // Archiver — close succeeded/cancelled jobs older than the
                // grace window so the status page doesn't grow unbounded.
                if let Ok(ids) = worker_for_lifecycle
                    .index_db
                    .list_archivable_index_jobs(close_grace)
                    .await
                {
                    for id in ids {
                        if let Err(e) = worker_for_lifecycle
                            .index_db
                            .close_index_job(id, None)
                            .await
                        {
                            tracing::error!(job_id = id, error = %e, "indexworker_close_index_job_failed");
                        }
                    }
                }
                if let Some(sdb) = semantic_for_lifecycle.as_ref() {
                    if let Ok(ids) = sdb.list_archivable_embed_jobs(close_grace).await {
                        for id in ids {
                            if let Err(e) = sdb.close_embed_job(id, None).await {
                                tracing::error!(job_id = id, error = %e, "indexworker_close_embed_job_failed");
                            }
                        }
                    }
                }

                // Reconciler — runs on its own coarser cadence.
                if last_reconcile.elapsed() >= Duration::from_secs(reconcile_secs) {
                    last_reconcile = std::time::Instant::now();
                    reconcile_failed_index_jobs(&worker_for_lifecycle.index_db, max_chain).await;
                    if let Some(sdb) = semantic_for_lifecycle.as_ref() {
                        reconcile_failed_embed_jobs(sdb, max_chain).await;
                    }
                }

                tokio::time::sleep(Duration::from_secs(
                    ARCHIVER_TICK_SECS.min(TIMEOUT_TICK_SECS),
                ))
                .await;
            }
        });

        // Periodic delta walk timer — independent task that just enqueues
        // a `delta` job at intervals. The poll loop picks it up and runs
        // the actual walk. Gated on a non-zero interval so operators can
        // disable polling entirely (webhook-only mode).
        if delta_interval_secs > 0 {
            tokio::spawn(async move {
                let interval = Duration::from_secs(delta_interval_secs);
                tracing::info!(
                    interval_s = delta_interval_secs,
                    "indexworker_delta_cron_active"
                );
                // Stagger the first delta so it doesn't race the initial
                // full walk.
                tokio::time::sleep(Duration::from_secs(60)).await;
                loop {
                    match worker_for_delta
                        .index_db
                        .create_index_job("delta", "both", "cron")
                        .await
                    {
                        Ok((id, is_new)) => {
                            if is_new {
                                tracing::info!(job_id = id, "indexworker_delta_cron_queued");
                            }
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "indexworker_delta_cron_enqueue_failed")
                        }
                    }
                    tokio::time::sleep(interval).await;
                }
            });
        } else {
            tracing::info!("indexworker_delta_cron_disabled");
        }

        // Main poll loop.
        tokio::spawn(async move {
            // Stagger initial check by a few seconds so server startup
            // isn't immediately followed by a heavy walk.
            tokio::time::sleep(Duration::from_secs(10)).await;

            if let Err(e) = worker.maybe_enqueue_initial_walk().await {
                tracing::error!(error = %e, "indexworker_initial_walk_enqueue_failed");
            }
            worker.poll_loop().await;
        })
    }

    /// On first start, queue a full walk if it's never been done.
    async fn maybe_enqueue_initial_walk(&self) -> Result<(), String> {
        if self
            .index_db
            .get_index_meta("last_full_walk_at")
            .await?
            .is_some()
        {
            return Ok(());
        }
        let (id, is_new) = self
            .index_db
            .create_index_job("all", "both", "startup")
            .await?;
        if is_new {
            tracing::info!(job_id = id, "indexworker_initial_full_walk_enqueued");
        }
        Ok(())
    }

    async fn poll_loop(&self) {
        loop {
            match self.index_db.claim_next_index_job().await {
                Ok(Some(job)) => self.handle_job(job).await,
                Ok(None) => tokio::time::sleep(POLL_INTERVAL).await,
                Err(e) => {
                    tracing::error!(error = %e, "indexworker_claim_next_index_job_error");
                    tokio::time::sleep(POLL_INTERVAL).await;
                }
            }
        }
    }

    async fn handle_job(&self, job: IndexJob) {
        tracing::info!(
            job_id = job.id,
            scope = %job.scope,
            kind = %job.kind,
            triggered_by = %job.triggered_by,
            "indexworker_processing_job"
        );
        let result = self.process_job(&job).await;
        // If the job was cancelled mid-walk, the cancel endpoint already
        // wrote the resolved state — don't overwrite it via complete_index_job.
        let stopped_externally =
            matches!(self.index_db.should_stop_index_job(job.id).await, Ok(true));
        let outcome = if stopped_externally {
            Ok(())
        } else {
            match &result {
                Ok(()) => self.index_db.complete_index_job(job.id, None).await,
                Err(e) => {
                    tracing::error!(job_id = job.id, error = %e, "indexworker_job_failed");
                    self.index_db.complete_index_job(job.id, Some(e)).await
                }
            }
        };
        if let Err(e) = outcome {
            tracing::error!(job_id = job.id, error = %e, "indexworker_job_completion_update_failed");
        }
    }

    async fn process_job(&self, job: &IndexJob) -> Result<(), String> {
        match job.scope.as_str() {
            "all" => self.walk_all(job.id).await,
            "delta" => self.walk_delta(job.id).await,
            scope if scope.starts_with("page:") => {
                let id = parse_scope_id(scope, "page:")?;
                self.reconcile_page(id).await
            }
            scope if scope.starts_with("chapter:") => {
                let id = parse_scope_id(scope, "chapter:")?;
                self.reconcile_chapter(id).await
            }
            scope if scope.starts_with("book:") => {
                let id = parse_scope_id(scope, "book:")?;
                self.reconcile_book(id).await
            }
            scope if scope.starts_with("shelf:") => {
                let id = parse_scope_id(scope, "shelf:")?;
                self.reconcile_shelf(id).await
            }
            other => Err(format!("unknown scope: {other}")),
        }
    }

    /// Initial full walk — every shelf in scope → books → chapters →
    /// pages. Also handles pages loose at the book root (no chapter).
    /// Sets `index_meta.last_full_walk_at` on success.
    ///
    /// Issue #119: scope is `globals.indexed_shelves` if non-empty; otherwise
    /// the worker enumerates every shelf the token can see via
    /// `BookStackClient::list_all_shelves` and walks all of them. The old
    /// `[hive_shelf_id, user_journals_shelf_id]` fields are gone — on a fresh
    /// deployment with no settings configured, semantic search now sees the
    /// whole instance instead of no-op'ing.
    async fn walk_all(&self, job_id: i64) -> Result<(), String> {
        let globals = self.db.get_global_settings().await.unwrap_or_default();
        let shelves: Vec<i64> = if globals.indexed_shelves.is_empty() {
            match self.bs_client.list_all_shelves().await {
                Ok(all) => {
                    tracing::info!(shelf_count = all.len(), "indexworker_full_walk_all_shelves");
                    all
                }
                Err(e) => {
                    tracing::error!(error = %e, "indexworker_full_walk_list_shelves_failed");
                    return Err(format!("list_all_shelves: {e}"));
                }
            }
        } else {
            tracing::info!(
                shelf_count = globals.indexed_shelves.len(),
                "indexworker_full_walk_configured_shelves"
            );
            globals.indexed_shelves.clone()
        };
        if shelves.is_empty() {
            tracing::warn!("indexworker_full_walk_no_shelves_visible");
            self.stamp_full_walk_done().await?;
            return Ok(());
        }
        let mut total_pages = 0usize;
        for shelf_id in shelves {
            // Per-shelf/book/chapter/page status check. This (and every
            // other should_stop_index_job call in walk_*) is a `WHERE id = ?`
            // PK lookup — cheap (microseconds on either backend) and
            // intentional, so a user-issued cancel takes effect at the
            // next yield point rather than waiting for the whole walk.
            if matches!(self.index_db.should_stop_index_job(job_id).await, Ok(true)) {
                tracing::info!(job_id, walk = "all", "indexworker_walk_stopped");
                return Ok(());
            }
            match self.walk_shelf(shelf_id, job_id).await {
                Ok(n) => total_pages += n,
                Err(e) => tracing::error!(shelf_id, error = %e, "indexworker_walk_shelf_failed"),
            }
        }
        self.stamp_full_walk_done().await?;
        tracing::info!(total_pages, "indexworker_full_walk_complete");
        Ok(())
    }

    async fn stamp_full_walk_done(&self) -> Result<(), String> {
        let now = current_unix();
        self.index_db
            .set_index_meta("last_full_walk_at", &now.to_string())
            .await
    }

    /// Periodic delta walk — list pages whose `updated_at` advanced past
    /// `last_delta_walk_at` (or `last_full_walk_at` on first run) and
    /// reconcile each. Advances `last_delta_walk_at` to the newest
    /// `updated_at` seen so a subsequent run resumes from the correct
    /// boundary even if some pages failed to reconcile.
    async fn walk_delta(&self, job_id: i64) -> Result<(), String> {
        let _ = job_id; // delta walk is short-lived; the per-page yield in walk_book/chapter handles cancel.
        let last_walk = match self.index_db.get_index_meta("last_delta_walk_at").await? {
            Some(v) => v,
            None => match self.index_db.get_index_meta("last_full_walk_at").await? {
                Some(v) => unix_to_iso(&v),
                None => {
                    tracing::info!("indexworker_walk_delta_skipped_no_full_walk_yet");
                    return Ok(());
                }
            },
        };

        let pages = self
            .bs_client
            .list_pages_updated_since(&last_walk, 250)
            .await?;
        tracing::info!(
            since = %last_walk,
            candidate_pages = pages.len(),
            "indexworker_walk_delta_started"
        );

        let mut newest_seen = last_walk.clone();
        let mut reconciled = 0usize;
        for page in pages {
            let Some(page_id) = page.get("id").and_then(|v| v.as_i64()) else {
                continue;
            };
            let updated_at = page
                .get("updated_at")
                .and_then(|v| v.as_str())
                .map(String::from);

            if let Err(e) = self.reconcile_page(page_id).await {
                tracing::error!(
                    walk = "delta",
                    page_id,
                    error = %e,
                    "indexworker_reconcile_page_failed"
                );
                continue;
            }
            reconciled += 1;
            if let Some(ts) = updated_at {
                if ts > newest_seen {
                    newest_seen = ts;
                }
            }
        }

        // Stamp the boundary even on a no-op pass so periodic runs don't
        // keep re-listing the same window. If newest_seen advanced, future
        // walks pick up from there; if no pages came back, we use `now`
        // so we don't redundantly query the same window.
        let advance_to = if newest_seen != last_walk {
            newest_seen
        } else {
            iso_now()
        };
        self.index_db
            .set_index_meta("last_delta_walk_at", &advance_to)
            .await?;
        tracing::info!(
            reconciled,
            advanced_to = %advance_to,
            "indexworker_walk_delta_complete"
        );
        Ok(())
    }

    async fn walk_shelf(&self, shelf_id: i64, job_id: i64) -> Result<usize, String> {
        let shelf = self.bs_client.get_shelf(shelf_id).await?;
        let name = string_field(&shelf, "name");
        let slug = string_field(&shelf, "slug");
        // Issue #119: every shelf is `Unclassified` now — see `classify_shelf`
        // for why the helper survives as a constant.
        let shelf_kind = classify_shelf(shelf_id);
        self.index_db
            .upsert_indexed_shelf(&IndexedShelf {
                shelf_id,
                name,
                slug,
                shelf_kind,
                indexed_at: current_unix(),
                deleted: false,
            })
            .await?;
        let books: Vec<i64> = shelf
            .get("books")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|b| b.get("id").and_then(|v| v.as_i64()))
                    .collect()
            })
            .unwrap_or_default();
        let mut count = 0usize;
        for book_id in books {
            if matches!(self.index_db.should_stop_index_job(job_id).await, Ok(true)) {
                tracing::info!(job_id, walk = "shelf", "indexworker_walk_stopped");
                return Ok(count);
            }
            match self
                .walk_book(book_id, Some(shelf_id), shelf_kind, job_id)
                .await
            {
                Ok(n) => count += n,
                Err(e) => tracing::error!(book_id, error = %e, "indexworker_walk_book_failed"),
            }
        }
        Ok(count)
    }

    async fn walk_book(
        &self,
        book_id: i64,
        shelf_id: Option<i64>,
        shelf_kind: ShelfKind,
        job_id: i64,
    ) -> Result<usize, String> {
        let book = self.bs_client.get_book(book_id).await?;
        let name = string_field(&book, "name");
        let slug = string_field(&book, "slug");
        let book_kind = classify_book(&name, shelf_kind);
        // For Identity/UserIdentity books, dig out the ouid from the
        // manifest page's frontmatter so it can be propagated to every
        // descendant we index.
        let identity_ouid = if matches!(book_kind, BookKind::Identity | BookKind::UserIdentity) {
            self.find_book_identity_ouid(&book).await
        } else {
            None
        };
        self.index_db
            .upsert_indexed_book(&IndexedBook {
                book_id,
                name,
                slug,
                shelf_id,
                identity_ouid: identity_ouid.clone(),
                book_kind,
                indexed_at: current_unix(),
                deleted: false,
            })
            .await?;

        // The book contents array mixes chapters and loose-at-root pages.
        // Dispatch each accordingly.
        let contents = book
            .get("contents")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut count = 0usize;
        let mut idx = 0usize;
        for item in contents {
            if matches!(self.index_db.should_stop_index_job(job_id).await, Ok(true)) {
                tracing::info!(job_id, walk = "book", "indexworker_walk_stopped");
                return Ok(count);
            }
            match item.get("type").and_then(|v| v.as_str()) {
                Some("chapter") => {
                    if let Some(chapter_id) = item.get("id").and_then(|v| v.as_i64()) {
                        match self
                            .walk_chapter(
                                chapter_id,
                                book_id,
                                book_kind,
                                identity_ouid.as_deref(),
                                job_id,
                            )
                            .await
                        {
                            Ok(n) => count += n,
                            Err(e) => tracing::error!(
                                chapter_id,
                                error = %e,
                                "indexworker_walk_chapter_failed"
                            ),
                        }
                    }
                }
                Some("page") => {
                    if let Some(page_id) = item.get("id").and_then(|v| v.as_i64()) {
                        if let Err(e) = self
                            .reconcile_page_with_parents(
                                page_id,
                                book_id,
                                None,
                                book_kind,
                                None,
                                None,
                                identity_ouid.as_deref(),
                            )
                            .await
                        {
                            tracing::error!(
                                page_id,
                                error = %e,
                                "indexworker_reconcile_page_failed"
                            );
                        }
                        count += 1;
                    }
                }
                _ => {}
            }
            idx += 1;
            if idx.is_multiple_of(YIELD_EVERY) {
                tokio::task::yield_now().await;
            }
        }
        Ok(count)
    }

    async fn walk_chapter(
        &self,
        chapter_id: i64,
        book_id: i64,
        parent_book_kind: BookKind,
        identity_ouid: Option<&str>,
        job_id: i64,
    ) -> Result<usize, String> {
        let chapter = self.bs_client.get_chapter(chapter_id).await?;
        let name = string_field(&chapter, "name");
        let slug = string_field(&chapter, "slug");
        let (chapter_kind, archive_year) = classify_chapter(&name, parent_book_kind);
        self.index_db
            .upsert_indexed_chapter(&IndexedChapter {
                chapter_id,
                book_id,
                name,
                slug,
                identity_ouid: identity_ouid.map(String::from),
                chapter_kind,
                archive_year,
                indexed_at: current_unix(),
                deleted: false,
            })
            .await?;
        let pages = chapter
            .get("pages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut count = 0usize;
        let mut idx = 0usize;
        for page in pages {
            if matches!(self.index_db.should_stop_index_job(job_id).await, Ok(true)) {
                tracing::info!(job_id, walk = "chapter", "indexworker_walk_stopped");
                return Ok(count);
            }
            if let Some(page_id) = page.get("id").and_then(|v| v.as_i64()) {
                if let Err(e) = self
                    .reconcile_page_with_parents(
                        page_id,
                        book_id,
                        Some(chapter_id),
                        parent_book_kind,
                        Some(chapter_kind),
                        archive_year,
                        identity_ouid,
                    )
                    .await
                {
                    tracing::error!(page_id, error = %e, "indexworker_reconcile_page_failed");
                }
                count += 1;
            }
            idx += 1;
            if idx.is_multiple_of(YIELD_EVERY) {
                tokio::task::yield_now().await;
            }
        }
        Ok(count)
    }

    /// Single-page reconcile keyed off page id. Looks up parent classification
    /// from the index (so the parent walk has to have happened first), then
    /// upserts the page row + cache row in one transaction. Used by:
    ///   - Phase 4c webhook handler (page event → enqueue page:{id} job)
    ///   - Phase 4c periodic delta walk (each delta page goes through here)
    ///
    /// If the parent book is missing from the index (initial walk hasn't
    /// reached it yet, or the index is partially-stale), this cascades a
    /// `book:{id}` reconcile job and fails the page job retryably so the
    /// #54 retry-chain reconciler picks it back up after the parent lands.
    async fn reconcile_page(&self, page_id: i64) -> Result<(), String> {
        let page = self.bs_client.get_page(page_id).await?;
        let book_id = page
            .get("book_id")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| format!("page {page_id} has no book_id"))?;
        let chapter_id = page
            .get("chapter_id")
            .and_then(|v| v.as_i64())
            .filter(|&id| id != 0);

        let parent_book = match self.index_db.get_indexed_book(book_id).await? {
            Some(b) => b,
            None => {
                return cascade_missing_parent(
                    &self.index_db,
                    &format!("book:{book_id}"),
                    &format!("cascade_from_page:{page_id}"),
                    &format!("page {page_id} parent book {book_id}"),
                )
                .await;
            }
        };
        let parent_chapter = if let Some(cid) = chapter_id {
            self.index_db.get_indexed_chapter(cid).await?
        } else {
            None
        };

        let parent_chapter_kind = parent_chapter.as_ref().map(|c| c.chapter_kind);
        let parent_chapter_archive_year = parent_chapter.as_ref().and_then(|c| c.archive_year);

        // The classify_page call inside reconcile_page_with_parents already
        // pulls the page name from `page`; but to keep that helper backend-
        // agnostic we re-fetch here. Given page is already in scope, pass it
        // directly via the lower-level helper.
        self.reconcile_page_inner(
            &page,
            book_id,
            chapter_id,
            parent_book.book_kind,
            parent_chapter_kind,
            parent_chapter_archive_year,
            parent_book.identity_ouid.as_deref(),
        )
        .await
    }

    /// Single-chapter reconcile keyed off chapter id. Cascades a `book:{id}`
    /// job (and fails retryably) when the parent book is missing from the
    /// index — same self-heal shape as `reconcile_page`. Upserts only the
    /// chapter row; descendant pages are reached by the next full/delta walk
    /// or by their own `page:{id}` jobs.
    async fn reconcile_chapter(&self, chapter_id: i64) -> Result<(), String> {
        let chapter = self.bs_client.get_chapter(chapter_id).await?;
        let book_id = chapter
            .get("book_id")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| format!("chapter {chapter_id} has no book_id"))?;

        let parent_book = match self.index_db.get_indexed_book(book_id).await? {
            Some(b) => b,
            None => {
                return cascade_missing_parent(
                    &self.index_db,
                    &format!("book:{book_id}"),
                    &format!("cascade_from_chapter:{chapter_id}"),
                    &format!("chapter {chapter_id} parent book {book_id}"),
                )
                .await;
            }
        };

        let name = string_field(&chapter, "name");
        let slug = string_field(&chapter, "slug");
        let (chapter_kind, archive_year) = classify_chapter(&name, parent_book.book_kind);
        self.index_db
            .upsert_indexed_chapter(&IndexedChapter {
                chapter_id,
                book_id,
                name,
                slug,
                identity_ouid: parent_book.identity_ouid.clone(),
                chapter_kind,
                archive_year,
                indexed_at: current_unix(),
                deleted: false,
            })
            .await
    }

    /// Single-book reconcile keyed off book id. When a configured indexed
    /// shelf claims this book and that shelf isn't in the index yet, cascades
    /// a `shelf:{id}` job and fails retryably. Otherwise upserts the book
    /// row. Descendants are not touched — the full walk or their own per-id
    /// jobs handle that.
    ///
    /// Issue #119: parent-shelf attribution probes
    /// `globals.indexed_shelves` when non-empty, otherwise falls back to the
    /// full shelf list. Same best-effort shape as before — BookStack's
    /// `/api/books/{id}` doesn't return the parent shelf, so we probe.
    async fn reconcile_book(&self, book_id: i64) -> Result<(), String> {
        let book = self.bs_client.get_book(book_id).await?;
        let globals = self.db.get_global_settings().await.unwrap_or_default();
        let candidate_shelves: Vec<i64> = if globals.indexed_shelves.is_empty() {
            // Configured to walk all — probe every visible shelf. Larger
            // probe than v0.12.x's two-shelf shape, but the cascade-on-miss
            // path is rare (only fires on a webhook reaching us before the
            // initial walk landed the parent).
            self.bs_client.list_all_shelves().await.unwrap_or_default()
        } else {
            globals.indexed_shelves.clone()
        };
        let mut shelf_id: Option<i64> = None;
        for sid in &candidate_shelves {
            let shelf = match self.bs_client.get_shelf(*sid).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!(
                        book_id,
                        shelf_id = sid,
                        error = %e,
                        "indexworker_reconcile_book_get_shelf_failed"
                    );
                    continue;
                }
            };
            let contains = shelf
                .get("books")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|b| b.get("id").and_then(|v| v.as_i64()))
                        .any(|bid| bid == book_id)
                })
                .unwrap_or(false);
            if contains {
                shelf_id = Some(*sid);
                break;
            }
        }

        if let Some(sid) = shelf_id {
            if self.index_db.get_indexed_shelf(sid).await?.is_none() {
                return cascade_missing_parent(
                    &self.index_db,
                    &format!("shelf:{sid}"),
                    &format!("cascade_from_book:{book_id}"),
                    &format!("book {book_id} parent shelf {sid}"),
                )
                .await;
            }
        }

        let shelf_kind = classify_shelf(shelf_id.unwrap_or(0));
        let name = string_field(&book, "name");
        let slug = string_field(&book, "slug");
        let book_kind = classify_book(&name, shelf_kind);
        let identity_ouid = if matches!(book_kind, BookKind::Identity | BookKind::UserIdentity) {
            self.find_book_identity_ouid(&book).await
        } else {
            None
        };
        self.index_db
            .upsert_indexed_book(&IndexedBook {
                book_id,
                name,
                slug,
                shelf_id,
                identity_ouid,
                book_kind,
                indexed_at: current_unix(),
                deleted: false,
            })
            .await
    }

    /// Single-shelf reconcile keyed off shelf id. Top of the cascade
    /// chain — no parent to check. Upserts only the shelf row.
    async fn reconcile_shelf(&self, shelf_id: i64) -> Result<(), String> {
        let shelf = self.bs_client.get_shelf(shelf_id).await?;
        let shelf_kind = classify_shelf(shelf_id);
        self.index_db
            .upsert_indexed_shelf(&IndexedShelf {
                shelf_id,
                name: string_field(&shelf, "name"),
                slug: string_field(&shelf, "slug"),
                shelf_kind,
                indexed_at: current_unix(),
                deleted: false,
            })
            .await
    }

    /// Same as `reconcile_page` but takes parent classification + ouid from
    /// the caller (the full walk already has them in hand without the extra
    /// index lookup).
    #[allow(clippy::too_many_arguments)]
    async fn reconcile_page_with_parents(
        &self,
        page_id: i64,
        book_id: i64,
        chapter_id: Option<i64>,
        parent_book_kind: BookKind,
        parent_chapter_kind: Option<ChapterKind>,
        parent_chapter_archive_year: Option<i32>,
        identity_ouid: Option<&str>,
    ) -> Result<(), String> {
        let page = self.bs_client.get_page(page_id).await?;
        self.reconcile_page_inner(
            &page,
            book_id,
            chapter_id,
            parent_book_kind,
            parent_chapter_kind,
            parent_chapter_archive_year,
            identity_ouid,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn reconcile_page_inner(
        &self,
        page: &Value,
        book_id: i64,
        chapter_id: Option<i64>,
        parent_book_kind: BookKind,
        parent_chapter_kind: Option<ChapterKind>,
        parent_chapter_archive_year: Option<i32>,
        identity_ouid: Option<&str>,
    ) -> Result<(), String> {
        let page_id = page
            .get("id")
            .and_then(|v| v.as_i64())
            .ok_or("page has no id")?;
        let name = string_field(page, "name");
        let (page_kind, page_key, archive_year_from_page) = classify_page(
            &name,
            parent_book_kind,
            parent_chapter_kind,
            parent_chapter_archive_year,
        );
        // archive_year priority: classifier's signal first (pages in archive
        // chapters); fall back to chapter's archive year if the classifier
        // didn't surface one (defensive — usually identical).
        let archive_year = archive_year_from_page.or(parent_chapter_archive_year);
        let now = current_unix();
        let raw_md = page
            .get("markdown")
            .and_then(|v| v.as_str())
            .map(String::from);
        let html = page.get("html").and_then(|v| v.as_str()).map(String::from);
        let page_updated_at = page
            .get("updated_at")
            .and_then(|v| v.as_str())
            .map(String::from);
        let stripped_md = raw_md.as_deref().map(strip_frontmatter);

        let indexed = IndexedPage {
            page_id,
            book_id,
            chapter_id,
            name,
            slug: string_field(page, "slug"),
            url: page.get("url").and_then(|v| v.as_str()).map(String::from),
            page_created_at: page
                .get("created_at")
                .and_then(|v| v.as_str())
                .map(String::from),
            page_updated_at: page_updated_at.clone(),
            identity_ouid: identity_ouid.map(String::from),
            page_kind,
            page_key,
            archive_year,
            indexed_at: now,
            deleted: false,
        };
        let cache = PageCache {
            page_id,
            markdown: stripped_md,
            raw_markdown: raw_md,
            html,
            cached_at: now,
            page_updated_at,
        };
        self.index_db
            .upsert_indexed_page(&indexed, Some(&cache))
            .await
    }

    /// For an Identity / UserIdentity book, find the manifest page (loose
    /// at book root, named "Identity") and pull its `ai_identity_ouid`
    /// frontmatter field. Best-effort — returns None if the book has no
    /// manifest yet, the manifest has no frontmatter, or the field is
    /// absent. Identity books without an ouid still index correctly; the
    /// dedup UNIQUE index just doesn't fire for them.
    async fn find_book_identity_ouid(&self, book: &Value) -> Option<String> {
        let contents = book.get("contents").and_then(|v| v.as_array())?;
        let manifest = contents.iter().find(|item| {
            item.get("type").and_then(|v| v.as_str()) == Some("page")
                && item.get("name").and_then(|v| v.as_str()).map(|n| n.trim()) == Some("Identity")
        })?;
        let manifest_id = manifest.get("id").and_then(|v| v.as_i64())?;
        let page = self.bs_client.get_page(manifest_id).await.ok()?;
        let md = page.get("markdown").and_then(|v| v.as_str())?;
        extract_ouid_from_frontmatter(md)
    }
}

// --- Job lifecycle reconciler (issue #54) ---

pub(crate) async fn reconcile_failed_index_jobs(db: &Arc<dyn IndexDb>, max_chain: usize) {
    let jobs = match db.list_failed_open_index_jobs().await {
        Ok(j) => j,
        Err(e) => {
            tracing::error!(error = %e, "indexworker_list_failed_open_index_jobs_failed");
            return;
        }
    };
    for j in jobs {
        match db.has_successor_index_job(&j.scope, j.id).await {
            Ok(true) => {
                if let Err(e) = db.close_index_job(j.id, Some("superseded")).await {
                    tracing::error!(
                        job_id = j.id,
                        error = %e,
                        "indexworker_close_index_job_superseded_failed"
                    );
                }
                continue;
            }
            Err(e) => {
                tracing::error!(
                    job_id = j.id,
                    error = %e,
                    "indexworker_has_successor_index_job_failed"
                );
                continue;
            }
            Ok(false) => {}
        }
        let chain_len = db.index_job_retry_chain_len(j.id).await.unwrap_or(1);
        if chain_len >= max_chain {
            tracing::warn!(
                job_id = j.id,
                chain_len,
                max_chain,
                "indexworker_index_job_giving_up"
            );
            if let Err(e) = db.close_index_job(j.id, Some("gave_up")).await {
                tracing::error!(
                    job_id = j.id,
                    error = %e,
                    "indexworker_close_index_job_gave_up_failed"
                );
            }
            continue;
        }
        match db.create_retry_index_job(&j.scope, &j.kind, j.id).await {
            Ok(new_id) => {
                tracing::info!(
                    new_job_id = new_id,
                    of_job_id = j.id,
                    chain_len,
                    "indexworker_retry_index_job_queued"
                );
                if let Err(e) = db.close_index_job(j.id, Some("retried")).await {
                    tracing::error!(
                        job_id = j.id,
                        error = %e,
                        "indexworker_close_index_job_retried_failed"
                    );
                }
            }
            Err(e) => tracing::error!(
                job_id = j.id,
                error = %e,
                "indexworker_create_retry_index_job_failed"
            ),
        }
    }
}

pub(crate) async fn reconcile_failed_embed_jobs(db: &Arc<dyn SemanticDb>, max_chain: usize) {
    let jobs = match db.list_failed_open_embed_jobs().await {
        Ok(j) => j,
        Err(e) => {
            tracing::error!(error = %e, "indexworker_list_failed_open_embed_jobs_failed");
            return;
        }
    };
    for j in jobs {
        match db.has_successor_embed_job(&j.scope, j.id).await {
            Ok(true) => {
                if let Err(e) = db.close_embed_job(j.id, Some("superseded")).await {
                    tracing::error!(
                        job_id = j.id,
                        error = %e,
                        "indexworker_close_embed_job_superseded_failed"
                    );
                }
                continue;
            }
            Err(e) => {
                tracing::error!(
                    job_id = j.id,
                    error = %e,
                    "indexworker_has_successor_embed_job_failed"
                );
                continue;
            }
            Ok(false) => {}
        }
        let chain_len = db.embed_job_retry_chain_len(j.id).await.unwrap_or(1);
        if chain_len >= max_chain {
            tracing::warn!(
                job_id = j.id,
                chain_len,
                max_chain,
                "indexworker_embed_job_giving_up"
            );
            if let Err(e) = db.close_embed_job(j.id, Some("gave_up")).await {
                tracing::error!(
                    job_id = j.id,
                    error = %e,
                    "indexworker_close_embed_job_gave_up_failed"
                );
            }
            continue;
        }
        match db.create_retry_embed_job(&j.scope, j.id).await {
            Ok(new_id) => {
                tracing::info!(
                    new_job_id = new_id,
                    of_job_id = j.id,
                    chain_len,
                    "indexworker_retry_embed_job_queued"
                );
                if let Err(e) = db.close_embed_job(j.id, Some("retried")).await {
                    tracing::error!(
                        job_id = j.id,
                        error = %e,
                        "indexworker_close_embed_job_retried_failed"
                    );
                }
            }
            Err(e) => tracing::error!(
                job_id = j.id,
                error = %e,
                "indexworker_create_retry_embed_job_failed"
            ),
        }
    }
}

// --- helpers ---

/// Parse a scoped job id like "page:123" into the trailing integer.
/// Used by `process_job` to dispatch per-id reconcile scopes.
fn parse_scope_id(scope: &str, prefix: &str) -> Result<i64, String> {
    scope
        .strip_prefix(prefix)
        .ok_or_else(|| format!("invalid scope: {scope}"))?
        .parse()
        .map_err(|e| format!("invalid scope id in {scope}: {e}"))
}

/// Self-heal entrypoint for missing-parent cases in `reconcile_page`,
/// `reconcile_chapter`, and `reconcile_book`. Idempotently enqueues a
/// reconcile job for the missing parent (the existing `create_index_job`
/// dedup collapses pending/running/failed-open same-scope jobs onto one),
/// emits a single log line, and returns a retryable error so the #54
/// retry-chain reconciler picks the original job back up after the parent
/// lands. The retry job created by that reconciler runs the same scope
/// again, by which time the parent is in the index.
///
/// `parent_scope` — e.g. `"book:2113"` — must be a valid scope string the
/// worker's `process_job` dispatch table understands. `triggered_by` is
/// stored on the enqueued job for provenance (e.g. `"cascade_from_page:2116"`).
/// `subject` is a short human-readable noun phrase ("page 2116 parent book 2113")
/// used only in the returned error / log message.
async fn cascade_missing_parent(
    index_db: &Arc<dyn IndexDb>,
    parent_scope: &str,
    triggered_by: &str,
    subject: &str,
) -> Result<(), String> {
    let (job_id, is_new) = index_db
        .create_index_job(parent_scope, "both", triggered_by)
        .await?;
    if is_new {
        tracing::info!(
            subject = %subject,
            parent_scope = %parent_scope,
            triggered_by = %triggered_by,
            job_id,
            "indexworker_cascade_missing_parent_enqueued"
        );
    } else {
        tracing::debug!(
            subject = %subject,
            parent_scope = %parent_scope,
            job_id,
            "indexworker_cascade_missing_parent_coalesced"
        );
    }
    Err(format!(
        "{subject} not in index — enqueued {parent_scope} (job {job_id}); will retry"
    ))
}

fn current_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Render the current UTC moment as ISO 8601 (e.g. `2026-04-27T03:14:15Z`).
/// Used by the delta walk's `last_delta_walk_at` checkpoint when no pages
/// came back in a polling pass.
fn iso_now() -> String {
    let secs = current_unix();
    let (y, mo, d, h, mi, s) = unix_to_components(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Convert a stored unix-seconds value (e.g., from `last_full_walk_at`) to
/// an ISO 8601 string suitable for the BookStack `filter[updated_at:gt]`
/// param. Falls back to "1970-01-01T00:00:00Z" on parse failure.
fn unix_to_iso(stored: &str) -> String {
    let secs: i64 = stored.parse().unwrap_or(0);
    let (y, mo, d, h, mi, s) = unix_to_components(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

fn unix_to_components(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let time = secs.rem_euclid(86_400) as u32;
    let h = time / 3600;
    let mi = (time % 3600) / 60;
    let s = time % 60;
    let (y, mo, d) = days_to_ymd(days);
    (y, mo, d, h, mi, s)
}

fn days_to_ymd(days: i64) -> (i64, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn string_field(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string()
}

/// Strip a leading YAML frontmatter block, if any. Inlined (rather than
/// reusing `crate::remember::frontmatter::strip`) to avoid a coupling
/// where the worker depends on the remember module.
fn strip_frontmatter(md: &str) -> String {
    let trimmed = md.trim_start();
    let Some(after_open) = trimmed.strip_prefix("---") else {
        return md.to_string();
    };
    let after_open = after_open
        .strip_prefix("\r\n")
        .or_else(|| after_open.strip_prefix('\n'))
        .unwrap_or(after_open);
    let mut pos = 0usize;
    for line in after_open.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']) == "---" {
            return after_open[pos + line.len()..].trim_start().to_string();
        }
        pos += line.len();
    }
    md.to_string()
}

fn extract_ouid_from_frontmatter(md: &str) -> Option<String> {
    let trimmed = md.trim_start();
    if !trimmed.starts_with("---") {
        return None;
    }
    let mut iter = trimmed.lines();
    iter.next(); // opening ---
    for line in iter {
        let line = line.trim();
        if line == "---" {
            break;
        }
        if let Some(rest) = line
            .strip_prefix("ai_identity_ouid:")
            .or_else(|| line.strip_prefix("ouid:"))
        {
            return Some(rest.trim().trim_matches('"').to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_ouid_present() {
        let md =
            "---\nai_identity_ouid: 019dc66e4dc27d329e4a4abd1bec0c80\nname: Pia\n---\n\n# Pia\n";
        assert_eq!(
            extract_ouid_from_frontmatter(md).as_deref(),
            Some("019dc66e4dc27d329e4a4abd1bec0c80")
        );
    }

    #[test]
    fn extract_ouid_missing() {
        let md = "---\nname: Pia\n---\n\n# Pia\n";
        assert!(extract_ouid_from_frontmatter(md).is_none());
    }

    #[test]
    fn extract_ouid_no_frontmatter() {
        assert!(extract_ouid_from_frontmatter("# Pia\n\nmanifest body").is_none());
    }

    #[test]
    fn extract_ouid_quoted() {
        let md = "---\nouid: \"abc-123\"\n---\nbody";
        assert_eq!(
            extract_ouid_from_frontmatter(md).as_deref(),
            Some("abc-123")
        );
    }

    #[test]
    fn strip_frontmatter_removes_block() {
        let md = "---\nfoo: bar\n---\n\nbody text\n";
        assert_eq!(strip_frontmatter(md), "body text\n");
    }

    #[test]
    fn strip_frontmatter_no_block() {
        assert_eq!(strip_frontmatter("just body"), "just body");
    }
}

#[cfg(test)]
mod reconcile_tests {
    use super::*;
    use bsmcp_common::db::IndexDb;
    use bsmcp_db_sqlite::SqliteDb;
    use std::env as std_env;

    fn temp_db() -> Arc<dyn IndexDb> {
        let dir = std_env::temp_dir();
        let unique = format!(
            "bsmcp-worker-test-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = dir.join(unique);
        Arc::new(SqliteDb::open(
            &path,
            "test-encryption-key-thirty-two-chars-long",
        )) as Arc<dyn IndexDb>
    }

    #[tokio::test]
    async fn reconcile_supersedes_when_successor_exists() {
        let db = temp_db();
        let (a, _) = db.create_index_job("page:1", "both", "test").await.unwrap();
        db.fail_index_job(a, "boom").await.unwrap();
        // Force-close the failed-open dedup gate so we can create a successor.
        db.close_index_job(a, Some("retried")).await.unwrap();
        let (b, _) = db.create_index_job("page:1", "both", "test").await.unwrap();
        // Re-fail a so it shows up as failed-open again, but b exists as successor.
        // Easier: create another failed-open job for the same scope. Since the
        // scope dedup with closed jobs lets us insert anew, fail b then put a third.
        db.fail_index_job(b, "boom").await.unwrap();
        db.close_index_job(b, Some("retried")).await.unwrap();
        let (c, _) = db.create_index_job("page:1", "both", "test").await.unwrap();
        let _ = c; // c is the successor for both a and b.
                   // Re-open a's failed-open marker by manually flipping b. Simpler:
                   // start fresh, this exercises has_successor on a's behalf.
                   // Reset a back to failed-open status via fail_index_job (which is
                   // a no-op once status is closed). Simpler approach: use a fresh db.
        let db = temp_db();
        let (j1, _) = db.create_index_job("page:1", "both", "test").await.unwrap();
        db.fail_index_job(j1, "boom").await.unwrap();
        // Manually insert a successor — bypass dedup by using a different scope key
        // then reset to the same scope. Easiest path: close j1 first, create
        // successor, then re-fail j1's status without changing resolved_status.
        // The reconciler shape expects "j1 in failed status, newer same-scope
        // job exists". We simulate by closing-as-superseded directly.
        // Since the reconcile function is what we're testing, we need j1 to
        // be failed-open AND have a successor. Easiest: close j1, create
        // successor j2, then UPDATE j1 back to failed-open via raw SQL.
        // We don't have raw SQL access through the trait; instead, the
        // close-and-reopen dance below works because close_index_job leaves
        // resolved_status (that's what we want to test isn't double-emitted).
        // Different approach: directly create j1, fail, j2 manually as retry
        // child without closing j1 — the dedup check skips retry inserts (no
        // dedup on create_retry).
        let db = temp_db();
        let (j1, _) = db.create_index_job("page:1", "both", "test").await.unwrap();
        db.fail_index_job(j1, "boom").await.unwrap();
        let _j2 = db
            .create_retry_index_job("page:1", "both", j1)
            .await
            .unwrap();
        // Now j2 is pending+same scope+ id > j1. Reconciler should see j1 as superseded.
        reconcile_failed_index_jobs(&db, 5).await;
        // Verify j1 closed with resolved_status='superseded'.
        let after = db.list_failed_open_index_jobs().await.unwrap();
        assert!(
            after.iter().all(|j| j.id != j1),
            "j1 should no longer be failed-open"
        );
    }

    #[tokio::test]
    async fn reconcile_retries_when_no_successor_and_chain_short() {
        let db = temp_db();
        let (j1, _) = db.create_index_job("page:9", "both", "test").await.unwrap();
        db.fail_index_job(j1, "boom").await.unwrap();
        reconcile_failed_index_jobs(&db, 5).await;
        // j1 should be closed with resolved_status='retried' and a fresh
        // pending job with retry_of=j1 should exist.
        let listed = db.list_index_jobs(20).await.unwrap();
        let original = listed.iter().find(|j| j.id == j1).unwrap();
        assert_eq!(original.status, "closed");
        assert_eq!(original.resolved_status.as_deref(), Some("retried"));
        let retry = listed
            .iter()
            .find(|j| j.retry_of == Some(j1))
            .expect("retry exists");
        assert_eq!(retry.status, "pending");
        assert_eq!(retry.scope, "page:9");
    }

    #[tokio::test]
    async fn reconcile_gives_up_when_chain_at_max() {
        let db = temp_db();
        let (j1, _) = db.create_index_job("page:7", "both", "test").await.unwrap();
        db.fail_index_job(j1, "1").await.unwrap();
        let j2 = db
            .create_retry_index_job("page:7", "both", j1)
            .await
            .unwrap();
        db.close_index_job(j1, Some("retried")).await.unwrap();
        db.fail_index_job(j2, "2").await.unwrap();
        // chain length at j2 is 2; with max=2 we should give up.
        reconcile_failed_index_jobs(&db, 2).await;
        let listed = db.list_index_jobs(20).await.unwrap();
        let j2_after = listed.iter().find(|j| j.id == j2).unwrap();
        assert_eq!(j2_after.status, "closed");
        assert_eq!(j2_after.resolved_status.as_deref(), Some("gave_up"));
    }

    /// Issue #72 — cascade-on-missing-parent.
    ///
    /// Concurrent page-level reconciles for two different pages whose
    /// shared parent book is not yet indexed must coalesce onto a single
    /// `book:N` cascade job (not one per child) and the original page
    /// reconciles must surface a retryable error so the #54 reconciler
    /// picks them up after the cascade book job lands.
    ///
    /// We test the helper directly rather than `reconcile_page` end-to-end
    /// because `BookStackClient` is a concrete reqwest-backed struct with
    /// no test stub — building one would balloon scope. The helper carries
    /// 100% of the cascade behavior (idempotency + retryable error + log
    /// line); the per-method call sites are thin wrappers around it.
    #[tokio::test]
    async fn cascade_missing_parent_coalesces_concurrent_children() {
        let db = temp_db();
        // Two different child pages discover the same missing parent book.
        let r1 = cascade_missing_parent(
            &db,
            "book:2113",
            "cascade_from_page:2116",
            "page 2116 parent book 2113",
        )
        .await;
        let r2 = cascade_missing_parent(
            &db,
            "book:2113",
            "cascade_from_page:2117",
            "page 2117 parent book 2113",
        )
        .await;
        // Both children must surface a retryable error mentioning the
        // cascade scope so the #54 reconciler treats them as failed-open.
        let e1 = r1.expect_err("first child must fail-retryable");
        let e2 = r2.expect_err("second child must fail-retryable");
        assert!(e1.contains("book:2113"), "err mentions cascade scope: {e1}");
        assert!(e2.contains("book:2113"), "err mentions cascade scope: {e2}");

        // Exactly one pending `book:2113` job exists — no duplicates.
        let pending = db.list_pending_index_jobs(50).await.unwrap();
        let book_jobs: Vec<_> = pending.iter().filter(|j| j.scope == "book:2113").collect();
        assert_eq!(
            book_jobs.len(),
            1,
            "concurrent children must coalesce onto one cascade job, got {book_jobs:?}"
        );

        // Provenance: the surviving cascade job records the FIRST child's
        // trigger string (subsequent children are no-ops via dedup, so
        // their `triggered_by` is intentionally not preserved).
        let cascade = book_jobs[0];
        assert_eq!(cascade.triggered_by, "cascade_from_page:2116");
        assert_eq!(cascade.kind, "both");
        assert_eq!(cascade.status, "pending");
    }

    /// A second cascade for the same parent after the first has already
    /// transitioned through `failed` (waiting on the reconciler) must
    /// still coalesce — `create_index_job`'s dedup rule covers
    /// failed-without-resolved-status as an active state.
    #[tokio::test]
    async fn cascade_coalesces_against_failed_open_parent_job() {
        let db = temp_db();
        cascade_missing_parent(
            &db,
            "book:9000",
            "cascade_from_page:1",
            "page 1 parent book 9000",
        )
        .await
        .unwrap_err();
        let pending_before = db.list_pending_index_jobs(50).await.unwrap();
        let parent_id = pending_before
            .iter()
            .find(|j| j.scope == "book:9000")
            .map(|j| j.id)
            .expect("first cascade enqueued");
        // Simulate the cascade job itself failing once (e.g. transient
        // BookStack 5xx) — it's now failed-open, the reconciler hasn't
        // run yet. A second child cascading the same parent must NOT
        // double-enqueue.
        db.fail_index_job(parent_id, "transient").await.unwrap();
        cascade_missing_parent(
            &db,
            "book:9000",
            "cascade_from_page:2",
            "page 2 parent book 9000",
        )
        .await
        .unwrap_err();
        let all_jobs = db.list_index_jobs(50).await.unwrap();
        let book_jobs: Vec<_> = all_jobs.iter().filter(|j| j.scope == "book:9000").collect();
        assert_eq!(
            book_jobs.len(),
            1,
            "failed-open parent job must absorb subsequent cascades, got {book_jobs:?}"
        );
    }

    #[tokio::test]
    async fn parse_scope_id_round_trip() {
        assert_eq!(parse_scope_id("page:42", "page:").unwrap(), 42);
        assert_eq!(parse_scope_id("chapter:7", "chapter:").unwrap(), 7);
        assert_eq!(parse_scope_id("book:2113", "book:").unwrap(), 2113);
        assert_eq!(parse_scope_id("shelf:927", "shelf:").unwrap(), 927);
        assert!(parse_scope_id("page:nope", "page:").is_err());
        assert!(parse_scope_id("all", "page:").is_err());
    }
}
