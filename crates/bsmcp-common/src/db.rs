use std::path::Path;

use async_trait::async_trait;

use crate::index::*;
use crate::settings::GlobalSettings;
use crate::types::*;

/// Core database operations (auth tokens, backups, global settings).
#[async_trait]
pub trait DbBackend: Send + Sync + 'static {
    /// Atomically insert an access token if under the 10k limit.
    /// Encrypts token_id and token_secret at rest.
    async fn insert_access_token(&self, token: &str, id: &str, secret: &str) -> Result<(), String>;

    /// Retrieve and decrypt an access token's BookStack credentials.
    async fn get_access_token(&self, token: &str) -> Result<Option<(String, String)>, String>;

    /// Delete expired access tokens and refresh tokens.
    async fn cleanup_expired_tokens(&self) -> Result<(), String>;

    /// Store a refresh token mapped to encrypted BookStack credentials.
    async fn insert_refresh_token(&self, token: &str, id: &str, secret: &str)
        -> Result<(), String>;

    /// Retrieve and decrypt a refresh token's BookStack credentials.
    /// Returns None if the token doesn't exist or has expired.
    async fn get_refresh_token(&self, token: &str) -> Result<Option<(String, String)>, String>;

    /// Delete a refresh token (used during rotation — old token is consumed).
    async fn delete_refresh_token(&self, token: &str) -> Result<(), String>;

    /// Create a database backup. SQLite: VACUUM INTO. Postgres: no-op (use pg_dump).
    async fn backup(&self, path: &Path) -> Result<(), String>;

    // --- Global settings (server-instance-wide) ---

    /// Load the singleton global settings row. Returns defaults if never set.
    async fn get_global_settings(&self) -> Result<GlobalSettings, String>;

    /// Upsert the singleton global settings row. Records the writer's token hash.
    async fn save_global_settings(
        &self,
        settings: &GlobalSettings,
        set_by_token_hash: &str,
    ) -> Result<(), String>;
}

/// Semantic search database operations.
#[async_trait]
pub trait SemanticDb: Send + Sync + 'static {
    /// Create semantic search tables if they don't exist.
    async fn init_semantic_tables(&self) -> Result<(), String>;

    // --- Pages ---

    async fn upsert_page(&self, meta: &PageMeta) -> Result<(), String>;
    async fn delete_page(&self, page_id: i64) -> Result<(), String>;
    async fn get_page_content_hash(&self, page_id: i64) -> Result<Option<String>, String>;
    async fn get_page_meta(&self, page_id: i64) -> Result<Option<PageMeta>, String>;
    async fn resolve_page_slug(&self, slug: &str) -> Result<Option<i64>, String>;

    // --- Chunks + embeddings ---

    async fn insert_chunks(&self, page_id: i64, chunks: &[ChunkInsert]) -> Result<(), String>;
    async fn get_chunk_details(&self, chunk_ids: &[i64]) -> Result<Vec<ChunkDetail>, String>;

    // --- Relationships ---

    async fn replace_relationships(
        &self,
        source: i64,
        targets: &[(i64, String)],
    ) -> Result<(), String>;
    async fn get_markov_blanket(&self, page_id: i64) -> Result<MarkovBlanket, String>;

    // --- Job queue ---

    /// Create a pending embed job. Returns `(job_id, is_new)`.
    /// If a pending job with the same scope exists, returns it (`is_new=false`).
    /// If a running job with the same scope exists, returns it (`is_new=false`).
    /// Only creates a new job if no active job with the same scope exists.
    async fn create_embed_job(&self, scope: &str) -> Result<(i64, bool), String>;

    /// Atomically claim the next pending job (set status to 'running'). Returns None if queue is empty.
    /// Stamps the job with `worker_id` to identify which embedder owns it.
    async fn claim_next_job(&self, worker_id: &str) -> Result<Option<EmbedJob>, String>;

    /// Reset jobs stuck in 'running' for longer than the given duration back to 'pending'.
    async fn expire_stale_jobs(&self, stale_secs: i64) -> Result<usize, String>;

    /// Recover jobs owned by this worker that are stuck in 'running' (e.g. after a crash).
    /// Resets them to 'pending' so they can be reclaimed.
    async fn recover_worker_jobs(&self, worker_id: &str) -> Result<usize, String>;

    async fn update_job_progress(&self, job_id: i64, done: i64, total: i64) -> Result<(), String>;
    async fn complete_job(&self, job_id: i64, error: Option<&str>) -> Result<(), String>;
    async fn get_latest_job(&self) -> Result<Option<EmbedJob>, String>;
    async fn get_stats(&self) -> Result<EmbedStats, String>;

    /// List all pending/running jobs, plus the most recent completed/failed jobs (up to `recent`).
    async fn list_jobs(&self, recent: usize) -> Result<Vec<EmbedJob>, String>;

    // --- Job lifecycle (issue #54) ---

    /// Cancel a job. Idempotent: pending/running flip to `cancelled`; jobs
    /// already in a resolved or closed state are left alone. Running workers
    /// observe via `should_stop_embed_job`.
    async fn cancel_embed_job(&self, job_id: i64) -> Result<(), String>;

    /// True when the job is no longer `running` — running pipelines poll this
    /// at yield points to short-circuit on cancel/timeout/external failure.
    async fn should_stop_embed_job(&self, job_id: i64) -> Result<bool, String>;

    /// Mark a job as failed (hard timeout or systemic error). Sets
    /// `status='failed'`, `resolved_status='failed'`, `resolved_at=now`.
    /// Reconciler decides whether to retry, supersede, or give up.
    async fn fail_embed_job(&self, job_id: i64, reason: &str) -> Result<(), String>;

    /// Reconciler input: failed jobs that haven't been closed yet.
    async fn list_failed_open_embed_jobs(&self) -> Result<Vec<EmbedJob>, String>;

    /// Reconciler input: any same-scope job with `id > excluded_id` whose
    /// status is in pending/running/succeeded/cancelled/closed. Used to
    /// detect supersedence — a newer job for the same scope already exists.
    async fn has_successor_embed_job(&self, scope: &str, excluded_id: i64) -> Result<bool, String>;

    /// Walk `retry_of` back to the chain root and return the chain length
    /// (1 = original, 2 = first retry, ...).
    async fn embed_job_retry_chain_len(&self, job_id: i64) -> Result<usize, String>;

    /// Close a job (`status='closed'`, `prev_status=status`). When
    /// `resolved_status` is `Some`, overwrite the existing resolved_status;
    /// when `None`, preserve whatever's there (archiver path for
    /// succeeded/cancelled).
    async fn close_embed_job(
        &self,
        job_id: i64,
        resolved_status: Option<&str>,
    ) -> Result<(), String>;

    /// Insert a fresh pending job that records `retry_of` as its
    /// predecessor. Returns the new job id.
    async fn create_retry_embed_job(&self, scope: &str, retry_of: i64) -> Result<i64, String>;

    /// Archiver input: ids of succeeded/cancelled jobs whose
    /// `resolved_at` is older than `older_than_secs` and which haven't
    /// already been closed.
    async fn list_archivable_embed_jobs(&self, older_than_secs: i64) -> Result<Vec<i64>, String>;

    /// List currently-running jobs whose `started_at` is older than
    /// `started_before_secs` (unix epoch). Background timeout watcher
    /// uses this to fail hung jobs.
    async fn list_running_embed_jobs_started_before(
        &self,
        started_before_secs: i64,
    ) -> Result<Vec<EmbedJob>, String>;

    // --- Vector search ---

    /// Backend-specific vector search. SQLite: brute-force cosine scan. Postgres: pgvector HNSW.
    ///
    /// `scope`: when `Some(&filter)` with at least one non-empty ID list,
    /// restrict candidates to chunks whose parent page matches the union of
    /// the supplied book/chapter/page ID lists. When `None` or an
    /// all-empty filter, search across the entire embedded corpus.
    ///
    /// Shelf-level filtering is **not** evaluated here — the embedding
    /// `pages` table doesn't carry `shelf_id`. The caller is responsible for
    /// expanding `shelf_ids` to the matching `book_ids` (via the structural
    /// index) before invoking `vector_search`.
    async fn vector_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        threshold: f32,
        scope: Option<&ScopeFilter>,
    ) -> Result<Vec<SearchHit>, String>;

    /// Look up the `book_id` for each requested page in one roundtrip.
    /// Returns the rows that matched, in unspecified order. Pages missing
    /// from the embedding store are simply omitted.
    async fn get_page_book_ids(&self, page_ids: &[i64]) -> Result<Vec<(i64, i64)>, String>;

    /// Batched variant of `get_page_meta`. Returns one entry per requested
    /// page that exists in the embedding store; missing pages are omitted.
    async fn get_page_metas(&self, page_ids: &[i64]) -> Result<Vec<PageMeta>, String>;

    /// Delete all pages, chunks, and relationships. Used for full re-index.
    async fn clear_all_embeddings(&self) -> Result<(), String>;

    /// Alter the embedding vector dimension (e.g. when switching models).
    /// PostgreSQL: alters the pgvector column type and rebuilds the HNSW index.
    /// SQLite: no-op (BLOB columns are dimensionless).
    async fn alter_embedding_dimension(&self, dims: usize) -> Result<(), String>;

    // --- Inferred relationships ---

    /// Compute page centroids from chunk embeddings and store top-N most similar
    /// pages per page as "similar" relationships. Called after a full reindex.
    async fn compute_similar_pages(&self, top_k: usize, threshold: f32) -> Result<usize, String>;

    // --- Metadata key-value store ---

    /// Get a metadata value by key. Used for storing chunk_version, etc.
    async fn get_meta(&self, key: &str) -> Result<Option<String>, String>;

    /// Set a metadata value by key.
    async fn set_meta(&self, key: &str, value: &str) -> Result<(), String>;

    // --- Permission ACL (page-level role visibility) ---

    /// Replace the ACL row for one page. Deletes any prior `page_view_acl`
    /// entries for `page_id` then inserts the new role list. `default_open`
    /// is stored on the `pages` row (`acl_default_open` column) so the
    /// query path can short-circuit role checks for fully-open pages.
    async fn upsert_page_acl(&self, acl: &PageAcl) -> Result<(), String>;

    /// Drop a page from the ACL store. Called on `page_delete` events.
    async fn delete_page_acl(&self, page_id: i64) -> Result<(), String>;

    /// Drop one role from every page's ACL. Called on `role_delete` events.
    async fn delete_role_from_acl(&self, role_id: i64) -> Result<(), String>;

    /// List page IDs that have a stored ACL. Used by the daily reconciliation
    /// job to know which pages to refresh.
    async fn list_acl_page_ids(&self) -> Result<Vec<i64>, String>;

    /// DB-side ACL prefilter (issue #58 lever a.5). Single query joins
    /// `pages` against `page_view_acl` to bucket each requested page into
    /// [`AclPrefilter`]. Lets the semantic-search read path drop
    /// definitely-denied pages and admit definitely-allowed pages without
    /// any BookStack HTTP fan-out.
    ///
    /// Decision tree per page row:
    /// - `acl_computed_at IS NULL` → `Uncomputed`
    /// - `acl_default_open = true` → `DefaultOpen` (HTTP still required for
    ///   system-perm check, but skipped from the deny short-circuit).
    /// - any row in `page_view_acl` with `role_id IN role_ids` → `Allow`
    /// - else → `Deny`
    ///
    /// Returns one verdict per requested page that exists in the
    /// `pages` table. Pages missing from the embedding store are omitted
    /// (they have no ACL signal yet — fall back to HTTP).
    async fn prefilter_pages_by_roles(
        &self,
        page_ids: &[i64],
        role_ids: &[i64],
    ) -> Result<Vec<(i64, AclPrefilter)>, String>;

    // --- Permission cache (per-token, per-page) ---
    //
    // Issue #58 lever (a): durable L2 below the in-memory L1 in
    // `SemanticState`. Survives restart, so cold-start no longer means
    // fan-out a `can_access_page` call per candidate. Keyed by
    // `SHA256(token_id)` so a raw credential identifier never lands on disk.

    /// Fetch cached permission verdicts for a list of pages. Returns
    /// `(page_id, viewable)` only for entries that exist AND whose
    /// `cached_at >= min_cached_at`. Missing or stale rows are omitted so
    /// the caller can fall through to HTTP.
    async fn get_permission_cache_batch(
        &self,
        token_hash: &str,
        page_ids: &[i64],
        min_cached_at: i64,
    ) -> Result<Vec<(i64, bool)>, String>;

    /// Upsert one or more permission cache entries for a token. Idempotent
    /// on `(token_hash, page_id)`; refreshes both `viewable` and
    /// `cached_at`. Batched into a single transaction.
    async fn upsert_permission_cache_batch(
        &self,
        token_hash: &str,
        entries: &[(i64, bool)],
        cached_at: i64,
    ) -> Result<(), String>;

    /// Delete cache rows whose `cached_at < older_than`. Returns the row
    /// count removed. Called by the periodic eviction task.
    async fn evict_stale_permission_cache(&self, older_than: i64) -> Result<usize, String>;
}

/// v1.0.0 reconciliation index — structural mirror of every BookStack item we
/// care about (shelves, books, chapters, pages) plus a page-body cache. Phase
/// 4 of the identity-book-restructure RFC. The Phase 4 worker calls upsert_*
/// to populate; Phase 5 cuts over read paths to use list/get_* in place of
/// BookStack API calls.
///
/// All "soft delete" methods set `deleted = TRUE` rather than removing rows
/// — that lets a subsequent reconcile distinguish "page never existed" from
/// "page was deleted upstream" without having to re-query BookStack.
#[async_trait]
pub trait IndexDb: Send + Sync + 'static {
    // --- Shelves ---

    async fn upsert_indexed_shelf(&self, shelf: &IndexedShelf) -> Result<(), String>;
    async fn get_indexed_shelf(&self, shelf_id: i64) -> Result<Option<IndexedShelf>, String>;
    async fn soft_delete_indexed_shelf(&self, shelf_id: i64) -> Result<(), String>;

    // --- Books ---

    async fn upsert_indexed_book(&self, book: &IndexedBook) -> Result<(), String>;
    async fn get_indexed_book(&self, book_id: i64) -> Result<Option<IndexedBook>, String>;
    async fn list_indexed_books_by_shelf(&self, shelf_id: i64) -> Result<Vec<IndexedBook>, String>;
    async fn list_indexed_books_by_identity(
        &self,
        identity_ouid: &str,
    ) -> Result<Vec<IndexedBook>, String>;
    async fn soft_delete_indexed_book(&self, book_id: i64) -> Result<(), String>;

    // --- Chapters ---

    async fn upsert_indexed_chapter(&self, chapter: &IndexedChapter) -> Result<(), String>;
    async fn get_indexed_chapter(&self, chapter_id: i64) -> Result<Option<IndexedChapter>, String>;
    async fn list_indexed_chapters_by_book(
        &self,
        book_id: i64,
    ) -> Result<Vec<IndexedChapter>, String>;
    async fn soft_delete_indexed_chapter(&self, chapter_id: i64) -> Result<(), String>;

    // --- Pages ---
    //
    // upsert_indexed_page writes both the index row AND the optional
    // page_cache row in the same transaction. Keeping them in lockstep
    // is what makes `bookstack_pages.page_updated_at == page_cache.page_updated_at`
    // a reliable cache-hit invariant.

    async fn upsert_indexed_page(
        &self,
        page: &IndexedPage,
        cache: Option<&PageCache>,
    ) -> Result<(), String>;
    async fn get_indexed_page(&self, page_id: i64) -> Result<Option<IndexedPage>, String>;
    async fn find_indexed_page_by_key(
        &self,
        identity_ouid: &str,
        page_kind: PageKind,
        page_key: &str,
    ) -> Result<Option<IndexedPage>, String>;
    async fn list_indexed_pages_by_chapter(
        &self,
        chapter_id: i64,
    ) -> Result<Vec<IndexedPage>, String>;
    async fn list_indexed_pages_by_book_root(
        &self,
        book_id: i64,
    ) -> Result<Vec<IndexedPage>, String>;
    /// Most-recently-updated pages within a book, sorted by `page_updated_at`
    /// descending.
    async fn list_indexed_pages_recent(
        &self,
        book_id: i64,
        limit: i64,
    ) -> Result<Vec<IndexedPage>, String>;
    async fn soft_delete_indexed_page(&self, page_id: i64) -> Result<(), String>;

    // --- Page cache ---

    async fn get_page_cache(&self, page_id: i64) -> Result<Option<PageCache>, String>;

    // --- Index jobs (mirrors embed_jobs shape) ---
    //
    // create_index_job dedupes on `scope` like create_embed_job does — if a
    // pending or running job with the same scope exists, it's returned with
    // `is_new = false` so the caller can decide whether to wait or no-op.

    async fn create_index_job(
        &self,
        scope: &str,
        kind: &str,
        triggered_by: &str,
    ) -> Result<(i64, bool), String>;
    async fn claim_next_index_job(&self) -> Result<Option<IndexJob>, String>;
    async fn update_index_job_progress(
        &self,
        job_id: i64,
        progress: i64,
        total: i64,
    ) -> Result<(), String>;
    async fn complete_index_job(&self, job_id: i64, error: Option<&str>) -> Result<(), String>;
    async fn list_pending_index_jobs(&self, limit: i64) -> Result<Vec<IndexJob>, String>;
    async fn get_latest_index_job(&self) -> Result<Option<IndexJob>, String>;

    // --- Job lifecycle (issue #54) ---

    async fn cancel_index_job(&self, job_id: i64) -> Result<(), String>;
    async fn should_stop_index_job(&self, job_id: i64) -> Result<bool, String>;
    async fn fail_index_job(&self, job_id: i64, reason: &str) -> Result<(), String>;
    async fn list_failed_open_index_jobs(&self) -> Result<Vec<IndexJob>, String>;
    async fn has_successor_index_job(&self, scope: &str, excluded_id: i64) -> Result<bool, String>;
    async fn index_job_retry_chain_len(&self, job_id: i64) -> Result<usize, String>;
    async fn close_index_job(
        &self,
        job_id: i64,
        resolved_status: Option<&str>,
    ) -> Result<(), String>;
    async fn create_retry_index_job(
        &self,
        scope: &str,
        kind: &str,
        retry_of: i64,
    ) -> Result<i64, String>;
    async fn list_archivable_index_jobs(&self, older_than_secs: i64) -> Result<Vec<i64>, String>;
    async fn list_running_index_jobs_started_before(
        &self,
        started_before_secs: i64,
    ) -> Result<Vec<IndexJob>, String>;
    /// List the set of pending/running/failed-open jobs for status-page rendering.
    async fn list_index_jobs(&self, recent: usize) -> Result<Vec<IndexJob>, String>;

    // --- Index meta (singleton key-value) ---

    async fn get_index_meta(&self, key: &str) -> Result<Option<String>, String>;
    async fn set_index_meta(&self, key: &str, value: &str) -> Result<(), String>;

    // --- Directory tree (issue #69) ---

    /// List every non-deleted shelf in the index. Used by `read_directory_tree`
    /// when scope is `All` to enumerate the top level.
    async fn list_indexed_shelves(&self) -> Result<Vec<IndexedShelf>, String>;

    /// List every non-deleted book that isn't assigned to any shelf. Returned
    /// alongside `list_indexed_shelves` for the `All` scope so the synthetic
    /// "Unshelved" group can wrap them.
    async fn list_indexed_books_unshelved(&self) -> Result<Vec<IndexedBook>, String>;

    /// Assemble a scoped, depth-limited directory tree from the bookstack_*
    /// index tables (NOT live BookStack — target ~10ms warm).
    ///
    /// Depth semantics: 0 returns just the root nodes (shelves or the
    /// requested book/chapter) with empty `children`; 1 expands one level;
    /// unbounded (`depth = None`) walks down to pages. `scope` chooses the
    /// root; `All` materializes every shelf plus a synthetic "Unshelved"
    /// pseudo-shelf (id = 0, kind = `Shelf`) that holds any book with
    /// `shelf_id IS NULL`.
    ///
    /// Soft-deleted rows (`deleted = TRUE`) are skipped at every level. ACL
    /// filtering is the caller's job — this method returns the structural
    /// candidate tree; the MCP layer filters pages via BookStack's
    /// `can_access_page` and prunes empty branches.
    async fn read_directory_tree(
        &self,
        scope: DirectoryScope,
        depth: Option<u32>,
    ) -> Result<Vec<DirectoryNode>, String>;
}
