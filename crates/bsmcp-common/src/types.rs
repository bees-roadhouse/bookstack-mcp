#[derive(Clone, Debug)]
pub struct PageMeta {
    pub page_id: i64,
    pub book_id: i64,
    pub chapter_id: Option<i64>,
    pub name: String,
    pub slug: String,
    pub content_hash: String,
    /// ISO 8601 timestamp from BookStack API (e.g. "2025-03-10T14:30:00.000000Z")
    pub updated_at: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ChunkInsert {
    pub chunk_index: usize,
    pub heading_path: String,
    pub content: String,
    pub content_hash: String,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct ChunkDetail {
    pub chunk_id: i64,
    pub page_id: i64,
    pub heading_path: String,
    pub content: String,
    pub page_name: String,
}

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub chunk_id: i64,
    pub page_id: i64,
    pub score: f32,
}

/// Scope filter applied across all stages of the precision cascade and the
/// stage-1 vector pass of any scoped search. Empty vectors mean "no filter
/// for this kind"; an entirely empty [`ScopeFilter`] is treated as "full
/// corpus" by callers (use [`Self::is_empty`] to test).
///
/// IDs are unioned across the four kinds: a page qualifies if it matches any
/// shelf/book/chapter/page list. Stage-1 intersection is done at the SQL
/// layer in the page table — the embedding `pages` table carries `book_id`
/// and `chapter_id` directly; `shelf_id` is resolved via the structural
/// index's `bookstack_books.shelf_id` column.
#[derive(Clone, Debug, Default)]
pub struct ScopeFilter {
    pub shelf_ids: Vec<i64>,
    pub book_ids: Vec<i64>,
    pub chapter_ids: Vec<i64>,
    pub page_ids: Vec<i64>,
}

impl ScopeFilter {
    pub fn is_empty(&self) -> bool {
        self.shelf_ids.is_empty()
            && self.book_ids.is_empty()
            && self.chapter_ids.is_empty()
            && self.page_ids.is_empty()
    }

    /// Merge another scope's IDs into this one (union semantics).
    pub fn merge(&mut self, other: &ScopeFilter) {
        self.shelf_ids.extend(other.shelf_ids.iter().copied());
        self.book_ids.extend(other.book_ids.iter().copied());
        self.chapter_ids.extend(other.chapter_ids.iter().copied());
        self.page_ids.extend(other.page_ids.iter().copied());
    }

    /// Drop duplicate IDs in-place. Useful after [`Self::merge`] from
    /// multiple named scopes.
    pub fn dedup(&mut self) {
        for ids in [
            &mut self.shelf_ids,
            &mut self.book_ids,
            &mut self.chapter_ids,
            &mut self.page_ids,
        ] {
            ids.sort_unstable();
            ids.dedup();
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MarkovBlanket {
    pub linked_from: Vec<RelatedPage>,
    pub links_to: Vec<RelatedPage>,
    pub co_linked: Vec<RelatedPage>,
    pub siblings: Vec<RelatedPage>,
}

#[derive(Clone, Debug)]
pub struct RelatedPage {
    pub page_id: i64,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct EmbedJob {
    pub id: i64,
    pub scope: String,
    pub status: String,
    pub total_pages: i64,
    pub done_pages: i64,
    pub started_at: Option<i64>,
    pub finished_at: Option<i64>,
    pub error: Option<String>,
    pub worker_id: Option<String>,
    pub resolved_status: Option<String>,
    pub prev_status: Option<String>,
    pub resolved_at: Option<i64>,
    pub retry_of: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct EmbedStats {
    pub total_pages: i64,
    pub total_chunks: i64,
    pub latest_job: Option<EmbedJob>,
}

/// Effective access-control snapshot for a single page. Populated at embed
/// time by walking BookStack's content-permissions inheritance chain
/// (page → chapter → book → role-level defaults). Used by `vector_search` to
/// filter out pages the requesting user can't view, eliminating the per-page
/// `GET /api/pages/{id}` HTTP fan-out that dominated cold-cache search latency.
#[derive(Clone, Debug, Default)]
pub struct PageAcl {
    pub page_id: i64,
    /// Role IDs that can view the page. Empty means "explicitly restricted —
    /// no roles" (only page owner + admins via system permissions).
    pub view_roles: Vec<i64>,
    /// True when the resolved permission level is "all-inheriting from book"
    /// AND no explicit role overrides exist anywhere in the chain. The HTTP
    /// fallback path uses this to skip the cache-hit short-circuit for
    /// system-level role permissions that BookStack evaluates dynamically.
    pub default_open: bool,
    /// Unix epoch seconds the ACL was computed.
    pub computed_at: i64,
}
