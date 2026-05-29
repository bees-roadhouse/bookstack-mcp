//! v1.0.0 architecture: types and classification for the structural index of
//! BookStack content.
//!
//! Phase 3 of the identity-book-restructure RFC. The DB tables defined in the
//! backends (`bsmcp-db-sqlite`, `bsmcp-db-postgres`) mirror these structs.
//! The `classify_*` functions are pure over (parent context + name + position)
//! — no DB access, no async, easy to unit-test. The reconciliation worker
//! (Phase 4) calls them while walking BookStack content; the existing
//! provisioning code (Phase 1) doesn't use them yet but can after Phase 5.

use std::str::FromStr;

// --- Kinds ---

/// What role a shelf plays in the Hive layout. The two recognized shelves are
/// configured globally; anything else is unclassified and ignored by the
/// structural reflection logic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShelfKind {
    /// Global Hive shelf — holds every AI identity's Identity book + Collage,
    /// plus the shared Cross-Identity Collage.
    Hive,
    /// Global User Journals shelf — holds each human user's Identity + Journal.
    UserJournals,
    Unclassified,
}

impl ShelfKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hive => "hive",
            Self::UserJournals => "user_journals",
            Self::Unclassified => "unclassified",
        }
    }
}

impl FromStr for ShelfKind {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "hive" => Ok(Self::Hive),
            "user_journals" => Ok(Self::UserJournals),
            "unclassified" => Ok(Self::Unclassified),
            _ => Err(()),
        }
    }
}

/// What role a book plays. AI-side books (Identity / Collage / SharedCollage)
/// live on the Hive shelf; per-user books (UserIdentity / UserJournal) live on
/// the User Journals shelf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BookKind {
    Identity,
    Collage,
    SharedCollage,
    UserIdentity,
    UserJournal,
    Unclassified,
}

impl BookKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Collage => "collage",
            Self::SharedCollage => "shared_collage",
            Self::UserIdentity => "user_identity",
            Self::UserJournal => "user_journal",
            Self::Unclassified => "unclassified",
        }
    }
}

impl FromStr for BookKind {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "identity" => Ok(Self::Identity),
            "collage" => Ok(Self::Collage),
            "shared_collage" => Ok(Self::SharedCollage),
            "user_identity" => Ok(Self::UserIdentity),
            "user_journal" => Ok(Self::UserJournal),
            "unclassified" => Ok(Self::Unclassified),
            _ => Err(()),
        }
    }
}

/// What role a chapter plays inside an Identity book. The four recognized
/// kinds are the ones the v1.0.0 structure mandates; anything else is left
/// unclassified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChapterKind {
    Agents,
    SubagentConversations,
    JournalActive,
    JournalArchive,
    Unclassified,
}

impl ChapterKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Agents => "agents",
            Self::SubagentConversations => "subagent_conversations",
            Self::JournalActive => "journal_active",
            Self::JournalArchive => "journal_archive",
            Self::Unclassified => "unclassified",
        }
    }
}

impl FromStr for ChapterKind {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "agents" => Ok(Self::Agents),
            "subagent_conversations" => Ok(Self::SubagentConversations),
            "journal_active" => Ok(Self::JournalActive),
            "journal_archive" => Ok(Self::JournalArchive),
            "unclassified" => Ok(Self::Unclassified),
            _ => Err(()),
        }
    }
}

/// What role a page plays. Determined by parent (book + chapter) and name.
/// Pages that don't match any known shape stay `Unclassified` — the index
/// still tracks them, but they don't participate in the dedup UNIQUE index
/// (which is conditional on a non-null `page_kind` + `page_key`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageKind {
    Manifest,
    Agent,
    JournalEntry,
    CollageTopic,
    SubagentConversation,
    Unclassified,
}

impl PageKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Manifest => "manifest",
            Self::Agent => "agent",
            Self::JournalEntry => "journal_entry",
            Self::CollageTopic => "collage_topic",
            Self::SubagentConversation => "subagent_conversation",
            Self::Unclassified => "unclassified",
        }
    }
}

impl FromStr for PageKind {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "manifest" => Ok(Self::Manifest),
            "agent" => Ok(Self::Agent),
            "journal_entry" => Ok(Self::JournalEntry),
            "collage_topic" => Ok(Self::CollageTopic),
            "subagent_conversation" => Ok(Self::SubagentConversation),
            "unclassified" => Ok(Self::Unclassified),
            _ => Err(()),
        }
    }
}

// --- Structs (mirror DB rows) ---

#[derive(Clone, Debug)]
pub struct IndexedShelf {
    pub shelf_id: i64,
    pub name: String,
    pub slug: String,
    pub shelf_kind: ShelfKind,
    pub indexed_at: i64,
    pub deleted: bool,
}

#[derive(Clone, Debug)]
pub struct IndexedBook {
    pub book_id: i64,
    pub name: String,
    pub slug: String,
    pub shelf_id: Option<i64>,
    pub identity_ouid: Option<String>,
    pub book_kind: BookKind,
    pub indexed_at: i64,
    pub deleted: bool,
}

#[derive(Clone, Debug)]
pub struct IndexedChapter {
    pub chapter_id: i64,
    pub book_id: i64,
    pub name: String,
    pub slug: String,
    pub identity_ouid: Option<String>,
    pub chapter_kind: ChapterKind,
    pub archive_year: Option<i32>,
    pub indexed_at: i64,
    pub deleted: bool,
}

#[derive(Clone, Debug)]
pub struct IndexedPage {
    pub page_id: i64,
    pub book_id: i64,
    pub chapter_id: Option<i64>,
    pub name: String,
    pub slug: String,
    pub url: Option<String>,
    pub page_created_at: Option<String>,
    pub page_updated_at: Option<String>,
    pub identity_ouid: Option<String>,
    pub page_kind: PageKind,
    pub page_key: Option<String>,
    pub archive_year: Option<i32>,
    pub indexed_at: i64,
    pub deleted: bool,
}

#[derive(Clone, Debug)]
pub struct PageCache {
    pub page_id: i64,
    pub markdown: Option<String>,
    pub raw_markdown: Option<String>,
    pub html: Option<String>,
    pub cached_at: i64,
    pub page_updated_at: Option<String>,
}

// --- Directory tree (issue #69) ---
//
// The `directory` MCP tool returns a scoped, depth-limited tree assembled
// from the bookstack_* index tables. Replaces the assemble-it-yourself
// pattern of calling list_shelves + list_books + list_chapters + list_pages
// and stitching on the client. See `IndexDb::read_directory_tree`.

/// Where to root the directory walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectoryScope {
    /// Full tree: every shelf, plus shelf-less books grouped under a
    /// synthetic "Unshelved" root.
    All,
    Shelf(i64),
    Book(i64),
    Chapter(i64),
}

/// What level a `DirectoryNode` sits at. The same struct is reused at every
/// level — `node_type` says which.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectoryNodeKind {
    Shelf,
    Book,
    Chapter,
    Page,
}

impl DirectoryNodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shelf => "shelf",
            Self::Book => "book",
            Self::Chapter => "chapter",
            Self::Page => "page",
        }
    }
}

/// One node in the directory tree returned by `read_directory_tree`.
/// `id` is the BookStack id of the underlying entity. `children` is empty
/// for pages and for any node trimmed by `depth`. Page nodes carry
/// `page_kind` since the structural index already classifies them.
#[derive(Clone, Debug)]
pub struct DirectoryNode {
    pub kind: DirectoryNodeKind,
    pub id: i64,
    pub name: String,
    pub slug: String,
    /// Only set on `Page` nodes (mirrors `IndexedPage.page_kind.as_str()`).
    pub page_kind: Option<String>,
    pub children: Vec<DirectoryNode>,
}

#[derive(Clone, Debug)]
pub struct IndexJob {
    pub id: i64,
    pub scope: String,
    pub kind: String,
    pub status: String,
    pub triggered_by: String,
    pub started_at: Option<i64>,
    pub finished_at: Option<i64>,
    pub progress: i64,
    pub total: i64,
    pub error: Option<String>,
    pub resolved_status: Option<String>,
    pub prev_status: Option<String>,
    pub resolved_at: Option<i64>,
    pub retry_of: Option<i64>,
}

// --- Classification ---
//
// All classify_* functions are pure — no DB access, no async, no IO.
// The reconciliation worker (Phase 4) walks BookStack content and calls these
// to compute (kind, key, archive_year) for each item. Test-friendly,
// branch-coverable, no mocks needed.

/// Classify a shelf by id.
///
/// Issue #119 (v0.13.0): with the generic `indexed_shelves` model, the
/// indexer no longer distinguishes "hive shelf" from "user journals shelf".
/// Every shelf the worker walks is now [`ShelfKind::Unclassified`], so this
/// helper is a constant — kept as a single call site (rather than inlined
/// at every walk path) for grep-ability and to make the future "we again
/// care about shelf roles" reintroduction cheap.
///
/// The downstream [`ShelfKind::Hive`] / [`ShelfKind::UserJournals`] arms in
/// [`classify_book`] / [`classify_chapter`] / [`classify_page`] remain on
/// the structs for serde back-compat with pre-#119 indexed rows; they're
/// effectively dead code post-#119 because no shelf classifies into them
/// anymore. They will get pruned in a follow-up cleanup once any lingering
/// rows have been re-walked.
pub fn classify_shelf(_shelf_id: i64) -> ShelfKind {
    ShelfKind::Unclassified
}

/// Classify a book by its name and the kind of shelf it lives on.
///
/// Hive-shelf books fall into `Identity` (e.g., `"Pia Identity"`), `Collage`
/// (e.g., `"Pia's Collage"`), or `SharedCollage` (the literal name
/// `"Cross-Identity Collage"`). User-journals-shelf books fall into
/// `UserIdentity` (name contains `" — Identity"` or `" - Identity"`) or
/// `UserJournal` (name is exactly `"Journal"` or ends with `" Journal"`).
pub fn classify_book(name: &str, parent_shelf_kind: ShelfKind) -> BookKind {
    let trimmed = name.trim();
    match parent_shelf_kind {
        ShelfKind::Hive => {
            if trimmed.ends_with(" Identity") || trimmed.ends_with("'s Identity") {
                BookKind::Identity
            } else if trimmed == "Cross-Identity Collage" {
                BookKind::SharedCollage
            } else if trimmed.ends_with("'s Collage") || trimmed.ends_with(" Collage") {
                BookKind::Collage
            } else {
                BookKind::Unclassified
            }
        }
        ShelfKind::UserJournals => {
            // Per-user identity book: "{user_id} — Identity" (em-dash) or hyphen variant.
            if trimmed.contains(" — Identity") || trimmed.contains(" - Identity") {
                BookKind::UserIdentity
            } else if trimmed == "Journal"
                || trimmed.ends_with(" Journal")
                || trimmed.ends_with("'s Journal")
            {
                BookKind::UserJournal
            } else {
                BookKind::Unclassified
            }
        }
        ShelfKind::Unclassified => BookKind::Unclassified,
    }
}

/// Classify a chapter by its name and the kind of book it lives in. Returns
/// `(kind, archive_year)` — `archive_year` is `Some` only for `JournalArchive`.
pub fn classify_chapter(name: &str, parent_book_kind: BookKind) -> (ChapterKind, Option<i32>) {
    let trimmed = name.trim();
    if !matches!(parent_book_kind, BookKind::Identity) {
        // Only Identity books have meaningful chapter classification under v1.0.0.
        // UserIdentity books may grow chapters later; for now, leave them
        // unclassified rather than guessing.
        return (ChapterKind::Unclassified, None);
    }
    match trimmed {
        "Agents" => (ChapterKind::Agents, None),
        "Subagent Conversations" => (ChapterKind::SubagentConversations, None),
        "Journal" => (ChapterKind::JournalActive, None),
        _ => {
            if let Some(year) = parse_archive_year(trimmed) {
                (ChapterKind::JournalArchive, Some(year))
            } else {
                (ChapterKind::Unclassified, None)
            }
        }
    }
}

/// Classify a page by its name and parent context. Returns `(kind, key,
/// archive_year)`. `key` is the natural identifier the /remember protocol uses
/// — date for journals, agent name for agents, slug-or-name for collage
/// topics, etc. `archive_year` is `Some` only for journal entries living in
/// an archive chapter.
pub fn classify_page(
    name: &str,
    parent_book_kind: BookKind,
    parent_chapter_kind: Option<ChapterKind>,
    parent_chapter_archive_year: Option<i32>,
) -> (PageKind, Option<String>, Option<i32>) {
    let trimmed = name.trim();

    // Pages inside chapters: chapter context dominates.
    match parent_chapter_kind {
        Some(ChapterKind::Agents) => {
            // "Agent: pia-journal-agent" → key = "pia-journal-agent"
            let key = trimmed
                .strip_prefix("Agent: ")
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            return (PageKind::Agent, key, None);
        }
        Some(ChapterKind::SubagentConversations) => {
            return (
                PageKind::SubagentConversation,
                Some(trimmed.to_string()).filter(|s| !s.is_empty()),
                None,
            );
        }
        Some(ChapterKind::JournalActive) => {
            if let Some(date) = match_iso_date(trimmed) {
                return (PageKind::JournalEntry, Some(date), None);
            }
            return (PageKind::Unclassified, None, None);
        }
        Some(ChapterKind::JournalArchive) => {
            if let Some(date) = match_iso_date(trimmed) {
                return (
                    PageKind::JournalEntry,
                    Some(date),
                    parent_chapter_archive_year,
                );
            }
            return (PageKind::Unclassified, None, None);
        }
        Some(ChapterKind::Unclassified) | None => {}
    }

    // No chapter (loose at book root) or unclassified chapter — book context decides.
    match parent_book_kind {
        BookKind::Identity | BookKind::UserIdentity => {
            // Identity manifest sits loose at book root, named exactly "Identity".
            if trimmed == "Identity" {
                (PageKind::Manifest, None, None)
            } else {
                (PageKind::Unclassified, None, None)
            }
        }
        BookKind::Collage | BookKind::SharedCollage => {
            // Topic pages sit loose at the collage book root. Key = the page name.
            (
                PageKind::CollageTopic,
                Some(trimmed.to_string()).filter(|s| !s.is_empty()),
                None,
            )
        }
        BookKind::UserJournal => {
            // Legacy shape: journal entries sat at book root, named YYYY-MM-DD.
            // (Post-v1.0.0 they live in the Journal chapter inside UserIdentity.
            // Recognize the legacy shape so the indexer can still classify
            // existing user journals before migration.)
            if let Some(date) = match_iso_date(trimmed) {
                (PageKind::JournalEntry, Some(date), None)
            } else {
                (PageKind::Unclassified, None, None)
            }
        }
        BookKind::Unclassified => (PageKind::Unclassified, None, None),
    }
}

// --- Helpers ---

/// Parse `"Journal Archive - 2025"` → `Some(2025)`. Tolerates leading/trailing
/// whitespace and an optional space around the hyphen, but the prefix
/// `"Journal Archive"` is required exactly.
fn parse_archive_year(name: &str) -> Option<i32> {
    let rest = name.trim().strip_prefix("Journal Archive")?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('-')?;
    let year_str = rest.trim();
    if year_str.len() != 4 {
        return None;
    }
    year_str
        .parse::<i32>()
        .ok()
        .filter(|y| (1900..=9999).contains(y))
}

/// Match `YYYY-MM-DD`. Returns the canonicalized date string if valid.
/// Doesn't validate calendar correctness (Feb 30 passes) — name-shape only.
fn match_iso_date(s: &str) -> Option<String> {
    let trimmed = s.trim();
    if trimmed.len() != 10 {
        return None;
    }
    let bytes = trimmed.as_bytes();
    if bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let year_ok = bytes[..4].iter().all(|b| b.is_ascii_digit());
    let month_ok = bytes[5..7].iter().all(|b| b.is_ascii_digit());
    let day_ok = bytes[8..10].iter().all(|b| b.is_ascii_digit());
    if !(year_ok && month_ok && day_ok) {
        return None;
    }
    Some(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #119 — post-refactor every shelf is `Unclassified`. The Hive /
    /// UserJournals variants survive on the enum for back-compat with rows
    /// already in the index but the classifier no longer produces them.
    #[test]
    fn shelf_classification_is_always_unclassified() {
        assert_eq!(classify_shelf(927), ShelfKind::Unclassified);
        assert_eq!(classify_shelf(2023), ShelfKind::Unclassified);
        assert_eq!(classify_shelf(0), ShelfKind::Unclassified);
    }

    #[test]
    fn book_classification_on_hive_shelf() {
        let h = ShelfKind::Hive;
        assert_eq!(classify_book("Pia Identity", h), BookKind::Identity);
        assert_eq!(classify_book("Apis's Identity", h), BookKind::Identity);
        assert_eq!(classify_book("  Pia Identity  ", h), BookKind::Identity); // trims
        assert_eq!(classify_book("Pia's Collage", h), BookKind::Collage);
        assert_eq!(
            classify_book("Cross-Identity Collage", h),
            BookKind::SharedCollage
        );
        assert_eq!(classify_book("Some Random Book", h), BookKind::Unclassified);
    }

    #[test]
    fn book_classification_on_user_journals_shelf() {
        let u = ShelfKind::UserJournals;
        assert_eq!(
            classify_book("nate@example.com — Identity", u),
            BookKind::UserIdentity
        );
        assert_eq!(
            classify_book("nate@example.com - Identity", u),
            BookKind::UserIdentity
        );
        assert_eq!(classify_book("Journal", u), BookKind::UserJournal);
        assert_eq!(classify_book("Nate's Journal", u), BookKind::UserJournal);
        assert_eq!(classify_book("Random Notes", u), BookKind::Unclassified);
    }

    #[test]
    fn book_classification_on_unclassified_shelf_is_always_unclassified() {
        let s = ShelfKind::Unclassified;
        assert_eq!(classify_book("Pia Identity", s), BookKind::Unclassified);
        assert_eq!(
            classify_book("Cross-Identity Collage", s),
            BookKind::Unclassified
        );
    }

    #[test]
    fn chapter_classification() {
        assert_eq!(
            classify_chapter("Agents", BookKind::Identity),
            (ChapterKind::Agents, None)
        );
        assert_eq!(
            classify_chapter("Subagent Conversations", BookKind::Identity),
            (ChapterKind::SubagentConversations, None)
        );
        assert_eq!(
            classify_chapter("Journal", BookKind::Identity),
            (ChapterKind::JournalActive, None)
        );
        assert_eq!(
            classify_chapter("Journal Archive - 2025", BookKind::Identity),
            (ChapterKind::JournalArchive, Some(2025))
        );
        assert_eq!(
            classify_chapter("Journal Archive - 2024", BookKind::Identity),
            (ChapterKind::JournalArchive, Some(2024))
        );
        // Tolerates whitespace.
        assert_eq!(
            classify_chapter("  Journal Archive - 2026  ", BookKind::Identity),
            (ChapterKind::JournalArchive, Some(2026))
        );
        assert_eq!(
            classify_chapter("Random Chapter", BookKind::Identity),
            (ChapterKind::Unclassified, None)
        );
    }

    #[test]
    fn chapter_classification_outside_identity_is_always_unclassified() {
        assert_eq!(
            classify_chapter("Agents", BookKind::Collage),
            (ChapterKind::Unclassified, None)
        );
        assert_eq!(
            classify_chapter("Journal", BookKind::UserJournal),
            (ChapterKind::Unclassified, None)
        );
    }

    #[test]
    fn page_classification_inside_agents_chapter() {
        let (k, key, year) = classify_page(
            "Agent: pia-journal-agent",
            BookKind::Identity,
            Some(ChapterKind::Agents),
            None,
        );
        assert_eq!(k, PageKind::Agent);
        assert_eq!(key.as_deref(), Some("pia-journal-agent"));
        assert_eq!(year, None);
    }

    #[test]
    fn page_classification_inside_journal_active() {
        let (k, key, year) = classify_page(
            "2026-04-27",
            BookKind::Identity,
            Some(ChapterKind::JournalActive),
            None,
        );
        assert_eq!(k, PageKind::JournalEntry);
        assert_eq!(key.as_deref(), Some("2026-04-27"));
        assert_eq!(year, None);
    }

    #[test]
    fn page_classification_inside_journal_archive_includes_year() {
        let (k, key, year) = classify_page(
            "2025-12-31",
            BookKind::Identity,
            Some(ChapterKind::JournalArchive),
            Some(2025),
        );
        assert_eq!(k, PageKind::JournalEntry);
        assert_eq!(key.as_deref(), Some("2025-12-31"));
        assert_eq!(year, Some(2025));
    }

    #[test]
    fn page_classification_non_iso_in_journal_chapter_is_unclassified() {
        let (k, key, _) = classify_page(
            "Random Note",
            BookKind::Identity,
            Some(ChapterKind::JournalActive),
            None,
        );
        assert_eq!(k, PageKind::Unclassified);
        assert!(key.is_none());
    }

    #[test]
    fn page_classification_loose_at_identity_root() {
        // "Identity" at book root → manifest.
        let (k, key, _) = classify_page("Identity", BookKind::Identity, None, None);
        assert_eq!(k, PageKind::Manifest);
        assert!(key.is_none());

        // Anything else at the identity book root is unclassified — it's
        // probably a stray page outside the agents chapter (the migration
        // tool will move it).
        let (k, _, _) = classify_page("Some Stray Page", BookKind::Identity, None, None);
        assert_eq!(k, PageKind::Unclassified);
    }

    #[test]
    fn page_classification_in_collage_book() {
        let (k, key, _) = classify_page("The Practice", BookKind::Collage, None, None);
        assert_eq!(k, PageKind::CollageTopic);
        assert_eq!(key.as_deref(), Some("The Practice"));
    }

    #[test]
    fn page_classification_legacy_user_journal_at_book_root() {
        // Pre-v1.0.0 shape: journal entries at the user-journal book root.
        let (k, key, _) = classify_page("2026-04-27", BookKind::UserJournal, None, None);
        assert_eq!(k, PageKind::JournalEntry);
        assert_eq!(key.as_deref(), Some("2026-04-27"));
    }

    #[test]
    fn iso_date_parser_rejects_garbage() {
        assert!(match_iso_date("2026-04-27").is_some());
        assert!(match_iso_date("2026/04/27").is_none()); // slashes not hyphens
        assert!(match_iso_date("2026-4-27").is_none()); // single-digit month
        assert!(match_iso_date("Random").is_none());
        assert!(match_iso_date("").is_none());
    }

    #[test]
    fn archive_year_parser_rejects_garbage() {
        assert_eq!(parse_archive_year("Journal Archive - 2025"), Some(2025));
        assert_eq!(parse_archive_year("Journal Archive  -  2025"), Some(2025));
        assert_eq!(parse_archive_year("Journal Archive - 25"), None); // too short
        assert_eq!(parse_archive_year("Archive - 2025"), None); // wrong prefix
        assert_eq!(parse_archive_year("Journal - 2025"), None); // wrong prefix
        assert_eq!(parse_archive_year("Journal Archive - abcd"), None);
    }

    #[test]
    fn kind_str_roundtrip() {
        for k in [
            ShelfKind::Hive,
            ShelfKind::UserJournals,
            ShelfKind::Unclassified,
        ] {
            assert_eq!(ShelfKind::from_str(k.as_str()), Ok(k));
        }
        for k in [
            BookKind::Identity,
            BookKind::Collage,
            BookKind::SharedCollage,
            BookKind::UserIdentity,
            BookKind::UserJournal,
            BookKind::Unclassified,
        ] {
            assert_eq!(BookKind::from_str(k.as_str()), Ok(k));
        }
        for k in [
            ChapterKind::Agents,
            ChapterKind::SubagentConversations,
            ChapterKind::JournalActive,
            ChapterKind::JournalArchive,
            ChapterKind::Unclassified,
        ] {
            assert_eq!(ChapterKind::from_str(k.as_str()), Ok(k));
        }
        for k in [
            PageKind::Manifest,
            PageKind::Agent,
            PageKind::JournalEntry,
            PageKind::CollageTopic,
            PageKind::SubagentConversation,
            PageKind::Unclassified,
        ] {
            assert_eq!(PageKind::from_str(k.as_str()), Ok(k));
        }
    }
}
