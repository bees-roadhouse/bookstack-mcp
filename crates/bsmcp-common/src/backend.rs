//! The content backend behind the 59 BookStack-shaped tools.
//!
//! `Backend` is the surface the tool dispatcher, the semantic-search ACL
//! path, the embedder pipeline and the reconciliation worker call. It is
//! shaped exactly like `BookStackClient` (issue #156): every method keeps
//! BookStack's argument list and returns BookStack's JSON shape, so the
//! dispatcher is backend-agnostic and every connector keeps working when the
//! household moves to LibStack.
//!
//! Two implementations:
//!
//! * [`crate::bookstack::BookStackClient`] — the original, unchanged in
//!   behaviour. Its trait impl delegates to the inherent methods.
//! * [`crate::libstack::LibStackClient`] — talks to LibStack's JSON `/api`
//!   and maps nested collections onto shelves / books / chapters. See that
//!   module for the mapping.
//!
//! Selection is `BSMCP_BACKEND=bookstack|libstack` (default `bookstack`),
//! resolved once at startup by [`BackendConfig::from_env`]; per-session
//! clients are minted by [`BackendConfig::client`].

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::bookstack::{BookStackClient, ContentType, CredentialCheck, ExportFormat, UserIdentity};
use crate::libstack::LibStackClient;

/// Which content system a client talks to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    BookStack,
    LibStack,
}

impl BackendKind {
    /// `BSMCP_BACKEND`, case-insensitive. Anything other than `libstack`
    /// is BookStack so existing deployments are untouched by the variable's
    /// existence.
    pub fn from_env() -> Self {
        Self::parse(&std::env::var("BSMCP_BACKEND").unwrap_or_default())
    }

    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "libstack" => Self::LibStack,
            _ => Self::BookStack,
        }
    }

    /// Lower-case config spelling (`bookstack` / `libstack`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BookStack => "bookstack",
            Self::LibStack => "libstack",
        }
    }

    /// Product name for messages shown to people and AI clients.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::BookStack => "BookStack",
            Self::LibStack => "LibStack",
        }
    }
}

/// The one error every unsupported operation returns, so a caller can match
/// on it and no backend ever fakes a success for a feature it lacks.
pub fn not_available(kind: BackendKind, feature: &str) -> String {
    format!("{feature} is not available on {}", kind.display_name())
}

/// Startup-resolved backend selection: which system, where the API is
/// dialed, and which URL a browser can open (for the links in tool output).
#[derive(Clone, Debug)]
pub struct BackendConfig {
    pub kind: BackendKind,
    /// Where the server dials the API; may be an internal host.
    pub api_url: String,
    /// Browser-reachable URL for human-facing links. Defaults to `api_url`.
    pub public_url: String,
}

impl BackendConfig {
    /// Read `BSMCP_BACKEND` and the matching URL pair:
    ///
    /// * BookStack: `BSMCP_BOOKSTACK_URL` (required) and
    ///   `BSMCP_BOOKSTACK_PUBLIC_URL` (optional, defaults to the API URL).
    /// * LibStack: `BSMCP_LIBSTACK_URL` (required) and
    ///   `BSMCP_LIBSTACK_PUBLIC_URL` (optional, defaults to the API URL).
    ///
    /// Returns a human-readable error naming the missing variable.
    pub fn from_env() -> Result<Self, String> {
        let kind = BackendKind::from_env();
        let (url_var, public_var) = match kind {
            BackendKind::BookStack => ("BSMCP_BOOKSTACK_URL", "BSMCP_BOOKSTACK_PUBLIC_URL"),
            BackendKind::LibStack => ("BSMCP_LIBSTACK_URL", "BSMCP_LIBSTACK_PUBLIC_URL"),
        };
        let api_url = std::env::var(url_var)
            .ok()
            .map(|v| v.trim().trim_end_matches('/').to_string())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| format!("{url_var} is required (BSMCP_BACKEND={})", kind.as_str()))?;
        let public_url = std::env::var(public_var)
            .ok()
            .map(|v| v.trim().trim_end_matches('/').to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| api_url.clone());
        Ok(Self {
            kind,
            api_url,
            public_url,
        })
    }

    /// Explicit constructor for tests and tools that don't read the env.
    pub fn new(kind: BackendKind, api_url: &str, public_url: Option<&str>) -> Self {
        let api_url = api_url.trim().trim_end_matches('/').to_string();
        let public_url = public_url
            .map(|p| p.trim().trim_end_matches('/').to_string())
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| api_url.clone());
        Self {
            kind,
            api_url,
            public_url,
        }
    }

    /// Mint a per-session client for this backend.
    ///
    /// Both backends take the same `(token_id, token_secret)` pair the
    /// OAuth form, the access-token store and the legacy `Bearer id:secret`
    /// header already carry. BookStack sends them as `Token id:secret`.
    /// LibStack has a single bearer token: see
    /// [`LibStackClient::bearer_from_pair`] for how the pair becomes one.
    pub fn client(
        &self,
        token_id: &str,
        token_secret: &str,
        http: reqwest::Client,
    ) -> Arc<dyn Backend> {
        match self.kind {
            BackendKind::BookStack => Arc::new(
                BookStackClient::new(&self.api_url, token_id, token_secret, http)
                    .with_public_url(&self.public_url),
            ),
            BackendKind::LibStack => Arc::new(
                LibStackClient::new(&self.api_url, token_id, token_secret, http)
                    .with_public_url(&self.public_url),
            ),
        }
    }
}

/// Backend-agnostic content API. Argument lists and JSON shapes are
/// BookStack's; see the module docs.
///
/// Methods a backend does not support return
/// [`not_available`]`(kind, feature)` as the `Err` string and never a
/// fabricated success.
#[async_trait]
pub trait Backend: Send + Sync + 'static {
    fn kind(&self) -> BackendKind;

    /// URL the API is dialed on. Never use for links shown to people.
    fn base_url(&self) -> &str;
    /// Browser-reachable URL for human-facing links.
    fn public_url(&self) -> &str;
    /// Non-secret identifier for the session's credential, used as a cache
    /// key (structure blurb, ACL caches). Never log the secret half.
    fn token_id(&self) -> &str;

    // --- Credentials and identity ---

    async fn validate(&self) -> CredentialCheck;
    async fn is_admin(&self) -> Result<bool, String>;
    async fn can_access_page(&self, page_id: i64) -> bool;
    /// `Ok(None)` disables the ACL prefilter (every page falls through to
    /// `can_access_page`); a backend without roles returns exactly that.
    async fn list_my_roles(&self) -> Result<Option<Vec<i64>>, String>;
    async fn whoami(&self) -> Result<Option<UserIdentity>, String>;

    // --- Shelves ---

    async fn list_shelves(&self, count: i64, offset: i64) -> Result<Value, String>;
    async fn list_all_shelves(&self) -> Result<Vec<i64>, String>;
    async fn get_shelf(&self, id: i64) -> Result<Value, String>;
    async fn create_shelf(&self, name: &str, description: &str) -> Result<Value, String>;
    async fn update_shelf(&self, id: i64, data: &Value) -> Result<Value, String>;
    async fn delete_shelf(&self, id: i64) -> Result<(), String>;

    // --- Books ---

    async fn list_books(&self, count: i64, offset: i64) -> Result<Value, String>;
    async fn list_all_books(&self) -> Result<Vec<i64>, String>;
    async fn get_book(&self, id: i64) -> Result<Value, String>;
    /// `shelf_id` is optional on BookStack (a book may sit on no shelf) and
    /// required on LibStack (a book is a collection inside a shelf).
    async fn create_book(
        &self,
        name: &str,
        description: &str,
        shelf_id: Option<i64>,
    ) -> Result<Value, String>;
    async fn update_book(&self, id: i64, data: &Value) -> Result<Value, String>;
    async fn delete_book(&self, id: i64) -> Result<(), String>;

    // --- Chapters ---

    async fn list_chapters(&self, count: i64, offset: i64) -> Result<Value, String>;
    async fn get_chapter(&self, id: i64) -> Result<Value, String>;
    async fn create_chapter(
        &self,
        book_id: i64,
        name: &str,
        description: &str,
    ) -> Result<Value, String>;
    async fn update_chapter(&self, id: i64, data: &Value) -> Result<Value, String>;
    async fn delete_chapter(&self, id: i64) -> Result<(), String>;

    // --- Pages ---

    async fn list_pages(&self, count: i64, offset: i64) -> Result<Value, String>;
    /// Pages with `updated_at` strictly after `since_iso_utc`, oldest first,
    /// at most `count` rows. Drives the index worker's delta walk.
    async fn list_pages_updated_since(
        &self,
        since_iso_utc: &str,
        count: i64,
    ) -> Result<Vec<Value>, String>;
    async fn get_page(&self, id: i64) -> Result<Value, String>;
    async fn create_page(&self, data: &Value) -> Result<Value, String>;
    async fn update_page(&self, id: i64, data: &Value) -> Result<Value, String>;
    async fn delete_page(&self, id: i64) -> Result<(), String>;

    // --- Search ---

    async fn search(&self, query: &str, page: i64, count: i64) -> Result<Value, String>;

    // --- Attachments ---

    /// `page_id` narrows the listing to one page. BookStack lists every
    /// attachment when it is `None`; LibStack stores attachments per page
    /// and requires it.
    async fn list_attachments(&self, page_id: Option<i64>) -> Result<Value, String>;
    async fn get_attachment(&self, id: i64) -> Result<Value, String>;
    async fn create_attachment(&self, data: &Value) -> Result<Value, String>;
    async fn update_attachment(&self, id: i64, data: &Value) -> Result<Value, String>;
    async fn delete_attachment(&self, id: i64) -> Result<(), String>;
    async fn create_file_attachment(
        &self,
        name: &str,
        uploaded_to: i64,
        filename: &str,
        bytes: Vec<u8>,
        mime_type: &str,
    ) -> Result<Value, String>;

    // --- Exports ---

    async fn export_page(&self, id: i64, format: ExportFormat) -> Result<String, String>;
    async fn export_chapter(&self, id: i64, format: ExportFormat) -> Result<String, String>;
    async fn export_book(&self, id: i64, format: ExportFormat) -> Result<String, String>;

    // --- Comments ---

    async fn list_comments(&self, query: &[(&str, &str)]) -> Result<Value, String>;
    async fn get_comment(&self, id: i64) -> Result<Value, String>;
    /// `data` carries `page_id`, `html` (rendered by the dispatcher) and,
    /// when the caller supplied it, the original `markdown`. BookStack sends
    /// only `html`; LibStack keeps `markdown`.
    async fn create_comment(&self, data: &Value) -> Result<Value, String>;
    async fn update_comment(&self, id: i64, data: &Value) -> Result<Value, String>;
    async fn delete_comment(&self, id: i64) -> Result<(), String>;

    // --- Recycle bin ---

    async fn list_recycle_bin(&self, count: i64, offset: i64) -> Result<Value, String>;
    async fn restore_recycle_bin_item(&self, id: i64) -> Result<Value, String>;
    async fn destroy_recycle_bin_item(&self, id: i64) -> Result<(), String>;

    // --- Users ---

    async fn list_users(&self, count: i64, offset: i64) -> Result<Value, String>;
    async fn get_user(&self, id: i64) -> Result<Value, String>;

    // --- Audit log / system ---

    async fn list_audit_log(&self, count: i64, offset: i64) -> Result<Value, String>;
    async fn get_system_info(&self) -> Result<Value, String>;

    // --- Images ---

    async fn list_images(
        &self,
        count: i64,
        offset: i64,
        filter: &[(&str, &str)],
    ) -> Result<Value, String>;
    async fn get_image(&self, id: i64) -> Result<Value, String>;
    async fn update_image(&self, id: i64, data: &Value) -> Result<Value, String>;
    async fn delete_image(&self, id: i64) -> Result<(), String>;
    async fn upload_image(
        &self,
        name: &str,
        image_type: &str,
        uploaded_to: i64,
        filename: &str,
        bytes: Vec<u8>,
        mime_type: &str,
    ) -> Result<Value, String>;

    // --- Content permissions / roles ---

    async fn get_content_permissions(
        &self,
        content_type: ContentType,
        content_id: i64,
    ) -> Result<Value, String>;
    async fn update_content_permissions(
        &self,
        content_type: ContentType,
        content_id: i64,
        data: &Value,
    ) -> Result<Value, String>;
    async fn list_roles(&self, count: i64, offset: i64) -> Result<Value, String>;
    async fn get_role(&self, id: i64) -> Result<Value, String>;

    // --- Collections (the real LibStack tree) ---
    //
    // BookStack has no nested containers; the defaults answer
    // `not_available`. LibStack overrides all three.

    async fn list_collections(&self) -> Result<Value, String> {
        Err(not_available(self.kind(), "collections"))
    }
    async fn get_collection(&self, id: i64) -> Result<Value, String> {
        let _ = id;
        Err(not_available(self.kind(), "collections"))
    }
    async fn create_collection(
        &self,
        parent_id: Option<i64>,
        name: &str,
        description: &str,
    ) -> Result<Value, String> {
        let _ = (parent_id, name, description);
        Err(not_available(self.kind(), "collections"))
    }
}

/// Shared handle type. Sessions, worker and pipeline hold one of these.
pub type DynBackend = Arc<dyn Backend>;

// --- BookStackClient: the first implementation, behaviour unchanged ---

#[async_trait]
impl Backend for BookStackClient {
    fn kind(&self) -> BackendKind {
        BackendKind::BookStack
    }
    fn base_url(&self) -> &str {
        BookStackClient::base_url(self)
    }
    fn public_url(&self) -> &str {
        BookStackClient::public_url(self)
    }
    fn token_id(&self) -> &str {
        BookStackClient::token_id(self)
    }

    async fn validate(&self) -> CredentialCheck {
        BookStackClient::validate(self).await
    }
    async fn is_admin(&self) -> Result<bool, String> {
        BookStackClient::is_admin(self).await
    }
    async fn can_access_page(&self, page_id: i64) -> bool {
        BookStackClient::can_access_page(self, page_id).await
    }
    async fn list_my_roles(&self) -> Result<Option<Vec<i64>>, String> {
        BookStackClient::list_my_roles(self).await
    }
    async fn whoami(&self) -> Result<Option<UserIdentity>, String> {
        BookStackClient::whoami(self).await
    }

    async fn list_shelves(&self, count: i64, offset: i64) -> Result<Value, String> {
        BookStackClient::list_shelves(self, count, offset).await
    }
    async fn list_all_shelves(&self) -> Result<Vec<i64>, String> {
        BookStackClient::list_all_shelves(self).await
    }
    async fn get_shelf(&self, id: i64) -> Result<Value, String> {
        BookStackClient::get_shelf(self, id).await
    }
    async fn create_shelf(&self, name: &str, description: &str) -> Result<Value, String> {
        BookStackClient::create_shelf(self, name, description).await
    }
    async fn update_shelf(&self, id: i64, data: &Value) -> Result<Value, String> {
        BookStackClient::update_shelf(self, id, data).await
    }
    async fn delete_shelf(&self, id: i64) -> Result<(), String> {
        BookStackClient::delete_shelf(self, id).await
    }

    async fn list_books(&self, count: i64, offset: i64) -> Result<Value, String> {
        BookStackClient::list_books(self, count, offset).await
    }
    async fn list_all_books(&self) -> Result<Vec<i64>, String> {
        BookStackClient::list_all_books(self).await
    }
    async fn get_book(&self, id: i64) -> Result<Value, String> {
        BookStackClient::get_book(self, id).await
    }
    /// The inherent `create_book` is unchanged. When a `shelf_id` is given
    /// the new book is appended to that shelf's `books` the same way
    /// `move_book_to_shelf` does it (read the shelf, write the list back).
    async fn create_book(
        &self,
        name: &str,
        description: &str,
        shelf_id: Option<i64>,
    ) -> Result<Value, String> {
        let book = BookStackClient::create_book(self, name, description).await?;
        if let Some(shelf_id) = shelf_id {
            let book_id = book
                .get("id")
                .and_then(|v| v.as_i64())
                .ok_or("create_book: BookStack returned a book without an id")?;
            let shelf = BookStackClient::get_shelf(self, shelf_id).await?;
            let mut books: Vec<i64> = shelf
                .get("books")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|b| b.get("id").and_then(|v| v.as_i64()))
                        .collect()
                })
                .unwrap_or_default();
            if !books.contains(&book_id) {
                books.push(book_id);
            }
            BookStackClient::update_shelf(self, shelf_id, &serde_json::json!({ "books": books }))
                .await?;
        }
        Ok(book)
    }
    async fn update_book(&self, id: i64, data: &Value) -> Result<Value, String> {
        BookStackClient::update_book(self, id, data).await
    }
    async fn delete_book(&self, id: i64) -> Result<(), String> {
        BookStackClient::delete_book(self, id).await
    }

    async fn list_chapters(&self, count: i64, offset: i64) -> Result<Value, String> {
        BookStackClient::list_chapters(self, count, offset).await
    }
    async fn get_chapter(&self, id: i64) -> Result<Value, String> {
        BookStackClient::get_chapter(self, id).await
    }
    async fn create_chapter(
        &self,
        book_id: i64,
        name: &str,
        description: &str,
    ) -> Result<Value, String> {
        BookStackClient::create_chapter(self, book_id, name, description).await
    }
    async fn update_chapter(&self, id: i64, data: &Value) -> Result<Value, String> {
        BookStackClient::update_chapter(self, id, data).await
    }
    async fn delete_chapter(&self, id: i64) -> Result<(), String> {
        BookStackClient::delete_chapter(self, id).await
    }

    async fn list_pages(&self, count: i64, offset: i64) -> Result<Value, String> {
        BookStackClient::list_pages(self, count, offset).await
    }
    async fn list_pages_updated_since(
        &self,
        since_iso_utc: &str,
        count: i64,
    ) -> Result<Vec<Value>, String> {
        BookStackClient::list_pages_updated_since(self, since_iso_utc, count).await
    }
    async fn get_page(&self, id: i64) -> Result<Value, String> {
        BookStackClient::get_page(self, id).await
    }
    async fn create_page(&self, data: &Value) -> Result<Value, String> {
        BookStackClient::create_page(self, data).await
    }
    async fn update_page(&self, id: i64, data: &Value) -> Result<Value, String> {
        BookStackClient::update_page(self, id, data).await
    }
    async fn delete_page(&self, id: i64) -> Result<(), String> {
        BookStackClient::delete_page(self, id).await
    }

    async fn search(&self, query: &str, page: i64, count: i64) -> Result<Value, String> {
        BookStackClient::search(self, query, page, count).await
    }

    /// BookStack's `/api/attachments` lists everything; a `page_id` becomes
    /// its `filter[uploaded_to]` so the wire call without one is unchanged.
    async fn list_attachments(&self, page_id: Option<i64>) -> Result<Value, String> {
        match page_id {
            None => BookStackClient::list_attachments(self).await,
            Some(pid) => BookStackClient::list_attachments_for_page(self, pid).await,
        }
    }
    async fn get_attachment(&self, id: i64) -> Result<Value, String> {
        BookStackClient::get_attachment(self, id).await
    }
    async fn create_attachment(&self, data: &Value) -> Result<Value, String> {
        BookStackClient::create_attachment(self, data).await
    }
    async fn update_attachment(&self, id: i64, data: &Value) -> Result<Value, String> {
        BookStackClient::update_attachment(self, id, data).await
    }
    async fn delete_attachment(&self, id: i64) -> Result<(), String> {
        BookStackClient::delete_attachment(self, id).await
    }
    async fn create_file_attachment(
        &self,
        name: &str,
        uploaded_to: i64,
        filename: &str,
        bytes: Vec<u8>,
        mime_type: &str,
    ) -> Result<Value, String> {
        BookStackClient::create_file_attachment(self, name, uploaded_to, filename, bytes, mime_type)
            .await
    }

    async fn export_page(&self, id: i64, format: ExportFormat) -> Result<String, String> {
        BookStackClient::export_page(self, id, format).await
    }
    async fn export_chapter(&self, id: i64, format: ExportFormat) -> Result<String, String> {
        BookStackClient::export_chapter(self, id, format).await
    }
    async fn export_book(&self, id: i64, format: ExportFormat) -> Result<String, String> {
        BookStackClient::export_book(self, id, format).await
    }

    async fn list_comments(&self, query: &[(&str, &str)]) -> Result<Value, String> {
        BookStackClient::list_comments(self, query).await
    }
    async fn get_comment(&self, id: i64) -> Result<Value, String> {
        BookStackClient::get_comment(self, id).await
    }
    /// BookStack's comments API takes `html`; the dispatcher's extra
    /// `markdown` key is dropped here so the wire payload is unchanged.
    async fn create_comment(&self, data: &Value) -> Result<Value, String> {
        BookStackClient::create_comment(self, &without_markdown(data)).await
    }
    async fn update_comment(&self, id: i64, data: &Value) -> Result<Value, String> {
        BookStackClient::update_comment(self, id, &without_markdown(data)).await
    }
    async fn delete_comment(&self, id: i64) -> Result<(), String> {
        BookStackClient::delete_comment(self, id).await
    }

    async fn list_recycle_bin(&self, count: i64, offset: i64) -> Result<Value, String> {
        BookStackClient::list_recycle_bin(self, count, offset).await
    }
    async fn restore_recycle_bin_item(&self, id: i64) -> Result<Value, String> {
        BookStackClient::restore_recycle_bin_item(self, id).await
    }
    async fn destroy_recycle_bin_item(&self, id: i64) -> Result<(), String> {
        BookStackClient::destroy_recycle_bin_item(self, id).await
    }

    async fn list_users(&self, count: i64, offset: i64) -> Result<Value, String> {
        BookStackClient::list_users(self, count, offset).await
    }
    async fn get_user(&self, id: i64) -> Result<Value, String> {
        BookStackClient::get_user(self, id).await
    }

    async fn list_audit_log(&self, count: i64, offset: i64) -> Result<Value, String> {
        BookStackClient::list_audit_log(self, count, offset).await
    }
    async fn get_system_info(&self) -> Result<Value, String> {
        BookStackClient::get_system_info(self).await
    }

    async fn list_images(
        &self,
        count: i64,
        offset: i64,
        filter: &[(&str, &str)],
    ) -> Result<Value, String> {
        BookStackClient::list_images(self, count, offset, filter).await
    }
    async fn get_image(&self, id: i64) -> Result<Value, String> {
        BookStackClient::get_image(self, id).await
    }
    async fn update_image(&self, id: i64, data: &Value) -> Result<Value, String> {
        BookStackClient::update_image(self, id, data).await
    }
    async fn delete_image(&self, id: i64) -> Result<(), String> {
        BookStackClient::delete_image(self, id).await
    }
    async fn upload_image(
        &self,
        name: &str,
        image_type: &str,
        uploaded_to: i64,
        filename: &str,
        bytes: Vec<u8>,
        mime_type: &str,
    ) -> Result<Value, String> {
        BookStackClient::upload_image(
            self,
            name,
            image_type,
            uploaded_to,
            filename,
            bytes,
            mime_type,
        )
        .await
    }

    async fn get_content_permissions(
        &self,
        content_type: ContentType,
        content_id: i64,
    ) -> Result<Value, String> {
        BookStackClient::get_content_permissions(self, content_type, content_id).await
    }
    async fn update_content_permissions(
        &self,
        content_type: ContentType,
        content_id: i64,
        data: &Value,
    ) -> Result<Value, String> {
        BookStackClient::update_content_permissions(self, content_type, content_id, data).await
    }
    async fn list_roles(&self, count: i64, offset: i64) -> Result<Value, String> {
        BookStackClient::list_roles(self, count, offset).await
    }
    async fn get_role(&self, id: i64) -> Result<Value, String> {
        BookStackClient::get_role(self, id).await
    }
}

/// Drop the dispatcher's `markdown` convenience key from a comment payload.
fn without_markdown(data: &Value) -> Value {
    let mut out = data.clone();
    if let Some(obj) = out.as_object_mut() {
        obj.remove("markdown");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn backend_kind_defaults_to_bookstack() {
        assert_eq!(BackendKind::parse(""), BackendKind::BookStack);
        assert_eq!(BackendKind::parse("bookstack"), BackendKind::BookStack);
        assert_eq!(BackendKind::parse("nonsense"), BackendKind::BookStack);
        assert_eq!(BackendKind::parse("libstack"), BackendKind::LibStack);
        assert_eq!(BackendKind::parse(" LibStack "), BackendKind::LibStack);
    }

    #[test]
    fn not_available_names_the_backend_and_feature() {
        assert_eq!(
            not_available(BackendKind::LibStack, "content permissions"),
            "content permissions is not available on LibStack"
        );
    }

    #[test]
    fn config_public_url_defaults_to_api_url() {
        let c = BackendConfig::new(BackendKind::LibStack, "http://libstack/", None);
        assert_eq!(c.api_url, "http://libstack");
        assert_eq!(c.public_url, "http://libstack");
        let c = BackendConfig::new(
            BackendKind::LibStack,
            "http://libstack/",
            Some("https://libstack.example.com/"),
        );
        assert_eq!(c.public_url, "https://libstack.example.com");
    }

    #[test]
    fn config_mints_the_selected_backend() {
        let http = reqwest::Client::new();
        let bs = BackendConfig::new(BackendKind::BookStack, "http://bookstack", None).client(
            "id",
            "secret",
            http.clone(),
        );
        assert_eq!(bs.kind(), BackendKind::BookStack);
        let ls = BackendConfig::new(BackendKind::LibStack, "http://libstack", None)
            .client("token", "", http);
        assert_eq!(ls.kind(), BackendKind::LibStack);
        assert_eq!(ls.public_url(), "http://libstack");
    }

    #[test]
    fn comment_payload_sent_to_bookstack_has_no_markdown_key() {
        let data = json!({ "page_id": 1, "html": "<p>x</p>", "markdown": "x" });
        let out = without_markdown(&data);
        assert_eq!(out, json!({ "page_id": 1, "html": "<p>x</p>" }));
    }
}
