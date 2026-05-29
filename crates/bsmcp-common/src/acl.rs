//! Per-page ACL resolution.
//!
//! Walks BookStack's content-permissions inheritance chain
//! (page → chapter → book → role-level defaults) and produces a `PageAcl`
//! describing exactly which roles can view the page. Used by:
//!   - The embedder, after each successful page embed → upsert into `page_view_acl`.
//!   - The server's webhook handler, on `*_update` / role events → recompute.
//!   - The daily reconciliation job, as a safety net for missed events.
//!
//! Why this matters: the cold-cache permission filter in semantic search makes
//! one HTTP call per candidate page. Pre-resolving role-level visibility at
//! embed time drops candidates the user can't view *before* the HTTP fan-out,
//! so cold-cache search latency goes from O(candidates) HTTP calls to ~zero
//! for users whose roles don't match restricted content.
//!
//! For the all-inheriting case (no overrides anywhere in the chain) we set
//! `default_open=true` and leave `view_roles` empty — the consumer treats
//! these as "candidate visible to anyone with system view permission" and
//! still runs the HTTP fallback to handle role-level system perms.
//!
//! Per-user content_permissions overrides aren't exposed by the BookStack
//! public API (only role_permissions + fallback_permissions), so they're
//! not modelled here. The HTTP fallback in `semantic.rs` catches them.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::bookstack::{BookStackClient, ContentType};
use crate::db::SemanticDb;
use crate::types::PageAcl;

/// Cached per-pipeline role state. Built once per pipeline run / webhook
/// handling pass to avoid hammering `/api/roles` for every page.
#[derive(Clone, Debug, Default)]
pub struct RoleContext {
    /// Every role ID known to BookStack.
    pub all_role_ids: Vec<i64>,
    /// Role IDs that hold the system-level `content-export` or `page-view-all`
    /// style permission — i.e. the roles that can view content inherited from
    /// instance defaults. When a page falls all the way through inheritance
    /// without overrides, these are the roles considered to have access.
    pub view_all_role_ids: Vec<i64>,
}

/// Build the role context by fetching `/api/roles` and inspecting each role's
/// `permissions` list. Cheap (~10 roles), safe to call once per pipeline run.
pub async fn build_role_context(client: &BookStackClient) -> Result<RoleContext, String> {
    let mut all_role_ids = Vec::new();
    let mut view_all_role_ids = Vec::new();

    let mut offset = 0i64;
    loop {
        let resp = client.list_roles(100, offset).await?;
        let data = resp
            .get("data")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        if data.is_empty() {
            break;
        }
        for role in &data {
            let id = match role.get("id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => continue,
            };
            all_role_ids.push(id);
        }
        let total = resp.get("total").and_then(|v| v.as_i64()).unwrap_or(0);
        offset += 100;
        if offset >= total {
            break;
        }
    }

    // Per-role detail fetch — needed because `list_roles` doesn't include the
    // `permissions` array. Small N (typically ≤ 10).
    for role_id in &all_role_ids {
        let detail = match client.get_role(*role_id).await {
            Ok(d) => d,
            Err(e) => {
                eprintln!("ACL: failed to fetch role {role_id} detail (skipping): {e}");
                continue;
            }
        };
        if has_view_all_permission(&detail) {
            view_all_role_ids.push(*role_id);
        }
    }

    eprintln!(
        "ACL: built role context — {} total roles, {} have system-level page view",
        all_role_ids.len(),
        view_all_role_ids.len()
    );
    Ok(RoleContext {
        all_role_ids,
        view_all_role_ids,
    })
}

/// True when the role detail JSON includes any of BookStack's system-level
/// "view all content" permissions. Conservative: include `content-export`
/// because export-capable roles by definition see everything.
fn has_view_all_permission(role: &Value) -> bool {
    let perms = match role.get("permissions").and_then(|v| v.as_array()) {
        Some(p) => p,
        None => return false,
    };
    for perm in perms {
        if let Some(name) = perm.as_str() {
            if matches!(
                name,
                "page-view-all" | "chapter-view-all" | "book-view-all" | "content-export"
            ) {
                return true;
            }
        }
    }
    false
}

/// Resolve effective view roles for a page by walking up the inheritance
/// chain (page → chapter → book) until a non-inheriting permission level is
/// found. Falls back to `RoleContext::view_all_role_ids` when nothing in the
/// chain has overrides — these pages are flagged `default_open=true` so the
/// query path can short-circuit role checks for them.
pub async fn resolve_page_acl(
    client: &BookStackClient,
    page_id: i64,
    chapter_id: Option<i64>,
    book_id: i64,
    role_ctx: &RoleContext,
) -> Result<PageAcl, String> {
    let now = now_secs();

    // Page-level override?
    let page_perms = client
        .get_content_permissions(ContentType::Page, page_id)
        .await
        .ok();
    if let Some(p) = page_perms.as_ref() {
        if !is_inheriting(p) {
            return Ok(PageAcl {
                page_id,
                view_roles: compute_effective_view(p, &role_ctx.all_role_ids),
                default_open: false,
                computed_at: now,
            });
        }
    }

    // Chapter-level override?
    if let Some(cid) = chapter_id {
        if let Ok(cp) = client
            .get_content_permissions(ContentType::Chapter, cid)
            .await
        {
            if !is_inheriting(&cp) {
                return Ok(PageAcl {
                    page_id,
                    view_roles: compute_effective_view(&cp, &role_ctx.all_role_ids),
                    default_open: false,
                    computed_at: now,
                });
            }
        }
    }

    // Book-level override?
    if let Ok(bp) = client
        .get_content_permissions(ContentType::Book, book_id)
        .await
    {
        if !is_inheriting(&bp) {
            return Ok(PageAcl {
                page_id,
                view_roles: compute_effective_view(&bp, &role_ctx.all_role_ids),
                default_open: false,
                computed_at: now,
            });
        }
    }

    // All inheriting — page is visible to roles with system-level view perm.
    Ok(PageAcl {
        page_id,
        view_roles: role_ctx.view_all_role_ids.clone(),
        default_open: true,
        computed_at: now,
    })
}

fn is_inheriting(perms: &Value) -> bool {
    perms
        .get("fallback_permissions")
        .and_then(|f| f.get("inheriting"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
}

/// Iterate every known role; for each, decide view access based on the
/// content_permissions response. Roles with explicit entries use that entry's
/// `view` flag; roles without entries fall back to `fallback_permissions.view`.
fn compute_effective_view(perms: &Value, all_role_ids: &[i64]) -> Vec<i64> {
    let role_perms = perms
        .get("role_permissions")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let fallback_view = perms
        .get("fallback_permissions")
        .and_then(|f| f.get("view"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mut explicit_view: std::collections::HashMap<i64, bool> = std::collections::HashMap::new();
    for rp in &role_perms {
        let role_id = match rp.get("role_id").and_then(|v| v.as_i64()) {
            Some(id) => id,
            None => continue,
        };
        let view = rp.get("view").and_then(|v| v.as_bool()).unwrap_or(false);
        explicit_view.insert(role_id, view);
    }

    let mut out = Vec::new();
    for &rid in all_role_ids {
        let allowed = match explicit_view.get(&rid) {
            Some(&v) => v,
            None => fallback_view,
        };
        if allowed {
            out.push(rid);
        }
    }
    out
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Recompute ACL for every page in the embedding store. Called by the daily
/// reconciliation job and by the webhook handler on role_* / bookshelf_update
/// events that may have changed effective visibility for a broad set of pages.
///
/// Returns `(processed, failed)`. Best-effort: per-page failures are logged
/// but don't abort the run.
pub async fn reconcile_all_pages(
    client: &BookStackClient,
    db: &Arc<dyn SemanticDb>,
) -> Result<(usize, usize), String> {
    let role_ctx = build_role_context(client).await?;
    let page_ids = db.list_acl_page_ids().await?;
    let mut processed = 0usize;
    let mut failed = 0usize;
    for page_id in page_ids {
        let meta = match db.get_page_meta(page_id).await {
            Ok(Some(m)) => m,
            Ok(None) => continue,
            Err(e) => {
                eprintln!("ACL reconcile: get_page_meta({page_id}) failed: {e}");
                failed += 1;
                continue;
            }
        };
        match resolve_page_acl(client, page_id, meta.chapter_id, meta.book_id, &role_ctx).await {
            Ok(acl) => {
                if let Err(e) = db.upsert_page_acl(&acl).await {
                    eprintln!("ACL reconcile: upsert page {page_id} failed: {e}");
                    failed += 1;
                } else {
                    processed += 1;
                }
            }
            Err(e) => {
                eprintln!("ACL reconcile: resolve page {page_id} failed: {e}");
                failed += 1;
            }
        }
    }
    Ok((processed, failed))
}

/// Recompute ACL for a single page. Called by the webhook handler on
/// page/chapter/book content_permissions changes.
pub async fn reconcile_page(
    client: &BookStackClient,
    db: &Arc<dyn SemanticDb>,
    page_id: i64,
    role_ctx: &RoleContext,
) -> Result<(), String> {
    let meta = match db.get_page_meta(page_id).await? {
        Some(m) => m,
        None => return Ok(()),
    };
    let acl = resolve_page_acl(client, page_id, meta.chapter_id, meta.book_id, role_ctx).await?;
    db.upsert_page_acl(&acl).await
}
