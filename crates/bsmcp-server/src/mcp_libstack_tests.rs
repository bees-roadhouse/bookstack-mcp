//! The dispatcher through `LibStackClient` against a small axum fake of the
//! LibStack API (issue #156). Proves the mapping end to end for the calls
//! every connector makes first: `list_shelves`, `get_book`, `get_page`,
//! `create_page`, `search_content` — plus the not-available error and the
//! three collection tools.

use super::*;

use std::net::SocketAddr;
use std::sync::Arc as StdArc;
use std::sync::Mutex as StdMutex;

use axum::extract::{Path as AxPath, Query as AxQuery, State as AxState};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json as AxJson, Router};
use bsmcp_common::backend::{Backend, BackendConfig, BackendKind};
use bsmcp_common::db::IndexDb;
use bsmcp_common::libstack::ids;
use bsmcp_db_sqlite::SqliteDb;
use uuid::Uuid;

const HOUSEHOLD: &str = "11111111-1111-4111-8111-111111111111";
const INFRA: &str = "22222222-2222-4222-8222-222222222222";
const OPS: &str = "33333333-3333-4333-8333-333333333333";
const WORK: &str = "44444444-4444-4444-8444-444444444444";
const PAGE_BACKUPS: &str = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
const PAGE_WELCOME: &str = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
const PAGE_NEW: &str = "cccccccc-cccc-4ccc-8ccc-cccccccccccc";
const USER: &str = "dddddddd-dddd-4ddd-8ddd-dddddddddddd";

#[derive(Clone, Default)]
struct Fake {
    created: StdArc<StdMutex<Vec<Value>>>,
}

fn coll(id: &str, parent: Option<&str>, name: &str) -> Value {
    json!({
        "id": id, "org_id": USER, "name": name, "description_md": format!("About {name}."),
        "parent_id": parent, "is_template": false, "sort_order": 0,
        "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-01T00:00:00Z", "deleted_at": null
    })
}

fn page_meta(id: &str, coll: &str, title: &str, updated: &str) -> Value {
    json!({
        "id": id, "org_id": USER, "collection_id": coll, "title": title, "is_draft": false,
        "sort_order": 0, "revision_count": 3, "created_by": USER, "updated_by": USER,
        "created_at": "2026-10-01T00:00:00Z", "updated_at": updated, "type": null, "fields": {}
    })
}

fn page_full(id: &str, coll: &str, title: &str, markdown: &str) -> Value {
    let mut v = page_meta(id, coll, title, "2026-10-01T01:00:00Z");
    v["markdown"] = json!(markdown);
    v["rendered_html"] = json!(format!("<p>{markdown}</p>"));
    v["tags"] = json!([{ "name": "kind", "value": "runbook" }]);
    v
}

fn envelope(code: u16, message: &str) -> axum::response::Response {
    (
        StatusCode::from_u16(code).unwrap(),
        AxJson(json!({ "error": { "code": code, "message": message } })),
    )
        .into_response()
}

async fn me(headers: axum::http::HeaderMap) -> axum::response::Response {
    if headers.get("authorization").and_then(|v| v.to_str().ok()) != Some("Bearer lst-test-token") {
        return envelope(401, "unauthenticated");
    }
    AxJson(json!({
        "user": { "id": USER, "email": "nate@example.com", "display_name": "Nate", "kind": "member", "must_change_password": false },
        "active_org": { "id": USER, "name": "Bee's Roadhouse", "type": "household", "role_id": 1, "current": true },
        "orgs": [], "capabilities": ["content.view", "content.edit", "org.manage_members"], "csrf_token": "x"
    }))
    .into_response()
}

async fn collections() -> AxJson<Value> {
    AxJson(json!([
        coll(HOUSEHOLD, None, "Household"),
        coll(INFRA, Some(HOUSEHOLD), "Infrastructure"),
        coll(OPS, Some(INFRA), "Ops"),
        coll(WORK, None, "Work"),
    ]))
}

#[derive(serde::Deserialize, Default)]
struct ListQ {
    collection_id: Option<String>,
}

async fn pages_list(AxQuery(q): AxQuery<ListQ>) -> AxJson<Value> {
    let all = vec![
        page_meta(PAGE_BACKUPS, OPS, "Backups", "2026-10-01T01:00:00Z"),
        page_meta(PAGE_WELCOME, HOUSEHOLD, "Welcome", "2026-09-30T01:00:00Z"),
    ];
    let data: Vec<Value> = all
        .into_iter()
        .filter(|p| match &q.collection_id {
            Some(c) => p["collection_id"] == json!(c),
            None => true,
        })
        .collect();
    AxJson(json!({ "total": data.len(), "data": data }))
}

async fn page_get(AxPath(id): AxPath<String>) -> axum::response::Response {
    match id.as_str() {
        PAGE_BACKUPS => AxJson(page_full(
            PAGE_BACKUPS,
            OPS,
            "Backups",
            "Nightly to Garage.",
        ))
        .into_response(),
        PAGE_WELCOME => {
            AxJson(page_full(PAGE_WELCOME, HOUSEHOLD, "Welcome", "Hi.")).into_response()
        }
        _ => envelope(404, "not found"),
    }
}

async fn page_create(
    AxState(fake): AxState<Fake>,
    AxJson(body): AxJson<Value>,
) -> axum::response::Response {
    if body
        .get("title")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .is_empty()
    {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            AxJson(json!({ "error": { "code": 422, "message": "validation failed", "validation": { "title": ["required"] } } })),
        )
            .into_response();
    }
    fake.created.lock().unwrap().push(body.clone());
    let coll = body["collection_id"].as_str().unwrap_or("");
    let page = page_full(
        PAGE_NEW,
        coll,
        body["title"].as_str().unwrap_or(""),
        body["markdown"].as_str().unwrap_or(""),
    );
    (StatusCode::CREATED, AxJson(page)).into_response()
}

#[derive(serde::Deserialize, Default)]
struct SearchQ {
    q: String,
}

async fn search(AxQuery(q): AxQuery<SearchQ>) -> AxJson<Value> {
    if q.q.contains("backup") {
        AxJson(json!({ "total": 2, "data": [
            { "id": PAGE_BACKUPS, "type": "page", "title": "Backups", "excerpt": "Nightly to <b>Garage</b>.", "collection_id": OPS, "rank": 0.9 },
            { "id": INFRA, "type": "collection", "title": "Infrastructure", "excerpt": "About Infrastructure.", "collection_id": INFRA, "rank": 0.2 },
        ]}))
    } else {
        AxJson(json!({ "total": 0, "data": [] }))
    }
}

async fn spawn_fake() -> (String, Fake) {
    let fake = Fake::default();
    let app = Router::new()
        .route("/api/auth/me", get(me))
        .route("/api/collections", get(collections))
        .route("/api/pages", get(pages_list).post(page_create))
        .route("/api/pages/{id}", get(page_get))
        .route("/api/search", get(search))
        .with_state(fake.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    (format!("http://{addr}"), fake)
}

struct Harness {
    client: Arc<dyn Backend>,
    index_db: Arc<dyn IndexDb>,
    staging: crate::staging::StagingStore,
    fake: Fake,
}

async fn harness() -> Harness {
    let (base, fake) = spawn_fake().await;
    let cfg = BackendConfig::new(
        BackendKind::LibStack,
        &base,
        Some("https://libstack.example.com"),
    );
    let client = cfg.client("lst-test-token", "", reqwest::Client::new());
    let path = std::env::temp_dir().join(format!(
        "bsmcp-libstack-{}-{}.db",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let sqlite = Arc::new(SqliteDb::open(
        &path,
        "test-encryption-key-thirty-two-chars-long",
    ));
    Harness {
        client,
        index_db: sqlite,
        staging: crate::staging::new_staging_store(),
        fake,
    }
}

impl Harness {
    async fn call(&self, tool: &str, args: Value) -> Result<String, String> {
        execute_tool(
            tool,
            &args,
            &self.client,
            None,
            self.index_db.as_ref(),
            &self.staging,
        )
        .await
    }

    async fn call_json(&self, tool: &str, args: Value) -> Value {
        let text = self.call(tool, args).await.unwrap();
        serde_json::from_str(&text).unwrap_or_else(|_| panic!("{tool} did not return JSON: {text}"))
    }
}

fn int(id: &str) -> i64 {
    ids::to_i64(&Uuid::parse_str(id).unwrap())
}

#[tokio::test]
async fn libstack_credentials_validate_through_auth_me() {
    let h = harness().await;
    assert!(matches!(
        h.client.validate().await,
        bsmcp_common::bookstack::CredentialCheck::Valid
    ));
    assert!(h.client.is_admin().await.unwrap());
    let bad = BackendConfig::new(BackendKind::LibStack, h.client.base_url(), None).client(
        "wrong",
        "",
        reqwest::Client::new(),
    );
    assert!(matches!(
        bad.validate().await,
        bsmcp_common::bookstack::CredentialCheck::Rejected(_)
    ));
}

#[tokio::test]
async fn list_shelves_are_the_root_collections() {
    let h = harness().await;
    let v = h.call_json("list_shelves", json!({})).await;
    assert_eq!(v["total"], 2);
    let names: Vec<&str> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["Household", "Work"]);
    assert_eq!(v["data"][0]["id"], int(HOUSEHOLD));
    assert_eq!(
        v["data"][0]["url"],
        "/c/11111111-1111-4111-8111-111111111111"
    );
    // The depth-1 collection is a book, reachable from its shelf.
    let shelf = h
        .call_json("get_shelf", json!({ "shelf_id": int(HOUSEHOLD) }))
        .await;
    assert_eq!(shelf["books"][0]["name"], "Infrastructure");
    assert_eq!(shelf["books"][0]["id"], int(INFRA));
}

#[tokio::test]
async fn get_book_lists_descendant_chapters_and_their_pages() {
    let h = harness().await;
    let book = h
        .call_json("get_book", json!({ "book_id": int(INFRA) }))
        .await;
    assert_eq!(book["name"], "Infrastructure");
    assert_eq!(book["shelf_id"], int(HOUSEHOLD));
    let contents = book["contents"].as_array().unwrap();
    assert_eq!(contents.len(), 1, "one chapter (Ops), no loose pages");
    assert_eq!(contents[0]["type"], "chapter");
    assert_eq!(contents[0]["id"], int(OPS));
    assert_eq!(contents[0]["book_id"], int(INFRA));
    assert_eq!(contents[0]["pages"][0]["name"], "Backups");
    assert_eq!(contents[0]["pages"][0]["id"], int(PAGE_BACKUPS));
    // A page directly in a root collection reports that root as its book.
    let root = h
        .call_json("get_book", json!({ "book_id": int(HOUSEHOLD) }))
        .await;
    let loose: Vec<&Value> = root["contents"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["type"] == "page")
        .collect();
    assert_eq!(loose.len(), 1);
    assert_eq!(loose[0]["name"], "Welcome");
}

#[tokio::test]
async fn get_page_carries_markdown_rendered_html_and_parents() {
    let h = harness().await;
    // The id must be known: list first, as a connector would.
    h.call_json("list_pages", json!({})).await;
    let page = h
        .call_json("get_page", json!({ "page_id": int(PAGE_BACKUPS) }))
        .await;
    assert_eq!(page["name"], "Backups");
    assert_eq!(page["editor"], "markdown");
    assert_eq!(page["markdown"], "Nightly to Garage.");
    assert_eq!(page["html"], "<p>Nightly to Garage.</p>");
    assert_eq!(page["book_id"], int(INFRA));
    assert_eq!(page["chapter_id"], int(OPS));
    assert_eq!(page["chapter"]["name"], "Ops");
    assert_eq!(page["url"], format!("/p/{PAGE_BACKUPS}"));
    assert_eq!(page["tags"][0]["name"], "kind");
    assert_eq!(page["revision_count"], 3);
}

#[tokio::test]
async fn create_page_posts_collection_id_title_and_markdown() {
    let h = harness().await;
    h.call_json("list_chapters", json!({})).await;
    let text = h
        .call(
            "create_page",
            json!({ "name": "Restore drill", "chapter_id": int(OPS), "markdown": "# Restore drill\n\nSteps." }),
        )
        .await
        .unwrap();
    assert!(text.starts_with("Page created successfully."), "{text}");
    assert!(
        text.contains(&format!("Page ID: {}", int(PAGE_NEW))),
        "{text}"
    );
    assert!(text.contains("https://libstack.example.com/p/"), "{text}");
    let sent = h.fake.created.lock().unwrap().clone();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0]["collection_id"], OPS);
    assert_eq!(sent[0]["title"], "Restore drill");
    // The duplicate H1 is stripped before the body reaches LibStack.
    assert_eq!(sent[0]["markdown"], "Steps.");
    // `html` bodies are refused, never silently converted.
    let err = h
        .call(
            "create_page",
            json!({ "name": "X", "chapter_id": int(OPS), "html": "<p>x</p>" }),
        )
        .await
        .unwrap_err();
    assert!(err.contains("not available on LibStack"), "{err}");
}

#[tokio::test]
async fn validation_errors_reach_the_caller_with_the_field() {
    let h = harness().await;
    h.call_json("list_chapters", json!({})).await;
    let err = h
        .call(
            "create_page",
            json!({ "name": "", "chapter_id": int(OPS), "markdown": "x" }),
        )
        .await
        .unwrap_err();
    assert_eq!(
        err,
        "LibStack API error: 422 validation failed — title: required"
    );
}

#[tokio::test]
async fn search_content_maps_hits_onto_bookstack_rows() {
    let h = harness().await;
    let text = h
        .call("search_content", json!({ "query": "{type:page} backup" }))
        .await
        .unwrap();
    assert!(text.starts_with("Found 2 results:"), "{text}");
    assert!(
        text.contains(&format!(
            "- [page] Backups (id: {}) — https://libstack.example.com/p/{PAGE_BACKUPS}",
            int(PAGE_BACKUPS)
        )),
        "{text}"
    );
    assert!(text.contains("Preview: Nightly to Garage."), "{text}");
    // A depth-1 collection hit is a book.
    assert!(
        text.contains(&format!("- [book] Infrastructure (id: {})", int(INFRA))),
        "{text}"
    );
    let none = h
        .call("search_content", json!({ "query": "nothing-here" }))
        .await
        .unwrap();
    assert_eq!(none, "No results found.");
}

#[tokio::test]
async fn unsupported_tools_say_so_instead_of_faking_success() {
    let h = harness().await;
    for (tool, args) in [
        (
            "get_content_permissions",
            json!({ "content_type": "page", "content_id": 1 }),
        ),
        ("list_roles", json!({})),
        ("get_role", json!({ "role_id": 1 })),
        ("list_recycle_bin", json!({})),
        ("restore_recycle_bin_item", json!({ "deletion_id": 1 })),
        ("destroy_recycle_bin_item", json!({ "deletion_id": 1 })),
        ("export_page", json!({ "page_id": 1 })),
        ("export_chapter", json!({ "chapter_id": 1 })),
        ("export_book", json!({ "book_id": 1 })),
    ] {
        let err = h.call(tool, args).await.unwrap_err();
        assert!(err.contains("not available on LibStack"), "{tool}: {err}");
    }
    let err = h
        .call(
            "update_content_permissions",
            json!({ "content_type": "page", "content_id": 1, "fallback_permissions": {} }),
        )
        .await
        .unwrap_err();
    assert!(err.contains("not available on LibStack"), "{err}");
}

#[tokio::test]
async fn collection_tools_expose_the_real_tree() {
    let h = harness().await;
    let all = h.call_json("list_collections", json!({})).await;
    assert_eq!(all["total"], 4);
    let ops = all["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "Ops")
        .unwrap();
    assert_eq!(ops["depth"], 2);
    assert_eq!(ops["appears_as"], "chapter");
    assert_eq!(ops["parent_id"], int(INFRA));
    assert_eq!(ops["path"], "Household / Infrastructure / Ops");
    let one = h
        .call_json("get_collection", json!({ "collection_id": int(OPS) }))
        .await;
    assert_eq!(one["pages"][0]["name"], "Backups");
}

#[tokio::test]
async fn delta_listing_cuts_client_side_and_sorts_oldest_first() {
    let h = harness().await;
    let rows = h
        .client
        .list_pages_updated_since("2026-09-30T00:00:00Z", 10)
        .await
        .unwrap();
    let names: Vec<&str> = rows.iter().map(|p| p["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["Welcome", "Backups"]);
    let rows = h
        .client
        .list_pages_updated_since("2026-10-01T00:30:00Z", 10)
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["name"], "Backups");
    assert_eq!(rows[0]["book_id"], int(INFRA));
}

#[tokio::test]
async fn whoami_and_users_are_the_caller() {
    let h = harness().await;
    let me = h.client.whoami().await.unwrap().unwrap();
    assert_eq!(me.email.as_deref(), Some("nate@example.com"));
    assert_eq!(me.bookstack_user_id, int(USER));
    let users = h.call_json("list_users", json!({})).await;
    assert_eq!(users["total"], 1);
    assert_eq!(users["data"][0]["name"], "Nate");
    let err = h
        .call("get_user", json!({ "user_id": 42 }))
        .await
        .unwrap_err();
    assert!(err.contains("only the calling user"), "{err}");
}
