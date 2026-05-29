//! `bsmcp-worker` — reconciliation worker for the v1.0.0 index.
//!
//! Background process that owns the `index_jobs` queue: walks every
//! configured shelf on first startup, polls for webhook-enqueued and
//! cron-enqueued jobs, and runs the periodic delta walk. Shares the same
//! database as `bsmcp-server` so the server's webhook handler can enqueue
//! jobs that this binary picks up.
//!
//! Why a separate binary: keeps `bsmcp-server` focused on serving MCP/HTTP
//! requests and lets the worker scale, restart, or run on a different
//! schedule than the API surface. The worker shipped inline in
//! `bsmcp-server` through Phase 4-7; this binary is the v1.1.0 split.
//!
//! Required env:
//!   BSMCP_BOOKSTACK_URL          — same as server
//!   BSMCP_ENCRYPTION_KEY         — same as server (32+ chars; needed
//!                                   even though the worker doesn't hold
//!                                   user tokens, because the DB layer
//!                                   initializes its encryption context
//!                                   regardless of read/write usage)
//!   BSMCP_INDEX_TOKEN_ID/SECRET  — admin BookStack API token. Falls back
//!                                   to BSMCP_EMBED_TOKEN_* if unset.
//!
//! Optional env:
//!   BSMCP_DB_BACKEND             — sqlite (default) | postgres
//!   BSMCP_DB_PATH                — SQLite path (default /data/bookstack-mcp.db)
//!   BSMCP_DATABASE_URL           — Postgres connection (required if backend=postgres)
//!   BSMCP_INDEX_DELTA_INTERVAL_SECONDS — default 300

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use bsmcp_common::config::DbBackendType;
use bsmcp_common::db::{DbBackend, IndexDb, SemanticDb};

#[tokio::main]
async fn main() {
    bsmcp_common::logging::init("bsmcp-worker");
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "worker_starting");

    let bookstack_url = env::var("BSMCP_BOOKSTACK_URL").expect("BSMCP_BOOKSTACK_URL is required");

    let encryption_key = env::var("BSMCP_ENCRYPTION_KEY")
        .expect("BSMCP_ENCRYPTION_KEY is required (32+ character key, must match the server's)");
    if encryption_key.len() < 32 {
        panic!("BSMCP_ENCRYPTION_KEY must be at least 32 characters");
    }

    // BookStack admin token — try BSMCP_INDEX_TOKEN_* first, fall back to
    // BSMCP_EMBED_TOKEN_*. The two-token-name pattern matches what the
    // server used pre-split so existing deployments work without renaming.
    let token_id = env::var("BSMCP_INDEX_TOKEN_ID")
        .or_else(|_| env::var("BSMCP_EMBED_TOKEN_ID"))
        .expect(
            "BSMCP_INDEX_TOKEN_ID (or BSMCP_EMBED_TOKEN_ID) is required — admin BookStack API token id",
        );
    let token_secret = env::var("BSMCP_INDEX_TOKEN_SECRET")
        .or_else(|_| env::var("BSMCP_EMBED_TOKEN_SECRET"))
        .expect(
            "BSMCP_INDEX_TOKEN_SECRET (or BSMCP_EMBED_TOKEN_SECRET) is required — admin BookStack API token secret",
        );

    // Select database backend — must point at the SAME database the server
    // uses so the index_jobs queue is shared.
    let backend_type = DbBackendType::from_env();
    let (db, index_db, semantic_db): (Arc<dyn DbBackend>, Arc<dyn IndexDb>, Arc<dyn SemanticDb>) =
        match backend_type {
            DbBackendType::Sqlite => {
                let db_path = env::var("BSMCP_DB_PATH")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("/data/bookstack-mcp.db"));
                tracing::info!(backend = "sqlite", path = %db_path.display(), "database_selected");
                let db = Arc::new(bsmcp_db_sqlite::SqliteDb::open(&db_path, &encryption_key));
                (
                    db.clone() as Arc<dyn DbBackend>,
                    db.clone() as Arc<dyn IndexDb>,
                    db as Arc<dyn SemanticDb>,
                )
            }
            DbBackendType::Postgres => {
                let database_url = env::var("BSMCP_DATABASE_URL")
                    .expect("BSMCP_DATABASE_URL is required when BSMCP_DB_BACKEND=postgres");
                tracing::info!(backend = "postgres", "database_selected");
                let db = Arc::new(
                    bsmcp_db_postgres::PostgresDb::new(&database_url, &encryption_key)
                        .await
                        .expect("Failed to connect to PostgreSQL"),
                );
                (
                    db.clone() as Arc<dyn DbBackend>,
                    db.clone() as Arc<dyn IndexDb>,
                    db as Arc<dyn SemanticDb>,
                )
            }
        };

    let bs_client = bsmcp_common::bookstack::BookStackClient::new(
        &bookstack_url,
        &token_id,
        &token_secret,
        reqwest::Client::new(),
    );

    let delta_interval: u64 = env::var("BSMCP_INDEX_DELTA_INTERVAL_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    tracing::info!(delta_interval_s = delta_interval, "worker_delta_interval");

    let worker = bsmcp_worker::IndexWorker::new(bs_client, db, index_db);
    let handle = worker.spawn(delta_interval, Some(semantic_db));

    // Block on the worker indefinitely. Crashes inside the spawned task
    // surface as a JoinError here so the container exits and the
    // orchestrator can restart us.
    if let Err(e) = handle.await {
        tracing::error!(error = %e, "worker_task_exited_unexpectedly");
        std::process::exit(1);
    }
}
