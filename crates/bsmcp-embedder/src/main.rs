mod embed;
mod pipeline;
mod rerank;

use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::{IntoResponse, Json};
use axum::{routing::get, Router};
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use bsmcp_common::bookstack::BookStackClient;
use bsmcp_common::config::DbBackendType;
use bsmcp_common::db::SemanticDb;

use embed::Embedder;
use rerank::Reranker;

const DEFAULT_LOCAL_MODEL: &str = "BAAI/bge-base-en-v1.5";
const DEFAULT_OLLAMA_MODEL: &str = "nomic-embed-text";
const DEFAULT_OPENAI_MODEL: &str = "text-embedding-3-small";
const DEFAULT_VOYAGE_MODEL: &str = "voyage-3-lite";

const DEFAULT_LOCAL_RERANK_MODEL: &str = "BAAI/bge-reranker-v2-m3";
const DEFAULT_VOYAGE_RERANK_MODEL: &str = "rerank-2";

struct AppState {
    embedder: Arc<dyn Embedder>,
    model_name: String,
    provider_name: String,
    db: Arc<dyn SemanticDb>,
    /// Reranker is optional — `BSMCP_RERANK_PROVIDER=none` (the default) leaves
    /// it unconfigured and `/rerank` returns 503.
    reranker: Option<Arc<dyn Reranker>>,
}

/// Load or generate a persistent worker UUID from a file in the data directory.
fn load_or_create_worker_id(data_dir: &Path) -> String {
    let id_file = data_dir.join("worker_id");
    if let Ok(id) = fs::read_to_string(&id_file) {
        let id = id.trim().to_string();
        if !id.is_empty() {
            return id;
        }
    }
    let id = Uuid::new_v4().to_string();
    fs::create_dir_all(data_dir).ok();
    fs::write(&id_file, &id).ok();
    id
}

#[derive(Deserialize)]
struct EmbedRequest {
    texts: Vec<String>,
}

#[derive(Deserialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    /// Optional cap; server returns at most this many sorted-by-score hits.
    /// Omit to receive all hits in input order (caller does its own cut).
    #[serde(default)]
    top_k: Option<usize>,
}

#[tokio::main]
async fn main() {
    bsmcp_common::logging::init("bsmcp-embedder");
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "embedder_starting");

    let encryption_key =
        env::var("BSMCP_ENCRYPTION_KEY").expect("BSMCP_ENCRYPTION_KEY is required");
    if encryption_key.len() < 32 {
        panic!("BSMCP_ENCRYPTION_KEY must be at least 32 characters");
    }

    // Select database backend
    let backend_type = DbBackendType::from_env();
    let db: Arc<dyn SemanticDb> = match backend_type {
        DbBackendType::Sqlite => {
            let db_path = env::var("BSMCP_DB_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/data/bookstack-mcp.db"));
            if let Some(parent) = db_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            tracing::info!(backend = "sqlite", path = %db_path.display(), "database_selected");
            Arc::new(bsmcp_db_sqlite::SqliteDb::open(&db_path, &encryption_key))
        }
        DbBackendType::Postgres => {
            let database_url = env::var("BSMCP_DATABASE_URL")
                .expect("BSMCP_DATABASE_URL is required when BSMCP_DB_BACKEND=postgres");
            tracing::info!(backend = "postgres", "database_selected");
            Arc::new(
                bsmcp_db_postgres::PostgresDb::new(&database_url, &encryption_key)
                    .await
                    .expect("Failed to connect to PostgreSQL"),
            )
        }
    };

    // Initialize semantic tables
    db.init_semantic_tables()
        .await
        .expect("Failed to initialize semantic tables");

    // Select embedding provider
    let provider = env::var("BSMCP_EMBED_PROVIDER")
        .unwrap_or_else(|_| "local".into())
        .to_lowercase();

    let (embedder, model_name, dims): (Arc<dyn Embedder>, String, usize) = match provider.as_str() {
        "openai" => {
            let api_key = env::var("BSMCP_EMBED_API_KEY")
                .expect("BSMCP_EMBED_API_KEY is required when BSMCP_EMBED_PROVIDER=openai");
            let model =
                env::var("BSMCP_EMBED_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.into());
            let base_url = env::var("BSMCP_EMBED_API_URL")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "https://api.openai.com".into());

            // Auto-detect dimensions unless explicitly set
            let dims: usize =
                if let Some(d) = env::var("BSMCP_EMBED_DIMS").ok().filter(|s| !s.is_empty()) {
                    d.parse().expect("BSMCP_EMBED_DIMS must be a valid number")
                } else {
                    tracing::info!(provider = "openai", model = %model, "embedder_detecting_dims");
                    match embed::OpenAIEmbedder::detect_dims(&api_key, &model, &base_url).await {
                        Ok(d) => {
                            tracing::info!(provider = "openai", dims = d, "embedder_detected_dims");
                            d
                        }
                        Err(e) => {
                            panic!("Embedder: OpenAI dimension detection failed: {e}");
                        }
                    }
                };

            tracing::info!(
                provider = "openai",
                model = %model,
                dims,
                url = %base_url,
                "embedder_provider_configured"
            );
            let e = embed::OpenAIEmbedder::new(&api_key, &model, &base_url, dims);
            (Arc::new(e), model, dims)
        }
        "voyage" => {
            let api_key = env::var("BSMCP_EMBED_API_KEY")
                .expect("BSMCP_EMBED_API_KEY is required when BSMCP_EMBED_PROVIDER=voyage");
            let model =
                env::var("BSMCP_EMBED_MODEL").unwrap_or_else(|_| DEFAULT_VOYAGE_MODEL.into());
            let base_url = env::var("BSMCP_EMBED_API_URL")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "https://api.voyageai.com".into());

            // Auto-detect dimensions unless explicitly set
            let dims: usize =
                if let Some(d) = env::var("BSMCP_EMBED_DIMS").ok().filter(|s| !s.is_empty()) {
                    d.parse().expect("BSMCP_EMBED_DIMS must be a valid number")
                } else {
                    tracing::info!(provider = "voyage", model = %model, "embedder_detecting_dims");
                    match embed::VoyageEmbedder::detect_dims(&api_key, &model, &base_url).await {
                        Ok(d) => {
                            tracing::info!(provider = "voyage", dims = d, "embedder_detected_dims");
                            d
                        }
                        Err(e) => {
                            panic!("Embedder: Voyage dimension detection failed: {e}");
                        }
                    }
                };

            tracing::info!(
                provider = "voyage",
                model = %model,
                dims,
                url = %base_url,
                "embedder_provider_configured"
            );
            let e = embed::VoyageEmbedder::new(&api_key, &model, &base_url, dims);
            (Arc::new(e), model, dims)
        }
        "ollama" => {
            let model =
                env::var("BSMCP_EMBED_MODEL").unwrap_or_else(|_| DEFAULT_OLLAMA_MODEL.into());
            let base_url =
                env::var("BSMCP_EMBED_API_URL").unwrap_or_else(|_| "http://localhost:11434".into());

            // Auto-detect dimensions unless explicitly set
            let dims: usize =
                if let Some(d) = env::var("BSMCP_EMBED_DIMS").ok().filter(|s| !s.is_empty()) {
                    d.parse().expect("BSMCP_EMBED_DIMS must be a valid number")
                } else {
                    tracing::info!(provider = "ollama", model = %model, "embedder_detecting_dims");
                    match embed::OllamaEmbedder::detect_dims(&model, &base_url).await {
                        Ok(d) => {
                            tracing::info!(provider = "ollama", dims = d, "embedder_detected_dims");
                            d
                        }
                        Err(e) => {
                            tracing::warn!(
                                provider = "ollama",
                                error = %e,
                                default = 768,
                                "embedder_dim_detect_failed_fallback"
                            );
                            768
                        }
                    }
                };

            tracing::info!(
                provider = "ollama",
                model = %model,
                dims,
                url = %base_url,
                "embedder_provider_configured"
            );
            let e = embed::OllamaEmbedder::new(&model, &base_url, dims);
            (Arc::new(e), model, dims)
        }
        _ => {
            // Local fastembed/ONNX model
            let model_path = env::var("BSMCP_MODEL_PATH").unwrap_or_else(|_| "/data/models".into());
            let model_name =
                env::var("BSMCP_EMBED_MODEL").unwrap_or_else(|_| DEFAULT_LOCAL_MODEL.into());

            tracing::info!(
                provider = "local",
                model = %model_name,
                cache = %model_path,
                "embedder_loading_local_model"
            );
            let local = pipeline::EmbedModel::new(&model_name, &model_path)
                .expect("Failed to load embedding model");
            let dims = local.dims();
            let local = Arc::new(local);
            tracing::info!(provider = "local", dims, "embedder_local_model_ready");
            let e = embed::LocalEmbedder::new(local);
            (Arc::new(e), model_name, dims)
        }
    };

    tracing::info!(
        provider = %provider,
        model = %model_name,
        dims,
        "embedder_ready"
    );

    // Optional reranker. None when BSMCP_RERANK_PROVIDER is unset, "none", or empty.
    let reranker: Option<Arc<dyn Reranker>> = init_reranker().await;

    // Start HTTP server for /embed endpoint
    let host = env::var("BSMCP_EMBED_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port: u16 = env::var("BSMCP_EMBED_PORT")
        .unwrap_or_else(|_| "8081".into())
        .parse()
        .expect("BSMCP_EMBED_PORT must be a valid port number");

    let state = Arc::new(AppState {
        embedder: embedder.clone(),
        model_name: model_name.clone(),
        provider_name: provider.clone(),
        db: db.clone(),
        reranker: reranker.clone(),
    });

    let app = Router::new()
        .route("/embed", axum::routing::post(handle_embed))
        .route("/rerank", axum::routing::post(handle_rerank))
        .route("/health", get(handle_health))
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}").parse().unwrap();

    // Worker identity — persistent UUID for job ownership
    let model_path = env::var("BSMCP_MODEL_PATH").unwrap_or_else(|_| "/data/models".into());
    let worker_data_dir = PathBuf::from(env::var("BSMCP_EMBED_DATA_DIR").unwrap_or(model_path));
    let worker_id = load_or_create_worker_id(&worker_data_dir);
    tracing::info!(worker_id = %worker_id, "embedder_worker_id");

    // Recover any jobs from a previous crash of this worker
    match db.recover_worker_jobs(&worker_id).await {
        Ok(0) => {}
        Ok(n) => tracing::info!(count = n, "embedder_recovered_jobs"),
        Err(e) => tracing::error!(error = %e, "embedder_recover_jobs_failed"),
    }

    // Spawn job queue worker
    let worker_db = db.clone();
    let worker_embedder = embedder.clone();
    let worker_model_name = model_name;
    let worker_dims = dims;
    tokio::spawn(async move {
        job_queue_worker(
            worker_db,
            worker_embedder,
            worker_id,
            worker_model_name,
            worker_dims,
        )
        .await;
    });

    tracing::info!(addr = %addr, "embedder_http_listening");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn handle_embed(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> impl IntoResponse {
    if req.texts.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({"error": "texts array must not be empty"})),
        )
            .into_response();
    }

    if req.texts.len() > 100 {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({"error": "maximum 100 texts per request"})),
        )
            .into_response();
    }

    match state.embedder.embed(req.texts).await {
        Ok(embeddings) => Json(json!({ "embeddings": embeddings })).into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Embedding failed: {e}")})),
        )
            .into_response(),
    }
}

async fn handle_health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let dims = state.embedder.dims();
    let stats = state.db.get_stats().await.ok();
    let reranker = state.reranker.as_ref().map(|r| {
        json!({
            "provider": r.provider_name(),
            "model": r.model_name(),
        })
    });
    Json(json!({
        "status": "ok",
        "provider": state.provider_name,
        "model": state.model_name,
        "dimensions": dims,
        "reranker": reranker,
        "stats": stats.map(|s| json!({
            "total_pages": s.total_pages,
            "total_chunks": s.total_chunks,
            "latest_job": s.latest_job.map(|j| json!({
                "id": j.id,
                "scope": j.scope,
                "status": j.status,
                "done_pages": j.done_pages,
                "total_pages": j.total_pages,
            })),
        })),
    }))
}

async fn handle_rerank(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RerankRequest>,
) -> impl IntoResponse {
    let Some(reranker) = state.reranker.as_ref() else {
        return (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "Reranker disabled (BSMCP_RERANK_PROVIDER unset or 'none')"})),
        )
            .into_response();
    };

    if req.query.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({"error": "query must not be empty"})),
        )
            .into_response();
    }
    if req.documents.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({"error": "documents array must not be empty"})),
        )
            .into_response();
    }
    if req.documents.len() > 200 {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({"error": "maximum 200 documents per rerank request"})),
        )
            .into_response();
    }

    match reranker.rerank(req.query, req.documents).await {
        Ok(mut hits) => {
            // Sort descending by score. Stable so ties keep input order.
            hits.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(k) = req.top_k {
                hits.truncate(k);
            }
            Json(json!({
                "results": hits,
                "provider": reranker.provider_name(),
                "model": reranker.model_name(),
            }))
            .into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Rerank failed: {e}")})),
        )
            .into_response(),
    }
}

/// Build a `Reranker` from `BSMCP_RERANK_*` env vars. Returns `None` when
/// the provider is unset, "none", or empty — `/rerank` then returns 503.
async fn init_reranker() -> Option<Arc<dyn Reranker>> {
    let provider = env::var("BSMCP_RERANK_PROVIDER")
        .unwrap_or_default()
        .to_lowercase();
    if provider.is_empty() || provider == "none" {
        tracing::info!("reranker_disabled");
        return None;
    }

    match provider.as_str() {
        "local" => {
            let model_path = env::var("BSMCP_MODEL_PATH").unwrap_or_else(|_| "/data/models".into());
            let model_name = env::var("BSMCP_RERANK_MODEL")
                .unwrap_or_else(|_| DEFAULT_LOCAL_RERANK_MODEL.into());
            tracing::info!(
                provider = "local",
                model = %model_name,
                cache = %model_path,
                "reranker_loading_local"
            );
            match pipeline::RerankModel::new(&model_name, &model_path) {
                Ok(m) => {
                    tracing::info!(provider = "local", "reranker_ready");
                    Some(Arc::new(rerank::LocalReranker::new(Arc::new(m))) as Arc<dyn Reranker>)
                }
                Err(e) => {
                    tracing::error!(
                        provider = "local",
                        error = %e,
                        "reranker_init_failed"
                    );
                    None
                }
            }
        }
        "voyage" => {
            let api_key = match env::var("BSMCP_RERANK_API_KEY") {
                Ok(k) if !k.is_empty() => k,
                _ => {
                    tracing::error!(
                        provider = "voyage",
                        env = "BSMCP_RERANK_API_KEY",
                        "reranker_missing_required_env"
                    );
                    return None;
                }
            };
            let model = env::var("BSMCP_RERANK_MODEL")
                .unwrap_or_else(|_| DEFAULT_VOYAGE_RERANK_MODEL.into());
            let base_url = env::var("BSMCP_RERANK_API_URL")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "https://api.voyageai.com".into());
            tracing::info!(
                provider = "voyage",
                model = %model,
                url = %base_url,
                "reranker_configured"
            );
            Some(
                Arc::new(rerank::VoyageReranker::new(&api_key, &model, &base_url))
                    as Arc<dyn Reranker>,
            )
        }
        "openai" => {
            let api_key = match env::var("BSMCP_RERANK_API_KEY") {
                Ok(k) if !k.is_empty() => k,
                _ => {
                    tracing::error!(
                        provider = "openai",
                        env = "BSMCP_RERANK_API_KEY",
                        "reranker_missing_required_env"
                    );
                    return None;
                }
            };
            let model = match env::var("BSMCP_RERANK_MODEL") {
                Ok(m) if !m.is_empty() => m,
                _ => {
                    tracing::error!(
                        provider = "openai",
                        env = "BSMCP_RERANK_MODEL",
                        "reranker_missing_required_env"
                    );
                    return None;
                }
            };
            let base_url = match env::var("BSMCP_RERANK_API_URL") {
                Ok(u) if !u.is_empty() => u,
                _ => {
                    tracing::error!(
                        provider = "openai",
                        env = "BSMCP_RERANK_API_URL",
                        "reranker_missing_required_env"
                    );
                    return None;
                }
            };
            tracing::info!(
                provider = "openai",
                model = %model,
                url = %base_url,
                "reranker_configured"
            );
            Some(
                Arc::new(rerank::OpenAIReranker::new(&api_key, &model, &base_url))
                    as Arc<dyn Reranker>,
            )
        }
        other => {
            tracing::error!(
                provider = %other,
                "reranker_unknown_provider"
            );
            None
        }
    }
}

/// Background job queue worker. Polls for pending embed jobs and processes them.
async fn job_queue_worker(
    db: Arc<dyn SemanticDb>,
    embedder: Arc<dyn Embedder>,
    worker_id: String,
    model_name: String,
    dims: usize,
) {
    let poll_interval: u64 = env::var("BSMCP_EMBED_POLL_INTERVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let delay_ms: u64 = env::var("BSMCP_EMBED_DELAY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let batch_size: usize = env::var("BSMCP_EMBED_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    // BSMCP_JOB_TIMEOUT_SECS supersedes the legacy BSMCP_EMBED_JOB_TIMEOUT.
    // The new var matches the worker + reconciler naming so all three share
    // one knob; the legacy env stays as a fallback for existing deployments.
    let job_timeout: i64 = env::var("BSMCP_JOB_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| {
            env::var("BSMCP_EMBED_JOB_TIMEOUT")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(3600);
    let failure_threshold: usize = env::var("BSMCP_EMBED_FAILURE_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(pipeline::DEFAULT_FAILURE_THRESHOLD);
    let consecutive_abort: usize = env::var("BSMCP_EMBED_CONSECUTIVE_ABORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(pipeline::DEFAULT_CONSECUTIVE_ABORT);

    let bookstack_url = env::var("BSMCP_BOOKSTACK_URL").expect("BSMCP_BOOKSTACK_URL is required");
    let embed_token_id =
        env::var("BSMCP_EMBED_TOKEN_ID").expect("BSMCP_EMBED_TOKEN_ID is required");
    let embed_token_secret =
        env::var("BSMCP_EMBED_TOKEN_SECRET").expect("BSMCP_EMBED_TOKEN_SECRET is required");

    let http_client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(120))
        .build()
        .expect("Failed to build HTTP client");

    let client = BookStackClient::new(
        &bookstack_url,
        &embed_token_id,
        &embed_token_secret,
        http_client,
    );

    tracing::info!(
        poll_interval_s = poll_interval,
        delay_ms,
        batch_size,
        job_timeout_s = job_timeout,
        failure_threshold,
        consecutive_abort,
        "embedder_job_queue_worker_started"
    );

    // Track whether we already triggered a reindex this startup (avoid double-triggers)
    let mut reindex_triggered = false;

    // Check chunk version — auto-reindex if chunking logic changed
    let current_chunk_version = bsmcp_common::chunking::CHUNK_VERSION.to_string();
    let stored_version = db.get_meta("chunk_version").await.unwrap_or(None);
    if stored_version.as_deref() != Some(&current_chunk_version) {
        tracing::info!(
            from = %stored_version.as_deref().unwrap_or("none"),
            to = %current_chunk_version,
            "embedder_chunk_version_changed"
        );
        trigger_clean_reindex(&db, dims).await;
        db.set_meta("chunk_version", &current_chunk_version)
            .await
            .ok();
        reindex_triggered = true;
    }

    // Check model or dimension change — auto-reindex if either changed
    let stored_model = db.get_meta("embed_model").await.unwrap_or(None);
    let stored_dims = db.get_meta("embed_dims").await.unwrap_or(None);
    let dims_str = dims.to_string();
    let model_changed = stored_model.as_deref() != Some(&model_name);
    let dims_changed = stored_dims.as_deref().is_some_and(|d| d != dims_str);
    if !reindex_triggered && (model_changed || dims_changed) {
        if model_changed {
            tracing::info!(
                from = %stored_model.as_deref().unwrap_or("none"),
                to = %model_name,
                "embedder_model_changed"
            );
        } else {
            tracing::info!(
                from = %stored_dims.as_deref().unwrap_or("?"),
                to = dims,
                "embedder_dims_changed"
            );
        }
        trigger_clean_reindex(&db, dims).await;
        reindex_triggered = true;
    }

    // Store current model metadata
    db.set_meta("embed_model", &model_name).await.ok();
    db.set_meta("embed_dims", &dims.to_string()).await.ok();

    // Auto-embed on startup if requested (and not already triggered above)
    if !reindex_triggered {
        let embed_on_startup = env::var("BSMCP_EMBED_ON_STARTUP").unwrap_or_default();
        if embed_on_startup == "true" || embed_on_startup == "clean" {
            if embed_on_startup == "clean" {
                match db.clear_all_embeddings().await {
                    Ok(()) => tracing::info!("embedder_cleared_all_embeddings"),
                    Err(e) => tracing::error!(error = %e, "embedder_clear_embeddings_failed"),
                }
            }
            match db.create_embed_job("all").await {
                Ok((job_id, true)) => tracing::info!(job_id, "embedder_auto_queued_full_job"),
                Ok((_, false)) => tracing::info!("embedder_auto_embed_skipped_active"),
                Err(e) => tracing::error!(error = %e, "embedder_auto_embed_queue_failed"),
            }
        }
    }

    loop {
        // Expire stale jobs before claiming
        if let Ok(expired) = db.expire_stale_jobs(job_timeout).await {
            if expired > 0 {
                tracing::warn!(
                    expired,
                    timeout_s = job_timeout,
                    "embedder_expired_stale_jobs"
                );
            }
        }

        match db.claim_next_job(&worker_id).await {
            Ok(Some(job)) => {
                tracing::info!(job_id = job.id, scope = %job.scope, "embedder_job_claimed");
                let result = pipeline::run_pipeline(
                    &db,
                    &embedder,
                    &client,
                    job.id,
                    &job.scope,
                    delay_ms,
                    batch_size,
                    consecutive_abort,
                )
                .await;
                match result {
                    Ok(pr) => {
                        let failed_count = pr.failed_pages.len();

                        if failed_count >= failure_threshold || pr.aborted {
                            // Systemic failure — mark job as failed, don't auto-requeue
                            let sample_errors: Vec<String> = pr
                                .failed_pages
                                .iter()
                                .take(3)
                                .map(|(pid, e)| format!("page {pid}: {e}"))
                                .collect();
                            let err_msg = format!(
                                "{failed_count} pages failed (aborted={}). Sample errors: {}",
                                pr.aborted,
                                sample_errors.join("; ")
                            );
                            tracing::error!(job_id = job.id, error = %err_msg, "embedder_job_failed");
                            if let Err(e) = db.complete_job(job.id, Some(&err_msg)).await {
                                tracing::error!(
                                    job_id = job.id,
                                    error = %e,
                                    "embedder_complete_job_failed"
                                );
                            }
                        } else if failed_count > 0 {
                            // Partial failure — mark job complete, queue retries for failed pages
                            if let Err(e) = db.complete_job(job.id, None).await {
                                tracing::error!(
                                    job_id = job.id,
                                    error = %e,
                                    "embedder_complete_job_failed"
                                );
                            }
                            tracing::warn!(
                                job_id = job.id,
                                failed_count,
                                "embedder_job_partial_failure_retrying"
                            );
                            for (page_id, err) in &pr.failed_pages {
                                let scope = format!("page:{page_id}");
                                match db.create_embed_job(&scope).await {
                                    Ok((retry_id, true)) => {
                                        tracing::info!(
                                            retry_job_id = retry_id,
                                            page_id,
                                            error = %err,
                                            "embedder_retry_queued"
                                        );
                                    }
                                    Ok((_, false)) => {
                                        tracing::debug!(page_id, "embedder_retry_already_queued");
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            page_id,
                                            error = %e,
                                            "embedder_retry_queue_failed"
                                        );
                                    }
                                }
                            }

                            // Still recompute relationships for the pages that succeeded
                            if pr.succeeded > 0 {
                                tracing::info!("embedder_computing_similar_pages");
                                match db.compute_similar_pages(5, 0.65).await {
                                    Ok(n) => {
                                        tracing::info!(count = n, "embedder_similar_pages_stored")
                                    }
                                    Err(e) => tracing::error!(
                                        error = %e,
                                        "embedder_similar_pages_failed"
                                    ),
                                }
                            }
                        } else {
                            // Full success
                            if let Err(e) = db.complete_job(job.id, None).await {
                                tracing::error!(
                                    job_id = job.id,
                                    error = %e,
                                    "embedder_complete_job_failed"
                                );
                            }
                            tracing::info!(
                                job_id = job.id,
                                pages = pr.succeeded,
                                "embedder_job_completed"
                            );

                            // Recompute similar-page relationships after any embedding job
                            tracing::info!("embedder_computing_similar_pages");
                            match db.compute_similar_pages(5, 0.65).await {
                                Ok(n) => tracing::info!(count = n, "embedder_similar_pages_stored"),
                                Err(e) => tracing::error!(
                                    error = %e,
                                    "embedder_similar_pages_failed"
                                ),
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(job_id = job.id, error = %e, "embedder_job_failed");
                        if let Err(e2) = db.complete_job(job.id, Some(&e)).await {
                            tracing::error!(
                                job_id = job.id,
                                error = %e2,
                                "embedder_complete_job_failed"
                            );
                        }
                    }
                }
            }
            Ok(None) => {
                // No pending jobs, sleep
                tokio::time::sleep(Duration::from_secs(poll_interval)).await;
            }
            Err(e) => {
                tracing::error!(error = %e, "embedder_job_queue_poll_error");
                tokio::time::sleep(Duration::from_secs(poll_interval)).await;
            }
        }
    }
}

/// Clear all embeddings, adjust pgvector column dimension, and queue a full reindex.
async fn trigger_clean_reindex(db: &Arc<dyn SemanticDb>, dims: usize) {
    match db.clear_all_embeddings().await {
        Ok(()) => tracing::info!("embedder_cleared_all_embeddings"),
        Err(e) => tracing::error!(error = %e, "embedder_clear_embeddings_failed"),
    }
    match db.alter_embedding_dimension(dims).await {
        Ok(()) => tracing::info!(dims, "embedder_alter_dim_set"),
        Err(e) => tracing::error!(error = %e, "embedder_alter_dim_failed"),
    }
    match db.create_embed_job("all").await {
        Ok((job_id, true)) => tracing::info!(job_id, "embedder_auto_queued_full_job"),
        Ok((_, false)) => tracing::info!("embedder_reindex_job_already_active"),
        Err(e) => tracing::error!(error = %e, "embedder_reindex_queue_failed"),
    }
}
