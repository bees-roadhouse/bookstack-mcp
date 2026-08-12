//! Embedding pipeline: fetch pages from BookStack, chunk, embed, store.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use fastembed::{
    EmbeddingModel, InitOptions, InitOptionsUserDefined, Pooling, RerankInitOptions, RerankerModel,
    TextEmbedding, TextRerank, TokenizerFiles, UserDefinedEmbeddingModel,
};

use bsmcp_common::acl::{build_role_context, reconcile_all_pages, resolve_page_acl, RoleContext};
use bsmcp_common::bookstack::BookStackClient;
use bsmcp_common::chunking;
use bsmcp_common::db::SemanticDb;
use bsmcp_common::types::{ChunkInsert, PageMeta};

use crate::embed::Embedder;

/// Known model configurations.
struct ModelConfig {
    builtin: Option<EmbeddingModel>,
    hf_repo: &'static str,
    dims: usize,
}

/// Resolve a model name to its configuration.
fn resolve_model(name: &str) -> Option<ModelConfig> {
    match name {
        "BAAI/bge-large-en-v1.5" => Some(ModelConfig {
            builtin: Some(EmbeddingModel::BGELargeENV15),
            hf_repo: "BAAI/bge-large-en-v1.5",
            dims: 1024,
        }),
        "BAAI/bge-base-en-v1.5" => Some(ModelConfig {
            builtin: Some(EmbeddingModel::BGEBaseENV15),
            hf_repo: "BAAI/bge-base-en-v1.5",
            dims: 768,
        }),
        "BAAI/bge-small-en-v1.5" => Some(ModelConfig {
            builtin: Some(EmbeddingModel::BGESmallENV15),
            hf_repo: "BAAI/bge-small-en-v1.5",
            dims: 384,
        }),
        "onnx-community/embeddinggemma-300m-ONNX"
        | "google/embeddinggemma-300m"
        | "embeddinggemma-300m" => Some(ModelConfig {
            builtin: None,
            hf_repo: "onnx-community/embeddinggemma-300m-ONNX",
            dims: 768,
        }),
        _ => None,
    }
}

/// Thread-safe wrapper around the fastembed TextEmbedding model.
pub struct EmbedModel {
    model: std::sync::Mutex<TextEmbedding>,
    dims: usize,
}

impl EmbedModel {
    pub fn new(model_name: &str, cache_dir: &str) -> Result<Self, String> {
        let config = resolve_model(model_name)
            .ok_or_else(|| format!("Unknown model: {model_name}. Supported: BAAI/bge-large-en-v1.5, BAAI/bge-base-en-v1.5, BAAI/bge-small-en-v1.5, embeddinggemma-300m"))?;

        let model = if let Some(builtin) = config.builtin {
            // Use fastembed's built-in model registry
            let options = InitOptions::new(builtin)
                .with_cache_dir(cache_dir.into())
                .with_show_download_progress(true);
            TextEmbedding::try_new(options).map_err(|e| format!("Model init failed: {e}"))?
        } else {
            // Custom model: download from HuggingFace and load via UserDefinedEmbeddingModel
            let downloaded = download_hf_model(config.hf_repo, cache_dir)?;
            load_custom_model(&downloaded)?
        };

        tracing::info!(
            repo = config.hf_repo,
            dims = config.dims,
            "pipeline_model_loaded"
        );
        Ok(Self {
            model: std::sync::Mutex::new(model),
            dims: config.dims,
        })
    }

    /// Return the embedding dimensions for this model.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Embed a batch of texts. Thread-safe (locks internally).
    pub fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        let mut m = self
            .model
            .lock()
            .map_err(|e| format!("Model lock poisoned: {e}"))?;
        m.embed(texts, None).map_err(|e| format!("{e}"))
    }
}

/// Resolve a reranker model name to its fastembed enum variant.
fn resolve_reranker(name: &str) -> Option<RerankerModel> {
    match name {
        "BAAI/bge-reranker-base" | "bge-reranker-base" => Some(RerankerModel::BGERerankerBase),
        "BAAI/bge-reranker-v2-m3" | "rozgo/bge-reranker-v2-m3" | "bge-reranker-v2-m3" => {
            Some(RerankerModel::BGERerankerV2M3)
        }
        "jinaai/jina-reranker-v1-turbo-en" | "jina-reranker-v1-turbo-en" => {
            Some(RerankerModel::JINARerankerV1TurboEn)
        }
        "jinaai/jina-reranker-v2-base-multilingual" | "jina-reranker-v2-base-multilingual" => {
            Some(RerankerModel::JINARerankerV2BaseMultiligual)
        }
        _ => None,
    }
}

/// Thread-safe wrapper around fastembed's `TextRerank` cross-encoder.
pub struct RerankModel {
    model: std::sync::Mutex<TextRerank>,
    model_name: String,
}

impl RerankModel {
    pub fn new(model_name: &str, cache_dir: &str) -> Result<Self, String> {
        let reranker = resolve_reranker(model_name).ok_or_else(|| {
            format!(
                "Unknown reranker model: {model_name}. Supported: \
                 BAAI/bge-reranker-base, BAAI/bge-reranker-v2-m3, \
                 jinaai/jina-reranker-v1-turbo-en, \
                 jinaai/jina-reranker-v2-base-multilingual"
            )
        })?;

        let options = RerankInitOptions::new(reranker)
            .with_cache_dir(cache_dir.into())
            .with_show_download_progress(true);
        let model =
            TextRerank::try_new(options).map_err(|e| format!("Reranker model init failed: {e}"))?;

        tracing::info!(model = %model_name, "pipeline_reranker_loaded");
        Ok(Self {
            model: std::sync::Mutex::new(model),
            model_name: model_name.to_string(),
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Score `documents` against `query`. Thread-safe (locks internally).
    pub fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<Vec<crate::rerank::RerankHit>, String> {
        let mut m = self
            .model
            .lock()
            .map_err(|e| format!("Reranker lock poisoned: {e}"))?;
        // fastembed's `rerank` shares one generic `S: AsRef<str>` between
        // query and documents — owning the query as `String` lets `S = String`
        // match `&Vec<String>: AsRef<[String]>`.
        let q = query.to_string();
        let results = m
            .rerank(q, &documents, false, None)
            .map_err(|e| format!("{e}"))?;
        Ok(results
            .into_iter()
            .map(|r| crate::rerank::RerankHit {
                index: r.index,
                score: r.score,
            })
            .collect())
    }
}

/// Downloaded model file paths.
struct DownloadedModel {
    onnx_path: PathBuf,
    onnx_data_path: Option<PathBuf>,
    tokenizer_path: PathBuf,
    config_path: PathBuf,
    special_tokens_path: PathBuf,
    tokenizer_config_path: PathBuf,
}

/// Download model files from HuggingFace Hub.
/// Handles repos where ONNX files are in a subfolder (e.g. onnx/model.onnx)
/// while tokenizer files are at root.
fn download_hf_model(repo_id: &str, cache_dir: &str) -> Result<DownloadedModel, String> {
    use hf_hub::api::sync::ApiBuilder;

    tracing::info!(repo = %repo_id, "pipeline_hf_download_starting");
    let api = ApiBuilder::new()
        .with_cache_dir(PathBuf::from(cache_dir))
        .build()
        .map_err(|e| format!("HF API init failed: {e}"))?;
    let repo = api.model(repo_id.to_string());

    let get = |filename: &str| -> Result<PathBuf, String> {
        let path = repo
            .get(filename)
            .map_err(|e| format!("Failed to download {filename}: {e}"))?;
        tracing::debug!(path = %path.display(), "pipeline_hf_cached");
        Ok(path)
    };

    // Try onnx/model.onnx first (onnx-community layout), fall back to model.onnx
    let onnx_path = get("onnx/model.onnx").or_else(|_| get("model.onnx"))?;

    // External weights (optional)
    let onnx_data_path = get("onnx/model.onnx_data")
        .or_else(|_| get("model.onnx_data"))
        .ok();

    // Tokenizer files are always at root
    let tokenizer_path = get("tokenizer.json")?;
    let config_path = get("config.json")?;
    let special_tokens_path = get("special_tokens_map.json")?;
    let tokenizer_config_path = get("tokenizer_config.json")?;

    Ok(DownloadedModel {
        onnx_path,
        onnx_data_path,
        tokenizer_path,
        config_path,
        special_tokens_path,
        tokenizer_config_path,
    })
}

/// Load a custom ONNX model from downloaded file paths.
fn load_custom_model(downloaded: &DownloadedModel) -> Result<TextEmbedding, String> {
    let read = |path: &Path| -> Result<Vec<u8>, String> {
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))
    };

    let onnx_file = read(&downloaded.onnx_path)?;
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read(&downloaded.tokenizer_path)?,
        config_file: read(&downloaded.config_path)?,
        special_tokens_map_file: read(&downloaded.special_tokens_path)?,
        tokenizer_config_file: read(&downloaded.tokenizer_config_path)?,
    };

    let mut user_model =
        UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files).with_pooling(Pooling::Mean);

    // Load external weights file if present (EmbeddingGemma uses this)
    if let Some(ref data_path) = downloaded.onnx_data_path {
        let data = read(data_path)?;
        user_model = user_model.with_external_initializer("model.onnx_data".to_string(), data);
    }

    let options = InitOptionsUserDefined::default();
    TextEmbedding::try_new_from_user_defined(user_model, options)
        .map_err(|e| format!("Custom model init failed: {e}"))
}

/// Build a book_id → shelf_name lookup by fetching all shelves from BookStack.
async fn build_shelf_lookup(client: &BookStackClient) -> HashMap<i64, String> {
    let mut lookup: HashMap<i64, String> = HashMap::new();
    let mut offset = 0i64;
    loop {
        let list = match client.list_shelves(100, offset).await {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(error = %e, "pipeline_list_shelves_failed");
                break;
            }
        };
        let data = list.get("data").and_then(|v| v.as_array());
        let Some(shelves) = data else { break };
        if shelves.is_empty() {
            break;
        }

        for shelf in shelves {
            let shelf_id = shelf.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
            if shelf_id == 0 {
                continue;
            }
            let shelf_name = shelf
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if shelf_name.is_empty() {
                continue;
            }

            // Fetch shelf detail to get its books
            match client.get_shelf(shelf_id).await {
                Ok(detail) => {
                    if let Some(books) = detail.get("books").and_then(|v| v.as_array()) {
                        for book in books {
                            let bid = book.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                            if bid > 0 {
                                lookup.insert(bid, shelf_name.clone());
                            }
                        }
                    }
                }
                Err(e) => tracing::error!(shelf_id, error = %e, "pipeline_get_shelf_failed"),
            }
        }

        let total = list.get("total").and_then(|v| v.as_i64()).unwrap_or(0);
        offset += 100;
        if offset >= total {
            break;
        }
    }
    tracing::info!(mappings = lookup.len(), "pipeline_shelf_lookup_built");
    lookup
}

/// Result of a pipeline run, including any per-page failures.
pub struct PipelineResult {
    pub succeeded: usize,
    pub failed_pages: Vec<(i64, String)>,
    /// True if the pipeline aborted early due to too many consecutive failures.
    pub aborted: bool,
}

/// Default threshold for total failures before marking the job as failed.
pub const DEFAULT_FAILURE_THRESHOLD: usize = 10;

/// Default threshold for consecutive failures before aborting early (systemic issue).
pub const DEFAULT_CONSECUTIVE_ABORT: usize = 10;

/// Run the embedding pipeline for a job.
// Tunables (delay/batch/abort thresholds) are pulled in from main rather
// than reread from env each call. Bundling them into a config struct would
// hide the call site, so the tradeoff favors keeping the explicit signature.
#[allow(clippy::too_many_arguments)]
pub async fn run_pipeline(
    db: &Arc<dyn SemanticDb>,
    embedder: &Arc<dyn Embedder>,
    client: &BookStackClient,
    job_id: i64,
    scope: &str,
    delay_ms: u64,
    _batch_size: usize,
    consecutive_abort: usize,
) -> Result<PipelineResult, String> {
    tracing::info!(scope = %scope, job_id, "pipeline_starting");

    // ACL-only reconciliation path. Triggered by webhooks for role events or
    // by the daily reconciliation cron — no embedding work, just refresh
    // page_view_acl for every previously-stored page.
    if scope == "acl_reconcile" {
        match reconcile_all_pages(client, db).await {
            Ok((processed, failed)) => {
                tracing::info!(processed, failed, "pipeline_acl_reconcile_complete");
                db.update_job_progress(job_id, processed as i64, (processed + failed) as i64)
                    .await?;
                return Ok(PipelineResult {
                    succeeded: processed,
                    failed_pages: Vec::new(),
                    aborted: false,
                });
            }
            Err(e) => {
                tracing::error!(error = %e, "pipeline_acl_reconcile_failed");
                return Err(e);
            }
        }
    }

    // Build shelf lookup for context prefix injection
    let shelf_lookup = build_shelf_lookup(client).await;

    // Resolve role context once per pipeline run for ACL stamping. If the
    // BookStack token can't list roles (rare — system-level perm), ACL
    // population is skipped for the whole run; semantic search falls back to
    // its existing per-page HTTP permission check.
    let role_ctx = match build_role_context(client).await {
        Ok(rc) => Some(rc),
        Err(e) => {
            tracing::warn!(error = %e, "pipeline_role_context_unavailable_acl_disabled");
            None
        }
    };

    // Collect page IDs to embed
    let mut offset = 0i64;
    let mut all_page_ids: Vec<i64> = Vec::new();

    loop {
        let list = client.list_pages(100, offset).await?;
        let data = list.get("data").and_then(|v| v.as_array());
        let Some(pages) = data else { break };
        if pages.is_empty() {
            break;
        }

        for page in pages {
            let pid = page.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
            if pid == 0 {
                continue;
            }

            // Scope filtering
            if scope != "all" {
                if let Some(book_prefix) = scope.strip_prefix("book:") {
                    if let Ok(target_book) = book_prefix.parse::<i64>() {
                        let bid = page.get("book_id").and_then(|v| v.as_i64()).unwrap_or(0);
                        if bid != target_book {
                            continue;
                        }
                    }
                } else if let Some(page_prefix) = scope.strip_prefix("page:") {
                    if let Ok(target_page) = page_prefix.parse::<i64>() {
                        if pid != target_page {
                            continue;
                        }
                    }
                }
            }

            all_page_ids.push(pid);
        }

        // Advance by rows actually returned, not by the count we asked
        // for — BookStack clamps `count` to API_MAX_ITEM_COUNT, and a
        // fixed stride would skip rows on an instance capped below it.
        // A response missing `total` falls through to the empty-page
        // check at the top of the loop instead of truncating here.
        offset += pages.len() as i64;
        let total_in_response = list
            .get("total")
            .and_then(|v| v.as_i64())
            .unwrap_or(i64::MAX);
        if offset >= total_in_response {
            break;
        }
    }

    let total_pages = all_page_ids.len();
    db.update_job_progress(job_id, 0, total_pages as i64)
        .await?;
    tracing::info!(total_pages, "pipeline_pages_found");

    // Always force re-embed (bypass content hash check).
    // Context prefix changes (shelf/book/chapter renames, moves) don't change
    // the HTML, so the content hash would incorrectly skip affected pages.
    // The shelf lookup is rebuilt each run, ensuring fresh context.
    let force = true;

    let mut failed_pages: Vec<(i64, String)> = Vec::new();
    let mut succeeded = 0usize;
    let mut consecutive_failures = 0usize;
    let mut aborted = false;

    for (i, page_id) in all_page_ids.iter().enumerate() {
        // Per-page status check. This is a `WHERE id = ?` PK lookup —
        // cheap (microseconds on either backend) and intentional. We pay
        // it on every page so a user-issued cancel takes effect within
        // one page rather than waiting for the whole walk to finish.
        if matches!(db.should_stop_embed_job(job_id).await, Ok(true)) {
            tracing::info!(job_id, page_index = i, "pipeline_job_cancelled_aborting");
            aborted = true;
            db.update_job_progress(job_id, i as i64, total_pages as i64)
                .await?;
            break;
        }
        match embed_single_page(
            db,
            embedder,
            client,
            *page_id,
            force,
            &shelf_lookup,
            role_ctx.as_ref(),
        )
        .await
        {
            Ok(outcome) => {
                succeeded += 1;
                consecutive_failures = 0;
                // One line per page at INFO — this is the progress readout
                // for a running job. `page`/`total_pages` make it obvious
                // how far along a multi-thousand-page walk is without
                // polling /status.
                tracing::info!(
                    job_id,
                    page = i + 1,
                    total_pages,
                    page_id = *page_id,
                    name = %outcome.name,
                    chunks = outcome.chunks,
                    skipped = outcome.skipped,
                    "pipeline_page_embedded"
                );
            }
            Err(e) => {
                tracing::error!(page_id = *page_id, error = %e, "pipeline_page_embed_failed");
                failed_pages.push((*page_id, e));
                consecutive_failures += 1;

                // Abort early if we hit too many consecutive failures — likely systemic
                if consecutive_failures >= consecutive_abort {
                    tracing::error!(
                        consecutive_failures,
                        remaining = total_pages - i - 1,
                        "pipeline_aborting_systemic_failures"
                    );
                    aborted = true;
                    db.update_job_progress(job_id, (i + 1) as i64, total_pages as i64)
                        .await?;
                    break;
                }
            }
        }

        db.update_job_progress(job_id, (i + 1) as i64, total_pages as i64)
            .await?;

        if delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }
    }

    let failed_count = failed_pages.len();
    if failed_count > 0 {
        tracing::warn!(
            failed = failed_count,
            total_pages,
            aborted,
            "pipeline_finished_with_failures"
        );
    } else {
        tracing::info!(total_pages, "pipeline_completed");
    }

    Ok(PipelineResult {
        succeeded,
        failed_pages,
        aborted,
    })
}

/// Embed a single page: fetch, check hash, chunk, embed, store relationships.
/// When `force` is true, skip the content hash check and always re-embed.
/// What one page's embed pass actually did, so the caller can log
/// per-page progress at INFO without re-fetching the page.
struct PageOutcome {
    name: String,
    chunks: usize,
    /// Content hash matched and the page was left alone. Only reachable
    /// when `force` is false.
    skipped: bool,
}

async fn embed_single_page(
    db: &Arc<dyn SemanticDb>,
    embedder: &Arc<dyn Embedder>,
    client: &BookStackClient,
    page_id: i64,
    force: bool,
    shelf_lookup: &HashMap<i64, String>,
    role_ctx: Option<&RoleContext>,
) -> Result<PageOutcome, String> {
    let page = client.get_page(page_id).await?;

    let html = page.get("html").and_then(|v| v.as_str()).unwrap_or("");
    let name = page.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let slug = page.get("slug").and_then(|v| v.as_str()).unwrap_or("");
    let book_id = page.get("book_id").and_then(|v| v.as_i64()).unwrap_or(0);
    let chapter_id = page
        .get("chapter_id")
        .and_then(|v| v.as_i64())
        .filter(|&id| id > 0);

    // Extract shelf/book/chapter names for chunk context injection
    let shelf_name = shelf_lookup.get(&book_id).map(|s| s.as_str()).unwrap_or("");
    let book_name = page
        .get("book")
        .and_then(|b| b.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let chapter_name = page
        .get("chapter")
        .and_then(|c| c.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Compute content hash to skip unchanged pages (unless forced)
    let content_hash = {
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(html.as_bytes());
        hash.iter().map(|b| format!("{b:02x}")).collect::<String>()
    };

    if !force {
        if let Ok(Some(existing_hash)) = db.get_page_content_hash(page_id).await {
            if existing_hash == content_hash {
                return Ok(PageOutcome {
                    name: name.to_string(),
                    chunks: 0,
                    skipped: true,
                });
            }
        }
    }

    let updated_at = page
        .get("updated_at")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let meta = PageMeta {
        page_id,
        book_id,
        chapter_id,
        name: name.to_string(),
        slug: slug.to_string(),
        content_hash: content_hash.clone(),
        updated_at,
    };

    // Chunk the HTML (skip first h1 if it matches page name to avoid duplication)
    let chunks = chunking::chunk_html_with_name(html, Some(name));
    if chunks.is_empty() {
        if html.len() > 100 {
            // Log a sample of the HTML to help diagnose why chunking failed
            let sample: String = html.chars().take(300).collect();
            tracing::warn!(
                page_id,
                name = %name,
                html_len = html.len(),
                sample = %sample,
                "pipeline_chunker_empty_result"
            );
        }
        db.upsert_page(&meta).await?;
        return Ok(PageOutcome {
            name: name.to_string(),
            chunks: 0,
            skipped: false,
        });
    }
    tracing::debug!(
        page_id,
        name = %name,
        chunks = chunks.len(),
        html_len = html.len(),
        "pipeline_page_chunked"
    );

    // Build context prefix for embedding (shelf > book > chapter > page hierarchy)
    let context_prefix = {
        let mut parts = Vec::new();
        if !shelf_name.is_empty() {
            parts.push(shelf_name.to_string());
        }
        if !book_name.is_empty() {
            parts.push(book_name.to_string());
        }
        if !chapter_name.is_empty() {
            parts.push(chapter_name.to_string());
        }
        parts.push(name.to_string());
        format!("[{}]\n\n", parts.join(" > "))
    };

    // Prepend context to chunk text for embedding (stored content stays clean for display)
    let texts: Vec<String> = chunks
        .iter()
        .map(|c| format!("{context_prefix}{}", c.content))
        .collect();
    let embeddings = embedder.embed(texts).await?;

    // Store chunks with embeddings
    let chunk_inserts: Vec<ChunkInsert> = chunks
        .iter()
        .zip(embeddings.iter())
        .map(|(chunk, embedding)| ChunkInsert {
            chunk_index: chunk.index,
            heading_path: chunk.heading_path.clone(),
            content: chunk.content.clone(),
            content_hash: chunk.content_hash.clone(),
            embedding: embedding.clone(),
        })
        .collect();

    // Upsert page row BEFORE chunks — the chunks table has a FK to pages(page_id).
    // Use an empty content_hash initially so a crash before insert_chunks completes
    // will cause a re-embed on the next run (hash won't match).
    let preliminary_meta = PageMeta {
        content_hash: String::new(),
        ..meta.clone()
    };
    db.upsert_page(&preliminary_meta).await?;

    db.insert_chunks(page_id, &chunk_inserts).await?;

    // Extract and store link relationships
    let links = chunking::extract_links(html);
    let mut targets: Vec<(i64, String)> = Vec::new();
    for link in &links {
        if link.contains("/page/") {
            if let Some(slug_part) = link.rsplit("/page/").next() {
                if let Ok(Some(target_id)) = db.resolve_page_slug(slug_part).await {
                    targets.push((target_id, "link".to_string()));
                }
            }
        } else if let Some(link_id_str) = link.strip_prefix("/link/") {
            if let Ok(link_id) = link_id_str.parse::<i64>() {
                targets.push((link_id, "link".to_string()));
            }
        }
    }
    db.replace_relationships(page_id, &targets).await?;
    let chunk_count = chunk_inserts.len();

    // Store final page metadata with real content_hash — this is the commit marker.
    db.upsert_page(&meta).await?;

    // Stamp the per-page ACL. Best-effort: a failure here doesn't fail the
    // embed (the page is already searchable), it just means the page falls
    // through to the HTTP permission check at query time until the next
    // pipeline run or webhook recompute.
    if let Some(rc) = role_ctx {
        match resolve_page_acl(client, page_id, chapter_id, book_id, rc).await {
            Ok(acl) => {
                if let Err(e) = db.upsert_page_acl(&acl).await {
                    tracing::warn!(page_id, error = %e, "pipeline_acl_upsert_failed");
                }
            }
            Err(e) => {
                tracing::warn!(page_id, error = %e, "pipeline_acl_resolve_failed");
            }
        }
    }

    Ok(PageOutcome {
        name: name.to_string(),
        chunks: chunk_count,
        skipped: false,
    })
}
