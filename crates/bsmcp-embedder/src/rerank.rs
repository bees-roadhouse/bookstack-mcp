//! Reranker provider abstraction. Sibling of `embed.rs`.
//!
//! Three providers, parallel to the embedder side:
//! - `LocalReranker` — in-process ONNX cross-encoder via fastembed (default
//!   `BAAI/bge-reranker-v2-m3`). Wraps `pipeline::RerankModel`.
//! - `VoyageReranker` — Voyage AI's `POST /v1/rerank`.
//! - `OpenAIReranker` — generic OpenAI-shape `POST /v1/rerank` for any
//!   compatible endpoint (Voyage/Jina/self-hosted). Configurable URL is
//!   required; OpenAI itself has not shipped a rerank API yet.

use std::sync::Arc;

use async_trait::async_trait;

/// One reranked result: original index into the documents array + score.
/// Higher score = more relevant. Callers apply their own sort / top-k.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RerankHit {
    pub index: usize,
    pub score: f32,
}

#[async_trait]
pub trait Reranker: Send + Sync + 'static {
    /// Score `documents` against `query`. Returns one hit per document, in
    /// input order. The caller does any sort / top-k cut.
    async fn rerank(&self, query: String, documents: Vec<String>)
        -> Result<Vec<RerankHit>, String>;

    fn provider_name(&self) -> &str;

    fn model_name(&self) -> &str;
}

// --- Local fastembed cross-encoder ---

pub struct LocalReranker {
    model: Arc<crate::pipeline::RerankModel>,
}

impl LocalReranker {
    pub fn new(model: Arc<crate::pipeline::RerankModel>) -> Self {
        Self { model }
    }
}

#[async_trait]
impl Reranker for LocalReranker {
    async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> Result<Vec<RerankHit>, String> {
        let model = self.model.clone();
        tokio::task::spawn_blocking(move || model.rerank(&query, documents))
            .await
            .map_err(|e| format!("Rerank task failed: {e}"))?
    }

    fn provider_name(&self) -> &str {
        "local"
    }

    fn model_name(&self) -> &str {
        self.model.model_name()
    }
}

// --- Voyage AI reranker ---

pub struct VoyageReranker {
    api_key: String,
    model: String,
    base_url: String,
    http: reqwest::Client,
}

impl VoyageReranker {
    pub fn new(api_key: &str, model: &str, base_url: &str) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            http,
        }
    }
}

#[async_trait]
impl Reranker for VoyageReranker {
    async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> Result<Vec<RerankHit>, String> {
        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents,
        });

        let resp = self
            .http
            .post(format!("{}/v1/rerank", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                let url = format!("{}/v1/rerank", self.base_url);
                format!(
                    "Voyage rerank request failed: {e:?} (url={url}, key_len={}, model={})",
                    self.api_key.len(),
                    self.model
                )
            })?;

        let status = resp.status();
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Voyage rerank response parse failed: {e}"))?;

        if !status.is_success() {
            let msg = json
                .get("detail")
                .and_then(|m| m.as_str())
                .or_else(|| {
                    json.get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                })
                .unwrap_or("unknown error");
            return Err(format!("Voyage rerank error {status}: {msg}"));
        }

        parse_voyage_shape(&json)
    }

    fn provider_name(&self) -> &str {
        "voyage"
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// --- OpenAI-compatible reranker (generic shape) ---

pub struct OpenAIReranker {
    api_key: String,
    model: String,
    base_url: String,
    http: reqwest::Client,
}

impl OpenAIReranker {
    pub fn new(api_key: &str, model: &str, base_url: &str) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            http,
        }
    }
}

#[async_trait]
impl Reranker for OpenAIReranker {
    async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> Result<Vec<RerankHit>, String> {
        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents,
        });

        let resp = self
            .http
            .post(format!("{}/v1/rerank", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                let url = format!("{}/v1/rerank", self.base_url);
                format!(
                    "OpenAI rerank request failed: {e:?} (url={url}, key_len={}, model={})",
                    self.api_key.len(),
                    self.model
                )
            })?;

        let status = resp.status();
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("OpenAI rerank response parse failed: {e}"))?;

        if !status.is_success() {
            let msg = json
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .or_else(|| json.get("detail").and_then(|m| m.as_str()))
                .unwrap_or("unknown error");
            return Err(format!("OpenAI rerank error {status}: {msg}"));
        }

        parse_voyage_shape(&json)
    }

    fn provider_name(&self) -> &str {
        "openai"
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// Parse the `{ data: [{ index, relevance_score }] }` shape used by Voyage
/// and most OpenAI-compatible rerank endpoints. Cohere uses `results` instead
/// of `data` — accept both so a Cohere endpoint plugged into the OpenAI-
/// compatible provider also works.
fn parse_voyage_shape(json: &serde_json::Value) -> Result<Vec<RerankHit>, String> {
    let arr = json
        .get("data")
        .and_then(|v| v.as_array())
        .or_else(|| json.get("results").and_then(|v| v.as_array()))
        .ok_or("rerank response missing 'data' or 'results' array")?;

    let mut hits = Vec::with_capacity(arr.len());
    for item in arr {
        let index = item
            .get("index")
            .and_then(|v| v.as_u64())
            .ok_or("rerank item missing 'index'")? as usize;
        let score = item
            .get("relevance_score")
            .and_then(|v| v.as_f64())
            .ok_or("rerank item missing 'relevance_score'")? as f32;
        hits.push(RerankHit { index, score });
    }
    Ok(hits)
}
