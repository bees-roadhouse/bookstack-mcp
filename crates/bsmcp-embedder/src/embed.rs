//! Embedding provider abstraction.
//! Supports local ONNX models (fastembed), OpenAI API, Ollama, and Voyage AI.

use std::sync::Arc;

use async_trait::async_trait;

/// Trait for embedding text into vectors.
#[async_trait]
pub trait Embedder: Send + Sync + 'static {
    /// Embed a batch of texts, returning one vector per text.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String>;

    /// Return the embedding dimension.
    fn dims(&self) -> usize;

    /// Return the provider name for logging/health endpoint.
    #[allow(dead_code)]
    fn provider_name(&self) -> &str;
}

// --- OpenAI-compatible embedder (works for OpenAI API) ---

pub struct OpenAIEmbedder {
    api_key: String,
    model: String,
    base_url: String,
    dims: usize,
    http: reqwest::Client,
}

impl OpenAIEmbedder {
    pub fn new(api_key: &str, model: &str, base_url: &str, dims: usize) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            dims,
            http,
        }
    }

    /// Auto-detect embedding dimensions by sending a test string.
    pub async fn detect_dims(api_key: &str, model: &str, base_url: &str) -> Result<usize, String> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| format!("HTTP client build failed: {e}"))?;

        let base = base_url.trim_end_matches('/');
        let body = serde_json::json!({
            "model": model,
            "input": "dimension test",
        });

        let resp = http
            .post(format!("{base}/v1/embeddings"))
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("OpenAI dim detection failed: {e}"))?;

        let status = resp.status();
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("OpenAI dim detection parse failed: {e}"))?;

        if !status.is_success() {
            let msg = json
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            return Err(format!("OpenAI dim detection error {status}: {msg}"));
        }

        let dims = json["data"][0]["embedding"]
            .as_array()
            .map(|a| a.len())
            .ok_or("Could not detect dimensions from OpenAI response")?;

        Ok(dims)
    }
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let resp = self
            .http
            .post(format!("{}/v1/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                let url = format!("{}/v1/embeddings", self.base_url);
                format!(
                    "OpenAI embed request failed: {e:?} (url={url}, key_len={}, model={})",
                    self.api_key.len(),
                    self.model
                )
            })?;

        let status = resp.status();
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("OpenAI embed response parse failed: {e}"))?;

        if !status.is_success() {
            return Err(format!(
                "OpenAI embed error {status}: {}",
                json.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error")
            ));
        }

        let data = json["data"]
            .as_array()
            .ok_or("No data array in OpenAI response")?;

        let mut embeddings: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
        for item in data {
            let index = item["index"].as_u64().unwrap_or(0) as usize;
            let embedding: Vec<f32> = item["embedding"]
                .as_array()
                .ok_or("No embedding array in response item")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            embeddings.push((index, embedding));
        }

        // Sort by index to match input order
        embeddings.sort_by_key(|(i, _)| *i);
        Ok(embeddings.into_iter().map(|(_, e)| e).collect())
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn provider_name(&self) -> &str {
        "openai"
    }
}

// --- Ollama embedder ---

pub struct OllamaEmbedder {
    model: String,
    base_url: String,
    dims: usize,
    http: reqwest::Client,
}

impl OllamaEmbedder {
    pub fn new(model: &str, base_url: &str, dims: usize) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            model: model.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            dims,
            http,
        }
    }

    /// Auto-detect embedding dimensions by sending a test string.
    pub async fn detect_dims(model: &str, base_url: &str) -> Result<usize, String> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| format!("HTTP client build failed: {e}"))?;

        let body = serde_json::json!({
            "model": model,
            "input": "dimension test",
        });

        let resp = http
            .post(format!("{}/api/embed", base_url.trim_end_matches('/')))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Ollama dim detection failed: {e}"))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Ollama dim detection parse failed: {e}"))?;

        let dims = json["embeddings"][0]
            .as_array()
            .map(|a| a.len())
            .ok_or("Could not detect dimensions from Ollama response")?;

        Ok(dims)
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        // Ollama's /api/embed supports batch input
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let resp = self
            .http
            .post(format!("{}/api/embed", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Ollama embed request failed: {e}"))?;

        let status = resp.status();
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Ollama embed response parse failed: {e}"))?;

        if !status.is_success() {
            return Err(format!(
                "Ollama embed error {status}: {}",
                json.get("error")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error")
            ));
        }

        let embeddings_arr = json["embeddings"]
            .as_array()
            .ok_or("No embeddings array in Ollama response")?;

        let embeddings: Vec<Vec<f32>> = embeddings_arr
            .iter()
            .map(|arr| {
                arr.as_array()
                    .unwrap_or(&Vec::new())
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .collect();

        Ok(embeddings)
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn provider_name(&self) -> &str {
        "ollama"
    }
}

// --- Voyage AI embedder ---

pub struct VoyageEmbedder {
    api_key: String,
    model: String,
    base_url: String,
    dims: usize,
    http: reqwest::Client,
}

impl VoyageEmbedder {
    pub fn new(api_key: &str, model: &str, base_url: &str, dims: usize) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            dims,
            http,
        }
    }

    /// Auto-detect embedding dimensions by sending a test string.
    pub async fn detect_dims(api_key: &str, model: &str, base_url: &str) -> Result<usize, String> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| format!("HTTP client build failed: {e}"))?;

        let base = base_url.trim_end_matches('/');
        let body = serde_json::json!({
            "model": model,
            "input": ["dimension test"],
        });

        let resp = http
            .post(format!("{base}/v1/embeddings"))
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Voyage dim detection failed: {e}"))?;

        let status = resp.status();
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Voyage dim detection parse failed: {e}"))?;

        if !status.is_success() {
            let msg = json
                .get("detail")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            return Err(format!("Voyage dim detection error {status}: {msg}"));
        }

        let dims = json["data"][0]["embedding"]
            .as_array()
            .map(|a| a.len())
            .ok_or("Could not detect dimensions from Voyage response")?;

        Ok(dims)
    }
}

#[async_trait]
impl Embedder for VoyageEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let resp = self
            .http
            .post(format!("{}/v1/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                let url = format!("{}/v1/embeddings", self.base_url);
                format!(
                    "Voyage embed request failed: {e:?} (url={url}, key_len={}, model={})",
                    self.api_key.len(),
                    self.model
                )
            })?;

        let status = resp.status();
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Voyage embed response parse failed: {e}"))?;

        if !status.is_success() {
            return Err(format!(
                "Voyage embed error {status}: {}",
                json.get("detail")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error")
            ));
        }

        let data = json["data"]
            .as_array()
            .ok_or("No data array in Voyage response")?;

        let mut embeddings: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
        for item in data {
            let index = item["index"].as_u64().unwrap_or(0) as usize;
            let embedding: Vec<f32> = item["embedding"]
                .as_array()
                .ok_or("No embedding array in response item")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            embeddings.push((index, embedding));
        }

        // Sort by index to match input order
        embeddings.sort_by_key(|(i, _)| *i);
        Ok(embeddings.into_iter().map(|(_, e)| e).collect())
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn provider_name(&self) -> &str {
        "voyage"
    }
}

// --- Local fastembed model wrapper (delegates to spawn_blocking) ---

pub struct LocalEmbedder {
    model: Arc<crate::pipeline::EmbedModel>,
}

impl LocalEmbedder {
    pub fn new(model: Arc<crate::pipeline::EmbedModel>) -> Self {
        Self { model }
    }
}

#[async_trait]
impl Embedder for LocalEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        let model = self.model.clone();
        tokio::task::spawn_blocking(move || model.embed(texts))
            .await
            .map_err(|e| format!("Embed task failed: {e}"))?
    }

    fn dims(&self) -> usize {
        self.model.dims()
    }

    fn provider_name(&self) -> &str {
        "local"
    }
}
