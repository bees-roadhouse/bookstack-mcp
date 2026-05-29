use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{Multipart, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use tokio::sync::RwLock;

use crate::sse::AppState;

const STAGING_TTL: Duration = Duration::from_secs(300); // 5 minutes
const MAX_STAGING_SIZE: usize = 50 * 1024 * 1024; // 50MB

pub struct StagingEntry {
    pub bytes: Vec<u8>,
    pub filename: String,
    pub mime_type: String,
    pub created_at: Instant,
}

pub type StagingStore = Arc<RwLock<HashMap<String, StagingEntry>>>;

pub fn new_staging_store() -> StagingStore {
    Arc::new(RwLock::new(HashMap::new()))
}

pub async fn consume_staged(store: &StagingStore, id: &str) -> Option<StagingEntry> {
    let mut map = store.write().await;
    let entry = map.remove(id)?;
    // Don't return empty pre-registered slots
    if entry.bytes.is_empty() {
        None
    } else {
        Some(entry)
    }
}

pub fn cleanup_expired_sync(store: &StagingStore) {
    if let Ok(mut map) = store.try_write() {
        let before = map.len();
        map.retain(|_, entry| entry.created_at.elapsed() < STAGING_TTL);
        let removed = before - map.len();
        if removed > 0 {
            eprintln!("Staging: cleaned up {removed} expired slot(s)");
        }
    }
}

pub async fn handle_stage_upload(
    State(state): State<AppState>,
    Path(staging_id): Path<String>,
    mut multipart: Multipart,
) -> Response {
    // Auth is the staging_id itself — only someone who called prepare_upload
    // (which requires MCP auth) has the UUID. Validate it exists in the store.
    {
        let store = state.staging.read().await;
        if !store.contains_key(&staging_id) {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Unknown staging_id — call prepare_upload first"})),
            )
                .into_response();
        }
    }

    // Extract the file from multipart
    let field = match multipart.next_field().await {
        Ok(Some(field)) => field,
        Ok(None) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "No file field in multipart body"})),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Multipart parse error: {e}")})),
            )
                .into_response();
        }
    };

    let filename = field.file_name().unwrap_or("upload").to_string();
    let mime_type = field
        .content_type()
        .unwrap_or("application/octet-stream")
        .to_string();

    // Read bytes with size limit
    let bytes = match field.bytes().await {
        Ok(b) => {
            if b.len() > MAX_STAGING_SIZE {
                return (
                    StatusCode::PAYLOAD_TOO_LARGE,
                    Json(json!({"error": format!("File exceeds maximum size of {}MB", MAX_STAGING_SIZE / 1024 / 1024)})),
                ).into_response();
            }
            b.to_vec()
        }
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Failed to read file: {e}")})),
            )
                .into_response();
        }
    };

    let size = bytes.len();

    // Replace the pre-registered slot with actual file data
    {
        let mut store = state.staging.write().await;
        store.insert(
            staging_id.clone(),
            StagingEntry {
                bytes,
                filename: filename.clone(),
                mime_type: mime_type.clone(),
                created_at: Instant::now(),
            },
        );
    }

    eprintln!("Staging: stored {staging_id} ({filename}, {mime_type}, {size} bytes)");

    (
        StatusCode::OK,
        Json(json!({
            "staging_id": staging_id,
            "filename": filename,
            "mime_type": mime_type,
            "size": size,
        })),
    )
        .into_response()
}
