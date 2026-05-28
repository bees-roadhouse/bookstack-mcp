//! Vector math utilities for semantic search embeddings.
//! BAAI/bge-large-en-v1.5 produces 1024-dim f32 vectors.
//! ~5000 chunks × 1024 dims × 4 bytes = ~20MB — brute-force cosine scan is <10ms.

pub fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        blob.extend_from_slice(&val.to_le_bytes());
    }
    blob
}

pub fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "embedding dimensions must match");
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Brute-force scan of all chunks, returning (chunk_id, page_id, score) sorted desc.
/// `chunks` is a slice of (chunk_id, page_id, embedding_blob).
pub fn search_embeddings(
    query_embedding: &[f32],
    chunks: &[(i64, i64, Vec<u8>)],
    limit: usize,
    threshold: f32,
) -> Vec<(i64, i64, f32)> {
    let mut results: Vec<(i64, i64, f32)> = chunks
        .iter()
        .filter_map(|(chunk_id, page_id, blob)| {
            let emb = blob_to_embedding(blob);
            let score = cosine_similarity(query_embedding, &emb);
            if score >= threshold {
                Some((*chunk_id, *page_id, score))
            } else {
                None
            }
        })
        .collect();
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_roundtrip() {
        let embedding = vec![1.0f32, -2.5, 0.0, std::f32::consts::PI, f32::MAX, f32::MIN];
        let blob = embedding_to_blob(&embedding);
        let recovered = blob_to_embedding(&blob);
        assert_eq!(embedding, recovered);
    }

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let score = cosine_similarity(&a, &a);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        let score = cosine_similarity(&a, &b);
        assert!((score + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_ordering_and_threshold() {
        let query = vec![1.0, 0.0, 0.0];
        let chunks = vec![
            (1, 10, embedding_to_blob(&[1.0, 0.0, 0.0])), // score ~1.0
            (2, 20, embedding_to_blob(&[0.5, 0.5, 0.0])), // score ~0.707
            (3, 30, embedding_to_blob(&[0.0, 1.0, 0.0])), // score ~0.0
            (4, 40, embedding_to_blob(&[0.9, 0.1, 0.0])), // score ~0.994
        ];
        let results = search_embeddings(&query, &chunks, 10, 0.5);
        assert_eq!(results.len(), 3); // chunk 3 below threshold
        assert_eq!(results[0].0, 1); // highest score first
        assert_eq!(results[1].0, 4);
        assert_eq!(results[2].0, 2);
    }
}
