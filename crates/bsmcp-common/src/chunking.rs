//! Heading-based HTML chunking for semantic search.
//! Splits page HTML into chunks by heading tags, with merge/split post-processing.
//! v2: tighter chunks (1200 chars) with paragraph overlap for better embedding precision.

use sha2::{Digest, Sha256};

/// Chunk format version. Increment when chunking logic changes to trigger re-indexing.
/// v5: added shelf to context prefix (Shelf > Book > Chapter > Page)
pub const CHUNK_VERSION: u32 = 5;

const MAX_CHUNK_LEN: usize = 1200;
const OVERLAP_LEN: usize = 150;

pub struct Chunk {
    pub index: usize,
    pub heading_path: String,
    pub content: String,
    pub content_hash: String,
}

/// Chunk HTML content by heading tags (h1-h3).
/// Each heading starts a new chunk; heading stack tracks nesting.
/// Post-processing merges tiny chunks and splits oversized ones.
///
/// If `page_name` is provided, the first h1 heading is skipped when it matches
/// the page name (BookStack pages typically repeat the title as the first h1).
pub fn chunk_html_with_name(html: &str, page_name: Option<&str>) -> Vec<Chunk> {
    let mut raw_chunks: Vec<(Vec<String>, String)> = Vec::new(); // (heading_stack, content)
    let mut heading_stack: Vec<(u8, String)> = Vec::new(); // (level, text)
    let mut skipped_title = false;
    let mut current_content = String::new();
    let mut in_tag = false;
    let mut tag_buf = String::new();
    let mut collecting_heading: Option<u8> = None;
    let mut heading_text = String::new();

    for ch in html.chars() {
        if ch == '<' {
            in_tag = true;
            tag_buf.clear();
            continue;
        }
        if ch == '>' {
            in_tag = false;
            let tag = tag_buf.trim().to_lowercase();

            // Check for heading open tags (extract tag name before any attributes)
            let tag_name = tag.split_whitespace().next().unwrap_or("");
            let heading_level = match tag_name {
                "h1" => Some(1u8),
                "h2" => Some(2),
                "h3" => Some(3),
                _ => None,
            };
            if let Some(level) = heading_level {
                // Save current chunk before starting new heading
                if !current_content.trim().is_empty() || !raw_chunks.is_empty() {
                    let path: Vec<String> = heading_stack.iter().map(|(_, t)| t.clone()).collect();
                    raw_chunks.push((path, std::mem::take(&mut current_content)));
                }
                collecting_heading = Some(level);
                heading_text.clear();
                continue;
            }

            // Check for heading close tags
            let is_closing_heading = matches!(tag.as_str(), "/h1" | "/h2" | "/h3");
            if is_closing_heading {
                if let Some(level) = collecting_heading.take() {
                    let text = heading_text.trim().to_string();
                    // Skip the first h1 if it matches the page name (avoids duplication)
                    if !skipped_title && level == 1 {
                        if let Some(pname) = page_name {
                            if text.eq_ignore_ascii_case(pname.trim()) {
                                skipped_title = true;
                                continue;
                            }
                        }
                    }
                    // Pop headings at same or deeper level
                    while heading_stack.last().is_some_and(|(l, _)| *l >= level) {
                        heading_stack.pop();
                    }
                    heading_stack.push((level, text));
                }
                continue;
            }

            // Other closing/self-closing tags — add space for word separation
            if tag.starts_with('/') || tag.ends_with('/') {
                if collecting_heading.is_some() {
                    heading_text.push(' ');
                } else {
                    current_content.push(' ');
                }
            }
            continue;
        }
        if in_tag {
            tag_buf.push(ch);
            continue;
        }

        // Collecting text
        if collecting_heading.is_some() {
            heading_text.push(ch);
        } else {
            current_content.push(ch);
        }
    }

    // Push final chunk
    let path: Vec<String> = heading_stack.iter().map(|(_, t)| t.clone()).collect();
    raw_chunks.push((path, current_content));

    // Post-process: merge tiny chunks, split oversized ones
    let mut merged: Vec<(String, String)> = Vec::new(); // (heading_path, content_text)
    for (path, content) in raw_chunks {
        let heading_path = path.join(" > ");
        let text = normalize_whitespace(content.trim());
        if text.is_empty() {
            continue;
        }
        // Merge tiny chunks into previous
        if text.len() < 50 && !merged.is_empty() {
            let last = merged.last_mut().unwrap();
            last.1.push_str("\n\n");
            last.1.push_str(&text);
        } else {
            merged.push((heading_path, text));
        }
    }

    // Split oversized chunks with paragraph overlap
    let mut final_chunks: Vec<Chunk> = Vec::new();
    for (heading_path, text) in merged {
        if text.len() > MAX_CHUNK_LEN {
            let parts = split_with_overlap(&text, MAX_CHUNK_LEN, OVERLAP_LEN);
            for part in parts {
                let full = if heading_path.is_empty() {
                    part.clone()
                } else {
                    format!("{heading_path}\n\n{part}")
                };
                let hash = sha256_hex(&full);
                final_chunks.push(Chunk {
                    index: final_chunks.len(),
                    heading_path: heading_path.clone(),
                    content: full,
                    content_hash: hash,
                });
            }
        } else {
            let full = if heading_path.is_empty() {
                text
            } else {
                format!("{heading_path}\n\n{text}")
            };
            let hash = sha256_hex(&full);
            final_chunks.push(Chunk {
                index: final_chunks.len(),
                heading_path: heading_path.clone(),
                content: full,
                content_hash: hash,
            });
        }
    }

    final_chunks
}

/// Chunk HTML without page name context (backward-compatible).
pub fn chunk_html(html: &str) -> Vec<Chunk> {
    chunk_html_with_name(html, None)
}

/// Extract internal BookStack links from HTML.
/// Matches href="/books/{slug}/page/{slug}" and href="/link/{id}" patterns.
/// Also handles absolute URLs (https://host/books/*/page/*).
pub fn extract_links(html: &str) -> Vec<String> {
    let mut links = Vec::new();
    let mut pos = 0;
    while pos < html.len() {
        if let Some(href_start) = html[pos..].find("href=\"") {
            let abs_start = pos + href_start + 6; // after href="
            if let Some(href_end) = html[abs_start..].find('"') {
                let href = &html[abs_start..abs_start + href_end];
                // Extract the path portion — handles both relative (/books/...) and
                // absolute (https://host/books/...) URLs
                let path = if let Some(idx) = href.find("/books/") {
                    &href[idx..]
                } else if let Some(idx) = href.find("/link/") {
                    &href[idx..]
                } else {
                    pos = abs_start + href_end + 1;
                    continue;
                };
                if (path.starts_with("/books/") && path.contains("/page/"))
                    || path.starts_with("/link/")
                {
                    links.push(path.to_string());
                }
                pos = abs_start + href_end + 1;
            } else {
                pos = abs_start;
            }
        } else {
            break;
        }
    }
    links
}

fn normalize_whitespace(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut prev_was_space = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_was_space {
                result.push(if ch == '\n' { '\n' } else { ' ' });
            }
            prev_was_space = true;
        } else {
            result.push(ch);
            prev_was_space = false;
        }
    }
    result
}

/// Split text at paragraph boundaries with overlap between consecutive parts.
/// Each part after the first includes up to `overlap` chars from the end of the previous part.
fn split_with_overlap(text: &str, max_len: usize, overlap: usize) -> Vec<String> {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut para_indices: Vec<usize> = Vec::new(); // track which paragraphs are in current

    for (i, paragraph) in paragraphs.iter().enumerate() {
        if current.len() + paragraph.len() + 2 > max_len && !current.is_empty() {
            parts.push(std::mem::take(&mut current));

            // Build overlap from trailing paragraphs of previous chunk
            let mut overlap_text = String::new();
            for &pi in para_indices.iter().rev() {
                let candidate = if overlap_text.is_empty() {
                    paragraphs[pi].to_string()
                } else {
                    format!("{}\n\n{overlap_text}", paragraphs[pi])
                };
                if candidate.len() > overlap {
                    break;
                }
                overlap_text = candidate;
            }
            if !overlap_text.is_empty() {
                current = overlap_text;
            }
            para_indices.clear();
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(paragraph);
        para_indices.push(i);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

fn sha256_hex(s: &str) -> String {
    let hash = Sha256::digest(s.as_bytes());
    hex_encode(&hash)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_heading_chunking() {
        let html = "<h1>Title</h1><p>Intro text here.</p><h2>Section A</h2><p>Content for section A with enough text to be a real chunk.</p><h2>Section B</h2><p>Content for section B with enough text to be a real chunk.</p>";
        let chunks = chunk_html(html);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].content.contains("Title"));
    }

    #[test]
    fn test_skip_duplicate_page_title() {
        let html = "<h1>My Page Title</h1><p>Introduction text that is long enough to form a real chunk on its own.</p><h2>Section A</h2><p>Content for section A with enough text to be a real chunk.</p>";
        // Without page name: h1 appears in heading_path
        let chunks = chunk_html(html);
        assert!(chunks
            .iter()
            .any(|c| c.heading_path.contains("My Page Title")));

        // With matching page name: h1 is skipped
        let chunks = chunk_html_with_name(html, Some("My Page Title"));
        assert!(
            !chunks
                .iter()
                .any(|c| c.heading_path.contains("My Page Title")),
            "Page title should be stripped from heading_path when it matches page name"
        );
        assert!(!chunks.is_empty());
        // Section A should still appear
        assert!(chunks.iter().any(|c| c.heading_path.contains("Section A")));
    }

    #[test]
    fn test_merge_tiny_chunks() {
        let html = "<h1>Title</h1><p>Hi</p><h2>Real Section</h2><p>This is a real section with enough content to stand on its own as a chunk.</p>";
        let chunks = chunk_html(html);
        // "Hi" is tiny (<50 chars) and should be merged
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_heading_path_nesting() {
        let html = "<h1>Top</h1><h2>Mid</h2><h3>Deep</h3><p>Content at the deepest level with enough text for a chunk.</p>";
        let chunks = chunk_html(html);
        let last = chunks.last().unwrap();
        assert!(last.heading_path.contains("Top"));
        assert!(last.heading_path.contains("Mid"));
        assert!(last.heading_path.contains("Deep"));
    }

    #[test]
    fn test_extract_links() {
        let html = r#"<a href="/books/tech/page/docker-setup">Docker</a> and <a href="/link/42">link</a> and <a href="https://external.com">ext</a>"#;
        let links = extract_links(html);
        assert_eq!(links.len(), 2);
        assert!(links[0].contains("/books/tech/page/docker-setup"));
        assert!(links[1].contains("/link/42"));
    }

    #[test]
    fn test_extract_links_absolute_urls() {
        let html = r#"<a href="https://kb.example.com/books/tech/page/docker-setup">Docker</a> and <a href="https://kb.example.com/link/42">link</a>"#;
        let links = extract_links(html);
        assert_eq!(links.len(), 2);
        assert_eq!(links[0], "/books/tech/page/docker-setup");
        assert_eq!(links[1], "/link/42");
    }

    #[test]
    fn test_split_oversized() {
        let para = "A".repeat(400);
        let text = format!("{para}\n\n{para}\n\n{para}\n\n{para}");
        let parts = split_with_overlap(&text, MAX_CHUNK_LEN, OVERLAP_LEN);
        assert!(parts.len() >= 2);
        // Parts after the first should start with overlap content
        if parts.len() > 1 {
            // The second part should contain some text from the end of the first
            let _first_end = &parts[0][parts[0].len().saturating_sub(100)..];
            // Overlap is paragraph-based, so second part should be larger than a single paragraph
            assert!(parts[1].len() > para.len());
        }
    }

    #[test]
    fn test_table_heavy_content() {
        // Simulate a BookStack markdown-rendered table (like page 1144)
        let html = r#"<p>Career progression roadmap from T2 to Junior Engineer.</p>
<table>
<thead><tr><th>Skill Area</th><th>Current (T2)</th><th>Target (Junior)</th></tr></thead>
<tbody>
<tr><td>PowerShell</td><td>Basic scripts</td><td>Advanced automation</td></tr>
<tr><td>Networking</td><td>IP basics</td><td>Subnetting, VLANs</td></tr>
<tr><td>Active Directory</td><td>User management</td><td>GPO, DNS integration</td></tr>
<tr><td>Cloud</td><td>Portal navigation</td><td>Azure AD, Intune</td></tr>
</tbody>
</table>"#;
        let chunks = chunk_html(html);
        assert!(
            !chunks.is_empty(),
            "Table-heavy HTML must produce at least one chunk"
        );
        // Verify table content was extracted
        let all_content: String = chunks.iter().map(|c| c.content.clone()).collect();
        assert!(
            all_content.contains("PowerShell"),
            "Table cell content should be extracted"
        );
        assert!(
            all_content.contains("Active Directory"),
            "Table cell content should be extracted"
        );
    }

    #[test]
    fn test_no_headings_content() {
        // Page with no headings at all — all content should go into one chunk
        let html = "<p>This is a page with no headings but enough content to be meaningful. It should produce at least one chunk even without any heading tags.</p>";
        let chunks = chunk_html(html);
        assert!(
            !chunks.is_empty(),
            "Content without headings must produce chunks"
        );
    }

    #[test]
    fn test_content_hash_deterministic() {
        let html = "<h1>Test</h1><p>Some content for hashing that is long enough to not be merged away.</p>";
        let chunks1 = chunk_html(html);
        let chunks2 = chunk_html(html);
        assert_eq!(chunks1.len(), chunks2.len());
        for (a, b) in chunks1.iter().zip(chunks2.iter()) {
            assert_eq!(a.content_hash, b.content_hash);
        }
    }
}
