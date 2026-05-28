use std::collections::HashMap;
use std::env;

use serde_json::{json, Value};

use pulldown_cmark::{html, Options, Parser};

use bsmcp_common::bookstack::{self, BookStackClient, ContentType, ExportFormat};
use crate::semantic::{trim_match, SearchMode, SemanticState};

const PROTOCOL_VERSION: &str = "2025-03-26";

pub async fn handle_request(
    request: &Value,
    client: &BookStackClient,
    semantic: Option<&SemanticState>,
    staging: &crate::staging::StagingStore,
) -> Option<Value> {
    let id = request.get("id");

    match request.get("jsonrpc").and_then(|v| v.as_str()) {
        Some("2.0") => {}
        _ => {
            return Some(json_rpc_error(id, -32600, "Invalid Request: missing or wrong jsonrpc version (must be \"2.0\")"));
        }
    }

    let method = request["method"].as_str().unwrap_or("");
    let params = request.get("params").cloned().unwrap_or(json!({}));

    match method {
        "initialize" => {
            let instructions = build_instructions(client, semantic.is_some()).await;
            Some(json_rpc_result(id, json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "BookStack MCP",
                    "version": env!("CARGO_PKG_VERSION"),
                },
                "instructions": instructions,
            })))
        }
        "notifications/initialized" => None,
        "tools/list" => Some(json_rpc_result(id, json!({ "tools": tool_definitions(semantic.is_some()) }))),
        "tools/call" => {
            let name = params["name"].as_str().unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            let result = execute_tool(name, &args, client, semantic, staging).await;

            let tool_result = match result {
                Ok(text) => json!({
                    "content": [{ "type": "text", "text": text }],
                }),
                Err(e) => json!({
                    "content": [{ "type": "text", "text": format!("Error: {e}") }],
                    "isError": true,
                }),
            };

            Some(json_rpc_result(id, tool_result))
        }
        _ => Some(json_rpc_error(id, -32601, "Method not found")),
    }
}

fn json_rpc_result(id: Option<&Value>, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id.cloned().unwrap_or(Value::Null),
        "result": result,
    })
}

fn json_rpc_error(id: Option<&Value>, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id.cloned().unwrap_or(Value::Null),
        "error": { "code": code, "message": message },
    })
}

async fn execute_tool(
    name: &str,
    args: &Value,
    client: &BookStackClient,
    semantic: Option<&SemanticState>,
    staging: &crate::staging::StagingStore,
) -> Result<String, String> {
    match name {
        // Semantic Search (conditional)
        "semantic_search" => {
            let sem = semantic.ok_or("Semantic search is not enabled")?;
            let query = arg_str(args, "query")?;
            let limit = arg_i64(args, "limit", 10).clamp(1, 50) as usize;
            let hybrid = args.get("hybrid").and_then(|v| v.as_bool()).unwrap_or(true);
            let default_threshold = if hybrid { 0.45 } else { 0.50 };
            let threshold = args.get("threshold").and_then(|v| v.as_f64()).unwrap_or(default_threshold) as f32;
            let verbose = args.get("verbose").and_then(|v| v.as_bool()).unwrap_or(false);
            let mode_str = args
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("standard");
            let mode = SearchMode::parse(mode_str).ok_or_else(|| {
                format!(
                    "invalid mode '{mode_str}' (expected: standard, rerank, precision)"
                )
            })?;
            // The HTTP `filter_by_permission` fallback inside `sem.search`
            // enforces per-page access control via BookStack's API.
            let mut result = sem
                .search(&query, limit, threshold, hybrid, verbose, client, None, mode)
                .await?;
            trim_semantic_search_payload(&mut result);
            format_json(&result)
        }
        "reembed" => {
            let sem = semantic.ok_or("Semantic search is not enabled")?;
            let scope = arg_str_default(args, "scope", "all");
            let result = sem.trigger_reembed(&scope).await?;
            format_json(&result)
        }
        "embedding_status" => {
            let sem = semantic.ok_or("Semantic search is not enabled")?;
            let result = sem.embedding_status().await?;
            format_json(&result)
        }

        // Search
        "search_content" => {
            let query = arg_str(args, "query")?;
            let page = arg_i64(args, "page", 1).max(1);
            let count = arg_count(args, 20);
            let result = client.search(&query, page, count).await?;
            Ok(format_search_results(&result, client.base_url()))
        }

        // Shelves
        "list_shelves" => {
            let count = arg_count(args, 50);
            let offset = arg_offset(args);
            format_json(&client.list_shelves(count, offset).await?)
        }
        "get_shelf" => {
            let id = arg_i64_required(args, "shelf_id")?;
            format_json(&client.get_shelf(id).await?)
        }
        "create_shelf" => {
            let name = arg_str(args, "name")?;
            let desc = require_description(args, "shelf")?;
            let result = client.create_shelf(&name, &desc).await?;
            Ok(format_shelf_success("Shelf created successfully.", &result, client.base_url()))
        }
        "update_shelf" => {
            let id = arg_i64_required(args, "shelf_id")?;
            let mut data = filter_string_update_fields(args, &["name", "description"]);
            if let Some(books) = args.get("books").and_then(|v| v.as_array()) {
                data["books"] = json!(books.iter().filter_map(|v| v.as_i64()).collect::<Vec<_>>());
            }
            let result = client.update_shelf(id, &data).await?;
            Ok(format_shelf_success("Shelf updated successfully.", &result, client.base_url()))
        }
        "delete_shelf" => {
            let id = arg_i64_required(args, "shelf_id")?;
            client.delete_shelf(id).await?;
            Ok(format!("Shelf {id} deleted."))
        }

        // Books
        "list_books" => {
            let count = arg_count(args, 50);
            let offset = arg_offset(args);
            format_json(&client.list_books(count, offset).await?)
        }
        "get_book" => {
            let id = arg_i64_required(args, "book_id")?;
            format_json(&client.get_book(id).await?)
        }
        "create_book" => {
            let name = arg_str(args, "name")?;
            let desc = require_description(args, "book")?;
            let result = client.create_book(&name, &desc).await?;
            Ok(format_book_success("Book created successfully.", &result, client.base_url()))
        }
        "update_book" => {
            let id = arg_i64_required(args, "book_id")?;
            let data = filter_string_update_fields(args, &["name", "description"]);
            let result = client.update_book(id, &data).await?;
            Ok(format_book_success("Book updated successfully.", &result, client.base_url()))
        }
        "delete_book" => {
            let id = arg_i64_required(args, "book_id")?;
            client.delete_book(id).await?;
            Ok(format!("Book {id} deleted."))
        }

        // Chapters
        "list_chapters" => {
            let count = arg_count(args, 50);
            let offset = arg_offset(args);
            format_json(&client.list_chapters(count, offset).await?)
        }
        "get_chapter" => {
            let id = arg_i64_required(args, "chapter_id")?;
            format_json(&client.get_chapter(id).await?)
        }
        "create_chapter" => {
            let book_id = arg_i64_required(args, "book_id")?;
            let name = arg_str(args, "name")?;
            let desc = require_description(args, "chapter")?;
            let result = client.create_chapter(book_id, &name, &desc).await?;
            Ok(format_chapter_success("Chapter created successfully.", &result, client.base_url()))
        }
        "update_chapter" => {
            let id = arg_i64_required(args, "chapter_id")?;
            let mut data = filter_string_update_fields(args, &["name", "description"]);
            if let Some(v) = arg_i64_opt(args, "book_id") {
                data["book_id"] = json!(v);
            }
            let result = client.update_chapter(id, &data).await?;
            Ok(format_chapter_success("Chapter updated successfully.", &result, client.base_url()))
        }
        "delete_chapter" => {
            let id = arg_i64_required(args, "chapter_id")?;
            client.delete_chapter(id).await?;
            Ok(format!("Chapter {id} deleted."))
        }

        // Pages
        "list_pages" => {
            let count = arg_count(args, 50);
            let offset = arg_offset(args);
            format_json(&client.list_pages(count, offset).await?)
        }
        "get_page" => {
            let id = arg_i64_required(args, "page_id")?;
            format_json(&client.get_page(id).await?)
        }
        "create_page" => {
            let mut data = json!({ "name": arg_str(args, "name")? });
            if let Some(v) = arg_i64_opt(args, "chapter_id") {
                data["chapter_id"] = json!(v);
            } else if let Some(v) = arg_i64_opt(args, "book_id") {
                data["book_id"] = json!(v);
            } else {
                return Err("Either book_id or chapter_id is required".to_string());
            }
            let page_name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if let Some(md) = args.get("markdown").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                data["markdown"] = json!(strip_duplicate_title(md, page_name));
            } else if let Some(v) = args.get("html").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                data["html"] = json!(strip_duplicate_title(v, page_name));
            }
            let result = client.create_page(&data).await?;
            Ok(format_page_success("Page created successfully.", &result, client.base_url()))
        }
        "update_page" => {
            let id = arg_i64_required(args, "page_id")?;
            let mut data = json!({});
            let has_content = args.get("markdown").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).is_some()
                || args.get("html").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).is_some();
            // Get the page name for duplicate title stripping
            let page_name = if let Some(n) = args.get("name").and_then(|v| v.as_str()) {
                n.to_string()
            } else if has_content {
                // Fetch current name so we can strip duplicate H1
                client.get_page(id).await?
                    .get("name").and_then(|v| v.as_str()).unwrap_or("").to_string()
            } else {
                String::new()
            };
            if let Some(v) = args.get("name").and_then(|v| v.as_str()) {
                data["name"] = json!(v);
            }
            if let Some(md) = args.get("markdown").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                data["markdown"] = json!(strip_duplicate_title(md, &page_name));
            } else if let Some(v) = args.get("html").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                data["html"] = json!(strip_duplicate_title(v, &page_name));
            }
            let move_chapter_id = arg_i64_opt(args, "chapter_id");
            let move_book_id = arg_i64_opt(args, "book_id");
            if move_chapter_id.is_some() && move_book_id.is_some() {
                return Err("Provide either chapter_id or book_id, not both".to_string());
            }
            if let Some(v) = move_chapter_id {
                data["chapter_id"] = json!(v);
            }
            if let Some(v) = move_book_id {
                data["book_id"] = json!(v);
            }
            let result = client.update_page(id, &data).await?;
            Ok(format_page_success("Page updated successfully.", &result, client.base_url()))
        }
        "edit_page" => {
            let id = arg_i64_required(args, "page_id")?;
            let old_text = args.get("old_text").and_then(|v| v.as_str())
                .ok_or("old_text is required")?;
            let new_text = args.get("new_text").and_then(|v| v.as_str())
                .ok_or("new_text is required")?;
            let replace_all = args.get("replace_all").and_then(|v| v.as_bool()).unwrap_or(false);

            // Fetch page in its native format
            let (editor, native_content) = get_page_content(client, id).await?;

            // Validate old_text exists in native content
            let count = native_content.matches(old_text).count();
            if count == 0 {
                return Err(format!("old_text not found in page {id}. This page uses the '{editor}' editor — make sure old_text matches the '{}' field from get_page.", if editor == "markdown" { "markdown" } else { "html" }));
            }
            if count > 1 && !replace_all {
                return Err(format!("old_text found {count} times in page {id}. Use replace_all=true to replace all, or provide more context to make it unique."));
            }

            // Apply replacement
            let updated = if replace_all {
                native_content.replace(old_text, new_text)
            } else {
                native_content.replacen(old_text, new_text, 1)
            };

            let data = if editor == "markdown" {
                json!({ "markdown": updated })
            } else {
                json!({ "html": updated })
            };
            let result = client.update_page(id, &data).await?;
            Ok(format_page_success("Page updated successfully.", &result, client.base_url()))
        }
        "append_to_page" => {
            let id = arg_i64_required(args, "page_id")?;
            let content = args.get("markdown").and_then(|v| v.as_str())
                .ok_or("markdown is required")?;
            let (editor, existing) = get_page_content(client, id).await?;

            let data = if editor == "markdown" {
                let updated = format!("{}\n\n{}", existing.trim_end(), content);
                json!({ "markdown": updated })
            } else {
                let html_content = markdown_to_html(content);
                let updated = format!("{}\n{}", existing.trim_end(), html_content);
                json!({ "html": updated })
            };
            let result = client.update_page(id, &data).await?;
            Ok(format_page_success("Content appended successfully.", &result, client.base_url()))
        }
        "replace_section" => {
            let id = arg_i64_required(args, "page_id")?;
            let heading = args.get("heading").and_then(|v| v.as_str())
                .ok_or("heading is required")?;
            let content = args.get("markdown").and_then(|v| v.as_str())
                .ok_or("markdown is required")?;
            let (editor, existing) = get_page_content(client, id).await?;

            let data = if editor == "markdown" {
                let updated = replace_section_markdown(&existing, heading, content, id)?;
                json!({ "markdown": updated })
            } else {
                let html_content = markdown_to_html(content);
                let updated = replace_section_html(&existing, heading, &html_content, id)?;
                json!({ "html": updated })
            };
            let result = client.update_page(id, &data).await?;
            Ok(format_page_success("Section replaced successfully.", &result, client.base_url()))
        }
        "insert_after" => {
            let id = arg_i64_required(args, "page_id")?;
            let after = args.get("after").and_then(|v| v.as_str())
                .ok_or("after is required")?;
            let content = args.get("markdown").and_then(|v| v.as_str())
                .ok_or("markdown is required")?;
            let (editor, existing) = get_page_content(client, id).await?;

            // Find the anchor — match by line content (trimmed)
            let lines: Vec<&str> = existing.lines().collect();
            let pos = lines.iter().position(|line| line.trim() == after.trim())
                .ok_or(format!("Anchor '{}' not found in page {id}. This page uses the '{editor}' editor — make sure the anchor matches a line from the '{}' field.", after, if editor == "markdown" { "markdown" } else { "html" }))?;

            let insert_content = if editor == "markdown" {
                content.to_string()
            } else {
                markdown_to_html(content)
            };

            // Insert after the matched line
            let mut updated = lines[..=pos].join("\n");
            updated.push('\n');
            updated.push_str(&insert_content);
            updated.push('\n');
            if pos + 1 < lines.len() {
                updated.push_str(&lines[pos + 1..].join("\n"));
            }

            let data = if editor == "markdown" {
                json!({ "markdown": updated })
            } else {
                json!({ "html": updated })
            };
            let result = client.update_page(id, &data).await?;
            Ok(format_page_success("Content inserted successfully.", &result, client.base_url()))
        }
        "delete_page" => {
            let id = arg_i64_required(args, "page_id")?;
            client.delete_page(id).await?;
            Ok(format!("Page {id} deleted."))
        }

        // Move operations
        "move_page" => {
            let id = arg_i64_required(args, "page_id")?;
            let chapter_id = arg_i64_opt(args, "chapter_id");
            let book_id = arg_i64_opt(args, "book_id");
            if chapter_id.is_none() && book_id.is_none() {
                return Err("Either chapter_id or book_id is required".to_string());
            }
            if chapter_id.is_some() && book_id.is_some() {
                return Err("Provide either chapter_id or book_id, not both".to_string());
            }
            let mut data = json!({});
            if let Some(v) = chapter_id {
                data["chapter_id"] = json!(v);
            }
            if let Some(v) = book_id {
                data["book_id"] = json!(v);
            }
            let result = client.update_page(id, &data).await?;
            Ok(format_page_success("Page moved successfully.", &result, client.base_url()))
        }
        "move_chapter" => {
            let id = arg_i64_required(args, "chapter_id")?;
            let book_id = arg_i64_required(args, "target_book_id")?;
            let data = json!({ "book_id": book_id });
            let result = client.update_chapter(id, &data).await?;
            Ok(format_chapter_success("Chapter moved successfully.", &result, client.base_url()))
        }
        // Note: This uses a GET-modify-PUT pattern which has a TOCTOU race if multiple
        // concurrent sessions modify the same shelf simultaneously. Acceptable for
        // single-user deployments; a per-shelf mutex would be needed for multi-user.
        "move_book_to_shelf" => {
            let book_id = arg_i64_required(args, "book_id")?;
            let target_shelf_id = arg_i64_required(args, "target_shelf_id")?;
            let remove_from_shelf_id = arg_i64_opt(args, "remove_from_shelf_id");

            // Add book to target shelf
            let target_shelf = client.get_shelf(target_shelf_id).await?;
            let mut target_books: Vec<i64> = target_shelf.get("books")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|b| b.get("id").and_then(|id| id.as_i64())).collect())
                .unwrap_or_default();
            if !target_books.contains(&book_id) {
                target_books.push(book_id);
            }
            client.update_shelf(target_shelf_id, &json!({ "books": target_books })).await?;

            // Remove from source shelf if specified
            let mut removed_from = String::new();
            if let Some(source_id) = remove_from_shelf_id {
                let source_shelf = client.get_shelf(source_id).await?;
                let source_name = source_shelf.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let source_books: Vec<i64> = source_shelf.get("books")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|b| b.get("id").and_then(|id| id.as_i64())).filter(|&id| id != book_id).collect())
                    .unwrap_or_default();
                client.update_shelf(source_id, &json!({ "books": source_books })).await?;
                removed_from = format!("\nRemoved from shelf: {} (ID: {})", source_name, source_id);
            }

            let target_name = target_shelf.get("name").and_then(|v| v.as_str()).unwrap_or("");
            Ok(format!("Book {book_id} moved to shelf \"{target_name}\" (ID: {target_shelf_id}).{removed_from}"))
        }

        // Attachments
        "list_attachments" => {
            format_json(&client.list_attachments().await?)
        }
        "get_attachment" => {
            let id = arg_i64_required(args, "attachment_id")?;
            format_json(&client.get_attachment(id).await?)
        }
        "create_attachment" => {
            let mut data = json!({
                "name": arg_str(args, "name")?,
                "uploaded_to": arg_i64_required(args, "uploaded_to")?,
            });
            if let Some(v) = args.get("link").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                data["link"] = json!(v);
            }
            format_json(&client.create_attachment(&data).await?)
        }
        "update_attachment" => {
            let id = arg_i64_required(args, "attachment_id")?;
            let data = filter_string_update_fields(args, &["name", "link"]);
            format_json(&client.update_attachment(id, &data).await?)
        }
        "delete_attachment" => {
            let id = arg_i64_required(args, "attachment_id")?;
            client.delete_attachment(id).await?;
            Ok(format!("Attachment {id} deleted."))
        }
        "upload_attachment" => {
            let name = arg_str(args, "name")?;
            let uploaded_to = arg_i64_required(args, "uploaded_to")?;
            let staging_id = args.get("staging_id").and_then(|v| v.as_str());
            let url = args.get("url").and_then(|v| v.as_str());
            let (bytes, auto_filename, resolved_mime) = if let Some(sid) = staging_id {
                let entry = crate::staging::consume_staged(staging, sid).await
                    .ok_or_else(|| format!("Staging slot '{}' not found or already consumed (slots expire after 5 minutes)", sid))?;
                (entry.bytes, entry.filename, entry.mime_type)
            } else if let Some(u) = url {
                let (b, f) = bookstack::resolve_file_content(None, Some(u)).await
                    .map_err(|e| e.to_string())?;
                (b, f, "application/octet-stream".to_string())
            } else {
                return Err("Either staging_id or url is required. Use prepare_upload to stage local files.".to_string());
            };
            let mime_type = arg_str_default(args, "mime_type", &resolved_mime);
            let filename = match args.get("filename").and_then(|v| v.as_str()) {
                Some(f) if !f.is_empty() => f.to_string(),
                _ => auto_filename,
            };
            format_json(&client.create_file_attachment(&name, uploaded_to, &filename, bytes, &mime_type).await?)
        }

        // Exports
        "export_page" => {
            let id = arg_i64_required(args, "page_id")?;
            let fmt = ExportFormat::parse_str(&arg_str_default(args, "format", "markdown"))?;
            client.export_page(id, fmt).await
        }
        "export_chapter" => {
            let id = arg_i64_required(args, "chapter_id")?;
            let fmt = ExportFormat::parse_str(&arg_str_default(args, "format", "markdown"))?;
            client.export_chapter(id, fmt).await
        }
        "export_book" => {
            let id = arg_i64_required(args, "book_id")?;
            let fmt = ExportFormat::parse_str(&arg_str_default(args, "format", "markdown"))?;
            client.export_book(id, fmt).await
        }

        // Comments
        "list_comments" => {
            let mut query: Vec<(&str, &str)> = vec![];
            let page_id_str;
            if let Some(v) = arg_i64_opt(args, "page_id") {
                page_id_str = v.to_string();
                query.push(("filter[page_id]", &page_id_str));
            }
            format_json(&client.list_comments(&query).await?)
        }
        "get_comment" => {
            let id = arg_i64_required(args, "comment_id")?;
            format_json(&client.get_comment(id).await?)
        }
        "create_comment" => {
            let mut data = json!({
                "page_id": arg_i64_required(args, "page_id")?,
            });
            if let Some(md) = args.get("markdown").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                data["html"] = json!(markdown_to_html(md));
            } else if let Some(v) = args.get("html").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                data["html"] = json!(v);
            }
            if let Some(v) = arg_i64_opt(args, "parent_id") {
                data["parent_id"] = json!(v);
            }
            format_json(&client.create_comment(&data).await?)
        }
        "update_comment" => {
            let id = arg_i64_required(args, "comment_id")?;
            let mut data = json!({});
            if let Some(md) = args.get("markdown").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                data["html"] = json!(markdown_to_html(md));
            } else if let Some(v) = args.get("html").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                data["html"] = json!(v);
            }
            format_json(&client.update_comment(id, &data).await?)
        }
        "delete_comment" => {
            let id = arg_i64_required(args, "comment_id")?;
            client.delete_comment(id).await?;
            Ok(format!("Comment {id} deleted."))
        }

        // Recycle Bin
        "list_recycle_bin" => {
            let count = arg_count(args, 50);
            let offset = arg_offset(args);
            format_json(&client.list_recycle_bin(count, offset).await?)
        }
        "restore_recycle_bin_item" => {
            let id = arg_i64_required(args, "deletion_id")?;
            format_json(&client.restore_recycle_bin_item(id).await?)
        }
        "destroy_recycle_bin_item" => {
            let id = arg_i64_required(args, "deletion_id")?;
            client.destroy_recycle_bin_item(id).await?;
            Ok(format!("Recycle bin item {id} permanently deleted."))
        }

        // Users
        "list_users" => {
            let count = arg_count(args, 50);
            let offset = arg_offset(args);
            format_json(&client.list_users(count, offset).await?)
        }
        "get_user" => {
            let id = arg_i64_required(args, "user_id")?;
            format_json(&client.get_user(id).await?)
        }

        // Audit Log
        "list_audit_log" => {
            let count = arg_count(args, 50);
            let offset = arg_offset(args);
            format_json(&client.list_audit_log(count, offset).await?)
        }

        // System
        "get_system_info" => {
            format_json(&client.get_system_info().await?)
        }

        // Image Gallery
        "list_images" => {
            let count = arg_count(args, 50);
            let offset = arg_offset(args);
            let mut filter: Vec<(&str, &str)> = vec![];
            let type_str;
            if let Some(v) = args.get("type").and_then(|v| v.as_str()) {
                validate_enum(v, &["gallery", "drawio"], "type")?;
                type_str = v.to_string();
                filter.push(("filter[type]", &type_str));
            }
            let uploaded_to_str;
            if let Some(v) = arg_i64_opt(args, "uploaded_to") {
                uploaded_to_str = v.to_string();
                filter.push(("filter[uploaded_to]", &uploaded_to_str));
            }
            format_json(&client.list_images(count, offset, &filter).await?)
        }
        "get_image" => {
            let id = arg_i64_required(args, "image_id")?;
            format_json(&client.get_image(id).await?)
        }
        "update_image" => {
            let id = arg_i64_required(args, "image_id")?;
            let data = filter_string_update_fields(args, &["name"]);
            format_json(&client.update_image(id, &data).await?)
        }
        "delete_image" => {
            let id = arg_i64_required(args, "image_id")?;
            client.delete_image(id).await?;
            Ok(format!("Image {id} deleted."))
        }
        "upload_image" => {
            let name = arg_str(args, "name")?;
            let image_type = arg_str_default(args, "type", "gallery");
            validate_enum(&image_type, &["gallery", "drawio"], "type")?;
            let uploaded_to = arg_i64_required(args, "uploaded_to")?;
            let embed = arg_bool(args, "embed", false);
            let staging_id = args.get("staging_id").and_then(|v| v.as_str());
            let url = args.get("url").and_then(|v| v.as_str());
            let (bytes, auto_filename, resolved_mime) = if let Some(sid) = staging_id {
                let entry = crate::staging::consume_staged(staging, sid).await
                    .ok_or_else(|| format!("Staging slot '{}' not found or already consumed (slots expire after 5 minutes)", sid))?;
                (entry.bytes, entry.filename, entry.mime_type)
            } else if let Some(u) = url {
                let (b, f) = bookstack::resolve_file_content(None, Some(u)).await
                    .map_err(|e| e.to_string())?;
                (b, f, "image/png".to_string())
            } else {
                return Err("Either staging_id or url is required. Use prepare_upload to stage local files.".to_string());
            };
            let mime_type = arg_str_default(args, "mime_type", &resolved_mime);
            let filename = match args.get("filename").and_then(|v| v.as_str()) {
                Some(f) if !f.is_empty() => f.to_string(),
                _ => auto_filename,
            };
            let result = client.upload_image(&name, &image_type, uploaded_to, &filename, bytes, &mime_type).await?;

            if embed {
                let display_url = result.get("thumbs")
                    .and_then(|t| t.get("display"))
                    .and_then(|v| v.as_str())
                    .or_else(|| result.get("url").and_then(|v| v.as_str()))
                    .unwrap_or("");
                let alt_text = result.get("name").and_then(|v| v.as_str()).unwrap_or(&name);
                let img_markdown = format!("![{}]({})", alt_text, display_url);

                let (editor, existing) = get_page_content(client, uploaded_to).await?;
                let data = if editor == "markdown" {
                    let updated = format!("{}\n\n{}", existing.trim_end(), img_markdown);
                    json!({ "markdown": updated })
                } else {
                    let html_content = markdown_to_html(&img_markdown);
                    let updated = format!("{}\n{}", existing.trim_end(), html_content);
                    json!({ "html": updated })
                };
                client.update_page(uploaded_to, &data).await?;
            }

            format_json(&result)
        }
        "prepare_upload" => {
            let staging_id = uuid::Uuid::new_v4().to_string();
            let base_url = env::var("BSMCP_PUBLIC_DOMAIN")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .map(|s| format!("https://{}", s.trim().trim_end_matches('/')))
                .unwrap_or_default();
            let upload_url = if base_url.is_empty() {
                format!("/stage/upload/{staging_id}")
            } else {
                format!("{base_url}/stage/upload/{staging_id}")
            };
            // Pre-register the slot so the staging_id acts as auth
            {
                let mut store = staging.write().await;
                store.insert(staging_id.clone(), crate::staging::StagingEntry {
                    bytes: Vec::new(),
                    filename: String::new(),
                    mime_type: String::new(),
                    created_at: std::time::Instant::now(),
                });
            }
            format_json(&json!({
                "staging_id": staging_id,
                "upload_url": upload_url,
                "instructions": "POST a multipart/form-data request with a 'file' field to the upload_url. No authorization header needed. Then pass the staging_id to upload_image or upload_attachment.",
                "ttl_seconds": 300
            }))
        }

        // Content Permissions
        "get_content_permissions" => {
            let content_type = ContentType::parse_str(&arg_str(args, "content_type")?)?;
            let content_id = arg_i64_required(args, "content_id")?;
            format_json(&client.get_content_permissions(content_type, content_id).await?)
        }
        "update_content_permissions" => {
            let content_type = ContentType::parse_str(&arg_str(args, "content_type")?)?;
            let content_id = arg_i64_required(args, "content_id")?;
            let data = filter_update_fields(args, &["owner_id", "role_permissions", "fallback_permissions"]);
            format_json(&client.update_content_permissions(content_type, content_id, &data).await?)
        }

        // Roles
        "list_roles" => {
            let count = arg_count(args, 50);
            let offset = arg_offset(args);
            format_json(&client.list_roles(count, offset).await?)
        }
        "get_role" => {
            let id = arg_i64_required(args, "role_id")?;
            format_json(&client.get_role(id).await?)
        }

        _ => Err(format!("Unknown tool: {name}")),
    }
}

// --- Arg helpers ---

fn arg_str(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("Missing required argument: {key}"))
}

fn arg_str_default(args: &Value, key: &str, default: &str) -> String {
    args.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}

/// Extract an integer from a JSON value, accepting both native numbers and
/// numeric strings (e.g. `1908` or `"1908"`). AI clients commonly serialize
/// IDs as strings and the server should accept both forms.
fn value_as_i64(v: &Value) -> Option<i64> {
    if let Some(n) = v.as_i64() {
        return Some(n);
    }
    if let Some(s) = v.as_str() {
        return s.trim().parse::<i64>().ok();
    }
    None
}

fn arg_i64_opt(args: &Value, key: &str) -> Option<i64> {
    args.get(key).and_then(value_as_i64)
}

fn arg_i64(args: &Value, key: &str, default: i64) -> i64 {
    arg_i64_opt(args, key).unwrap_or(default)
}

fn arg_count(args: &Value, default: i64) -> i64 {
    arg_i64(args, "count", default).clamp(1, 500)
}

fn arg_offset(args: &Value) -> i64 {
    arg_i64(args, "offset", 0).max(0)
}

fn arg_i64_required(args: &Value, key: &str) -> Result<i64, String> {
    arg_i64_opt(args, key)
        .ok_or_else(|| format!("Missing required argument: {key}"))
}

fn arg_bool(args: &Value, key: &str, default: bool) -> bool {
    args.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

/// Join a path/URL fragment from a BookStack API response with the base URL.
/// If the fragment is already absolute (http:// or https://), return it as-is
/// to avoid producing malformed URLs like `http://bookstack-apphttps://kb.example.com/...`.
fn join_base_url(base_url: &str, path: &str) -> String {
    if path.starts_with("http://") || path.starts_with("https://") {
        path.to_string()
    } else {
        format!("{base_url}{path}")
    }
}

/// Require a non-empty, meaningful description when creating shelves/books/chapters.
/// Descriptions are surfaced to AI clients in the server's structure listing on connect,
/// so missing or placeholder descriptions actively degrade future routing decisions.
fn require_description(args: &Value, kind: &str) -> Result<String, String> {
    let raw = args.get("description").and_then(|v| v.as_str()).unwrap_or("").trim();
    if raw.is_empty() {
        return Err(format!(
            "description is required when creating a {kind}. \
             Descriptions are surfaced to all Claude clients that connect to this BookStack, \
             so they shape placement decisions for every future page created here. \
             Provide a 1-2 sentence description that answers (1) what kind of content lives in \
             this {kind}, and (2) what it's for. Avoid placeholders like 'TODO' or 'description'."
        ));
    }
    if raw.len() < 15 {
        return Err(format!(
            "description is too short ({} chars) — write a meaningful 1-2 sentence description \
             that tells future clients what content belongs in this {kind} and what it's for.",
            raw.len()
        ));
    }
    let lowered = raw.to_lowercase();
    let placeholders = ["todo", "tbd", "placeholder", "description", "xxx", "fixme", "n/a"];
    if placeholders.iter().any(|p| lowered == *p || lowered.starts_with(&format!("{p} "))) {
        return Err(format!(
            "description looks like a placeholder ('{raw}'). Write a real description that \
             describes the {kind}'s purpose and contents — it will be shown to every future \
             Claude client that connects."
        ));
    }
    Ok(raw.to_string())
}

fn validate_enum(value: &str, allowed: &[&str], name: &str) -> Result<(), String> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(format!("Invalid {name}: '{value}'. Must be one of: {}", allowed.join(", ")))
    }
}

fn filter_update_fields(args: &Value, fields: &[&str]) -> Value {
    let mut data = json!({});
    for &field in fields {
        if let Some(v) = args.get(field) {
            if !v.is_null() {
                data[field] = v.clone();
            }
        }
    }
    data
}

fn filter_string_update_fields(args: &Value, fields: &[&str]) -> Value {
    let mut data = json!({});
    for &field in fields {
        if let Some(v) = args.get(field) {
            if v.is_string() {
                data[field] = v.clone();
            }
        }
    }
    data
}

fn format_json(v: &Value) -> Result<String, String> {
    serde_json::to_string_pretty(v).map_err(|e| e.to_string())
}

/// `semantic_search` MCP-tool payload trim. Caps each result's chunks and
/// truncates each chunk's content so a wide query doesn't blow past Claude
/// Code's response-size budget (which would force the response to spill to a
/// local file). Slightly more generous than the briefing's per-section trim
/// because the caller asked for these results explicitly and gets one shot at
/// them; the briefing pulls every session and amortizes across many tools.
///
/// Truncation logic itself lives in `crate::semantic::trim_match` — this
/// function only owns the budget and the response-envelope hint.
const SEMANTIC_SEARCH_CHUNK_LIMIT: usize = 5;
const SEMANTIC_SEARCH_CHUNK_CHARS: usize = 200;
const SEMANTIC_SEARCH_HINT: &str =
    "Each result returns up to 5 chunks of ~200 chars (truncated chunks have `truncated: true` and end with …). \
     These are search-result previews, not full page content — call `get_page(page_id)` to read the full markdown when a match looks relevant.";

fn trim_semantic_search_payload(payload: &mut Value) {
    let Some(obj) = payload.as_object_mut() else { return; };
    if let Some(results) = obj.get_mut("results").and_then(|v| v.as_array_mut()) {
        for result in results.iter_mut() {
            // Drop into the shared helper. take() leaves Value::Null in the slot;
            // we immediately overwrite it with the trimmed result so no consumer
            // ever sees the placeholder.
            let owned = std::mem::take(result);
            *result = trim_match(owned, SEMANTIC_SEARCH_CHUNK_LIMIT, SEMANTIC_SEARCH_CHUNK_CHARS);
        }
    }
    obj.insert("hint".to_string(), Value::String(SEMANTIC_SEARCH_HINT.to_string()));
}

fn format_search_results(data: &Value, base_url: &str) -> String {
    let results = data.get("data").and_then(|v| v.as_array());
    let total = data.get("total").and_then(|v| v.as_i64()).unwrap_or(0);

    let Some(results) = results else {
        return "No results found.".into();
    };

    if results.is_empty() {
        return "No results found.".into();
    }

    let mut lines = vec![format!("Found {total} results:\n")];
    for item in results {
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let id = item.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
        let url = item.get("url").and_then(|v| v.as_str())
            .map(|u| join_base_url(base_url, u))
            .unwrap_or_default();
        if url.is_empty() {
            lines.push(format!("- [{item_type}] {name} (id: {id})"));
        } else {
            lines.push(format!("- [{item_type}] {name} (id: {id}) — {url}"));
        }

        if let Some(preview) = item.get("preview_html") {
            let raw = if let Some(content) = preview.get("content").and_then(|v| v.as_str()) {
                content.to_string()
            } else if let Some(s) = preview.as_str() {
                s.to_string()
            } else {
                String::new()
            };
            if !raw.is_empty() {
                let clean = strip_html_tags(&raw);
                let truncated: String = clean.chars().take(200).collect();
                lines.push(format!("  Preview: {truncated}"));
            }
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

fn strip_html_tags(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }
    result
}

/// Strip a leading H1 heading from content if it matches the page name.
/// BookStack automatically renders the page name as H1, so including it in
/// content causes a duplicate title. Handles both markdown (`# Title`) and
/// HTML (`<h1>Title</h1>`).
fn strip_duplicate_title(content: &str, page_name: &str) -> String {
    let trimmed = content.trim_start();

    // Markdown: lines starting with "# Title"
    if let Some(rest) = trimmed.strip_prefix('#') {
        // Make sure it's H1 (not ## or ###)
        if !rest.starts_with('#') {
            let heading_text = rest.trim();
            // Check first line only
            let first_line = heading_text.lines().next().unwrap_or("");
            if first_line.trim().eq_ignore_ascii_case(page_name.trim()) {
                // Remove the H1 line and any immediately following blank lines
                let after_heading = heading_text.strip_prefix(first_line).unwrap_or("");
                return after_heading.trim_start_matches('\n').trim_start_matches('\r').to_string();
            }
        }
    }

    // HTML: <h1>Title</h1> or <h1 ...>Title</h1>
    if trimmed.starts_with("<h1") {
        if let Some(close_pos) = trimmed.find("</h1>") {
            let tag_content = &trimmed[..close_pos + 5]; // include </h1>
            let text = strip_html_tags(tag_content);
            if text.trim().eq_ignore_ascii_case(page_name.trim()) {
                let after = &trimmed[close_pos + 5..];
                return after.trim_start_matches('\n').trim_start_matches('\r').to_string();
            }
        }
    }

    content.to_string()
}

/// Truncate a description to a reasonable length for the structure tree.
/// Strips HTML tags, collapses whitespace, and caps at 150 chars.
fn truncate_desc(desc: &str) -> String {
    let clean = strip_html_tags(desc);
    // Collapse whitespace and newlines into single spaces
    let collapsed: String = clean.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.len() <= 150 {
        collapsed
    } else {
        let truncated: String = collapsed.chars().take(147).collect();
        format!("{truncated}...")
    }
}

fn markdown_to_html(md: &str) -> String {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TASKLISTS);
    let parser = Parser::new_ext(md, opts);
    let mut out = String::new();
    html::push_html(&mut out, parser);
    out
}

/// Fetch page and return (editor_type, native_content).
/// For markdown pages: returns ("markdown", markdown_source).
/// For WYSIWYG pages: returns ("wysiwyg", html_content).
async fn get_page_content(client: &BookStackClient, id: i64) -> Result<(String, String), String> {
    let page = client.get_page(id).await?;
    let editor = page.get("editor")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if editor == "markdown" {
        let content = page.get("markdown")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Ok(("markdown".to_string(), content))
    } else {
        // "wysiwyg" or "" (system default) — use HTML
        let content = page.get("html")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Ok(("wysiwyg".to_string(), content))
    }
}

/// Slim success response for page create/update operations.
fn format_page_success(action: &str, result: &Value, base_url: &str) -> String {
    let id = result.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
    let name = result.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let slug = result.get("slug").and_then(|v| v.as_str()).unwrap_or("");
    let editor = result.get("editor").and_then(|v| v.as_str()).unwrap_or("unknown");
    let book_id = result.get("book_id").and_then(|v| v.as_i64()).unwrap_or(0);
    let revision = result.get("revision_count").and_then(|v| v.as_i64()).unwrap_or(0);
    let url = if let Some(rel) = result.get("url").and_then(|v| v.as_str()) {
        join_base_url(base_url, rel)
    } else {
        let book_slug = result.get("book_slug").and_then(|v| v.as_str()).unwrap_or("");
        if !book_slug.is_empty() && !slug.is_empty() {
            format!("{base_url}/books/{book_slug}/page/{slug}")
        } else {
            String::new()
        }
    };
    let url_line = if url.is_empty() {
        String::new()
    } else {
        format!("\nURL: {url}")
    };
    format!("{action}\nPage ID: {id}\nBook ID: {book_id}\nName: {name}\nEditor: {editor}\nSlug: {slug}\nRevision: {revision}{url_line}\nUse get_page({id}) to verify content if needed.")
}

/// Slim success response for shelf create/update operations.
fn format_shelf_success(action: &str, result: &Value, base_url: &str) -> String {
    let id = result.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
    let name = result.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let slug = result.get("slug").and_then(|v| v.as_str()).unwrap_or("");
    let desc = result.get("description").and_then(|v| v.as_str()).unwrap_or("");
    let url = format!("{base_url}/shelves/{slug}");
    let desc_line = if desc.is_empty() {
        String::new()
    } else {
        format!("\nDescription: {desc}")
    };
    format!("{action}\nShelf ID: {id}\nName: {name}\nSlug: {slug}{desc_line}\nURL: {url}")
}

/// Slim success response for book create/update operations.
fn format_book_success(action: &str, result: &Value, base_url: &str) -> String {
    let id = result.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
    let name = result.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let slug = result.get("slug").and_then(|v| v.as_str()).unwrap_or("");
    let desc = result.get("description").and_then(|v| v.as_str()).unwrap_or("");
    let url = format!("{base_url}/books/{slug}");
    let desc_line = if desc.is_empty() {
        String::new()
    } else {
        format!("\nDescription: {desc}")
    };
    format!("{action}\nBook ID: {id}\nName: {name}\nSlug: {slug}{desc_line}\nURL: {url}")
}

/// Slim success response for chapter create/update operations.
fn format_chapter_success(action: &str, result: &Value, base_url: &str) -> String {
    let id = result.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
    let name = result.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let slug = result.get("slug").and_then(|v| v.as_str()).unwrap_or("");
    let book_id = result.get("book_id").and_then(|v| v.as_i64()).unwrap_or(0);
    let desc = result.get("description").and_then(|v| v.as_str()).unwrap_or("");
    let book_slug = result.get("book_slug").and_then(|v| v.as_str()).unwrap_or("");
    let url = if !book_slug.is_empty() && !slug.is_empty() {
        format!("{base_url}/books/{book_slug}/chapter/{slug}")
    } else {
        String::new()
    };
    let url_line = if url.is_empty() {
        String::new()
    } else {
        format!("\nURL: {url}")
    };
    let desc_line = if desc.is_empty() {
        String::new()
    } else {
        format!("\nDescription: {desc}")
    };
    format!("{action}\nChapter ID: {id}\nBook ID: {book_id}\nName: {name}\nSlug: {slug}{desc_line}{url_line}")
}

/// Replace a section in markdown content by heading.
fn replace_section_markdown(md: &str, heading: &str, content: &str, page_id: i64) -> Result<String, String> {
    let lines: Vec<&str> = md.lines().collect();
    let heading_pattern = heading.trim_start_matches('#').trim();

    let start = lines.iter().position(|line| {
        let trimmed = line.trim_start_matches('#').trim();
        trimmed.eq_ignore_ascii_case(heading_pattern)
    }).ok_or(format!("Heading '{}' not found in page {page_id}", heading))?;

    let level = lines[start].chars().take_while(|c| *c == '#').count();

    let end = lines[start + 1..].iter().position(|line| {
        let l = line.chars().take_while(|c| *c == '#').count();
        l > 0 && l <= level
    }).map(|p| p + start + 1).unwrap_or(lines.len());

    let mut updated = lines[..=start].join("\n");
    updated.push('\n');
    updated.push_str(content);
    updated.push('\n');
    if end < lines.len() {
        updated.push('\n');
        updated.push_str(&lines[end..].join("\n"));
    }

    Ok(updated)
}

/// Replace a section in HTML content by heading.
/// Finds <hN>heading</hN> and replaces content up to the next heading of same or higher level.
fn replace_section_html(html: &str, heading: &str, new_content: &str, page_id: i64) -> Result<String, String> {
    let heading_pattern = heading.trim_start_matches('#').trim();

    // Find the heading element
    let mut start_pos = None;
    let mut heading_level = 0usize;
    let mut search_from = 0;

    while search_from < html.len() {
        let Some(h_pos) = html[search_from..].find("<h") else { break };
        let abs_pos = search_from + h_pos;
        let rest = &html[abs_pos..];

        if rest.len() > 2 {
            let level_char = rest.as_bytes()[2];
            if (b'1'..=b'6').contains(&level_char) {
                let level = (level_char - b'0') as usize;
                let close_tag = format!("</h{}>", level);
                if let Some(close_pos) = rest.find(&close_tag) {
                    let tag_content = &rest[..close_pos + close_tag.len()];
                    let text = strip_html_tags(tag_content);
                    if text.trim().eq_ignore_ascii_case(heading_pattern) {
                        start_pos = Some(abs_pos);
                        heading_level = level;
                        break;
                    }
                }
            }
        }
        search_from = abs_pos + 1;
    }

    let start = start_pos.ok_or(format!("Heading '{}' not found in page {page_id}", heading))?;

    // Find end of the heading tag
    let close_tag = format!("</h{}>", heading_level);
    let heading_end = html[start..].find(&close_tag)
        .map(|p| start + p + close_tag.len())
        .ok_or("Malformed heading HTML".to_string())?;

    // Find next heading of same or higher level
    let mut end_pos = html.len();
    let mut search_from = heading_end;

    while search_from < html.len() {
        let Some(h_pos) = html[search_from..].find("<h") else { break };
        let abs_pos = search_from + h_pos;
        let rest = &html[abs_pos..];

        if rest.len() > 2 {
            let level_char = rest.as_bytes()[2];
            if (b'1'..=b'6').contains(&level_char) {
                let level = (level_char - b'0') as usize;
                if level <= heading_level {
                    end_pos = abs_pos;
                    break;
                }
            }
        }
        search_from = abs_pos + 1;
    }

    // Rebuild: heading + new content + rest
    let mut updated = html[..heading_end].to_string();
    updated.push('\n');
    updated.push_str(new_content);
    updated.push('\n');
    updated.push_str(&html[end_pos..]);

    Ok(updated)
}

// --- Dynamic instructions (sent on initialize) ---

async fn build_instructions(client: &BookStackClient, semantic_enabled: bool) -> String {
    let instance_name = env::var("BSMCP_INSTANCE_NAME").unwrap_or_default();
    let instance_desc = env::var("BSMCP_INSTANCE_DESC").unwrap_or_default();

    let mut instructions = String::new();

    if !instance_name.is_empty() {
        instructions.push_str(&instance_name);
        if !instance_desc.is_empty() {
            instructions.push_str(&format!(" - {instance_desc}"));
        }
        instructions.push_str("\n\n");
    }

    instructions.push_str(
        "BookStack knowledge management server. Content is organized as: \
         Shelves > Books > Chapters > Pages. ",
    );

    if semantic_enabled {
        instructions.push_str(
            "Use search_content to find content by keyword or tag, \
             or navigate the hierarchy using the IDs below.\n\n",
        );
    } else {
        instructions.push_str(
            "Use search_content to find content, \
             or navigate the hierarchy using the IDs below.\n\n",
        );
    }

    instructions.push_str(
        "IMPORTANT: Before creating or updating any page, first retrieve an existing page \
         from the same book or chapter using get_page to identify the writing style, \
         formatting conventions, heading structure, and markdown patterns already in use. \
         Match the established style of the surrounding content.\n\n\
         IMPORTANT: Validate content placement before creating pages. Each shelf, book, and \
         chapter has a specific purpose described in the structure below. Do NOT place content \
         where it doesn't belong — for example, do not mix SOPs with design documents, general \
         reference knowledge with company-specific knowledge, or personal content with work \
         content. If the user asks to create content in a location that doesn't match the \
         target's purpose, push back and suggest the correct location. When unsure, check the \
         shelf/book/chapter descriptions using get_shelf, get_book, or get_chapter.\n\n\
         IMPORTANT: Descriptions on shelves, books, and chapters are REQUIRED, not optional. \
         When you call create_shelf, create_book, or create_chapter, you MUST provide a \
         meaningful 1-2 sentence description. Descriptions are surfaced to every Claude \
         client that connects to this BookStack — they literally shape how future content \
         gets routed. A good description answers: (1) what kind of content lives here, and \
         (2) what is this container for (so a future AI can decide whether new content \
         belongs here vs elsewhere). Do NOT use placeholders like 'TODO', 'description', or \
         'n/a' — the server will reject them. If you don't yet know what the container is \
         for, ask the user before creating it. When you update existing shelves/books/chapters \
         via update_shelf, update_book, or update_chapter and notice the description is \
         missing or weak, offer to improve it.\n\n\
         Markdown content is automatically converted to HTML server-side. \
         You can send markdown via the 'markdown' parameter for pages and comments — \
         the server handles conversion reliably, avoiding JSON escaping issues with \
         complex markdown. Use 'html' only when you need precise HTML control.\n\n\
         IMPORTANT: BookStack automatically displays the page name as an H1 title at the top \
         of every page. Do NOT include the page title as a heading (e.g. '# Page Name') in \
         the markdown/html content — this causes a duplicate title. Start content directly with \
         body text or a sub-heading (## or lower).\n\n\
         All editing tools (edit_page, replace_section, append_to_page, insert_after) work on \
         ALL pages regardless of editor type (markdown or WYSIWYG). They use BookStack's \
         markdown export API which converts HTML content to markdown automatically. Prefer \
         these targeted tools over update_page for partial edits — update_page rewrites the \
         entire page and should only be used when the whole page needs replacing.\n\n\
         IMPORTANT: Pages have an 'editor' field ('markdown' or 'wysiwyg'). \
         For edit_page, old_text/new_text must match the page's native format: \
         the 'markdown' field for markdown pages, the 'html' field for WYSIWYG pages. \
         Check the editor type via get_page before using edit_page. \
         For append_to_page, replace_section, and insert_after, always pass markdown content — \
         it is automatically converted to HTML for WYSIWYG pages.\n\n\
         To upload images or file attachments from local files, use the staging upload flow: \
         (1) call prepare_upload to get a staging_id and upload_url, \
         (2) POST the file to the upload_url using curl: \
         `curl -X POST -F 'file=@/path/to/file' <upload_url>` (no auth header needed), \
         (3) call upload_image or upload_attachment with the staging_id. \
         Alternatively, if the file is at a public URL, pass the url parameter directly \
         to upload_image or upload_attachment without staging.\n\n",
    );

    // Include BookStack URL so AI can construct clickable links for users.
    // Uses BSMCP_BOOKSTACK_URL (the actual BookStack instance), NOT BSMCP_PUBLIC_DOMAIN
    // (which is the MCP server's own domain for OAuth).
    if let Ok(url) = env::var("BSMCP_BOOKSTACK_URL") {
        let public_url = url.trim().trim_end_matches('/').to_string();
        if !public_url.is_empty() {
            instructions.push_str(&format!(
                "BookStack URL: {public_url}\n\
                 When you create or update a page, present a clickable link to the user so they can \
                 review it. Page URLs follow the pattern: {public_url}/books/{{book_slug}}/page/{{page_slug}}\n\
                 The slug is returned in the API response. For other content types:\n\
                 - Books: {public_url}/books/{{slug}}\n\
                 - Chapters: {public_url}/books/{{book_slug}}/chapter/{{slug}}\n\
                 - Shelves: {public_url}/shelves/{{slug}}\n\n"
            ));
        }
    }

    match build_structure(client).await {
        Some(structure) => {
            instructions.push_str("Current structure:\n\n");
            instructions.push_str(&structure);
        }
        None => {
            instructions
                .push_str("Use list_shelves and list_books to explore the structure.");
        }
    }

    if semantic_enabled {
        instructions.push_str(
            "\n\nSemantic vector search is available and should be your PRIMARY search method. \
             Prefer `semantic_search` with `mode: \"precision\"` over `search_content` for most queries — \
             precision picks the most-relevant page ~1s faster and more accurately than keyword search \
             or the standard mode. Drop to `mode: \"standard\"` only when you need a broader sweep \
             (more results, blanket-adjacent pages via Markov-blanket boost). Fall back to \
             `search_content` only for exact keyword/tag matches or when semantic_search returns \
             nothing. Use `reembed` to re-index after bulk changes and `embedding_status` to check \
             progress.",
        );
    }

    instructions
}

async fn build_structure(client: &BookStackClient) -> Option<String> {
    let shelves = client.list_shelves(500, 0).await.ok()?;
    let shelf_list = shelves["data"].as_array()?;

    let shelf_futures: Vec<_> = shelf_list
        .iter()
        .filter_map(|s| s["id"].as_i64())
        .map(|id| client.get_shelf(id))
        .collect();
    let shelf_details = futures::future::join_all(shelf_futures).await;

    let chapters = client
        .list_chapters(500, 0)
        .await
        .ok()
        .and_then(|v| v["data"].as_array().cloned())
        .unwrap_or_default();

    let mut chapters_by_book: HashMap<i64, Vec<(i64, String, String)>> = HashMap::new();
    for ch in &chapters {
        if let (Some(book_id), Some(id), Some(name)) = (
            ch["book_id"].as_i64(),
            ch["id"].as_i64(),
            ch["name"].as_str(),
        ) {
            let desc = ch["description"].as_str().unwrap_or("").to_string();
            chapters_by_book
                .entry(book_id)
                .or_default()
                .push((id, name.to_string(), desc));
        }
    }

    let mut output = String::new();
    for shelf in shelf_details.iter().flatten() {
        let name = shelf["name"].as_str().unwrap_or("?");
        let id = shelf["id"].as_i64().unwrap_or(0);
        let desc = truncate_desc(shelf["description"].as_str().unwrap_or(""));
        if desc.is_empty() {
            output.push_str(&format!("Shelf: {name} (ID: {id})\n"));
        } else {
            output.push_str(&format!("Shelf: {name} (ID: {id}) — {desc}\n"));
        }

        if let Some(books) = shelf["books"].as_array() {
            for book in books {
                let bname = book["name"].as_str().unwrap_or("?");
                let bid = book["id"].as_i64().unwrap_or(0);
                let bdesc = truncate_desc(book["description"].as_str().unwrap_or(""));
                if bdesc.is_empty() {
                    output.push_str(&format!("  Book: {bname} (ID: {bid})\n"));
                } else {
                    output.push_str(&format!("  Book: {bname} (ID: {bid}) — {bdesc}\n"));
                }

                if let Some(chs) = chapters_by_book.get(&bid) {
                    for (cid, cname, cdesc) in chs {
                        let cdesc = truncate_desc(cdesc);
                        if cdesc.is_empty() {
                            output.push_str(&format!("    Chapter: {cname} (ID: {cid})\n"));
                        } else {
                            output.push_str(&format!("    Chapter: {cname} (ID: {cid}) — {cdesc}\n"));
                        }
                    }
                }
            }
        }
        output.push('\n');
    }

    if output.is_empty() {
        None
    } else {
        Some(output)
    }
}

// --- Tool definitions ---

pub fn tool_definitions(semantic_enabled: bool) -> Vec<Value> {
    let mut tools = vec![
        tool("search_content",
            "Search across all BookStack content (pages, chapters, books, shelves). Supports operators: {type:page}, [tag_name=value], {in_name:term}, {created_by:me}, exact match with quotes.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "page": { "type": "integer", "description": "Page number", "default": 1 },
                    "count": { "type": "integer", "description": "Results per page", "default": 20 }
                },
                "required": ["query"]
            })),

        // Shelves
        tool("list_shelves", "List all shelves.", paginated_schema()),
        tool("get_shelf", "Get a shelf by ID, including its books.",
            id_schema("shelf_id")),
        tool("create_shelf", "Create a new shelf.", name_desc_schema()),
        tool("update_shelf", "Update a shelf's name, description, or set which books it contains via the 'books' array (replaces all existing book assignments on this shelf).", json!({
            "type": "object",
            "properties": {
                "shelf_id": { "type": "integer", "description": "The shelf_id" },
                "name": { "type": "string", "description": "New name" },
                "description": { "type": "string", "description": "New description" },
                "books": { "type": "array", "items": { "type": "integer" }, "description": "Array of book IDs to assign to this shelf (replaces current assignments)" }
            },
            "required": ["shelf_id"]
        })),
        tool("delete_shelf", "Delete a shelf. This does NOT delete the books inside it.",
            id_schema("shelf_id")),

        // Books
        tool("list_books", "List all books.", paginated_schema()),
        tool("get_book", "Get a book by ID, including its chapters and pages.",
            id_schema("book_id")),
        tool("create_book", "Create a new book.", name_desc_schema()),
        tool("update_book", "Update a book.",
            update_schema("book_id", &["name", "description"])),
        tool("delete_book", "Delete a book and all its chapters/pages.",
            id_schema("book_id")),

        // Chapters
        tool("list_chapters", "List all chapters across all books.", paginated_schema()),
        tool("get_chapter", "Get a chapter by ID, including its pages.",
            id_schema("chapter_id")),
        tool("create_chapter", "Create a new chapter within a book.", json!({
            "type": "object",
            "properties": {
                "book_id": { "type": "integer", "description": "Book ID to create chapter in" },
                "name": { "type": "string", "description": "Chapter name" },
                "description": {
                    "type": "string",
                    "description": "REQUIRED. 1-2 sentences on what this chapter is for. No placeholders."
                }
            },
            "required": ["book_id", "name", "description"]
        })),
        tool("update_chapter", "Update a chapter's name, description, or move it to a different book by providing book_id.", json!({
            "type": "object",
            "properties": {
                "chapter_id": { "type": "integer", "description": "The chapter_id" },
                "name": { "type": "string", "description": "New name" },
                "description": { "type": "string", "description": "New description" },
                "book_id": { "type": "integer", "description": "Move chapter to a different book by providing the target book ID" }
            },
            "required": ["chapter_id"]
        })),
        tool("delete_chapter", "Delete a chapter. Pages inside become book-level pages.",
            id_schema("chapter_id")),

        // Pages
        tool("list_pages", "List all pages across all books.", paginated_schema()),
        tool("get_page", "Get a page by ID with full content. Response carries `editor` ('markdown'|'wysiwyg'), `markdown` source (empty for WYSIWYG pages), and rendered `html`.",
            id_schema("page_id")),
        tool("create_page", "Create a new page. Must provide either book_id or chapter_id. Pass content via `markdown` (creates a markdown-editor page) or `html` (creates a WYSIWYG page).", json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Page name" },
                "book_id": { "type": "integer", "description": "Book ID (if not in a chapter)" },
                "chapter_id": { "type": "integer", "description": "Chapter ID (preferred over book_id)" },
                "markdown": { "type": "string", "description": "Page content in markdown (converted to HTML server-side)", "default": "" },
                "html": { "type": "string", "description": "Page content in HTML (use if you need precise HTML control)", "default": "" }
            },
            "required": ["name"]
        })),
        tool("update_page", "Replace a page's name and/or content, or move it to a different chapter/book. Full rewrite — for surgical edits prefer edit_page, replace_section, append_to_page, or insert_after.", json!({
            "type": "object",
            "properties": {
                "page_id": { "type": "integer", "description": "The page_id" },
                "name": { "type": "string", "description": "New name" },
                "markdown": { "type": "string", "description": "New markdown content (for markdown-editor pages)" },
                "html": { "type": "string", "description": "New HTML content (for WYSIWYG pages)" },
                "chapter_id": { "type": "integer", "description": "Move page to a different chapter by providing the target chapter ID" },
                "book_id": { "type": "integer", "description": "Move page to a different book (at book level, not in any chapter) by providing the target book ID" }
            },
            "required": ["page_id"]
        })),
        tool("edit_page", "Exact-string replace in a page's native content. old_text/new_text must match the page's native format: `markdown` for markdown-editor pages, `html` for WYSIWYG (check `editor` via get_page). Fails if old_text is not found or is ambiguous (multiple matches without replace_all).", json!({
            "type": "object",
            "properties": {
                "page_id": { "type": "integer", "description": "The page_id" },
                "old_text": { "type": "string", "description": "The exact text to find and replace" },
                "new_text": { "type": "string", "description": "The replacement text" },
                "replace_all": { "type": "boolean", "description": "Replace all occurrences (default false)", "default": false }
            },
            "required": ["page_id", "old_text", "new_text"]
        })),
        tool("append_to_page", "Append markdown content to the end of a page. Works on markdown and WYSIWYG pages. No need to read the page first.", json!({
            "type": "object",
            "properties": {
                "page_id": { "type": "integer", "description": "The page_id" },
                "markdown": { "type": "string", "description": "Markdown content to append" }
            },
            "required": ["page_id", "markdown"]
        })),
        tool("replace_section", "Replace all content under a heading (up to the next heading of same or higher level). Works on markdown and WYSIWYG pages. No need to read the page first.", json!({
            "type": "object",
            "properties": {
                "page_id": { "type": "integer", "description": "The page_id" },
                "heading": { "type": "string", "description": "The heading text to find (e.g. '## Related' or just 'Related')" },
                "markdown": { "type": "string", "description": "New content for the section (replaces everything between this heading and the next)" }
            },
            "required": ["page_id", "heading", "markdown"]
        })),
        tool("insert_after", "Insert markdown content after a specific line in a page. Anchor matches exact line content (trimmed). Works on markdown and WYSIWYG pages. No need to read the page first.", json!({
            "type": "object",
            "properties": {
                "page_id": { "type": "integer", "description": "The page_id" },
                "after": { "type": "string", "description": "The exact line content to insert after (e.g. a heading like '## Notes')" },
                "markdown": { "type": "string", "description": "Markdown content to insert" }
            },
            "required": ["page_id", "after", "markdown"]
        })),
        tool("delete_page", "Delete a page (moves to recycle bin).",
            id_schema("page_id")),

        // Move operations
        tool("move_page", "Move a page to a different chapter or book. Only moves — does not modify content. Provide chapter_id to move into a chapter, or book_id to move to book level (not in any chapter).", json!({
            "type": "object",
            "properties": {
                "page_id": { "type": "integer", "description": "The page to move" },
                "chapter_id": { "type": "integer", "description": "Target chapter ID (moves page into this chapter)" },
                "book_id": { "type": "integer", "description": "Target book ID (moves page to book level, outside any chapter)" }
            },
            "required": ["page_id"]
        })),
        tool("move_chapter", "Move a chapter (with all its pages) to a different book.", json!({
            "type": "object",
            "properties": {
                "chapter_id": { "type": "integer", "description": "The chapter to move" },
                "target_book_id": { "type": "integer", "description": "The book to move the chapter into" }
            },
            "required": ["chapter_id", "target_book_id"]
        })),
        tool("move_book_to_shelf", "Move a book to a different shelf. Optionally remove it from a source shelf. Books can appear on multiple shelves — this adds to the target and optionally removes from the source. Note: concurrent calls targeting the same shelf may silently drop book assignments; use sequentially in multi-user environments.", json!({
            "type": "object",
            "properties": {
                "book_id": { "type": "integer", "description": "The book to move" },
                "target_shelf_id": { "type": "integer", "description": "The shelf to add the book to" },
                "remove_from_shelf_id": { "type": "integer", "description": "Optional: shelf to remove the book from (for a true move rather than just adding)" }
            },
            "required": ["book_id", "target_shelf_id"]
        })),

        // Attachments
        tool("list_attachments", "List all attachments.", json!({
            "type": "object", "properties": {}
        })),
        tool("get_attachment", "Get an attachment by ID, including its content or download link.",
            id_schema("attachment_id")),
        tool("create_attachment", "Create a link attachment on a page. uploaded_to is the page ID.", json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Attachment name" },
                "uploaded_to": { "type": "integer", "description": "Page ID to attach to" },
                "link": { "type": "string", "description": "URL for link attachment", "default": "" }
            },
            "required": ["name", "uploaded_to"]
        })),
        tool("update_attachment", "Update an attachment.",
            update_schema("attachment_id", &["name", "link"])),
        tool("delete_attachment", "Delete an attachment.",
            id_schema("attachment_id")),
        tool("upload_attachment", "Upload a file attachment to a page. Use staging_id from prepare_upload for local files, or url to fetch from a remote URL.", json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Attachment name" },
                "uploaded_to": { "type": "integer", "description": "Page ID to attach to" },
                "staging_id": { "type": "string", "description": "Staging slot ID from prepare_upload — use for local file uploads" },
                "url": { "type": "string", "description": "URL to fetch the file from" },
                "filename": { "type": "string", "description": "Override the auto-detected filename" },
                "mime_type": { "type": "string", "description": "MIME type of the file", "default": "application/octet-stream" }
            },
            "required": ["name", "uploaded_to"]
        })),

        // Exports
        tool("export_page", "Export a page as markdown, plaintext, or html. Returns the raw exported content.", json!({
            "type": "object",
            "properties": {
                "page_id": { "type": "integer", "description": "Page ID to export" },
                "format": { "type": "string", "enum": ["markdown", "plaintext", "html"], "description": "Export format", "default": "markdown" }
            },
            "required": ["page_id"]
        })),
        tool("export_chapter", "Export a chapter as markdown, plaintext, or html. Returns all pages in the chapter.", json!({
            "type": "object",
            "properties": {
                "chapter_id": { "type": "integer", "description": "Chapter ID to export" },
                "format": { "type": "string", "enum": ["markdown", "plaintext", "html"], "description": "Export format", "default": "markdown" }
            },
            "required": ["chapter_id"]
        })),
        tool("export_book", "Export a book as markdown, plaintext, or html. Returns all chapters and pages.", json!({
            "type": "object",
            "properties": {
                "book_id": { "type": "integer", "description": "Book ID to export" },
                "format": { "type": "string", "enum": ["markdown", "plaintext", "html"], "description": "Export format", "default": "markdown" }
            },
            "required": ["book_id"]
        })),

        // Comments
        tool("list_comments", "List comments, optionally filtered by page.", json!({
            "type": "object",
            "properties": {
                "page_id": { "type": "integer", "description": "Filter comments by page ID" }
            }
        })),
        tool("get_comment", "Get a comment by ID.",
            id_schema("comment_id")),
        tool("create_comment", "Create a comment on a page. Provide content as markdown (preferred) or html.", json!({
            "type": "object",
            "properties": {
                "page_id": { "type": "integer", "description": "Page ID to comment on" },
                "markdown": { "type": "string", "description": "Comment content in markdown (converted to HTML server-side)" },
                "html": { "type": "string", "description": "Comment content in HTML" },
                "parent_id": { "type": "integer", "description": "Parent comment ID for replies" }
            },
            "required": ["page_id"]
        })),
        tool("update_comment", "Update a comment. Provide content as markdown (preferred) or html.", json!({
            "type": "object",
            "properties": {
                "comment_id": { "type": "integer", "description": "The comment_id" },
                "markdown": { "type": "string", "description": "New comment content in markdown (converted to HTML server-side)" },
                "html": { "type": "string", "description": "New comment content in HTML" }
            },
            "required": ["comment_id"]
        })),
        tool("delete_comment", "Delete a comment.",
            id_schema("comment_id")),

        // Recycle Bin
        tool("list_recycle_bin", "List items in the recycle bin.",
            paginated_schema()),
        tool("restore_recycle_bin_item", "Restore an item from the recycle bin.",
            id_schema("deletion_id")),
        tool("destroy_recycle_bin_item", "Permanently delete an item from the recycle bin. Cannot be undone.",
            id_schema("deletion_id")),

        // Users
        tool("list_users", "List all users.",
            paginated_schema()),
        tool("get_user", "Get a user by ID.",
            id_schema("user_id")),

        // Audit Log
        tool("list_audit_log", "List audit log entries showing recent activity.",
            paginated_schema()),

        // System
        tool("get_system_info", "Get BookStack instance information (version, etc.).", json!({
            "type": "object", "properties": {}
        })),

        // Image Gallery
        tool("list_images", "List images in the gallery.", json!({
            "type": "object",
            "properties": {
                "count": { "type": "integer", "description": "Number of results", "default": 50 },
                "offset": { "type": "integer", "description": "Number to skip", "default": 0 },
                "type": { "type": "string", "enum": ["gallery", "drawio"], "description": "Filter by image type" },
                "uploaded_to": { "type": "integer", "description": "Filter by page ID the image was uploaded to" }
            }
        })),
        tool("get_image", "Get image details by ID. Returns metadata and URLs.",
            id_schema("image_id")),
        tool("update_image", "Update image metadata (name).", json!({
            "type": "object",
            "properties": {
                "image_id": { "type": "integer", "description": "The image_id" },
                "name": { "type": "string", "description": "New image name" }
            },
            "required": ["image_id"]
        })),
        tool("delete_image", "Delete an image from the gallery.",
            id_schema("image_id")),
        tool("upload_image", "Upload an image to the BookStack image gallery. Use staging_id from prepare_upload for local files, or url to fetch from a remote URL. Set embed=true to automatically append the image to the target page's content.", json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Image name" },
                "uploaded_to": { "type": "integer", "description": "Page ID the image is associated with" },
                "staging_id": { "type": "string", "description": "Staging slot ID from prepare_upload — use for local file uploads" },
                "url": { "type": "string", "description": "URL to fetch the image from" },
                "filename": { "type": "string", "description": "Override the auto-detected filename" },
                "type": { "type": "string", "enum": ["gallery", "drawio"], "description": "Image type", "default": "gallery" },
                "mime_type": { "type": "string", "description": "MIME type of the image", "default": "image/png" },
                "embed": { "type": "boolean", "description": "Automatically append the image to the page content after uploading", "default": false }
            },
            "required": ["name", "uploaded_to"]
        })),
        tool("prepare_upload", "Create a staging slot for a local-file upload. Returns `staging_id` + `upload_url`. POST the file as multipart/form-data with field name `file` to upload_url, then pass `staging_id` to upload_image or upload_attachment.", json!({
            "type": "object",
            "properties": {}
        })),

        // Content Permissions
        tool("get_content_permissions", "Get permissions for a content item.", json!({
            "type": "object",
            "properties": {
                "content_type": { "type": "string", "enum": ["page", "chapter", "book", "shelf"], "description": "Content type" },
                "content_id": { "type": "integer", "description": "Content item ID" }
            },
            "required": ["content_type", "content_id"]
        })),
        tool("update_content_permissions", "Update permissions for a content item.", json!({
            "type": "object",
            "properties": {
                "content_type": { "type": "string", "enum": ["page", "chapter", "book", "shelf"], "description": "Content type" },
                "content_id": { "type": "integer", "description": "Content item ID" },
                "owner_id": { "type": "integer", "description": "New owner user ID" },
                "role_permissions": { "type": "array", "description": "Array of role permission objects" },
                "fallback_permissions": { "type": "object", "description": "Fallback permission settings" }
            },
            "required": ["content_type", "content_id"]
        })),

        // Roles
        tool("list_roles", "List all roles.",
            paginated_schema()),
        tool("get_role", "Get a role by ID, including its permissions.",
            id_schema("role_id")),
    ];

    if semantic_enabled {
        tools.push(tool("semantic_search",
            "Semantic search with cross-encoder relevance ranking. \
             **Default to `mode: \"precision\"`** — it's faster (~1s less than standard on typical queries), picks better hits, and surfaces the most-relevant pages first. \
             Use `mode: \"standard\"` only when you need a broader sweep (more results, blanket-adjacent pages) — standard keeps pages that precision's strict cross-encoder ordering drops. Both modes return the same JSON shape, so A/B is just a `mode` swap. \
             \n\nMode reference: \
             `precision` (recommended) — wider vector pass (5× limit), cross-encoder ranks. Best for \"find the right page.\" \
             `standard` — vector + keyword + Markov-blanket boost + blended sort. Best for \"find everything relevant.\" \
             `rerank` — standard's candidate selection, then cross-encoder reorders the top-N. Middle ground; rarely the right pick over precision. \
             \n\n`precision` and `rerank` need `BSMCP_RERANK_PROVIDER` configured on the embedder. If you get \"Reranker is disabled,\" retry with `mode: \"standard\"`. \
             \n\nInclude synonyms and domain vocabulary in your query for better recall (e.g., \"security breach incident response ransomware\" beats \"office got hacked\").",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural language search query. Include synonyms and related terms for better results." },
                    "limit": { "type": "integer", "description": "Max number of page results to return", "default": 10 },
                    "threshold": { "type": "number", "description": "Minimum cosine similarity score (0.0-1.0). Default: 0.45 for hybrid, 0.50 for pure vector.", "default": 0.45 },
                    "hybrid": { "type": "boolean", "description": "Combine vector + keyword search (default true). Set false for pure vector. Forced to false when mode='precision'.", "default": true },
                    "verbose": { "type": "boolean", "description": "Include full Markov blanket data in results. Default false returns slim results (scores, chunks, scoring breakdown). Set true for full graph context.", "default": false },
                    "mode": { "type": "string", "description": "Ranking strategy. **`precision` recommended** (faster, more accurate). `standard` for wider sweep. `rerank` is the middle ground. Schema default stays `standard` for instances without rerank configured.", "enum": ["standard", "rerank", "precision"], "default": "standard" }
                },
                "required": ["query"]
            })));
        tools.push(tool("reembed",
            "Trigger re-embedding of page content. Runs in the background. Use 'all' to re-embed everything, 'book:ID' for a specific book, or 'page:ID' for a single page.",
            json!({
                "type": "object",
                "properties": {
                    "scope": { "type": "string", "description": "Scope: 'all', 'book:ID', or 'page:ID'", "default": "all" }
                }
            })));
        tools.push(tool("embedding_status",
            "Get the current status of the semantic search index, including total indexed pages, chunks, and latest embedding job progress.",
            json!({
                "type": "object",
                "properties": {}
            })));
    }

    tools
}

fn tool(name: &str, description: &str, input_schema: Value) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": input_schema,
    })
}

fn paginated_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "count": { "type": "integer", "description": "Number of results", "default": 50 },
            "offset": { "type": "integer", "description": "Number to skip", "default": 0 }
        }
    })
}

fn id_schema(id_name: &str) -> Value {
    json!({
        "type": "object",
        "properties": {
            id_name: { "type": "integer", "description": format!("The {id_name}") }
        },
        "required": [id_name]
    })
}

fn name_desc_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": { "type": "string", "description": "Name" },
            "description": {
                "type": "string",
                "description": "REQUIRED. 1-2 sentences on what lives here and what it's for. No placeholders ('TODO', 'description', 'n/a')."
            }
        },
        "required": ["name", "description"]
    })
}

fn update_schema(id_name: &str, fields: &[&str]) -> Value {
    let mut props = json!({ id_name: { "type": "integer", "description": format!("The {id_name}") } });
    for &field in fields {
        props[field] = json!({ "type": "string", "description": format!("New {field}") });
    }
    json!({
        "type": "object",
        "properties": props,
        "required": [id_name]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// v0.10.0 surface lock: 59 BookStack CRUD tools + 3 semantic tools = 62.
    /// Anything extra means a briefing-era surface leaked back in.
    #[test]
    fn tools_list_count_is_62_with_semantic() {
        let tools = tool_definitions(true);
        assert_eq!(tools.len(), 62, "expected 59 CRUD + 3 semantic = 62 tools");
    }

    #[test]
    fn tools_list_count_is_59_without_semantic() {
        let tools = tool_definitions(false);
        assert_eq!(tools.len(), 59, "expected 59 CRUD tools");
    }

    /// Locks the precise tool name set so a briefing/session/dismiss-style
    /// addition trips this assertion before it ships.
    #[test]
    fn tools_list_names_match_expected_set() {
        let tools = tool_definitions(true);
        let names: HashSet<String> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(|v| v.as_str()).map(String::from))
            .collect();
        let expected: HashSet<String> = [
            "search_content",
            "list_shelves",
            "get_shelf",
            "create_shelf",
            "update_shelf",
            "delete_shelf",
            "list_books",
            "get_book",
            "create_book",
            "update_book",
            "delete_book",
            "list_chapters",
            "get_chapter",
            "create_chapter",
            "update_chapter",
            "delete_chapter",
            "list_pages",
            "get_page",
            "create_page",
            "update_page",
            "edit_page",
            "append_to_page",
            "replace_section",
            "insert_after",
            "delete_page",
            "move_page",
            "move_chapter",
            "move_book_to_shelf",
            "list_attachments",
            "get_attachment",
            "create_attachment",
            "update_attachment",
            "delete_attachment",
            "upload_attachment",
            "export_page",
            "export_chapter",
            "export_book",
            "list_comments",
            "get_comment",
            "create_comment",
            "update_comment",
            "delete_comment",
            "list_recycle_bin",
            "restore_recycle_bin_item",
            "destroy_recycle_bin_item",
            "list_users",
            "get_user",
            "list_audit_log",
            "get_system_info",
            "list_images",
            "get_image",
            "update_image",
            "delete_image",
            "upload_image",
            "prepare_upload",
            "get_content_permissions",
            "update_content_permissions",
            "list_roles",
            "get_role",
            "semantic_search",
            "reembed",
            "embedding_status",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let extra: Vec<&String> = names.difference(&expected).collect();
        assert!(extra.is_empty(), "unexpected tools in tools/list: {extra:?}");
        let missing: Vec<&String> = expected.difference(&names).collect();
        assert!(missing.is_empty(), "missing tools from tools/list: {missing:?}");
    }
}
