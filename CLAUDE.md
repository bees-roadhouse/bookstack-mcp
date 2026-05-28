# BookStack MCP Server

Rust MCP server that bridges Claude to a BookStack instance via SSE and Streamable HTTP transports. Organized as a Cargo workspace with pluggable database backends and a separate embedder service.

> Build, branching, CI/CD, versioning, and contributor workflow live in [DEVELOPMENT.md](DEVELOPMENT.md). This file is the architectural map and project-shape context for AI agents working in the codebase.

## Architecture

```
crates/
  bsmcp-common/          Shared types, traits, config
    src/lib.rs           Re-exports
    src/config.rs        Env var parsing, DbBackendType
    src/db.rs            DbBackend + SemanticDb + IndexDb traits (async)
    src/types.rs         PageMeta, ChunkData, EmbedJob, MarkovBlanket, SearchHit, etc.
    src/settings.rs      GlobalSettings (server-wide config) + token-hash helper
    src/chunking.rs      Markdown chunking (heading-aware, ~500 token chunks)
    src/vector.rs        BLOB↔embedding conversion, cosine similarity
    src/acl.rs           Per-page ACL resolution against BookStack permissions
    src/rate_limit.rs    Token-bucket rate limiter

  bsmcp-db-sqlite/       SQLite backend
    src/lib.rs           impl DbBackend + SemanticDb + IndexDb for SqliteDb
                         rusqlite wrapped in spawn_blocking for async
                         Brute-force cosine scan for vector search

  bsmcp-db-postgres/     PostgreSQL + pgvector backend
    src/lib.rs           impl DbBackend + SemanticDb + IndexDb for PostgresDb
                         sqlx async driver
                         HNSW index for vector search (embedding <=> operator)
                         FOR UPDATE SKIP LOCKED for concurrent job queue

  bsmcp-server/          MCP server binary
    src/main.rs          Axum server, routes, env config, CORS, db backend selection, auto-migration
    src/sse.rs           SSE session management, Streamable HTTP, multi-user auth
    src/mcp.rs           MCP protocol handler, tool definitions, tool execution
    src/oauth.rs         OAuth 2.1 + refresh tokens, login form, token exchange
    src/semantic.rs      Semantic search (calls embedder /embed, queries db), webhook handler
    src/settings_ui.rs   Browser-based /settings form (token-gated via cookie)
    src/staging.rs       File staging for upload_image / upload_attachment
    src/migrate.rs       SQLite → PostgreSQL migration tool

  bsmcp-embedder/        Embedder binary (pluggable backends)
    src/main.rs          Job queue worker + HTTP /embed endpoint + provider selection
    src/embed.rs         Embedder trait + implementations (LocalEmbedder, OllamaEmbedder, OpenAIEmbedder)
    src/pipeline.rs      Embedding pipeline (fetch pages, chunk, embed, store)

  bsmcp-worker/          Reconciliation worker binary
    src/main.rs          Env wiring, db init, BookStackClient, IndexWorker spawn
    src/lib.rs           IndexWorker — owns the index_jobs queue. Initial full
                         walk on cold start, polls for webhook + cron jobs,
                         runs the periodic delta walk. Same database as the
                         server (server's webhook handler enqueues; worker
                         consumes).
```

**Two transports:**
1. **SSE (MCP 2024-11-05):** Client connects GET `/mcp/sse` with Bearer token -> validates -> creates session -> client sends JSON-RPC to `/mcp/messages/?sessionId=<id>` -> response via SSE event.
2. **Streamable HTTP (MCP 2025-03-26):** Client POSTs JSON-RPC to `/mcp/sse` with Bearer token -> validates -> returns JSON response directly. Used by claude.ai.

**Key patterns:**
- Tool definitions use helper fns: `tool()`, `paginated_schema()`, `id_schema()`, `name_desc_schema()`, `update_schema()`
- `bookstack.rs` has 4 HTTP methods (`get`, `post`, `put`, `delete`) that all follow the same pattern
- Sessions stored in `Arc<RwLock<HashMap<String, Session>>>` with 30s cleanup loop
- Database operations go through `dyn DbBackend` / `dyn SemanticDb` / `dyn IndexDb` trait objects
- Server selects backend at startup via `BSMCP_DB_BACKEND` env var

**Semantic search flow:**

`semantic_search` accepts `mode: "standard" | "rerank" | "precision"`. Schema default is `"standard"` for backward compat (instances without `BSMCP_RERANK_PROVIDER` configured fall through cleanly), but **the tool description and initialize-instructions blurb both push callers toward `mode: "precision"`** — it runs ~1s faster than standard on typical queries and the cross-encoder picks better hits than the heuristic blend. Standard is positioned as the "wider sweep" option (more results, blanket-adjacent pages). All three modes return the same JSON shape so a caller can A/B by swapping the mode value on the same query.

1. Server receives `semantic_search` tool call.
2. Server POSTs query text to embedder's `/embed` endpoint → query embedding.
3. Server calls `db.vector_search()` → SQLite does brute-force cosine, PostgreSQL uses pgvector HNSW. Candidate pool is `limit*2` for standard/rerank, `limit*5` for precision.
4. Server filters hits by per-page ACL via BookStack's API (`filter_by_permission`).
5. Mode-specific ranking step:
   - **Standard:** `db.get_markov_blanket()` for contextual relationships → vector + keyword + blanket-boost blend → top-N.
   - **Rerank:** standard pipeline produces the top-N, then a cross-encoder pass via `/rerank` re-orders just those N. Cheap refinement (~10-30ms for N≤50). The candidate selection is unchanged; only the final ordering changes.
   - **Precision:** wider pool, no blend. The cross-encoder is the ranker of record — picks the best chunk per page, POSTs `(query, [doc])` to `/rerank`, uses the score directly. `hybrid` is forced false in this mode.
6. Returns ranked results with content snippets, per-result `scoring` breakdown (`vector` / `keyword` / `blanket_boost` / `rerank` when applicable), and (in verbose mode) full Markov-blanket context.

Both `rerank` and `precision` modes require `BSMCP_RERANK_PROVIDER` configured on the embedder. Without it, `/rerank` returns 503 and the call surfaces a clear error so the caller can retry with `mode: "standard"`.

**Embedding flow:**
1. `reembed` tool or webhook inserts a job into `embed_jobs` table
2. Embedder polls `embed_jobs` for pending jobs
3. Embedder fetches pages from BookStack API, chunks content, generates embeddings
4. Embedder stores chunks + embeddings in database, updates job progress

## Environment Variables

All prefixed `BSMCP_`. See `.env.example` for full list. Key ones:

**Server:**
- `BSMCP_BOOKSTACK_URL` (required)
- `BSMCP_ENCRYPTION_KEY` (required, 32+ chars)
- `BSMCP_DB_BACKEND` — `sqlite` (default) or `postgres`
- `BSMCP_DATABASE_URL` — PostgreSQL connection string (required if postgres)
- `BSMCP_DB_PATH` — SQLite path (default: `/data/bookstack-mcp.db`)
- `BSMCP_PUBLIC_DOMAIN` (for OAuth redirect URLs)
- `BSMCP_SEMANTIC_SEARCH` — `true` to enable semantic tools
- `BSMCP_EMBEDDER_URL` — embedder HTTP endpoint (default: `http://bsmcp-embedder:8081`)
- `BSMCP_WEBHOOK_SECRET` — constant-time verified webhook secret

**Embedder:**
- `BSMCP_EMBED_TOKEN_ID` / `BSMCP_EMBED_TOKEN_SECRET` — BookStack API token for crawling
- `BSMCP_EMBED_PROVIDER` — `local` (default ONNX), `ollama`, `openai`
- `BSMCP_EMBED_MODEL` — model name (default per provider)
- `BSMCP_EMBED_API_KEY` — API key (openai provider only)
- `BSMCP_EMBED_API_URL` — base URL for ollama/openai
- `BSMCP_EMBED_DIMS` — embedding dimensions (auto-detected for ollama)
- `BSMCP_EMBED_BATCH_SIZE`, `BSMCP_EMBED_DELAY_MS` — performance tuning

## Settings UI (`/settings`)

Browser-based admin config page. Token-gated via the `/authorize` form — when `?return_to=/settings` is set, the server validates the BookStack API token and issues a settings-session cookie (HttpOnly, 8h TTL, in-memory store) instead of running the full OAuth code dance.

The page is admin-only — non-admin saves silently drop every field. Surfaces only the global server fields the index worker still needs:

- `hive_shelf_id`
- `user_journals_shelf_id`

There is no MCP write path for global settings — they must be configured via `/settings` by an admin. Per-user settings have been removed; the server holds no per-caller state beyond OAuth tokens.

## Auth-gated `/status`

The semantic-search status page accepts either a Bearer token (programmatic) or a settings-session cookie (browser). Unauthenticated requests get a 401 with a link to `/settings`.

## Implemented Tools (59 BookStack + 3 semantic = 62)

- **search_content** - Full-text search with BookStack query operators
- **semantic_search** - Natural language vector search (when semantic enabled)
- **reembed** - Trigger re-embedding of all pages (when semantic enabled)
- **embedding_status** - Check semantic index status (when semantic enabled)
- **Shelves** - list, get, create, update (assign books), delete (5)
- **Books** - list, get, create, update, delete (5)
- **Chapters** - list, get, create, update (move between books), delete (5)
- **Pages** - list, get, create, update (move between chapters/books), delete (5)
- **Page edits (partial)** - edit_page, append_to_page, replace_section, insert_after (4)
- **Move** - move_page, move_chapter, move_book_to_shelf (3) - dedicated move operations
- **Attachments** - list, get, create, upload, update, delete (6)
- **Staging** - prepare_upload (1) - create a staging slot for local-file uploads
- **Exports** - export_page, export_chapter, export_book (3) - markdown, plaintext, or html
- **Comments** - list, get, create, update, delete (5)
- **Recycle Bin** - list, restore, destroy (3)
- **Users** - list, get (2) - read-only
- **Audit Log** - list (1)
- **System** - get_system_info (1)
- **Image Gallery** - list, get, upload, update, delete (5)
- **Content Permissions** - get, update (2)
- **Roles** - list, get (2) - read-only

## Not Implemented

- **Imports** - ZIP file handling doesn't work well over MCP text protocol.
- **User/Role CRUD** - Creating/deleting users/roles is admin-level; read-only is sufficient.
- **PDF/ZIP export** - Binary formats can't be returned as MCP text content.

## Adding a New Tool

1. **bookstack.rs** - Add the API method(s) to `BookStackClient`
2. **mcp.rs** - Add match arm in `execute_tool()`, add tool def in `tool_definitions()`, update the `tools_list_*` test set
3. Use existing helpers: `arg_str`, `arg_i64`, `arg_i64_required`, `arg_str_default`, `filter_update_fields`, `format_json`

For GET endpoints that need a raw text response (like export), add a `get_text()` method to `BookStackClient` that returns `String` instead of `Value`.

## OAuth / Claude Desktop Custom Connector

The server implements OAuth 2.1 (authorization code + PKCE) with a browser-based login form for BookStack API token authentication.

**MCP endpoint URL:** `https://your-host/mcp/sse` (must include the `/mcp/sse` path)

**OAuth endpoints:**
- `GET /.well-known/oauth-authorization-server` — RFC 8414 metadata (MCP 2025-03-26)
- `GET /.well-known/oauth-protected-resource` — RFC 9728 metadata (MCP 2025-06-18)
- `GET /authorize` — Serves login form for API token entry
- `POST /authorize` — Validates token against BookStack, issues auth code
- `POST /token` — Token exchange

**Two auth flows:**
1. **Form-based (primary):** Claude opens /authorize → user enters BookStack API token → server validates → stores credentials with auth code → redirects → code exchange issues access token.
2. **Legacy Bearer:** `Authorization: Bearer token_id:token_secret` on SSE/messages endpoints (Claude Code direct connection).
