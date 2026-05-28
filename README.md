# BookStack MCP Server

[![SafeSkill 50/100](https://img.shields.io/badge/SafeSkill-50%2F100_Use%20with%20Caution-orange)](https://safeskill.dev/scan/bees-roadhouse-bookstack-mcp)
An MCP (Model Context Protocol) server that gives Claude full access to a [BookStack](https://www.bookstackapp.com/) instance. Built in Rust with tokio/axum as a Cargo workspace with pluggable database backends and optional semantic vector search.

## Features

- Full CRUD on all core BookStack resources (shelves, books, chapters, pages, attachments)
- Full-text search with BookStack query operators
- **Semantic vector search** — natural language search across all content via embeddings (optional). Three modes on `semantic_search`: `standard` (default, vector + keyword + Markov-blanket blend), `rerank` (standard pool, cross-encoder refines top-N), `precision` (wider pool, cross-encoder replaces the blend). Per-page access control enforced via BookStack's API on every result.
- **Settings UI (`/settings`)** — browser-based admin configuration page (token-gated via the same `/authorize` flow). Surfaces only the global server fields the index worker needs (`hive_shelf_id`, `user_journals_shelf_id`).
- **Pluggable database** — SQLite for simple deployments, PostgreSQL + pgvector for production
- **Separate embedder** — background embedding service with pluggable backends (local ONNX, Ollama, OpenAI, Voyage)
- **Cross-encoder reranker (optional)** — embedder exposes `POST /rerank` when `BSMCP_RERANK_PROVIDER` is configured. Three providers: `local` (in-process ONNX cross-encoder via fastembed, default `BAAI/bge-reranker-v2-m3`), `voyage` (Voyage's `/v1/rerank`), `openai` (any OpenAI-shape rerank endpoint — covers Voyage/Jina/Cohere-via-shim/self-hosted). Off by default; consumed by `semantic_search`'s `mode: "rerank"` (refinement) and `mode: "precision"` (cascade) modes.
- **Server-side markdown to HTML conversion** — send markdown, server converts before sending to BookStack
- **Staging upload flow** — upload local images and attachments through a two-step staging endpoint without exposing local paths to the container ([see below](#uploading-local-files-images--attachments))
- **OAuth 2.1 support** — use as a Claude.ai or Claude Desktop custom connector without config files
- **Encrypted token storage** — OAuth tokens encrypted at rest with AES-256-GCM
- **Dual transport** — SSE (MCP 2024-11-05) and Streamable HTTP (MCP 2025-03-26)
- **Dynamic structure discovery** — AI automatically learns your BookStack hierarchy on connect
- **Auto-migration** — seamlessly migrate from SQLite to PostgreSQL on startup
- Multi-user support via per-session BookStack API tokens
- Multi-arch Docker images (amd64 + arm64)

## Architecture

```
crates/
  bsmcp-common/       Shared types, traits, config, chunking, vector utils
  bsmcp-db-sqlite/    SQLite backend (rusqlite, bundled)
  bsmcp-db-postgres/  PostgreSQL + pgvector backend (sqlx)
  bsmcp-server/       MCP server binary (axum, no ONNX dependency)
  bsmcp-embedder/     Embedder binary (local ONNX / Ollama / OpenAI / Voyage, job queue worker + HTTP /embed + optional /rerank)
  bsmcp-worker/       Reconciliation worker (initial walk + webhook/cron-driven delta walk; same DB as the server)

docker/
  Dockerfile.server       Lightweight server image (~35MB)
  Dockerfile.embedder     Embedder image with ONNX Runtime (~45MB)
  Dockerfile.worker       Reconciliation worker image
  docker-compose.yml      PostgreSQL deployment (production)
  docker-compose.sqlite.yml  SQLite deployment (simple)
```

The MCP server handles all client-facing protocol, OAuth, and search. The embedder runs separately, polling a database-backed job queue to embed pages and serving a `/embed` HTTP endpoint for query-time embedding (and `/rerank` when a reranker provider is configured). The embedder supports four embedding backends: local ONNX models (fastembed), Ollama, OpenAI-compatible APIs, and Voyage. The worker owns the `index_jobs` queue — runs the initial full walk on cold start, then consumes webhook + cron jobs and the periodic delta walk.

## Available Tools (59 BookStack + 3 semantic = 62)

| Category | Tools |
|----------|-------|
| **Search** | `search_content` |
| **Semantic** | `semantic_search`, `reembed`, `embedding_status` |
| **Shelves** | `list_shelves`, `get_shelf`, `create_shelf`, `update_shelf`, `delete_shelf` |
| **Books** | `list_books`, `get_book`, `create_book`, `update_book`, `delete_book` |
| **Chapters** | `list_chapters`, `get_chapter`, `create_chapter`, `update_chapter`, `delete_chapter` |
| **Pages** | `list_pages`, `get_page`, `create_page`, `update_page`, `delete_page`, `edit_page`, `append_to_page`, `replace_section`, `insert_after` |
| **Move** | `move_page`, `move_chapter`, `move_book_to_shelf` |
| **Attachments** | `list_attachments`, `get_attachment`, `create_attachment`, `update_attachment`, `delete_attachment`, `upload_attachment` |
| **Staging** | `prepare_upload` (used with `upload_image` / `upload_attachment` for local file uploads) |
| **Exports** | `export_page`, `export_chapter`, `export_book` (markdown, plaintext, html) |
| **Comments** | `list_comments`, `get_comment`, `create_comment`, `update_comment`, `delete_comment` |
| **Recycle Bin** | `list_recycle_bin`, `restore_recycle_bin_item`, `destroy_recycle_bin_item` |
| **Users** | `list_users`, `get_user` |
| **Audit Log** | `list_audit_log` |
| **System** | `get_system_info` |
| **Images** | `list_images`, `get_image`, `upload_image`, `update_image`, `delete_image` |
| **Permissions** | `get_content_permissions`, `update_content_permissions` |
| **Roles** | `list_roles`, `get_role` |

Semantic tools (`semantic_search`, `reembed`, `embedding_status`) only appear when `BSMCP_SEMANTIC_SEARCH=true` and an embedder is running. Without semantic search: 59 BookStack tools.

The server is a thin BookStack CRUD facade plus semantic-search enrichment, OAuth, audit, and the reconciliation worker. Personal-memory primitives (journals, identities, reminders) and the v0.8.0/v0.9.0 briefing surface were removed in v0.10.0; v0.11.0 adds the optional cross-encoder reranker on the embedder side and the three-mode `semantic_search` shape on the server side. See the migration notes below.

## Setup

### Prerequisites

- A BookStack instance with API access enabled
- A BookStack API token (created in your BookStack user profile under "API Tokens")
- Docker and Docker Compose (for container deployment)

### Quick Start (PostgreSQL — recommended)

```bash
cp .env.example .env
# Edit .env with your BookStack URL, encryption key, and database password

docker compose -f docker/docker-compose.yml up -d
```

This starts three containers:
- **bsmcp-postgres** — PostgreSQL 17 with pgvector extension
- **bsmcp-server** — MCP server (port 8080)
- **bsmcp-embedder** — Background embedding service

### Quick Start (SQLite — simple)

```bash
cp .env.example .env
# Edit .env with your BookStack URL and encryption key

docker compose -f docker/docker-compose.sqlite.yml up -d
```

This starts two containers sharing a SQLite database file.

### Run from source

The project distributes as multi-arch (`linux/amd64` + `linux/arm64`) container images on GHCR — `ghcr.io/bees-roadhouse/bsmcp-server` and `ghcr.io/bees-roadhouse/bsmcp-embedder`. Native binaries for **`bsmcp-server` only** are attached to each GitHub Release for `linux-x86_64`, `linux-aarch64`, `darwin-x86_64`, `darwin-aarch64`, and `windows-x86_64`. The embedder is **not** distributed as a bare binary — it depends on ONNX Runtime (a per-platform C++ shared library), so running it outside Docker is awkward. Either run the published embedder container, or build from source:

```bash
# Server
cargo run --release -p bsmcp-server

# Embedder (separate terminal)
cargo run --release -p bsmcp-embedder
```

The server is pure Rust + bundled SQLite and builds cleanly on any target the Rust toolchain supports. The embedder depends on [`fastembed`](https://crates.io/crates/fastembed), which links ONNX Runtime; the crate downloads a matching prebuilt at build time for common targets, but cross-compiling or running on uncommon platforms may require installing ONNX Runtime separately. For most users, running the embedder from the published container avoids that complexity entirely.

### Configuration

#### Server Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BSMCP_BOOKSTACK_URL` | Yes | - | Your BookStack instance URL |
| `BSMCP_ENCRYPTION_KEY` | Yes | - | 32+ char key for AES-256-GCM token encryption |
| `BSMCP_DB_BACKEND` | No | `sqlite` | Database backend: `sqlite` or `postgres` |
| `BSMCP_DATABASE_URL` | If postgres | - | PostgreSQL connection string |
| `BSMCP_DB_PATH` | No | `/data/bookstack-mcp.db` | SQLite database path |
| `BSMCP_PUBLIC_DOMAIN` | No | - | Public domain for OAuth redirects (e.g. `mcp.example.com`) |
| `BSMCP_INTERNAL_DOMAIN` | No | - | Internal/Docker-network domain |
| `BSMCP_HOST` | No | `0.0.0.0` | Bind address |
| `BSMCP_PORT` | No | `8080` | Bind port |
| `BSMCP_INSTANCE_NAME` | No | - | Instance name shown to AI |
| `BSMCP_INSTANCE_DESC` | No | - | Instance description shown to AI |
| `BSMCP_SEMANTIC_SEARCH` | No | `false` | Enable semantic search tools |
| `BSMCP_EMBEDDER_URL` | No | `http://bsmcp-embedder:8081` | Embedder HTTP endpoint |
| `BSMCP_WEBHOOK_SECRET` | If semantic | - | BookStack webhook secret |
| `BSMCP_ACCESS_TOKEN_TTL` | No | `86400` | Access token TTL in seconds (24h) |
| `BSMCP_REFRESH_TOKEN_TTL` | No | `7776000` | Refresh token TTL in seconds (90d) |
| `BSMCP_BACKUP_INTERVAL` | No | - | Hours between backups (0 = disabled) |
| `BSMCP_BACKUP_PATH` | No | `/data/backups` | Backup directory |

#### Embedder Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BSMCP_EMBED_TOKEN_ID` | Yes | - | BookStack API token ID for crawling |
| `BSMCP_EMBED_TOKEN_SECRET` | Yes | - | BookStack API token secret |
| `BSMCP_EMBED_PROVIDER` | No | `local` | Embedding backend: `local`, `ollama`, `openai`, `voyage` |
| `BSMCP_EMBED_MODEL` | No | (per provider) | Model name (see [Embedding Providers](#embedding-providers)) |
| `BSMCP_EMBED_API_KEY` | If openai | - | API key for OpenAI embedding provider |
| `BSMCP_EMBED_API_URL` | No | (per provider) | Base URL for Ollama or OpenAI-compatible endpoint |
| `BSMCP_EMBED_DIMS` | No | (auto) | Embedding dimensions (auto-detected for Ollama) |
| `BSMCP_MODEL_PATH` | No | `/data/models` | ONNX model cache directory (local provider only) |
| `BSMCP_EMBED_CPUS` | No | `0` (unlimited) | Docker CPU limit for embedder |
| `BSMCP_EMBED_JOB_TIMEOUT` | No | `14400` | Seconds before stuck jobs reset |
| `BSMCP_EMBED_BATCH_SIZE` | No | `32` | Chunks per embedding batch |
| `BSMCP_EMBED_DELAY_MS` | No | `50` | Delay between pages (API throttle) |
| `BSMCP_EMBED_POLL_INTERVAL` | No | `5` | Seconds between job queue polls |
| `BSMCP_EMBED_ON_STARTUP` | No | `false` | `true` = auto-embed on startup, `clean` = clear all embeddings first |
| `BSMCP_EMBED_HOST` | No | `0.0.0.0` | Embedder listen address |
| `BSMCP_EMBED_PORT` | No | `8081` | Embedder listen port |
| `BSMCP_RERANK_PROVIDER` | No | (unset) | Cross-encoder rerank provider: `local`, `voyage`, `openai`, `none`. Off by default; enables `POST /rerank` on the embedder. |
| `BSMCP_RERANK_MODEL` | If reranker on | (per provider) | Reranker model. Defaults: `BAAI/bge-reranker-v2-m3` (local), `rerank-2` (voyage). Required for `openai`. |
| `BSMCP_RERANK_API_KEY` | If voyage/openai | - | API key for external rerank provider. |
| `BSMCP_RERANK_API_URL` | If openai | (per provider) | Base URL. Voyage defaults to `https://api.voyageai.com`; openai requires explicit URL. |

See `.env.example` for the full list with comments.

### Semantic Search Setup

1. Set `BSMCP_SEMANTIC_SEARCH=true` in your server env
2. Set `BSMCP_WEBHOOK_SECRET` to a random string (16+ characters)
3. Create a BookStack API token with read access for the embedder (`BSMCP_EMBED_TOKEN_ID` / `BSMCP_EMBED_TOKEN_SECRET`)
4. Start the embedder container — it downloads the ONNX model (~1.3GB) on first run
5. Use the `reembed` tool (via Claude) to trigger initial embedding of all pages
6. Configure a webhook in BookStack for automatic re-embedding on page changes:

#### BookStack Webhook Configuration

Go to **Settings > Webhooks > Create Webhook** in your BookStack instance:

| Field | Value |
|-------|-------|
| **Name** | MCP Semantic Search |
| **Endpoint URL** | `https://your-mcp-host/webhooks/bookstack` |
| **Active** | Yes |

**Events to select:**
- Page Create
- Page Update
- Page Delete

**Custom header** (required for verification):
```
X-Webhook-Secret: YOUR_WEBHOOK_SECRET
```

The `YOUR_WEBHOOK_SECRET` value must match `BSMCP_WEBHOOK_SECRET` in your server environment. The server uses constant-time comparison to verify the header.

After saving, any page create/update/delete in BookStack automatically queues a re-embedding job. The embedder picks it up within seconds (configurable via `BSMCP_EMBED_POLL_INTERVAL`).

## Connecting

The MCP endpoint URL is:

```
https://your-host/mcp/sse
```

> **Important:** Use the full path including `/mcp/sse` — not just the base domain.

### Claude.ai (Custom Connector)

1. Go to **Settings > Integrations > Add custom MCP** in Claude.ai
2. Enter the MCP endpoint URL: `https://your-host/mcp/sse`
3. A login form opens in your browser — enter your BookStack API **Token ID** and **Token Secret**
4. Once authorized, BookStack tools appear automatically in your conversations

### Claude Desktop (Custom Connector)

1. Add a custom connector with URL: `https://your-host/mcp/sse`
2. When connecting, a login form opens in your browser with instructions
3. Enter your BookStack API **Token ID** and **Token Secret**

No config files needed — authentication happens entirely through the browser via OAuth 2.1.

### Claude Code (Direct Bearer Token)

Add to your MCP server configuration:

```json
{
  "mcpServers": {
    "bookstack": {
      "url": "https://your-host/mcp/sse",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN_ID:YOUR_TOKEN_SECRET"
      }
    }
  }
}
```

The token ID and secret come from your BookStack API token (created under **My Account > Access & Security > API Tokens**).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/mcp/sse` | SSE connection (MCP 2024-11-05) |
| `POST` | `/mcp/sse` | Streamable HTTP (MCP 2025-03-26) |
| `POST` | `/mcp/messages/?sessionId=<id>` | Send MCP JSON-RPC messages (SSE transport) |
| `GET` | `/health` | Health check |
| `POST` | `/webhooks/bookstack` | BookStack webhook receiver (semantic search) |
| `GET` | `/status` | Embedding progress page with live progress bar |
| `GET` | `/.well-known/oauth-authorization-server` | OAuth metadata (RFC 8414) |
| `GET` | `/.well-known/oauth-protected-resource` | Protected resource metadata (RFC 9728) |
| `GET` | `/authorize` | Login form for BookStack API token |
| `POST` | `/authorize` | Validate credentials and issue auth code |
| `POST` | `/token` | OAuth token exchange |
| `POST` | `/register` | Dynamic client registration (RFC 7591) |

## Upgrading

All schema migrations are automatic on startup (CREATE TABLE IF NOT EXISTS, ALTER TABLE for new columns). No manual SQL is needed.

> **Heads up.** v0.10.0 stripped the briefing layer + per-user settings; v0.11.0 (this release) adds the optional cross-encoder reranker. Older entries describe functionality that no longer ships and are kept only for upgrade-path archaeology.

### From v0.10.0 to v0.11.0 (this release)

#### What's new

- **Cross-encoder reranker on the embedder.** New `POST /rerank` endpoint when `BSMCP_RERANK_PROVIDER` is set on the embedder. Three providers: `local` (in-process ONNX cross-encoder via fastembed; default `BAAI/bge-reranker-v2-m3`), `voyage` (Voyage's `/v1/rerank`), `openai` (any OpenAI-shape rerank endpoint). Off by default — `BSMCP_RERANK_PROVIDER=none` (or unset) leaves the endpoint disabled and returns 503.
- **Three-mode `semantic_search`.** Replaces the prior single-shape behavior with `mode: "standard" | "rerank" | "precision"`, defaulting to `"standard"`:
  - `standard` — vector + keyword + Markov-blanket blend (the v0.10.0 default behavior).
  - `rerank` — same candidate pool as standard, but the final top-N is re-ordered by the cross-encoder. Cheap refinement (~10–30 ms for top-10 against a local cross-encoder).
  - `precision` — wider initial vector pool (5× limit), no keyword/blanket blend, cross-encoder is the ranker of record. More expensive, can rescue hits the blend would miss.
- **Per-result `scoring.rerank` and `stats.rerank_*`** in the search response when either rerank-enabled mode fires (`mode`, `hybrid`, `rerank_ms`, `rerank_provider`, `rerank_model`, `candidates_reranked`).

#### What's automatic

- No schema changes. Drop-in at the database layer.
- Existing `semantic_search` callers keep working unchanged — `mode` defaults to `"standard"` and reproduces v0.10.0 behavior.

#### What you must do (only to use the new modes)

1. Pull new images: `ghcr.io/bees-roadhouse/bsmcp-server:0.11.0` + `ghcr.io/bees-roadhouse/bsmcp-embedder:0.11.0` + `ghcr.io/bees-roadhouse/bsmcp-worker:0.11.0`.
2. Set `BSMCP_RERANK_PROVIDER` (and the matching `BSMCP_RERANK_MODEL` / `BSMCP_RERANK_API_KEY` / `BSMCP_RERANK_API_URL`) on the embedder.
3. Pass `mode: "rerank"` or `mode: "precision"` on `semantic_search` calls. If the reranker is disabled, the embedder returns 503 and the server surfaces a clear error pointing at `BSMCP_RERANK_PROVIDER` so callers can drop back to `mode: "standard"`.

### From v0.9.0 to v0.10.0

#### What's removed

- **`briefing` MCP tool, `POST /briefing/v1/read` HTTP route, and the auto-injected `meta.briefing` envelope** are all gone. The single-call reconstitution shell from v0.8.0 / v0.9.0 turned out to fan out 5+ parallel BookStack page fetches per request and fail open on stale `system_prompt_page_ids` config. Removed wholesale.
- **Per-user `UserSettings`** struct and the `user_settings` table (both Postgres and SQLite) — every consumer was the briefing path or related setup nudges. No per-user state to persist after the cut.
- **Per-user role-level ACL filtering on semantic search and `tools/list`** — depended on `UserSettings.bookstack_user_id`. Semantic search becomes user-anonymous on the embedder side; per-page access control still runs through BookStack's API on every result.
- **`user_role_cache` table** — fed only the per-user role-level filter.
- **Briefing-only `GlobalSettings` fields** (`org_required_instructions_page_ids`, `org_ai_usage_policy_page_ids`, `org_identity_page_id`, `org_domains`, `guide_page_id`, `policies_scope`, `sops_scope`, `best_practices_scope`, `friendly_structure`, `full_content_in_briefing`, `strict_setup`) and the matching `/settings` UI sections.
- **Instance summary subsystem** (Ollama caller + the `Summary: …` log lines + `BSMCP_LLM_*` / `BSMCP_SUMMARY_*` env vars).
- **`session_event` and `dismiss_setup_nudge` MCP tools** (briefing-only).
- **`try_auto_populate_bookstack_user_id` in the OAuth flow** — no settings row to populate.
- **v0.7.x `extras` migration shims.**

#### What survives

- 59 BookStack CRUD tools (`create_*` / `update_*` / `delete_*` / `get_*` / `list_*` + `search_content`).
- `semantic_search`, `reembed`, `embedding_status`.
- `bsmcp-embedder` + `bsmcp-worker` images and the reconciler.
- Rate limiter + audit log (#54 infra).
- OAuth 2.1 / `/authorize` flow.
- `/settings` admin UI for the surviving global server config (`hive_shelf_id`, `user_journals_shelf_id`).

#### Tool count

- With semantic search: **62 tools** (59 CRUD + 3 semantic). Without semantic: 59.

#### What's automatic

- `user_settings`, `user_role_cache`, `remember_audit`, `token_bindings`, `sessions` tables are dropped on first startup (idempotent).
- Briefing-only `global_settings` columns are dropped via `ALTER TABLE DROP COLUMN` (Postgres native; SQLite ≥ 3.35).
- Existing org-level page-id config is discarded — re-enter via the trimmed `/settings` page if any of the surviving fields apply.

#### Breaking changes

- Clients calling `briefing` or `/briefing/v1/read` get `tool not found` / `404`.
- Tool responses no longer carry `meta.briefing`.
- `tools/list` no longer filters by role.

### From v0.8.0 to v0.9.0

#### What's new

- **v1.0.0 rollback.** The Phase 2 re-introduction of personal-memory MCP tools (`user`, `config`, `directory`, `identity`, `journal`, `migrate`, `reminders`, `events`, `sessions`, `session_event`, `dismiss_setup_nudge`) is gone. The single `briefing` tool from v0.8.0 stays. The codebase is back to v0.8.0's posture plus the issue #54 rate-limiter / job-lifecycle infrastructure.
- **DB tables `token_bindings` and `sessions` are no longer created** on fresh installs. Existing v1.0.0 deployments upgrading to v0.9.0 keep the on-disk tables (inert; `DROP TABLE` manually if cleanup matters).
- **`UserSettings` shed the per-account-settings + journal-resolver fields** added in v1.0.0 (`account_label`, `use_org_identity`, `journaling_enabled`, `chosen_ai_identity`, `setup_complete`, `tool_overrides`, `user_journal_book_id`, `cached_user_email*`, `cached_first_name*`, `cached_is_admin*`). The `extras` JSON catch-all silently preserves any leftover keys until the briefing's migration handler clears them.
- **`GlobalSettings.tool_defaults` and `admin_setup_complete` dropped** — admin-only defaults followed the per-tool toggle infrastructure into the bin.
- **`/setup/user` and `/setup/admin` browser wizards removed.** The `/settings` page is the only browser-side configuration surface.
- **`oauth.rs::ensure_token_binding` reverted to v0.8.0's `try_auto_populate_bookstack_user_id` shape.** Tokens key the `user_settings` row directly via `token_id_hash` again; no binding indirection.

#### What survives from the v0.8.0 → v1.0.0 era

- Rate limiter + job lifecycle (`bsmcp_common::rate_limit`, `embed_jobs` / `index_jobs` lifecycle columns, `/jobs/{embed,index}/{id}/cancel` endpoints, the lifecycle housekeeper in `bsmcp-worker`). Issue #54 work is general infra and is kept verbatim.

#### What's automatic

- **No DB migration ships for v1.0.0 → v0.9.0 downgraders.** `CREATE TABLE IF NOT EXISTS` won't re-shape the v1.0.0 `user_settings` PK (`stable_id` → `token_id_hash`). If a clean reset is needed, drop the table manually before first start, or wait for a follow-up one-shot migration.
- v0.8.0 → v0.9.0 is a no-op schema-wise.

### From v0.7.4 to v0.8.0

#### What's new

- **Personal-memory layer moved to memberberry.ai.** All 12 `remember_*` MCP tools (`remember_briefing` / `remember_journal` / `remember_collage` / `remember_shared_collage` / `remember_user_journal` / `remember_whoami` / `remember_user` / `remember_identity` / `remember_directory` / `remember_config` / `remember_audit` / `remember_search`) no longer ship. The `POST /remember/v1/{resource}/{action}` HTTP namespace is gone.
- **Single `briefing` MCP tool** replaces the 12 remember tools. Same response shape as the old `remember_briefing action=read`, no `action` arg. HTTP form: `POST /briefing/v1/read`.
- **`meta.briefing` auto-injection** on every MCP tool response — full content on the first call per `(token_hash, session_id)`, sticky bits (time + setup summary) thereafter. Calling `briefing` explicitly resets the session for the next response — useful after the AI's harness compacts the conversation.
- **Typed setup slots on global settings** — `guide_page_id`, `org_identity_page_id`, `policies_scope`, `sops_scope`, `best_practices_scope`, plus org-wide booleans `friendly_structure`, `full_content_in_briefing`, `strict_setup`. Idempotent `ALTER TABLE ADD COLUMN` migrations on first startup.
- **Removed:** `default_ai_identity_*` global columns (dropped via `ALTER TABLE DROP COLUMN`), `remember_audit` table (`DROP TABLE IF EXISTS`), and most per-user pointer fields from `UserSettings` (`ai_*_book_id`, `user_journal_book_id`, `recent_*_count`, etc.). The settings UI shrank ~1,300 lines to match.

#### What's automatic

- Idempotent `ALTER TABLE ADD COLUMN` for the new global slots.
- `ALTER TABLE DROP COLUMN [IF EXISTS]` for `default_ai_identity_*` (Postgres native; SQLite swallows duplicate-drop errors via `.ok()`, requires SQLite ≥ 3.35).
- `DROP TABLE IF EXISTS remember_audit` on startup.
- `user_settings` is a JSON blob — old keys are silently ignored on read and dropped on next save.

### From v0.7.3 to v0.7.4

#### What's new

- **Briefing payload trimmed** — `*_semantic_matches` entries now cap at 3 chunks of ~100 chars each (kb_semantic_matches: 4 × 150). Truncated chunks carry `truncated: true` and a `…` suffix. A new top-level `semantic_matches_hint` field tells consumers to call `get_page(page_id)` for full content. Briefing responses shrink ~50% in typical use, well under Claude Code's response-size threshold.
- **`semantic_search` tool trimmed** — same shape, slightly more headroom (5 chunks × ~200 chars). New top-level `hint` field on the tool response.
- **Shared trim helper** — chunk-truncation logic lives in one place (`semantic::trim_match`), each caller passes its own budget.

#### What's automatic

- No env vars, no schema changes, drop-in.

### From v0.7.2 to v0.7.3

#### What's new

- **`meta.time` block** on every `/remember` response — `now_unix`, `now_utc`, `now_local`, `now_human`, `timezone`, `timezone_source`, `timezone_refresh_due`. Per-user timezone cached server-side; refresh by passing `client_timezone` (IANA name) on any `remember_*` call.
- **Actionable setup errors** — `settings_not_configured` errors now carry an `error.fix` block with the exact MCP call to make.
- **Per-call elapsed timing** exposed in `meta.elapsed_ms`.

### From v0.7.0 to v0.7.2

#### What's new

- **Briefing latency cut** — book-scoped semantic-search candidate filtering, parallel KB fetches, batched DB lookups. `remember_briefing` is significantly faster on instances with large embedded corpora.
- **`kb_semantic_matches` reshape** — now an envelope `{enabled, reason, detail, results}` so consumers can branch on `enabled` rather than guessing whether an empty list means opt-out vs. zero hits.
- **Permission ACL filtering** for semantic search — set `bookstack_user_id` in user settings to enable role-based filtering at the candidate-pool layer (much faster than the per-page HTTP fallback).
- **Per-user identity auto-provisioning** — first call to `remember_user action=read` creates the per-user Identity book + Identity page + journal-agent page if missing, returning what was created in `auto_provisioned`.
- **Global org settings** — admin-only `org_identity_page_id`, `org_domains`, `org_required_instructions_page_ids`, `org_ai_usage_policy_page_ids` shared across every user on the instance. First-write-wins for the structural IDs.
- **Owner-only journal pages** — auto-applied content permission lock so journal entries are visible only to the owning agent/user.

### From v0.6.x to v0.7.0

#### What's new

- **`/remember` protocol** — server-side reconstitution + memory CRUD. 12 MCP tools: `remember_briefing`, `remember_whoami`, `remember_user`, `remember_config`, `remember_identity`, `remember_directory`, `remember_journal`, `remember_collage`, `remember_shared_collage`, `remember_user_journal`, `remember_audit`, `remember_search`. HTTP form: `POST /remember/v1/{resource}/{action}`.
- **`/settings` UI** — browser-based configuration page, token-gated via `/authorize`. Settings session cookie stored server-side (in-memory, 8h TTL).
- **YAML-frontmatter provenance** — every collection write stamps `written_by`, `ai_identity_ouid`, `user_id`, `written_at`, `trace_id`, `resource`, `key`, `supersedes_page` at the top of the page body. Invisible in BookStack's renderer; readable by tools.
- **Soft delete** — `remember_*_collection action=delete` prepends `[archived]` to the page name and stamps `deleted: true` in frontmatter rather than hard-deleting.
- **`remember_audit` log** — server-side audit table, scoped to the calling user, captures every write with trace_id and target_page_id.

### From v0.5.3 to v0.6.x

#### What's new

- **Image upload & file attachment tools** — `upload_image` and `upload_attachment` accept either a `staging_id` (from `prepare_upload`), a public `url`, or BookStack's standard direct-upload form data.
- **Staging upload flow** — two-step `prepare_upload` → POST file to returned URL → call `upload_image`/`upload_attachment` with the staging ID. Lets containerized servers receive local files without exposing client paths. 5-minute TTL, single-use, 50MB cap.
- **Move operations** — dedicated `move_page`, `move_chapter`, `move_book_to_shelf` tools (cleaner than the implicit move via update operations).
- **`embed` parameter** on `upload_image` — auto-appends the uploaded image into the target page's content.
- **DNS rebinding protection** — reqwest client pins validated DNS addresses to prevent SSRF via DNS rebinding attacks.

### From v0.5.2 to v0.5.3

v0.5.3 fixes embedding dimension detection, adds Ollama LLM support for summaries, and improves hybrid search scoring.

#### What's new

- **Ollama LLM support** — `BSMCP_LLM_PROVIDER=ollama` for instance summaries using local models (no API key needed)
- **Configurable summary refresh** — `BSMCP_SUMMARY_INTERVAL` (hours) for periodic regeneration instead of one-time only
- **Configurable LLM base URL** — `BSMCP_LLM_API_URL` for remote Ollama instances or custom endpoints
- **Hybrid search scoring fix** — keyword-only results no longer inflate above actual semantic matches via blanket boost. Pages with zero vector similarity are capped below real semantic results.
- **Embedding dimension auto-detection fix** — empty `BSMCP_EMBED_DIMS` env var no longer bypasses Ollama dimension detection (was silently defaulting to 768)
- **Auto-reindex on dimension change** — embedder now detects stored vs actual dimensions and triggers clean reindex automatically

#### What you must do

1. **Pull new images**: `ghcr.io/bees-roadhouse/bsmcp-server:0.5.3` + `ghcr.io/bees-roadhouse/bsmcp-embedder:0.5.3`
2. **Restart** — dimension mismatch auto-reindexes if needed

### From v0.5.1 to v0.5.2

v0.5.2 adds pluggable embedding providers, AI instance summaries, OAuth refresh tokens, and several quality-of-life improvements.

#### What's new

- **Embedding providers** — choose between local ONNX (`local`), Ollama (`ollama`), or OpenAI (`openai`) via `BSMCP_EMBED_PROVIDER`. Ollama auto-detects dimensions. OpenAI works with any compatible endpoint.
- **AI instance summary** — optional LLM call at startup generates a contextual summary of the knowledge base, included in MCP instructions so connecting AI assistants immediately understand what this BookStack is about. Supports OpenRouter, Anthropic, and OpenAI.
- **OAuth refresh tokens** — clients no longer need to re-enter API credentials every 24 hours. Refresh tokens silently issue new access tokens as long as BookStack credentials remain valid.
- **Configurable token TTLs** — `BSMCP_ACCESS_TOKEN_TTL` and `BSMCP_REFRESH_TOKEN_TTL` env vars.
- **Job queue status page** — `/status` now shows all pending/running jobs with progress bars plus recent completed/failed jobs.
- **Similar-page computation** — runs after every embedding job, not just full reindexes.
- **WYSIWYG editing** — all editing tools (`edit_page`, `replace_section`, `append_to_page`, `insert_after`) now explicitly documented to work on WYSIWYG pages.
- **Duplicate title prevention** — instructions tell AI not to include page title as H1 in content.
- **Auto-migration fix** — handles pre-semantic SQLite databases that lack `pages` table.

#### What's automatic

- All schema changes (refresh_tokens table, etc.) are applied on startup
- Existing deployments continue working with no env var changes
- Local ONNX embedding remains the default if `BSMCP_EMBED_PROVIDER` is not set

#### What you must do

1. **Pull new images**: `ghcr.io/bees-roadhouse/bsmcp-server:0.5.2` + `ghcr.io/bees-roadhouse/bsmcp-embedder:0.5.2` (or use `latest`)
2. **Restart** — that's it for the base upgrade

**Optional: Enable AI instance summary** — add LLM env vars:
```bash
BSMCP_LLM_PROVIDER=openrouter  # or: anthropic, openai, ollama
BSMCP_LLM_API_KEY=your-api-key  # not needed for ollama
BSMCP_SUMMARY_INTERVAL=24       # regenerate every 24h (0 = only on first startup)
# Uses BSMCP_EMBED_TOKEN_ID/SECRET for BookStack API access
```

**Optional: Switch to Ollama/OpenAI embeddings** — set `BSMCP_EMBED_PROVIDER`:
```bash
BSMCP_EMBED_PROVIDER=ollama
BSMCP_EMBED_MODEL=nomic-embed-text
BSMCP_EMBED_API_URL=http://ollama:11434
```
Switching provider triggers an automatic clean re-index.

### From v0.5.0 to v0.5.1

v0.5.1 switches the default embedding model and adds automatic model change detection.

#### What's new

- **Default model: EmbeddingGemma-300M** — Google's lightweight embedding model (768 dims, 300M params). Faster and lighter than BGE-large, especially on ARM.
- **Model change detection** — embedder detects model changes via meta table and auto-triggers clean re-index with pgvector dimension adjustment
- **Configurable embedding dimensions** — pgvector column type automatically adjusts when switching models
- **HuggingFace model downloads** — custom ONNX models download automatically from HuggingFace Hub

#### What's automatic

- **Full re-index** — switching from BGE-large (1024 dims) to EmbeddingGemma (768 dims) triggers automatic clean re-index. PostgreSQL column type is altered automatically.
- No env var changes required unless you want to keep the old model

#### What you must do

1. **Pull new images**: `ghcr.io/bees-roadhouse/bsmcp-server:0.5.1` + `ghcr.io/bees-roadhouse/bsmcp-embedder:0.5.1`
2. **Restart** — the embedder auto-detects the model change and re-indexes. Check progress at `/status`.
3. **To keep the old model**: Set `BSMCP_EMBED_MODEL=BAAI/bge-large-en-v1.5` in your embedder env.

### From v0.4.0 to v0.5.0

v0.5.0 is a search quality release — no infrastructure changes, just better results.

#### What's new

- **Hybrid search** — combines vector similarity with BookStack keyword search, weighted blend (70% vector + 20% keyword + blanket boost)
- **Markov blanket re-ranking** — pages whose graph neighbors also scored get a relevance boost (up to +0.15)
- **Tighter chunking** — max chunk size reduced from 2000 to 1200 chars with 150-char paragraph overlap between chunks
- **Higher default threshold** — raised from 0.50 to 0.65 to filter out low-quality matches
- **Auto-reindex on upgrade** — chunk version tracking triggers automatic clean re-index when chunking logic changes
- **`meta` table** — new key-value metadata table in both SQLite and PostgreSQL backends

#### What's automatic

- **Full re-index** — the embedder detects the chunk version change (v1 → v2) and automatically clears all embeddings and re-indexes everything on first startup. No manual `reembed` needed.
- Schema migration — `meta` table created automatically on startup
- All existing env vars and compose files are compatible

#### What you must do

1. **Pull new images**: `ghcr.io/bees-roadhouse/bsmcp-server:0.5.0` + `ghcr.io/bees-roadhouse/bsmcp-embedder:0.5.0`
2. **Restart** — the embedder auto-detects the chunk version change and re-indexes. Check progress at `/status`.
3. **No env var changes required** — new `hybrid` parameter defaults to `true` in the `semantic_search` tool

#### New `semantic_search` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.65` | Minimum score (was 0.50 in v0.4.0) |
| `hybrid` | `true` | Enable keyword + vector blended search |

Results now include a `scoring` breakdown when hybrid mode is on, showing vector, keyword, and blanket_boost components.

### From v0.3.x to v0.4.0

v0.4.0 splits the monolithic `bookstack-mcp` container into separate **server** and **embedder** binaries with a pluggable database layer (SQLite or PostgreSQL + pgvector).

#### What's new

- **Separate containers** — `bsmcp-server` (MCP protocol, OAuth, search) and `bsmcp-embedder` (ONNX model, background embedding, `/embed` HTTP endpoint)
- **PostgreSQL + pgvector** — optional production backend with native HNSW vector indexing
- **Database-backed job queue** — embedding jobs persist across restarts
- **Auto-migration** — switch `BSMCP_DB_BACKEND=postgres` and the server migrates SQLite data automatically
- **Dual MCP transport** — SSE (2024-11-05) and Streamable HTTP (2025-03-26)
- **New page editing tools** — `edit_page`, `append_to_page`, `replace_section`, `insert_after`

#### What's automatic

- SQLite schema is compatible — same tables, same columns
- `worker_id` column auto-added to `embed_jobs` if missing
- Existing embeddings preserved (same model: `BAAI/bge-large-en-v1.5`, same 1024 dimensions)
- Auto-migration from SQLite to PostgreSQL when switching backends

#### What you must do

1. **Replace compose file and images**:
   - Old: single `ghcr.io/bees-roadhouse/bookstack-mcp:latest` container
   - New: `ghcr.io/bees-roadhouse/bsmcp-server:latest` + `ghcr.io/bees-roadhouse/bsmcp-embedder:latest`
   - Use `docker/docker-compose.sqlite.yml` (simple) or `docker/docker-compose.yml` (PostgreSQL)

2. **Add new env vars**:
   ```bash
   # Database backend (required)
   BSMCP_DB_BACKEND=sqlite   # or postgres

   # Embedder connection (required for semantic search)
   BSMCP_EMBEDDER_URL=http://bsmcp-embedder:8081

   # Separate BookStack API token for the embedder (required for semantic search)
   BSMCP_EMBED_TOKEN_ID=<BookStack API token ID>
   BSMCP_EMBED_TOKEN_SECRET=<BookStack API token secret>

   # PostgreSQL (only if switching to postgres)
   BSMCP_DATABASE_URL=postgres://bsmcp:yourpassword@bsmcp-postgres/bsmcp
   BSMCP_DB_PASSWORD=yourpassword
   ```

3. **`BSMCP_EMBED_THREADS` is removed** — use `BSMCP_EMBED_CPUS` (Docker CPU limit) instead.

4. **Update webhook** to use `X-Webhook-Secret` header instead of `?secret=` query param (query param still works but is deprecated).

#### Migrating to PostgreSQL

Set `BSMCP_DB_BACKEND=postgres` and keep the SQLite file accessible at `BSMCP_DB_PATH`. The server auto-migrates all data on startup and renames the SQLite file to `.db.migrated`.

Manual migration is also available:
```bash
docker exec bsmcp-server bsmcp-server migrate \
  --from-sqlite /data/bookstack-mcp.db \
  --to-postgres postgres://bsmcp:yourpassword@bsmcp-postgres/bsmcp
```

Migration copies encrypted tokens as-is (portable when `BSMCP_ENCRYPTION_KEY` matches), converts embeddings from BLOB to pgvector format, and fixes PostgreSQL sequences.

### From v0.1.x to v0.4.0

This is the largest jump — from a single monolithic container with no encryption and no semantic search to the full multi-container architecture.

#### What's automatic

- Plaintext tokens from v0.1.0-0.1.2 are transparently encrypted on first access (the server detects unencrypted values and re-encrypts them in place)
- All database tables are created on startup via `CREATE TABLE IF NOT EXISTS`

#### What you must do

1. **Docker volume rename** (v0.1.0-0.1.2 only — skip if already on v0.1.3+):
   ```bash
   docker compose down
   docker volume create bsmcp-data
   docker run --rm -v mcp-data:/source:ro -v bsmcp-data:/dest alpine cp -a /source/. /dest/
   docker volume rm mcp-data  # after verification
   ```

2. **Update env vars**:
   ```bash
   # REMOVE (no longer recognized):
   # BSMCP_PUBLIC_URL=https://mcp.example.com

   # ADD (required):
   BSMCP_ENCRYPTION_KEY=<generate: openssl rand -base64 48>
   BSMCP_PUBLIC_DOMAIN=mcp.example.com  # domain only, no https://

   # ADD (for semantic search):
   BSMCP_SEMANTIC_SEARCH=true
   BSMCP_WEBHOOK_SECRET=<random string, 16+ chars>
   BSMCP_EMBED_TOKEN_ID=<BookStack API token ID>
   BSMCP_EMBED_TOKEN_SECRET=<BookStack API token secret>
   BSMCP_EMBEDDER_URL=http://bsmcp-embedder:8081

   # ADD (for PostgreSQL — recommended):
   BSMCP_DB_BACKEND=postgres
   BSMCP_DATABASE_URL=postgres://bsmcp:yourpassword@bsmcp-postgres/bsmcp
   BSMCP_DB_PASSWORD=yourpassword
   ```

3. **Replace compose file entirely**:
   - Old: `docker-compose.yml` with `ghcr.io/bees-roadhouse/bookstack-mcp:latest`
   - New (SQLite): `docker/docker-compose.sqlite.yml`
   - New (PostgreSQL): `docker/docker-compose.yml`
   - Images: `ghcr.io/bees-roadhouse/bsmcp-server:latest` + `ghcr.io/bees-roadhouse/bsmcp-embedder:latest`

4. **Create a BookStack API token** for the embedder with read access to all content

5. **Configure webhook** in BookStack (see [Semantic Search Setup](#semantic-search-setup))

6. **Trigger initial embedding** via the `reembed` MCP tool

### From v0.1.2 to v0.1.3

See the [v0.1.3 release notes](https://github.com/bees-roadhouse/bookstack-mcp/releases/tag/v0.1.3):
- New required `BSMCP_ENCRYPTION_KEY` env var
- `BSMCP_PUBLIC_URL` renamed to `BSMCP_PUBLIC_DOMAIN`
- Docker volume renamed `mcp-data` to `bsmcp-data`
- PKCE enforcement for OAuth

## Embedding Providers

Set via `BSMCP_EMBED_PROVIDER`. Changing provider or model triggers an automatic clean re-index.

### Local (default)

Uses fastembed with ONNX Runtime. No external API needed but requires the heavier embedder container.

| Model Name | Dimensions | Parameters | Notes |
|------------|-----------|------------|-------|
| `BAAI/bge-base-en-v1.5` | 768 | 110M | **Default.** Good balance of speed and quality. |
| `BAAI/bge-large-en-v1.5` | 1024 | 335M | Highest quality, heavier. |
| `BAAI/bge-small-en-v1.5` | 384 | 33M | Fastest, lower quality. |
| `embeddinggemma-300m` | 768 | 300M | Google's lightweight model. |

### Ollama

Uses a local or remote Ollama instance. Dimensions auto-detected. No API key needed.

```bash
BSMCP_EMBED_PROVIDER=ollama
BSMCP_EMBED_MODEL=nomic-embed-text        # or any Ollama embedding model
BSMCP_EMBED_API_URL=http://ollama:11434    # default: http://localhost:11434
```

### OpenAI

Uses OpenAI's embedding API or any OpenAI-compatible endpoint.

```bash
BSMCP_EMBED_PROVIDER=openai
BSMCP_EMBED_MODEL=text-embedding-3-small   # default
BSMCP_EMBED_API_KEY=sk-...
BSMCP_EMBED_DIMS=1536                      # must match model output
BSMCP_EMBED_API_URL=https://api.openai.com # or any compatible endpoint
```

## Search Operators

The `search_content` tool supports BookStack's search operators:

- `"exact phrase"` - Exact match
- `{type:page}` - Filter by type (page, chapter, book, shelf)
- `{in_name:term}` - Search within names only
- `{created_by:me}` - Filter by creator
- `[tag_name=value]` - Filter by tag

## Uploading Local Files (Images & Attachments)

The MCP server runs in a container and cannot read files from the client machine's filesystem directly. To upload local images or file attachments, use the two-step **staging upload flow**:

**Step 1:** Call `prepare_upload` — returns a `staging_id` and a full `upload_url`:

```json
{
  "staging_id": "f0103f6c-7c98-46c2-adbe-606ba26937c3",
  "upload_url": "https://your-mcp-host/stage/upload/f0103f6c-7c98-46c2-adbe-606ba26937c3",
  "ttl_seconds": 300
}
```

**Step 2:** POST the file to `upload_url` as multipart form-data. No auth header needed — the `staging_id` (a UUID that can only be generated via an authenticated MCP call) acts as the auth token for the one-time upload:

```bash
curl -X POST -F "file=@/path/to/image.jpg" \
  "https://your-mcp-host/stage/upload/f0103f6c-7c98-46c2-adbe-606ba26937c3"
```

**Step 3:** Call `upload_image` (or `upload_attachment`) with the `staging_id`:

```json
{
  "name": "Banner Logo",
  "uploaded_to": 1908,
  "staging_id": "f0103f6c-7c98-46c2-adbe-606ba26937c3",
  "mime_type": "image/jpeg",
  "embed": true
}
```

The staging slot is **consumed on first use** (destructively removed from the store) and **auto-expires after 5 minutes**. Maximum file size is 50MB.

### The `embed` parameter

`upload_image` accepts an `embed` boolean parameter (default `false`). When `embed=true`, the image is automatically appended to the target page's content after uploading, so you don't need a separate `edit_page` or `append_to_page` call. Works for both markdown and WYSIWYG pages.

### Alternative: `url` parameter

If the file is already hosted at a public URL the MCP server can reach, you can skip the staging flow entirely and pass the `url` parameter directly to `upload_image` or `upload_attachment`. The server will fetch the file and forward it to BookStack.

### Currently Claude Code only

**The staging upload flow currently only works from [Claude Code](https://claude.com/claude-code) (the CLI tool).** It does not work from Claude.ai's web custom connectors or Claude Desktop custom connectors.

The reason: Step 2 requires the MCP client to make an outbound HTTP POST to the MCP server's staging endpoint with the file bytes. Claude Code runs locally and has shell access (via its `Bash` tool), so it can `curl` the file directly. Claude.ai's remote MCP connector runs the MCP client inside Anthropic's sandboxed proxy infrastructure, which does not expose a mechanism for the client to make arbitrary HTTP file uploads to third-party hosts. Claude Desktop has similar limitations today.

If you're using Claude.ai or Claude Desktop, you can still use `upload_image` with the `url` parameter for files that are already web-accessible, or upload through the BookStack web UI directly.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for build instructions, branching model, CI/CD (artifact-before-merge), versioning, and the workflow for adding new tools.

## License

MIT
