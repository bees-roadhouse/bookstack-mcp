# Repository Structure

_Auto-generated on 2026-05-28 18:57 UTC from commit `bd87131`._

```
.
├── crates
│   ├── bsmcp-common
│   │   ├── src
│   │   │   ├── acl.rs
│   │   │   ├── bookstack.rs
│   │   │   ├── chunking.rs
│   │   │   ├── config.rs
│   │   │   ├── db.rs
│   │   │   ├── index.rs
│   │   │   ├── lib.rs
│   │   │   ├── rate_limit.rs
│   │   │   ├── settings.rs
│   │   │   ├── types.rs
│   │   │   └── vector.rs
│   │   └── Cargo.toml
│   ├── bsmcp-db-postgres
│   │   ├── src
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   ├── bsmcp-db-sqlite
│   │   ├── src
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   ├── bsmcp-embedder
│   │   ├── src
│   │   │   ├── embed.rs
│   │   │   ├── main.rs
│   │   │   ├── pipeline.rs
│   │   │   └── rerank.rs
│   │   └── Cargo.toml
│   ├── bsmcp-server
│   │   ├── src
│   │   │   ├── main.rs
│   │   │   ├── mcp.rs
│   │   │   ├── migrate.rs
│   │   │   ├── oauth.rs
│   │   │   ├── semantic.rs
│   │   │   ├── settings_ui.rs
│   │   │   ├── sse.rs
│   │   │   └── staging.rs
│   │   └── Cargo.toml
│   └── bsmcp-worker
│       ├── src
│       │   ├── lib.rs
│       │   └── main.rs
│       └── Cargo.toml
├── docker
│   ├── Dockerfile.embedder
│   ├── Dockerfile.server
│   ├── Dockerfile.worker
│   ├── docker-compose.sqlite.yml
│   └── docker-compose.yml
├── scripts
│   └── publish-pr-image.sh
├── CLAUDE.md
├── Cargo.lock
├── Cargo.toml
├── DEVELOPMENT.md
├── README.md
├── SBOM.md
├── STRUCTURE.md
└── entrypoint.sh

16 directories, 47 files
```
