# Development

How to build, test, and ship changes to `bookstack-mcp`. This document is the contributor entry point — README is the user-facing project overview, this file is for engineers.

## Prerequisites

- Rust toolchain (stable)
- Docker + Docker Buildx (for multi-arch builds)
- A BookStack instance with API token

## Local Build

```bash
# Full workspace
cargo build --release

# Individual crates
cargo build --release -p bsmcp-server
cargo build --release -p bsmcp-embedder
cargo build --release -p bsmcp-worker

# Check without building
cargo check
```

## Running Locally

Copy `.env.example` to `.env` and configure. At minimum:

```
BSMCP_BOOKSTACK_URL=https://your-bookstack.example.com
BSMCP_ENCRYPTION_KEY=your-32-char-key-here
```

Then:

```bash
cargo run -p bsmcp-server
cargo run -p bsmcp-embedder  # optional, for semantic search
cargo run -p bsmcp-worker    # optional, for the reconciliation index
```

## Docker Compose

Two deployment options:

```bash
# PostgreSQL backend (recommended for production)
docker compose -f docker/docker-compose.yml up -d

# SQLite backend (simpler, single-node)
docker compose -f docker/docker-compose.sqlite.yml up -d
```

## Branching

The canonical reference for the org-wide branching standard is the [Branching Strategy](https://kb.beesroadhouse.com/books/developer-operations-devops/page/branching-strategy) page in the Bee's Roadhouse DevOps book. This file mirrors the policy that applies to this repo specifically.

- `development` — default branch; all active work lands here. **PR required** (org-level `Default Branch Protection` ruleset: 1 approving review, thread resolution, no force-push, no deletion). PR merges trigger CI build/package.
- `release` — stable/production; merged from development when ready to ship. PR required (org-level `Release Branch Protection` ruleset: 1 approving review, merge-commit only, no force-push, no deletion).
- Work branches use the four-prefix taxonomy below.

No `main` or `master` branches exist.

### Work branch prefixes

| Prefix | Use for | GitHub labels | Default semver bump | Example |
|--------|---------|---------------|---------------------|---------|
| `feature/{name}` | New capability that didn't exist | `type:enhancement` + `category:feature` | minor | `feature/export-api` |
| `improvement/{name}` | Existing capability, done better | `type:enhancement` + `category:improvement` | minor | `improvement/search-relevance` |
| `refactor/{name}` | Design or structure redo | `type:problem` + `category:refactor` | patch (or minor if external behavior changes) | `refactor/auth-flow` |
| `bug/{name}` | Implementation mistake, something broken | `type:problem` + `category:bug` | patch | `bug/oauth-token-refresh` |

Breaking changes are orthogonal to type — prefix the **PR title** with `BREAKING:` regardless of the branch prefix to force a major-version bump.

### Workflow

```
1. git checkout development && git pull
2. git checkout -b improvement/my-change      # or feature/, refactor/, bug/
3. ... commit work (signed via SSH; see Commit Signing below) ...
4. git push -u origin improvement/my-change
5. Open PR against development; apply the matching type: + category: labels
6. CI runs verify-pr.yml (cargo test + clippy) and generate-artifacts.yml (SBOM/STRUCTURE auto-commit)
7. Squash-merge PR into development; delete the work branch
8. build-development.yml builds + pushes the :dev family of image tags
9. When ready to ship: open PR from development -> release
```

All changes go through a PR — direct pushes are blocked by the org ruleset (returns `GH013`). For CI emergencies (workflow-bootstrap gap, broken build), use `workflow_dispatch` on `build-development.yml` rather than bypassing the ruleset. Org admins can `gh pr merge --admin` for bootstrap PRs and small docs touchups, but `--admin` still routes through the PR machinery (CI runs, audit trail preserved) — it is not a direct push.

## CI/CD

Build-on-merge pattern. Reference docs:

- BR DevOps [Docker Image Build Workflows (1905)](https://kb.beesroadhouse.com/books/developer-operations-devops/page/docker-image-build-workflows) — canonical trigger / tag / cache shape.
- BR DevOps [Branching Strategy (1860)](https://kb.beesroadhouse.com/books/developer-operations-devops/page/branching-strategy) — branch model and direct-push authorization.

**PR-time gating is fast (cargo test + clippy). Images build on merge.** PRs trigger `verify-pr.yml` — `cargo check`, `cargo clippy`, `cargo test --workspace` — on `ubuntu-latest`. No image build on PRs. After squash-merge, `build-development.yml` (push to `development`) or `release.yml` (push to `release` / `v*` tag) builds + pushes the appropriate multi-arch image set. All builds run on GitHub-hosted `ubuntu-latest` — no self-hosted dependency.

### Contributor flow (per PR)

```
1. git checkout -b improvement/my-change
2. ... commit work, sign each commit ...
3. git push -u origin improvement/my-change
4. Open PR; verify-pr.yml runs cargo test + clippy
5. Squash-merge into development; delete the work branch
6. build-development.yml builds + pushes :dev / :{version}-dev family of tags
```

No local image build needed. `scripts/publish-pr-image.sh` is still in the repo as an out-of-band escape hatch when CI is unavailable, but it's not part of the normal flow.

### Cargo target / registry caching

Both Dockerfiles use BuildKit `--mount=type=cache` for `target/`, `~/.cargo/registry`, and `~/.cargo/git`. CI uses scoped GHA cache (`scope=server`, `scope=embedder`, `scope=worker`) so parallel jobs don't evict each other's layers. Cache mount IDs include `$TARGETPLATFORM` so linux/amd64 and linux/arm64 don't poison each other's caches.

### Embedder is opt-in for deployments

`bsmcp-embedder` is required only when running the **built-in** embedder provider (the default `BSMCP_EMBED_PROVIDER=local` ONNX model). Deployments configured for external providers (`ollama`, `openai`) don't need the embedder container at all — `bsmcp-server` talks to the external endpoint directly.

### What runs on what

| Event | Workflow | What happens |
|-------|----------|-------------|
| Push to a work branch with **no open PR** | nothing | test locally |
| `pull_request: opened/synchronize/reopened` against `development` or `release` | `verify-pr.yml` | `cargo check` + `cargo clippy -- -D warnings` + `cargo test --workspace` on `ubuntu-latest`. Fast, image-free. |
| Same trigger | `generate-artifacts.yml` | regenerates `SBOM.md` + `STRUCTURE.md`, commits to PR source branch (re-fire loop broken by `paths-ignore`). SBOM/STRUCTURE conflicts on rebase resolve via `merge=ours` in `.gitattributes`. |
| `push` to `development` (PR-merge commit or otherwise) | `build-development.yml` | multi-arch build + push of all three images on `ubuntu-latest`. Tags: `:dev`, `:dev-{sha}`, `:{version}-dev`, `:{version}-dev-{sha}`. |
| `workflow_dispatch` on `build-development.yml` | `build-development.yml` | manual rebuild at the current `development` HEAD. Same tag set as the push trigger. |
| `push` to `release` (always a PR-merge from development) | `release.yml` (`build-release-images` + `github-release-on-merge` + `release-binaries-on-merge`) | builds + pushes `:{version}` / `:{version}-{sha}` / `:release` / `:latest` on `ubuntu-latest`; creates the `v{version}` git tag and GitHub Release entry; builds `bsmcp-server` native binaries for 5 targets and attaches them. |
| `v*` tag push (emergency hotfix only) | `release.yml` (`tag-release` + `github-release-on-tag` + `release-binaries-on-tag`) | builds & pushes semver-tagged images on `ubuntu-latest`, creates the Release, attaches the server binaries. Use only when the normal PR flow isn't available. |
| `workflow_dispatch` on `release.yml` | `release.yml` | manual recovery path for the release stream |

### Why this shape

- **Build on merge, not on PR.** PR-time gating is fast (`cargo test` + `clippy`); image builds run once per merge. Removes the failure mode where a PR-time build has to complete for a downstream retag step to find an artifact — there is no downstream retag step.
- **Pinned to `ubuntu-latest`.** GitHub-hosted runners are always available. Self-hosted runners can return as a per-job opt-in once a runner pool is reliable; for now no `[self-hosted, ...]` label appears in any workflow.
- **Native binaries: server only.** `bsmcp-server` is pure Rust + bundled SQLite and cross-compiles cleanly. `bsmcp-embedder` depends on `fastembed` → ONNX Runtime → a per-platform C++ shared library; bare binaries would need ONNX Runtime installed on the host. Container is the only supported distribution for the embedder.
- **External fork PRs are skipped.** Forks can't push to `ghcr.io/bees-roadhouse/*`. `verify-pr.yml` and `generate-artifacts.yml` gate on `head.repo.full_name == github.repository`.

### Tag conventions on GHCR

No per-PR image tag. PRs don't build images. Commit-level pinning during review is unnecessary — the PR's source tree IS the artifact to review; reviewers can `cargo build` locally if they want to test.

Development stream (pushed by `build-development.yml` on push to `development`):
- `dev` — rolling, latest dev build
- `dev-{sha}` — immutable per-commit
- `{version}-dev` — version-level dev rolling
- `{version}-dev-{sha}` — version-level dev immutable

Release stream (pushed by `release.yml`'s `build-release-images` on push to `release`):
- `latest` — rolling, latest release
- `release` — alias for `latest`
- `{version}` — pinned semver (e.g., `0.11.0`)
- `{version}-{sha}` — immutable per-release-merge

Tag-push hotfix (`v*` tag → `release.yml` `tag-release`):
- `{version}`, `{major}.{minor}`, `{major}` — full semver hierarchy

Images are published to `ghcr.io/bees-roadhouse/bsmcp-server`, `ghcr.io/bees-roadhouse/bsmcp-embedder`, and `ghcr.io/bees-roadhouse/bsmcp-worker` for `linux/amd64` and `linux/arm64`.

### Native binary release artifacts

Each GitHub Release attaches `bsmcp-server` archives for these targets:

| Target | Archive | Runner |
|--------|---------|--------|
| `x86_64-unknown-linux-gnu` | `.tar.gz` | ubuntu-22.04 (glibc ≥ 2.35) |
| `aarch64-unknown-linux-gnu` | `.tar.gz` | ubuntu-22.04 + cross-linker |
| `x86_64-apple-darwin` | `.tar.gz` | macos-13 |
| `aarch64-apple-darwin` | `.tar.gz` | macos-14 |
| `x86_64-pc-windows-msvc` | `.zip` | windows-2022 |

Each archive contains the `bsmcp-server` (or `.exe`) binary plus `README.md` and `LICENSE`.

### Branch protection

Protection lives at the **organization level** via two GitHub Rulesets that apply to every repo in `bees-roadhouse`:

- `Default Branch Protection` (`~DEFAULT_BRANCH`) — `pull_request` (1 approval, thread resolution), `non_fast_forward`, `deletion`. Bypass: `OrganizationAdmin` in `pull_request` mode.
- `Release Branch Protection` (`refs/heads/release`, `refs/heads/release/*`, `refs/heads/release-*`) — `pull_request` (1 approval, merge-commit only, thread resolution), `non_fast_forward`, `deletion`. Bypass: `OrganizationAdmin` in `pull_request` mode.

Both rulesets enforce on every ref update on the targeted branches — direct pushes are rejected with `GH013`. CI runs on every PR push, so regressions are caught before merge. The `OrganizationAdmin` bypass uses `bypass_mode: pull_request` (skip review on a PR via `gh pr merge --admin`), not `repository` (which would allow direct push) — direct push is intentionally not configured.

Required status check for `verify-pr / verify` (cargo test + clippy on PR) is **not** wired up yet. After this CI rework lands and the check name stabilizes, a follow-up will add it to both rulesets.

### Commit signing

Every commit must be signed via SSH using 1Password's SSH agent. See the [Commit Signing](https://kb.beesroadhouse.com/books/developer-operations-devops/page/commit-signing) page in the DevOps book for full configuration.

## Versioning

Semantic versioning (`MAJOR.MINOR.PATCH`). Version lives in workspace `Cargo.toml`.

Default semver bump per branch prefix (override with `BREAKING:` in the PR title for a major bump):

- `feature/*` — minor
- `improvement/*` — minor
- `refactor/*` — patch (minor if external behavior changes)
- `bug/*` — patch

Release: open PR from development -> release. The release-merge fires `build-release-images` (builds + pushes `:{version}` / `:release` / `:latest`) and `github-release-on-merge` (creates the GitHub Release with native binaries).

## Testing

```bash
cargo test
cargo clippy
```

## Adding a New Tool

1. Add API method to `BookStackClient` in `crates/bsmcp-server/src/bookstack.rs`
2. Add match arm in `execute_tool()` in `crates/bsmcp-server/src/mcp.rs`
3. Add tool definition in `tool_definitions()` in the same file
4. Use existing helpers: `arg_str`, `arg_i64`, `arg_i64_required`, `arg_str_default`, `format_json`

## Migration

**SQLite -> PostgreSQL auto-migration:** When `BSMCP_DB_BACKEND=postgres` and a SQLite DB exists at `BSMCP_DB_PATH`, the server auto-migrates on startup and renames the file to `.db.migrated`.

**Manual migration:**

```bash
bsmcp-server migrate --from-sqlite /path/to/db --to-postgres postgres://user:pass@host/db
```

Migrates `access_tokens`, `pages`, `chunks` (BLOB -> pgvector), `relationships`, `embed_jobs`. Validates row counts.

## Multi-arch Docker builds (manual)

Normally CI handles this. For local multi-arch testing:

```bash
docker buildx build --builder multiarch --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.server \
  -t ghcr.io/bees-roadhouse/bsmcp-server:VERSION --push .

docker buildx build --builder multiarch --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.embedder \
  -t ghcr.io/bees-roadhouse/bsmcp-embedder:VERSION --push .

docker buildx build --builder multiarch --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.worker \
  -t ghcr.io/bees-roadhouse/bsmcp-worker:VERSION --push .
```
