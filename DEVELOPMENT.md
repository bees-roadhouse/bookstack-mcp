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
cargo run -p bsmcp-embedder                       # optional, for semantic search
cargo run -p bsmcp-embedder -- --role=worker      # optional, reconciliation index worker
cargo run -p bsmcp-embedder -- --role=both        # both loops in one process
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

- `main` — the default branch and the only long-lived one. All work lands here by PR. **PR required** (org-level `Default Branch Protection` ruleset, targeting `~DEFAULT_BRANCH`: 0 required approvals as of 2026-05-25, thread resolution, no force-push, no deletion). Merging to `main` publishes the rolling `:dev` image family.
- Work branches use the four-prefix taxonomy below.

No `development`, `release`, or `master` branches exist. **Merging to `main` is the release** for the rolling stream — there is no promote step. Versioned releases are cut by pushing a `v*` tag.

The org moved off the `development` / `release` model on **2026-07-05**; this repo's workflows and this file were swept to match on 2026-08-11.

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
1. git checkout main && git pull
2. git checkout -b improvement/my-change      # or feature/, refactor/, bug/
3. ... commit work (signed via SSH; see Commit Signing below) ...
4. git push -u origin improvement/my-change
5. Open PR against main; apply the matching type: + category: labels
6. CI runs verify-pr.yml (cargo test + clippy) and generate-artifacts.yml (SBOM/STRUCTURE auto-commit)
7. Squash-merge PR into main; delete the work branch
8. build-main.yml builds + pushes the :dev family of image tags
```

There is no step 9. The merge is the deploy — any Portainer git stack tracking `main` picks the new image up on its next poll.

To cut a versioned release, bump the workspace version in a normal PR, then tag the merge commit:

```bash
git tag v0.14.0 && git push origin v0.14.0
```

`release.yml` fires on the tag: multi-arch `:{version}` / `:{version}-{sha}` / `:release` / `:latest` images, the GitHub Release entry, and native `bsmcp-server` binaries for 5 targets. The tag must match the `Cargo.toml` workspace version or the run fails.

All changes go through a PR — direct pushes are blocked by the org ruleset (returns `GH013`). For CI emergencies (workflow-bootstrap gap, broken build), use `workflow_dispatch` on `build-main.yml` rather than bypassing the ruleset. Org admins can `gh pr merge --admin` for bootstrap PRs and small docs touchups, but `--admin` still routes through the PR machinery (CI runs, audit trail preserved) — it is not a direct push.

## CI/CD

Build-on-merge pattern. Reference docs:

- BR DevOps [Docker Image Build Workflows (1905)](https://kb.beesroadhouse.com/books/developer-operations-devops/page/docker-image-build-workflows) — canonical trigger / tag / cache shape.
- BR DevOps [Branching Strategy (1860)](https://kb.beesroadhouse.com/books/developer-operations-devops/page/branching-strategy) — branch model and direct-push authorization.

**PR-time gating is fast (cargo test + clippy). Images build on merge.** PRs trigger `verify-pr.yml` — `cargo check`, `cargo clippy`, `cargo test --workspace` — on `ubuntu-latest`. No image build on PRs. After squash-merge, `build-main.yml` (push to `main`) builds + pushes the rolling `:dev` image family; `release.yml` (push of a `v*` tag) builds + pushes the versioned `:latest` / `:release` set. All builds run on GitHub-hosted `ubuntu-latest` — no self-hosted dependency.

The `:dev` tag name predates the single-`main` migration and is kept deliberately: `docker/docker-compose.yml`, `docker/docker-compose.sqlite.yml`, and the Portainer stacks all pin `:dev`. It now means "the rolling build off `main`", not "built from a development branch".

### Contributor flow (per PR)

```
1. git checkout -b improvement/my-change
2. ... commit work, sign each commit ...
3. git push -u origin improvement/my-change
4. Open PR; verify-pr.yml runs cargo test + clippy
5. Squash-merge into main; delete the work branch
6. build-main.yml builds + pushes :dev / :{version}-dev family of tags
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
| `pull_request: opened/synchronize/reopened` against `main` | `verify-pr.yml` | `cargo check` + `cargo clippy -- -D warnings` + `cargo test --workspace` on `ubuntu-latest`. Fast, image-free. |
| Same trigger | `generate-artifacts.yml` | regenerates `SBOM.md` + `STRUCTURE.md`, commits to PR source branch (re-fire loop broken by `paths-ignore`). SBOM/STRUCTURE conflicts on rebase resolve via `merge=ours` in `.gitattributes`. |
| `push` to `main` (PR-merge commit or otherwise) | `build-main.yml` | multi-arch build + push of both images on `ubuntu-latest`. Tags: `:dev`, `:dev-{sha}`, `:{version}-dev`, `:{version}-dev-{sha}`. |
| `workflow_dispatch` on `build-main.yml` | `build-main.yml` | manual rebuild at the current `main` HEAD. Same tag set as the push trigger. |
| `v*` tag push | `release.yml` (`tag-release` + `alias-worker-to-embedder-on-tag` + `github-release-on-tag` + `release-binaries-on-tag`) | builds + pushes `:{version}` / `:{version}-{sha}` / `:release` / `:latest` on `ubuntu-latest`; creates the GitHub Release entry; builds `bsmcp-server` native binaries for 5 targets and attaches them. **This is the release mechanism** — there is no release branch. |
| `workflow_dispatch` on `release.yml` | `release.yml` | manual recovery path for the release stream |

### Why this shape

- **Build on merge, not on PR.** PR-time gating is fast (`cargo test` + `clippy`); image builds run once per merge. Removes the failure mode where a PR-time build has to complete for a downstream retag step to find an artifact — there is no downstream retag step.
- **Pinned to `ubuntu-latest`.** GitHub-hosted runners are always available. Self-hosted runners can return as a per-job opt-in once a runner pool is reliable; for now no `[self-hosted, ...]` label appears in any workflow.
- **Native binaries: server only.** `bsmcp-server` is pure Rust + bundled SQLite and cross-compiles cleanly. `bsmcp-embedder` depends on `fastembed` → ONNX Runtime → a per-platform C++ shared library; bare binaries would need ONNX Runtime installed on the host. Container is the only supported distribution for the embedder.
- **External fork PRs are skipped.** Forks can't push to `ghcr.io/bees-roadhouse/*`. `verify-pr.yml` and `generate-artifacts.yml` gate on `head.repo.full_name == github.repository`.

### Tag conventions on GHCR

No per-PR image tag. PRs don't build images. Commit-level pinning during review is unnecessary — the PR's source tree IS the artifact to review; reviewers can `cargo build` locally if they want to test.

Rolling stream (pushed by `build-main.yml` on push to `main`). The `dev`
prefix is historical — these are builds off `main`, not off a development
branch — and is kept because the compose files and Portainer stacks pin it:
- `dev` — rolling, latest build off `main`
- `dev-{sha}` — immutable per-commit
- `{version}-dev` — version-level rolling
- `{version}-dev-{sha}` — version-level immutable

Release stream (pushed by `release.yml`'s `tag-release` on a `v*` tag push):
- `latest` — rolling, latest release
- `release` — alias for `latest`
- `{version}` — pinned semver (e.g., `0.11.0`)
- `{version}-{sha}` — immutable per-release-merge

Tag-push hotfix (`v*` tag → `release.yml` `tag-release`):
- `{version}`, `{major}.{minor}`, `{major}` — full semver hierarchy

Images are published to `ghcr.io/bees-roadhouse/bsmcp-server` and `ghcr.io/bees-roadhouse/bsmcp-embedder` for `linux/amd64` and `linux/arm64`. For v0.13.0 only, `ghcr.io/bees-roadhouse/bsmcp-worker:<tag>` is published as a transitional registry alias of `ghcr.io/bees-roadhouse/bsmcp-embedder:<tag>` so existing compose files keep pulling — operators must still set `--role=worker` on the running container. The alias is removed in v0.14.0.

### Native binary release artifacts

Each GitHub Release attaches `bsmcp-server` archives for these targets:

| Target | Archive | Runner |
|--------|---------|--------|
| `x86_64-unknown-linux-gnu` | `.tar.gz` | ubuntu-22.04 (glibc ≥ 2.35) |
| `aarch64-unknown-linux-gnu` | `.tar.gz` | ubuntu-22.04 + cross-linker |
| `x86_64-apple-darwin` | `.tar.gz` | macos-14 (cross-compiled from Apple Silicon) |
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

Release: bump the workspace version in a normal PR to `main`, then tag the merge commit — `git tag v0.14.0 && git push origin v0.14.0`. The tag fires `tag-release` (builds + pushes `:{version}` / `:release` / `:latest`) and `github-release-on-tag` (creates the GitHub Release with native binaries). The tag must match the `Cargo.toml` version or the run fails.

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

# v0.13.0: bsmcp-worker is the same image as bsmcp-embedder, selected
# via --role=worker at run-time. No separate Dockerfile.
```
