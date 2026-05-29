use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{AeadCore, Aes256Gcm};
use async_trait::async_trait;
use base64::Engine;
use pgvector::Vector;
use sha2::Digest;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use zeroize::Zeroizing;

use bsmcp_common::config::{access_token_ttl, refresh_token_ttl};
use bsmcp_common::db::{DbBackend, IndexDb, SemanticDb};
use bsmcp_common::index::*;
use bsmcp_common::settings::GlobalSettings;
use bsmcp_common::types::*;

const BASE64: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::STANDARD;

pub struct PostgresDb {
    pool: PgPool,
    encryption_key: Zeroizing<[u8; 32]>,
}

impl PostgresDb {
    pub async fn new(database_url: &str, encryption_key: &str) -> Result<Self, String> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await
            .map_err(|e| format!("Failed to connect to PostgreSQL: {e}"))?;

        // Create pgvector extension and access_tokens table
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&pool)
            .await
            .map_err(|e| format!("Failed to create vector extension: {e}"))?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS access_tokens (
                token TEXT PRIMARY KEY,
                token_id TEXT NOT NULL,
                token_secret TEXT NOT NULL,
                created_at BIGINT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to create access_tokens table: {e}"))?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_tokens_created ON access_tokens(created_at)")
            .execute(&pool)
            .await
            .ok();

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS refresh_tokens (
                token TEXT PRIMARY KEY,
                token_id TEXT NOT NULL,
                token_secret TEXT NOT NULL,
                created_at BIGINT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to create refresh_tokens table: {e}"))?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_refresh_tokens_created ON refresh_tokens(created_at)",
        )
        .execute(&pool)
        .await
        .ok();

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS global_settings (
                id INT PRIMARY KEY CHECK (id = 1),
                hive_shelf_id BIGINT,
                user_journals_shelf_id BIGINT,
                set_by_token_hash TEXT,
                updated_at BIGINT NOT NULL DEFAULT 0,
                /* Issue #80 — named scope cuts (HashMap<String, ScopeDef>)
                   and the precision-cascade pool multipliers, persisted as
                   JSON blobs. Empty/NULL means: use defaults. */
                kb_scopes_json TEXT,
                cascade_multipliers_json TEXT
            )",
        )
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to create global_settings table: {e}"))?;

        // v0.10.0 cleanup migrations — drop the briefing/per-user-settings
        // surface stripped in #78, plus leftovers from earlier reverts.
        // Idempotent via IF EXISTS.
        for sql in [
            "DROP TABLE IF EXISTS user_settings CASCADE",
            "DROP TABLE IF EXISTS user_role_cache CASCADE",
            "DROP TABLE IF EXISTS remember_audit CASCADE",
            "DROP TABLE IF EXISTS token_bindings CASCADE",
            "DROP TABLE IF EXISTS sessions CASCADE",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS org_required_instructions_page_ids",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS org_ai_usage_policy_page_ids",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS org_identity_page_id",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS org_domains",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS guide_page_id",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS policies_scope",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS sops_scope",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS best_practices_scope",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS friendly_structure",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS full_content_in_briefing",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS strict_setup",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS default_ai_identity_page_id",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS default_ai_identity_name",
            "ALTER TABLE global_settings DROP COLUMN IF EXISTS default_ai_identity_ouid",
            // Issue #80 — additive JSON columns for kb_scopes + cascade
            // multipliers. IF NOT EXISTS keeps the migration idempotent.
            "ALTER TABLE global_settings ADD COLUMN IF NOT EXISTS kb_scopes_json TEXT",
            "ALTER TABLE global_settings ADD COLUMN IF NOT EXISTS cascade_multipliers_json TEXT",
        ] {
            sqlx::query(sql).execute(&pool).await.ok();
        }

        sqlx::query("INSERT INTO global_settings (id, updated_at) VALUES (1, 0) ON CONFLICT (id) DO NOTHING")
            .execute(&pool).await.ok();

        // v1.0.0 — DB-as-index schema. Mirror of every BookStack content item
        // we care about. Phase 3 ships the schema only; the reconciliation
        // worker (Phase 4) populates these tables.
        for sql in [
            "CREATE TABLE IF NOT EXISTS bookstack_shelves (
                shelf_id BIGINT PRIMARY KEY,
                name TEXT NOT NULL,
                slug TEXT NOT NULL,
                shelf_kind TEXT NOT NULL,
                indexed_at BIGINT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT FALSE
            )",
            "CREATE TABLE IF NOT EXISTS bookstack_books (
                book_id BIGINT PRIMARY KEY,
                name TEXT NOT NULL,
                slug TEXT NOT NULL,
                shelf_id BIGINT,
                identity_ouid TEXT,
                book_kind TEXT NOT NULL,
                indexed_at BIGINT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT FALSE
            )",
            "CREATE INDEX IF NOT EXISTS idx_bookstack_books_shelf ON bookstack_books(shelf_id)",
            "CREATE INDEX IF NOT EXISTS idx_bookstack_books_identity ON bookstack_books(identity_ouid)",
            "CREATE TABLE IF NOT EXISTS bookstack_chapters (
                chapter_id BIGINT PRIMARY KEY,
                book_id BIGINT NOT NULL,
                name TEXT NOT NULL,
                slug TEXT NOT NULL,
                identity_ouid TEXT,
                chapter_kind TEXT NOT NULL,
                archive_year INTEGER,
                indexed_at BIGINT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT FALSE
            )",
            "CREATE INDEX IF NOT EXISTS idx_bookstack_chapters_book ON bookstack_chapters(book_id)",
            "CREATE INDEX IF NOT EXISTS idx_bookstack_chapters_identity ON bookstack_chapters(identity_ouid)",
            "CREATE TABLE IF NOT EXISTS bookstack_pages (
                page_id BIGINT PRIMARY KEY,
                book_id BIGINT NOT NULL,
                chapter_id BIGINT,
                name TEXT NOT NULL,
                slug TEXT NOT NULL,
                url TEXT,
                page_created_at TEXT,
                page_updated_at TEXT,
                identity_ouid TEXT,
                page_kind TEXT NOT NULL,
                page_key TEXT,
                archive_year INTEGER,
                indexed_at BIGINT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT FALSE
            )",
            "CREATE INDEX IF NOT EXISTS idx_bookstack_pages_book ON bookstack_pages(book_id)",
            "CREATE INDEX IF NOT EXISTS idx_bookstack_pages_chapter ON bookstack_pages(chapter_id)",
            "CREATE INDEX IF NOT EXISTS idx_bookstack_pages_identity_kind ON bookstack_pages(identity_ouid, page_kind)",
            // Dedup enforcement: at most one non-deleted classified page per
            // (identity, kind, key). NULL identity or NULL key are excluded so
            // unclassified pages never trip the constraint.
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_bookstack_pages_dedup
                ON bookstack_pages(identity_ouid, page_kind, page_key)
                WHERE deleted = FALSE AND identity_ouid IS NOT NULL AND page_key IS NOT NULL",
            // Page-body cache. One row per page; refreshed when BookStack's
            // page_updated_at advances. Cache hit when our row's
            // page_updated_at equals bookstack_pages.page_updated_at.
            "CREATE TABLE IF NOT EXISTS page_cache (
                page_id BIGINT PRIMARY KEY,
                markdown TEXT,
                raw_markdown TEXT,
                html TEXT,
                cached_at BIGINT NOT NULL,
                page_updated_at TEXT
            )",
            // Reconciliation job queue. Mirrors `embed_jobs` in shape so the
            // worker pattern stays familiar.
            "CREATE TABLE IF NOT EXISTS index_jobs (
                id BIGSERIAL PRIMARY KEY,
                scope TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                triggered_by TEXT NOT NULL,
                started_at BIGINT,
                finished_at BIGINT,
                progress BIGINT NOT NULL DEFAULT 0,
                total BIGINT NOT NULL DEFAULT 0,
                error TEXT,
                resolved_status TEXT,
                prev_status TEXT,
                resolved_at BIGINT,
                retry_of BIGINT
            )",
            "CREATE INDEX IF NOT EXISTS idx_index_jobs_pending ON index_jobs(status) WHERE status = 'pending'",
            "ALTER TABLE index_jobs ADD COLUMN IF NOT EXISTS resolved_status TEXT",
            "ALTER TABLE index_jobs ADD COLUMN IF NOT EXISTS prev_status TEXT",
            "ALTER TABLE index_jobs ADD COLUMN IF NOT EXISTS resolved_at BIGINT",
            "ALTER TABLE index_jobs ADD COLUMN IF NOT EXISTS retry_of BIGINT",
            "UPDATE index_jobs SET status = 'failed' \
             WHERE status = 'error' AND resolved_status IS NULL",
            // Singleton bookkeeping for the indexer (last_full_walk_at,
            // last_delta_walk_at, etc.).
            "CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
        ] {
            sqlx::query(sql)
                .execute(&pool)
                .await
                .map_err(|e| format!("Failed to create v1.0.0 index schema: {e}"))?;
        }

        let hash = sha2::Sha256::digest(encryption_key.as_bytes());
        let mut key = Zeroizing::new([0u8; 32]);
        key.copy_from_slice(&hash);

        Ok(Self {
            pool,
            encryption_key: key,
        })
    }

    /// SHA-256 hash a bearer token before storing as primary key.
    fn hash_token(token: &str) -> String {
        let hash = sha2::Sha256::digest(token.as_bytes());
        format!("{hash:x}")
    }

    fn encrypt(&self, plaintext: &str) -> String {
        let cipher = Aes256Gcm::new((&*self.encryption_key).into());
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(&nonce, plaintext.as_bytes())
            .expect("AES-GCM encryption failed");
        let mut combined = nonce.to_vec();
        combined.extend_from_slice(&ciphertext);
        BASE64.encode(&combined)
    }

    fn decrypt(&self, stored: &str) -> Option<String> {
        let combined = BASE64.decode(stored).ok()?;
        if combined.len() < 12 {
            return None;
        }
        let (nonce_bytes, ciphertext) = combined.split_at(12);
        let nonce = aes_gcm::Nonce::from_slice(nonce_bytes);
        let cipher = Aes256Gcm::new((&*self.encryption_key).into());
        let plaintext = cipher.decrypt(nonce, ciphertext).ok()?;
        String::from_utf8(plaintext).ok()
    }

    fn now_secs() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }
}

#[async_trait]
impl DbBackend for PostgresDb {
    async fn insert_access_token(&self, token: &str, id: &str, secret: &str) -> Result<(), String> {
        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM access_tokens")
            .fetch_one(&self.pool)
            .await
            .unwrap_or((0,));
        if count.0 >= 10000 {
            let cutoff = Self::now_secs() - access_token_ttl().as_secs() as i64;
            sqlx::query("DELETE FROM access_tokens WHERE created_at <= $1")
                .bind(cutoff)
                .execute(&self.pool)
                .await
                .ok();
        }
        let token_hash = Self::hash_token(token);
        let enc_id = self.encrypt(id);
        let enc_secret = self.encrypt(secret);
        sqlx::query(
            "INSERT INTO access_tokens (token, token_id, token_secret, created_at)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (token) DO UPDATE SET token_id = $2, token_secret = $3, created_at = $4",
        )
        .bind(&token_hash)
        .bind(&enc_id)
        .bind(&enc_secret)
        .bind(Self::now_secs())
        .execute(&self.pool)
        .await
        .map_err(|e| format!("insert_access_token failed: {e}"))?;
        Ok(())
    }

    async fn get_access_token(&self, token: &str) -> Result<Option<(String, String)>, String> {
        let cutoff = Self::now_secs() - access_token_ttl().as_secs() as i64;
        let token_hash = Self::hash_token(token);

        // Try hashed token first, then fall back to raw token (pre-hash migration)
        let row = sqlx::query(
            "SELECT token_id, token_secret FROM access_tokens WHERE token = $1 AND created_at > $2",
        )
        .bind(&token_hash)
        .bind(cutoff)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("get_access_token failed: {e}"))?;

        let row = match row {
            Some(r) => r,
            None => {
                // Fallback: try raw token (pre-hash tokens from migration or older versions)
                match sqlx::query(
                    "SELECT token_id, token_secret FROM access_tokens WHERE token = $1 AND created_at > $2"
                )
                .bind(token)
                .bind(cutoff)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| format!("get_access_token fallback failed: {e}"))? {
                    Some(r) => r,
                    None => return Ok(None),
                }
            }
        };

        let stored_id: String = row.get("token_id");
        let stored_secret: String = row.get("token_secret");

        if let (Some(tid), Some(tsec)) = (self.decrypt(&stored_id), self.decrypt(&stored_secret)) {
            return Ok(Some((tid, tsec)));
        }

        // Plaintext fallback — re-encrypt in place
        let enc_id = self.encrypt(&stored_id);
        let enc_secret = self.encrypt(&stored_secret);
        sqlx::query("UPDATE access_tokens SET token_id = $1, token_secret = $2 WHERE token = $3")
            .bind(&enc_id)
            .bind(&enc_secret)
            .bind(token)
            .execute(&self.pool)
            .await
            .ok();

        Ok(Some((stored_id, stored_secret)))
    }

    async fn cleanup_expired_tokens(&self) -> Result<(), String> {
        let cutoff = Self::now_secs() - access_token_ttl().as_secs() as i64;
        sqlx::query("DELETE FROM access_tokens WHERE created_at <= $1")
            .bind(cutoff)
            .execute(&self.pool)
            .await
            .ok();
        let refresh_cutoff = Self::now_secs() - refresh_token_ttl().as_secs() as i64;
        sqlx::query("DELETE FROM refresh_tokens WHERE created_at <= $1")
            .bind(refresh_cutoff)
            .execute(&self.pool)
            .await
            .ok();
        Ok(())
    }

    async fn insert_refresh_token(
        &self,
        token: &str,
        id: &str,
        secret: &str,
    ) -> Result<(), String> {
        let token_hash = Self::hash_token(token);
        let enc_id = self.encrypt(id);
        let enc_secret = self.encrypt(secret);
        sqlx::query(
            "INSERT INTO refresh_tokens (token, token_id, token_secret, created_at)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (token) DO UPDATE SET token_id = $2, token_secret = $3, created_at = $4",
        )
        .bind(&token_hash)
        .bind(&enc_id)
        .bind(&enc_secret)
        .bind(Self::now_secs())
        .execute(&self.pool)
        .await
        .map_err(|e| format!("insert_refresh_token failed: {e}"))?;
        Ok(())
    }

    async fn get_refresh_token(&self, token: &str) -> Result<Option<(String, String)>, String> {
        let cutoff = Self::now_secs() - refresh_token_ttl().as_secs() as i64;
        let token_hash = Self::hash_token(token);
        let row = sqlx::query(
            "SELECT token_id, token_secret FROM refresh_tokens WHERE token = $1 AND created_at > $2"
        )
        .bind(&token_hash)
        .bind(cutoff)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("get_refresh_token failed: {e}"))?;

        let row = match row {
            Some(r) => r,
            None => return Ok(None),
        };

        let stored_id: String = row.get("token_id");
        let stored_secret: String = row.get("token_secret");

        match (self.decrypt(&stored_id), self.decrypt(&stored_secret)) {
            (Some(tid), Some(tsec)) => Ok(Some((tid, tsec))),
            _ => Ok(None),
        }
    }

    async fn delete_refresh_token(&self, token: &str) -> Result<(), String> {
        let token_hash = Self::hash_token(token);
        sqlx::query("DELETE FROM refresh_tokens WHERE token = $1")
            .bind(&token_hash)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("delete_refresh_token failed: {e}"))?;
        Ok(())
    }

    async fn backup(&self, _path: &Path) -> Result<(), String> {
        eprintln!("Backup: PostgreSQL backups should use pg_dump externally");
        Ok(())
    }

    async fn get_global_settings(&self) -> Result<GlobalSettings, String> {
        let row = sqlx::query(
            "SELECT hive_shelf_id, user_journals_shelf_id,
                    set_by_token_hash, updated_at,
                    kb_scopes_json, cascade_multipliers_json
             FROM global_settings WHERE id = 1",
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("get_global_settings: {e}"))?;

        Ok(row
            .map(|r| {
                let kb_scopes_json: Option<String> = r.get("kb_scopes_json");
                let cascade_json: Option<String> = r.get("cascade_multipliers_json");
                let kb_scopes = kb_scopes_json
                    .as_deref()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or_default();
                let cascade_multipliers = cascade_json
                    .as_deref()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or_default();
                GlobalSettings {
                    hive_shelf_id: r.get("hive_shelf_id"),
                    user_journals_shelf_id: r.get("user_journals_shelf_id"),
                    set_by_token_hash: r.get("set_by_token_hash"),
                    updated_at: r.get("updated_at"),
                    kb_scopes,
                    cascade_multipliers,
                }
            })
            .unwrap_or_default())
    }

    async fn save_global_settings(
        &self,
        settings: &GlobalSettings,
        set_by_token_hash: &str,
    ) -> Result<(), String> {
        let existing_setter: Option<String> = sqlx::query_scalar(
            "SELECT set_by_token_hash FROM global_settings WHERE id = 1 AND updated_at > 0",
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("save_global_settings preflight: {e}"))?
        .flatten();
        let final_setter = existing_setter.unwrap_or_else(|| set_by_token_hash.to_string());

        let kb_scopes_json = if settings.kb_scopes.is_empty() {
            None
        } else {
            Some(
                serde_json::to_string(&settings.kb_scopes)
                    .map_err(|e| format!("save_global_settings: serialize kb_scopes: {e}"))?,
            )
        };
        let cascade_multipliers_json = Some(
            serde_json::to_string(&settings.cascade_multipliers)
                .map_err(|e| format!("save_global_settings: serialize cascade_multipliers: {e}"))?,
        );

        sqlx::query(
            "UPDATE global_settings
             SET hive_shelf_id = $1,
                 user_journals_shelf_id = $2,
                 set_by_token_hash = $3,
                 updated_at = $4,
                 kb_scopes_json = $5,
                 cascade_multipliers_json = $6
             WHERE id = 1",
        )
        .bind(settings.hive_shelf_id)
        .bind(settings.user_journals_shelf_id)
        .bind(&final_setter)
        .bind(Self::now_secs())
        .bind(&kb_scopes_json)
        .bind(&cascade_multipliers_json)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("save_global_settings: {e}"))?;
        Ok(())
    }
}

#[async_trait]
impl SemanticDb for PostgresDb {
    async fn init_semantic_tables(&self) -> Result<(), String> {
        let statements = [
            "CREATE TABLE IF NOT EXISTS pages (
                page_id BIGINT PRIMARY KEY,
                book_id BIGINT NOT NULL,
                chapter_id BIGINT,
                name TEXT NOT NULL,
                slug TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedded_at BIGINT NOT NULL
            )",
            "CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL PRIMARY KEY,
                page_id BIGINT NOT NULL REFERENCES pages(page_id) ON DELETE CASCADE,
                chunk_index INT NOT NULL,
                heading_path TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding vector(1024) NOT NULL,
                UNIQUE(page_id, chunk_index)
            )",
            "CREATE TABLE IF NOT EXISTS relationships (
                source_page_id BIGINT NOT NULL,
                target_page_id BIGINT NOT NULL,
                link_type TEXT NOT NULL DEFAULT 'link',
                PRIMARY KEY (source_page_id, target_page_id, link_type)
            )",
            "CREATE TABLE IF NOT EXISTS embed_jobs (
                id BIGSERIAL PRIMARY KEY,
                scope TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                total_pages BIGINT DEFAULT 0,
                done_pages BIGINT DEFAULT 0,
                started_at BIGINT,
                finished_at BIGINT,
                error TEXT,
                worker_id TEXT,
                resolved_status TEXT,
                prev_status TEXT,
                resolved_at BIGINT,
                retry_of BIGINT
            )",
        ];
        for sql in statements {
            sqlx::query(sql)
                .execute(&self.pool)
                .await
                .map_err(|e| format!("Failed to create table: {e}"))?;
        }

        // Create indexes (ignore errors if they exist)
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)")
            .execute(&self.pool).await.ok();
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_embed_jobs_pending ON embed_jobs(status) WHERE status = 'pending'")
            .execute(&self.pool).await.ok();

        // Migration: add worker_id column if missing (existing databases)
        sqlx::query("ALTER TABLE embed_jobs ADD COLUMN IF NOT EXISTS worker_id TEXT")
            .execute(&self.pool)
            .await
            .ok();

        // Issue #54 — job lifecycle columns.
        for sql in [
            "ALTER TABLE embed_jobs ADD COLUMN IF NOT EXISTS resolved_status TEXT",
            "ALTER TABLE embed_jobs ADD COLUMN IF NOT EXISTS prev_status TEXT",
            "ALTER TABLE embed_jobs ADD COLUMN IF NOT EXISTS resolved_at BIGINT",
            "ALTER TABLE embed_jobs ADD COLUMN IF NOT EXISTS retry_of BIGINT",
            // Migrate v0.7.x rows that used 'error' as the failed sentinel.
            // resolved_status stays NULL so the reconciler picks them up.
            "UPDATE embed_jobs SET status = 'failed' \
             WHERE status = 'error' AND resolved_status IS NULL",
        ] {
            sqlx::query(sql).execute(&self.pool).await.ok();
        }

        // Metadata key-value store (v0.5.0+)
        sqlx::query("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            .execute(&self.pool)
            .await
            .ok();

        // Schema migration: add updated_at column if missing
        sqlx::query("ALTER TABLE pages ADD COLUMN IF NOT EXISTS updated_at TEXT")
            .execute(&self.pool)
            .await
            .ok();

        // Permission ACL: per-page role visibility, populated at embed time
        // by walking BookStack content_permissions inheritance.
        sqlx::query("ALTER TABLE pages ADD COLUMN IF NOT EXISTS acl_default_open BOOLEAN")
            .execute(&self.pool)
            .await
            .ok();
        sqlx::query("ALTER TABLE pages ADD COLUMN IF NOT EXISTS acl_computed_at BIGINT")
            .execute(&self.pool)
            .await
            .ok();
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS page_view_acl (
                page_id BIGINT NOT NULL,
                role_id BIGINT NOT NULL,
                PRIMARY KEY (page_id, role_id)
            )",
        )
        .execute(&self.pool)
        .await
        .map_err(|e| format!("Failed to create page_view_acl: {e}"))?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_page_view_acl_role ON page_view_acl(role_id, page_id)",
        )
        .execute(&self.pool)
        .await
        .ok();

        // Reconciliation tracker — single-row table for the daily ACL refresh job.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS acl_reconcile_state (
                scope TEXT PRIMARY KEY,
                last_full_run BIGINT NOT NULL DEFAULT 0
            )",
        )
        .execute(&self.pool)
        .await
        .map_err(|e| format!("Failed to create acl_reconcile_state: {e}"))?;

        eprintln!("Semantic: PostgreSQL tables initialized");
        Ok(())
    }

    async fn upsert_page(&self, meta: &PageMeta) -> Result<(), String> {
        sqlx::query(
            "INSERT INTO pages (page_id, book_id, chapter_id, name, slug, content_hash, embedded_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             ON CONFLICT (page_id) DO UPDATE SET
                book_id = EXCLUDED.book_id,
                chapter_id = EXCLUDED.chapter_id,
                name = EXCLUDED.name,
                slug = EXCLUDED.slug,
                content_hash = EXCLUDED.content_hash,
                embedded_at = EXCLUDED.embedded_at,
                updated_at = EXCLUDED.updated_at"
        )
        .bind(meta.page_id)
        .bind(meta.book_id)
        .bind(meta.chapter_id)
        .bind(&meta.name)
        .bind(&meta.slug)
        .bind(&meta.content_hash)
        .bind(Self::now_secs())
        .bind(&meta.updated_at)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("upsert_page failed: {e}"))?;
        Ok(())
    }

    async fn delete_page(&self, page_id: i64) -> Result<(), String> {
        // CASCADE handles chunks; manually delete relationships
        sqlx::query("DELETE FROM relationships WHERE source_page_id = $1 OR target_page_id = $1")
            .bind(page_id)
            .execute(&self.pool)
            .await
            .ok();
        sqlx::query("DELETE FROM pages WHERE page_id = $1")
            .bind(page_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("delete_page failed: {e}"))?;
        Ok(())
    }

    async fn get_page_content_hash(&self, page_id: i64) -> Result<Option<String>, String> {
        let row = sqlx::query("SELECT content_hash FROM pages WHERE page_id = $1")
            .bind(page_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| format!("get_page_content_hash failed: {e}"))?;
        Ok(row.map(|r| r.get("content_hash")))
    }

    async fn get_page_meta(&self, page_id: i64) -> Result<Option<PageMeta>, String> {
        let row = sqlx::query(
            "SELECT page_id, book_id, chapter_id, name, slug, content_hash, updated_at FROM pages WHERE page_id = $1"
        )
        .bind(page_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("get_page_meta failed: {e}"))?;

        Ok(row.map(|r| PageMeta {
            page_id: r.get("page_id"),
            book_id: r.get("book_id"),
            chapter_id: r.get("chapter_id"),
            name: r.get("name"),
            slug: r.get("slug"),
            content_hash: r.get("content_hash"),
            updated_at: r.get("updated_at"),
        }))
    }

    async fn resolve_page_slug(&self, slug: &str) -> Result<Option<i64>, String> {
        let row = sqlx::query("SELECT page_id FROM pages WHERE slug = $1")
            .bind(slug)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| format!("resolve_page_slug failed: {e}"))?;
        Ok(row.map(|r| r.get("page_id")))
    }

    async fn get_page_book_ids(&self, page_ids: &[i64]) -> Result<Vec<(i64, i64)>, String> {
        if page_ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<i64> = page_ids.to_vec();
        let rows = sqlx::query("SELECT page_id, book_id FROM pages WHERE page_id = ANY($1)")
            .bind(&ids)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| format!("get_page_book_ids failed: {e}"))?;
        Ok(rows
            .iter()
            .map(|r| (r.get("page_id"), r.get("book_id")))
            .collect())
    }

    async fn get_page_metas(&self, page_ids: &[i64]) -> Result<Vec<PageMeta>, String> {
        if page_ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<i64> = page_ids.to_vec();
        let rows = sqlx::query(
            "SELECT page_id, book_id, chapter_id, name, slug, content_hash, updated_at
             FROM pages WHERE page_id = ANY($1)",
        )
        .bind(&ids)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("get_page_metas failed: {e}"))?;

        Ok(rows
            .iter()
            .map(|r| PageMeta {
                page_id: r.get("page_id"),
                book_id: r.get("book_id"),
                chapter_id: r.get("chapter_id"),
                name: r.get("name"),
                slug: r.get("slug"),
                content_hash: r.get("content_hash"),
                updated_at: r.get("updated_at"),
            })
            .collect())
    }

    async fn insert_chunks(&self, page_id: i64, chunks: &[ChunkInsert]) -> Result<(), String> {
        // Wrap DELETE + INSERTs in a transaction to prevent partial state
        // (without this, queries hitting the table between DELETE and INSERT see zero chunks)
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| format!("insert_chunks transaction begin failed: {e}"))?;

        sqlx::query("DELETE FROM chunks WHERE page_id = $1")
            .bind(page_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("insert_chunks delete failed: {e}"))?;

        for chunk in chunks {
            let vec = Vector::from(chunk.embedding.clone());
            sqlx::query(
                "INSERT INTO chunks (page_id, chunk_index, heading_path, content, content_hash, embedding)
                 VALUES ($1, $2, $3, $4, $5, $6)"
            )
            .bind(page_id)
            .bind(chunk.chunk_index as i32)
            .bind(&chunk.heading_path)
            .bind(&chunk.content)
            .bind(&chunk.content_hash)
            .bind(vec)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("insert_chunks failed for chunk {}: {e}", chunk.chunk_index))?;
        }

        tx.commit()
            .await
            .map_err(|e| format!("insert_chunks commit failed: {e}"))?;
        Ok(())
    }

    async fn get_chunk_details(&self, chunk_ids: &[i64]) -> Result<Vec<ChunkDetail>, String> {
        if chunk_ids.is_empty() {
            return Ok(Vec::new());
        }
        let rows = sqlx::query(
            "SELECT c.id, c.page_id, c.heading_path, c.content, p.name
             FROM chunks c JOIN pages p ON c.page_id = p.page_id
             WHERE c.id = ANY($1)",
        )
        .bind(chunk_ids)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("get_chunk_details failed: {e}"))?;

        Ok(rows
            .iter()
            .map(|r| ChunkDetail {
                chunk_id: r.get("id"),
                page_id: r.get("page_id"),
                heading_path: r.get("heading_path"),
                content: r.get("content"),
                page_name: r.get("name"),
            })
            .collect())
    }

    async fn replace_relationships(
        &self,
        source: i64,
        targets: &[(i64, String)],
    ) -> Result<(), String> {
        // Only delete explicit link relationships; preserve inferred "similar" ones
        sqlx::query("DELETE FROM relationships WHERE source_page_id = $1 AND link_type = 'link'")
            .bind(source)
            .execute(&self.pool)
            .await
            .ok();

        for (target_id, link_type) in targets {
            sqlx::query(
                "INSERT INTO relationships (source_page_id, target_page_id, link_type)
                 VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            )
            .bind(source)
            .bind(target_id)
            .bind(link_type)
            .execute(&self.pool)
            .await
            .ok();
        }
        Ok(())
    }

    async fn get_markov_blanket(&self, page_id: i64) -> Result<MarkovBlanket, String> {
        let query_related = |pool: &PgPool, sql: &str, page_id: i64| {
            let pool = pool.clone();
            let sql = sql.to_string();
            async move {
                sqlx::query(&sql)
                    .bind(page_id)
                    .fetch_all(&pool)
                    .await
                    .map(|rows| {
                        rows.iter()
                            .map(|r| RelatedPage {
                                page_id: r.get(0),
                                name: r.get(1),
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
            }
        };

        let linked_from = query_related(
            &self.pool,
            "SELECT r.source_page_id, p.name FROM relationships r
             JOIN pages p ON r.source_page_id = p.page_id
             WHERE r.target_page_id = $1 LIMIT 20",
            page_id,
        )
        .await;

        let links_to = query_related(
            &self.pool,
            "SELECT r.target_page_id, p.name FROM relationships r
             JOIN pages p ON r.target_page_id = p.page_id
             WHERE r.source_page_id = $1 LIMIT 20",
            page_id,
        )
        .await;

        let co_linked = query_related(
            &self.pool,
            "SELECT DISTINCT r2.source_page_id, p.name FROM relationships r1
             JOIN relationships r2 ON r1.target_page_id = r2.target_page_id
             JOIN pages p ON r2.source_page_id = p.page_id
             WHERE r1.source_page_id = $1 AND r2.source_page_id != $1
             LIMIT 10",
            page_id,
        )
        .await;

        // Siblings: same chapter or same book
        let siblings = {
            let meta = sqlx::query("SELECT chapter_id, book_id FROM pages WHERE page_id = $1")
                .bind(page_id)
                .fetch_optional(&self.pool)
                .await
                .ok()
                .flatten();

            if let Some(meta) = meta {
                let chapter_id: Option<i64> = meta.get("chapter_id");
                let book_id: i64 = meta.get("book_id");

                if let Some(cid) = chapter_id {
                    let result: Vec<RelatedPage> = sqlx::query(
                        "SELECT page_id, name FROM pages WHERE chapter_id = $1 AND page_id != $2 LIMIT 20"
                    )
                    .bind(cid)
                    .bind(page_id)
                    .fetch_all(&self.pool)
                    .await
                    .map(|rows| rows.iter().map(|r| RelatedPage { page_id: r.get(0), name: r.get(1) }).collect())
                    .unwrap_or_default();

                    if !result.is_empty() {
                        result
                    } else {
                        sqlx::query("SELECT page_id, name FROM pages WHERE book_id = $1 AND page_id != $2 LIMIT 20")
                            .bind(book_id).bind(page_id)
                            .fetch_all(&self.pool).await
                            .map(|rows| rows.iter().map(|r| RelatedPage { page_id: r.get(0), name: r.get(1) }).collect())
                            .unwrap_or_default()
                    }
                } else {
                    sqlx::query("SELECT page_id, name FROM pages WHERE book_id = $1 AND page_id != $2 LIMIT 20")
                        .bind(book_id).bind(page_id)
                        .fetch_all(&self.pool).await
                        .map(|rows| rows.iter().map(|r| RelatedPage { page_id: r.get(0), name: r.get(1) }).collect())
                        .unwrap_or_default()
                }
            } else {
                Vec::new()
            }
        };

        Ok(MarkovBlanket {
            linked_from,
            links_to,
            co_linked,
            siblings,
        })
    }

    async fn create_embed_job(&self, scope: &str) -> Result<(i64, bool), String> {
        // Atomic check-and-insert in a serializable transaction to prevent duplicates
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| format!("create_embed_job transaction failed: {e}"))?;

        // Dedup: pending/running collapse onto the existing job. Failed jobs
        // not yet touched by the reconciler also count as active so a webhook
        // firing mid-retry-window doesn't double-enqueue.
        let existing = sqlx::query(
            "SELECT id FROM embed_jobs \
             WHERE scope = $1 \
               AND (status IN ('pending', 'running') \
                    OR (status = 'failed' AND resolved_status IS NULL)) \
             ORDER BY id DESC LIMIT 1 FOR UPDATE",
        )
        .bind(scope)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| format!("create_embed_job check failed: {e}"))?;

        if let Some(row) = existing {
            tx.commit()
                .await
                .map_err(|e| format!("create_embed_job commit failed: {e}"))?;
            return Ok((row.get("id"), false));
        }

        let row = sqlx::query(
            "INSERT INTO embed_jobs (scope, status, started_at) VALUES ($1, 'pending', $2) RETURNING id"
        )
        .bind(scope)
        .bind(Self::now_secs())
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| format!("create_embed_job insert failed: {e}"))?;

        tx.commit()
            .await
            .map_err(|e| format!("create_embed_job commit failed: {e}"))?;
        Ok((row.get("id"), true))
    }

    async fn claim_next_job(&self, worker_id: &str) -> Result<Option<EmbedJob>, String> {
        // FOR UPDATE SKIP LOCKED enables concurrent embedder workers
        let row = sqlx::query(&format!(
            "UPDATE embed_jobs SET status = 'running', started_at = $1, worker_id = $2 \
                 WHERE id = ( \
                    SELECT id FROM embed_jobs WHERE status = 'pending' \
                    ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED \
                 ) \
                 RETURNING {EMBED_JOB_COLS}"
        ))
        .bind(Self::now_secs())
        .bind(worker_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("claim_next_job failed: {e}"))?;

        Ok(row.map(embed_job_from_row))
    }

    async fn recover_worker_jobs(&self, worker_id: &str) -> Result<usize, String> {
        // Process restart: jobs left running by this worker (or orphans
        // pre-0.3.1) flip to failed-open. resolved_status stays NULL so the
        // reconciler picks them up.
        let failed = sqlx::query(
            "UPDATE embed_jobs \
             SET status = 'failed', finished_at = $1, error = 'worker_restart' \
             WHERE status = 'running' AND (worker_id = $2 OR worker_id IS NULL)",
        )
        .bind(Self::now_secs())
        .bind(worker_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("recover_worker_jobs failed: {e}"))?
        .rows_affected() as usize;

        Ok(failed)
    }

    async fn expire_stale_jobs(&self, stale_secs: i64) -> Result<usize, String> {
        let cutoff = Self::now_secs() - stale_secs;
        let failed = sqlx::query(
            "UPDATE embed_jobs \
             SET status = 'failed', finished_at = $1, error = 'timeout' \
             WHERE status = 'running' AND started_at < $2",
        )
        .bind(Self::now_secs())
        .bind(cutoff)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("expire_stale_jobs failed: {e}"))?
        .rows_affected() as usize;
        Ok(failed)
    }

    async fn update_job_progress(&self, job_id: i64, done: i64, total: i64) -> Result<(), String> {
        sqlx::query("UPDATE embed_jobs SET done_pages = $1, total_pages = $2 WHERE id = $3")
            .bind(done)
            .bind(total)
            .bind(job_id)
            .execute(&self.pool)
            .await
            .ok();
        Ok(())
    }

    async fn complete_job(&self, job_id: i64, error: Option<&str>) -> Result<(), String> {
        let now = Self::now_secs();
        if let Some(reason) = error {
            // Failed-open: resolved_status stays NULL until the reconciler
            // closes us with superseded/retried/gave_up. Status guard makes
            // this idempotent — if the user cancelled the job mid-flight,
            // the cancel wins and this is a no-op.
            sqlx::query(
                "UPDATE embed_jobs \
                 SET status = 'failed', finished_at = $1, error = $2 \
                 WHERE id = $3 AND status IN ('pending', 'running')",
            )
            .bind(now)
            .bind(reason)
            .bind(job_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("complete_job failed: {e}"))?;
        } else {
            // Status guard: a cancel that arrived after the pipeline's last
            // should_stop_embed_job poll but before this write must not be
            // silently overwritten back to 'succeeded'.
            sqlx::query(
                "UPDATE embed_jobs \
                 SET status = 'succeeded', finished_at = $1, \
                     resolved_status = 'succeeded', resolved_at = $1, error = NULL \
                 WHERE id = $2 AND status IN ('pending', 'running')",
            )
            .bind(now)
            .bind(job_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("complete_job failed: {e}"))?;
        }
        Ok(())
    }

    async fn get_latest_job(&self) -> Result<Option<EmbedJob>, String> {
        let row = sqlx::query(&format!(
            "SELECT {EMBED_JOB_COLS} FROM embed_jobs ORDER BY id DESC LIMIT 1"
        ))
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("get_latest_job failed: {e}"))?;

        Ok(row.map(embed_job_from_row))
    }

    async fn get_stats(&self) -> Result<EmbedStats, String> {
        let pages: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM pages")
            .fetch_one(&self.pool)
            .await
            .unwrap_or((0,));
        let chunks: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM chunks")
            .fetch_one(&self.pool)
            .await
            .unwrap_or((0,));

        let latest_job = self.get_latest_job().await?;

        Ok(EmbedStats {
            total_pages: pages.0,
            total_chunks: chunks.0,
            latest_job,
        })
    }

    async fn list_jobs(&self, recent: usize) -> Result<Vec<EmbedJob>, String> {
        let rows = sqlx::query(&format!(
            "(SELECT {EMBED_JOB_COLS} FROM embed_jobs \
                  WHERE status IN ('pending', 'running', 'failed') ORDER BY id ASC) \
                 UNION ALL \
                 (SELECT {EMBED_JOB_COLS} FROM embed_jobs \
                  WHERE status NOT IN ('pending', 'running', 'failed') \
                  ORDER BY id DESC LIMIT {recent})"
        ))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_jobs failed: {e}"))?;

        Ok(rows.into_iter().map(embed_job_from_row).collect())
    }

    async fn cancel_embed_job(&self, job_id: i64) -> Result<(), String> {
        let now = Self::now_secs();
        sqlx::query(
            "UPDATE embed_jobs \
             SET status = 'cancelled', resolved_status = 'cancelled', \
                 resolved_at = $1, finished_at = $1 \
             WHERE id = $2 AND status IN ('pending', 'running')",
        )
        .bind(now)
        .bind(job_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("cancel_embed_job: {e}"))?;
        Ok(())
    }

    async fn should_stop_embed_job(&self, job_id: i64) -> Result<bool, String> {
        let row: Option<(String,)> = sqlx::query_as("SELECT status FROM embed_jobs WHERE id = $1")
            .bind(job_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| format!("should_stop_embed_job: {e}"))?;
        Ok(matches!(row.as_ref().map(|r| r.0.as_str()), Some(s) if s != "running"))
    }

    async fn fail_embed_job(&self, job_id: i64, reason: &str) -> Result<(), String> {
        let now = Self::now_secs();
        sqlx::query(
            "UPDATE embed_jobs \
             SET status = 'failed', finished_at = $1, error = $2 \
             WHERE id = $3 AND status IN ('pending', 'running')",
        )
        .bind(now)
        .bind(reason)
        .bind(job_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("fail_embed_job: {e}"))?;
        Ok(())
    }

    async fn list_failed_open_embed_jobs(&self) -> Result<Vec<EmbedJob>, String> {
        let rows = sqlx::query(&format!(
            "SELECT {EMBED_JOB_COLS} FROM embed_jobs \
                 WHERE status = 'failed' ORDER BY id ASC"
        ))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_failed_open_embed_jobs: {e}"))?;
        Ok(rows.into_iter().map(embed_job_from_row).collect())
    }

    async fn has_successor_embed_job(&self, scope: &str, excluded_id: i64) -> Result<bool, String> {
        let row: Option<(i64,)> = sqlx::query_as(
            "SELECT 1::BIGINT FROM embed_jobs \
             WHERE scope = $1 AND id > $2 \
               AND status IN ('pending','running','succeeded','cancelled','closed') \
             LIMIT 1",
        )
        .bind(scope)
        .bind(excluded_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("has_successor_embed_job: {e}"))?;
        Ok(row.is_some())
    }

    async fn embed_job_retry_chain_len(&self, job_id: i64) -> Result<usize, String> {
        let mut len = 1usize;
        let mut current = job_id;
        for _ in 0..1024 {
            let parent: Option<(Option<i64>,)> =
                sqlx::query_as("SELECT retry_of FROM embed_jobs WHERE id = $1")
                    .bind(current)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| format!("embed_job_retry_chain_len: {e}"))?;
            match parent.and_then(|(p,)| p) {
                Some(p) => {
                    len += 1;
                    current = p;
                }
                None => break,
            }
        }
        Ok(len)
    }

    async fn close_embed_job(
        &self,
        job_id: i64,
        resolved_status: Option<&str>,
    ) -> Result<(), String> {
        if let Some(rs) = resolved_status {
            sqlx::query(
                "UPDATE embed_jobs \
                 SET prev_status = status, status = 'closed', resolved_status = $1 \
                 WHERE id = $2 AND status != 'closed'",
            )
            .bind(rs)
            .bind(job_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("close_embed_job: {e}"))?;
        } else {
            sqlx::query(
                "UPDATE embed_jobs \
                 SET prev_status = status, status = 'closed' \
                 WHERE id = $1 AND status != 'closed'",
            )
            .bind(job_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("close_embed_job: {e}"))?;
        }
        Ok(())
    }

    async fn create_retry_embed_job(&self, scope: &str, retry_of: i64) -> Result<i64, String> {
        let row: (i64,) = sqlx::query_as(
            "INSERT INTO embed_jobs (scope, status, retry_of) VALUES ($1, 'pending', $2) RETURNING id"
        )
        .bind(scope)
        .bind(retry_of)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| format!("create_retry_embed_job: {e}"))?;
        Ok(row.0)
    }

    async fn list_archivable_embed_jobs(&self, older_than_secs: i64) -> Result<Vec<i64>, String> {
        let cutoff = Self::now_secs() - older_than_secs;
        let rows: Vec<(i64,)> = sqlx::query_as(
            "SELECT id FROM embed_jobs \
             WHERE status IN ('succeeded', 'cancelled') AND resolved_at IS NOT NULL \
               AND resolved_at <= $1 \
             ORDER BY id ASC",
        )
        .bind(cutoff)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_archivable_embed_jobs: {e}"))?;
        Ok(rows.into_iter().map(|(id,)| id).collect())
    }

    async fn list_running_embed_jobs_started_before(
        &self,
        started_before_secs: i64,
    ) -> Result<Vec<EmbedJob>, String> {
        let rows = sqlx::query(&format!(
            "SELECT {EMBED_JOB_COLS} FROM embed_jobs \
                 WHERE status = 'running' AND started_at IS NOT NULL AND started_at < $1"
        ))
        .bind(started_before_secs)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_running_embed_jobs_started_before: {e}"))?;
        Ok(rows.into_iter().map(embed_job_from_row).collect())
    }

    async fn vector_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        threshold: f32,
        scope: Option<&ScopeFilter>,
    ) -> Result<Vec<SearchHit>, String> {
        // Sanity check: detect garbage embeddings (all zeros, NaN, etc.)
        let magnitude: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude < 0.01 || magnitude.is_nan() {
            return Err(format!("vector_search: query embedding appears invalid (magnitude={magnitude:.4}, dims={})", query_embedding.len()));
        }

        let vec = Vector::from(query_embedding.to_vec());

        // Use a transaction so SET LOCAL ef_search applies to the search query
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| format!("vector_search transaction failed: {e}"))?;

        // Increase HNSW ef_search for better recall (default 40 can miss results after bulk ops)
        sqlx::query("SET LOCAL hnsw.ef_search = 100")
            .execute(&mut *tx)
            .await
            .ok();

        // Optional scope filter. Union semantics: a chunk qualifies if its
        // parent page matches any of the supplied book/chapter/page lists.
        // Shelf-level IDs are not evaluated here — the caller must resolve
        // shelves → books before invoking the search.
        let scope_owned: Option<ScopeFilter> = scope.filter(|s| !s.is_empty()).map(|s| {
            let mut f = s.clone();
            f.dedup();
            f
        });

        let rows = match scope_owned {
            Some(f) => {
                // Build a union of `p.<col> = ANY(...)` clauses for the
                // non-empty ID lists. Bind each list as its own $N to keep
                // sqlx's parameter typing happy.
                let mut clauses: Vec<String> = Vec::new();
                let mut next_param: usize = 4; // $1=vec, $2=threshold, $3=limit
                let book_param = if !f.book_ids.is_empty() {
                    let p = next_param;
                    next_param += 1;
                    clauses.push(format!("p.book_id = ANY(${p})"));
                    Some(p)
                } else {
                    None
                };
                let chapter_param = if !f.chapter_ids.is_empty() {
                    let p = next_param;
                    next_param += 1;
                    clauses.push(format!("p.chapter_id = ANY(${p})"));
                    Some(p)
                } else {
                    None
                };
                let page_param = if !f.page_ids.is_empty() {
                    let p = next_param;
                    clauses.push(format!("p.page_id = ANY(${p})"));
                    Some(p)
                } else {
                    None
                };

                if clauses.is_empty() {
                    // shelf-only filter that the caller didn't expand —
                    // return zero rows rather than silently widening.
                    tx.commit().await.ok();
                    return Ok(Vec::new());
                }

                let scope_clause = clauses.join(" OR ");
                let sql = format!(
                    "SELECT c.id, c.page_id, (1 - (c.embedding <=> $1::vector))::FLOAT4 AS score
                     FROM chunks c
                     JOIN pages p ON c.page_id = p.page_id
                     WHERE 1 - (c.embedding <=> $1::vector) > $2::FLOAT8
                       AND ({scope_clause})
                     ORDER BY c.embedding <=> $1::vector
                     LIMIT $3"
                );

                let mut q = sqlx::query(&sql)
                    .bind(&vec)
                    .bind(threshold)
                    .bind(limit as i64);
                if book_param.is_some() {
                    q = q.bind(&f.book_ids);
                }
                if chapter_param.is_some() {
                    q = q.bind(&f.chapter_ids);
                }
                if page_param.is_some() {
                    q = q.bind(&f.page_ids);
                }
                q.fetch_all(&mut *tx).await
            }
            None => {
                sqlx::query(
                    "SELECT id, page_id, (1 - (embedding <=> $1::vector))::FLOAT4 AS score
                 FROM chunks
                 WHERE 1 - (embedding <=> $1::vector) > $2::FLOAT8
                 ORDER BY embedding <=> $1::vector
                 LIMIT $3",
                )
                .bind(&vec)
                .bind(threshold)
                .bind(limit as i64)
                .fetch_all(&mut *tx)
                .await
            }
        };
        let rows = rows.map_err(|e| format!("vector_search failed: {e}"))?;

        tx.commit().await.ok();

        let results: Vec<SearchHit> = rows
            .iter()
            .map(|r| SearchHit {
                chunk_id: r.get("id"),
                page_id: r.get("page_id"),
                score: r.get("score"),
            })
            .collect();

        // Diagnostic: if zero results, check why
        if results.is_empty() {
            let chunk_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM chunks")
                .fetch_one(&self.pool)
                .await
                .unwrap_or((0,));
            if chunk_count.0 == 0 {
                eprintln!("vector_search: 0 results — chunks table is EMPTY (embeddings may have been cleared)");
            } else {
                // Check max score to see if threshold is the issue
                let top: Option<(f32,)> = sqlx::query_as(
                    "SELECT (1 - (embedding <=> $1::vector))::FLOAT4 FROM chunks ORDER BY embedding <=> $1::vector LIMIT 1"
                )
                .bind(&vec)
                .fetch_optional(&self.pool)
                .await
                .ok()
                .flatten();
                let max_score = top.map(|t| t.0).unwrap_or(0.0);
                eprintln!("vector_search: 0 results — {count} chunks exist, threshold={threshold:.3}, max_score={max_score:.3}, dims={dims}",
                    count = chunk_count.0, dims = query_embedding.len());
            }
        }

        Ok(results)
    }

    async fn clear_all_embeddings(&self) -> Result<(), String> {
        // Log what's being cleared for debugging intermittent search death
        let pages: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM pages")
            .fetch_one(&self.pool)
            .await
            .unwrap_or((0,));
        let chunks: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM chunks")
            .fetch_one(&self.pool)
            .await
            .unwrap_or((0,));
        eprintln!(
            "clear_all_embeddings: clearing {} pages, {} chunks",
            pages.0, chunks.0
        );

        sqlx::query("DELETE FROM relationships")
            .execute(&self.pool)
            .await
            .map_err(|e| format!("clear relationships: {e}"))?;
        sqlx::query("DELETE FROM chunks")
            .execute(&self.pool)
            .await
            .map_err(|e| format!("clear chunks: {e}"))?;
        sqlx::query("DELETE FROM pages")
            .execute(&self.pool)
            .await
            .map_err(|e| format!("clear pages: {e}"))?;
        Ok(())
    }

    async fn alter_embedding_dimension(&self, dims: usize) -> Result<(), String> {
        // Drop the HNSW index, alter column type, recreate index
        sqlx::query("DROP INDEX IF EXISTS idx_chunks_embedding")
            .execute(&self.pool)
            .await
            .map_err(|e| format!("drop index: {e}"))?;
        let alter_sql = format!(
            "ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({dims}) USING embedding::vector({dims})"
        );
        sqlx::query(&alter_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("alter column: {e}"))?;
        sqlx::query(
            "CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)",
        )
        .execute(&self.pool)
        .await
        .map_err(|e| format!("recreate index: {e}"))?;
        eprintln!("PostgreSQL: embedding column altered to vector({dims})");
        Ok(())
    }

    async fn compute_similar_pages(&self, top_k: usize, threshold: f32) -> Result<usize, String> {
        // Clear existing "similar" relationships
        sqlx::query("DELETE FROM relationships WHERE link_type = 'similar'")
            .execute(&self.pool)
            .await
            .map_err(|e| format!("clear similar rels: {e}"))?;

        // Compute page centroids (average of chunk embeddings) and find top-K
        // most similar pages per page using pgvector cosine distance.
        // This uses a CTE to build centroids, then a lateral join for nearest neighbors.
        let sql = format!(
            "WITH centroids AS (
                SELECT page_id, AVG(embedding)::vector AS centroid
                FROM chunks
                GROUP BY page_id
            )
            INSERT INTO relationships (source_page_id, target_page_id, link_type)
            SELECT c1.page_id, nn.page_id, 'similar'
            FROM centroids c1
            CROSS JOIN LATERAL (
                SELECT c2.page_id, 1 - (c1.centroid <=> c2.centroid) AS sim
                FROM centroids c2
                WHERE c2.page_id != c1.page_id
                ORDER BY c1.centroid <=> c2.centroid
                LIMIT {top_k}
            ) nn
            WHERE nn.sim > $1
            ON CONFLICT DO NOTHING"
        );

        let result = sqlx::query(&sql)
            .bind(threshold)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("compute_similar_pages: {e}"))?;

        let count = result.rows_affected() as usize;
        eprintln!("Semantic: computed {count} similar-page relationships (top_k={top_k}, threshold={threshold})");
        Ok(count)
    }

    async fn get_meta(&self, key: &str) -> Result<Option<String>, String> {
        let row: Option<(String,)> = sqlx::query_as("SELECT value FROM meta WHERE key = $1")
            .bind(key)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| format!("get_meta: {e}"))?;
        Ok(row.map(|r| r.0))
    }

    async fn set_meta(&self, key: &str, value: &str) -> Result<(), String> {
        sqlx::query("INSERT INTO meta (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value = $2")
            .bind(key)
            .bind(value)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("set_meta: {e}"))?;
        Ok(())
    }

    async fn upsert_page_acl(&self, acl: &PageAcl) -> Result<(), String> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| format!("upsert_page_acl tx: {e}"))?;
        sqlx::query("DELETE FROM page_view_acl WHERE page_id = $1")
            .bind(acl.page_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("upsert_page_acl delete: {e}"))?;
        for &role_id in &acl.view_roles {
            sqlx::query(
                "INSERT INTO page_view_acl (page_id, role_id) VALUES ($1, $2)
                 ON CONFLICT (page_id, role_id) DO NOTHING",
            )
            .bind(acl.page_id)
            .bind(role_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("upsert_page_acl insert: {e}"))?;
        }
        sqlx::query(
            "UPDATE pages SET acl_default_open = $1, acl_computed_at = $2 WHERE page_id = $3",
        )
        .bind(acl.default_open)
        .bind(acl.computed_at)
        .bind(acl.page_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("upsert_page_acl flag: {e}"))?;
        tx.commit()
            .await
            .map_err(|e| format!("upsert_page_acl commit: {e}"))?;
        Ok(())
    }

    async fn delete_page_acl(&self, page_id: i64) -> Result<(), String> {
        sqlx::query("DELETE FROM page_view_acl WHERE page_id = $1")
            .bind(page_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("delete_page_acl: {e}"))?;
        Ok(())
    }

    async fn delete_role_from_acl(&self, role_id: i64) -> Result<(), String> {
        sqlx::query("DELETE FROM page_view_acl WHERE role_id = $1")
            .bind(role_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("delete_role_from_acl: {e}"))?;
        Ok(())
    }

    async fn list_acl_page_ids(&self) -> Result<Vec<i64>, String> {
        let rows: Vec<(i64,)> =
            sqlx::query_as("SELECT page_id FROM pages WHERE acl_computed_at IS NOT NULL")
                .fetch_all(&self.pool)
                .await
                .map_err(|e| format!("list_acl_page_ids: {e}"))?;
        Ok(rows.into_iter().map(|r| r.0).collect())
    }
}

// --- IndexDb impl (closes #36) ---
//
// Mirrors the SQLite impl in `bsmcp-db-sqlite/src/lib.rs`. Postgres
// differences from SQLite worth noting:
//   - $1/$2 placeholders, not ?1/?2.
//   - Real BOOLEAN type (no 0/1 conversion needed).
//   - BIGSERIAL on `index_jobs.id`, so create_index_job uses RETURNING id
//     instead of last_insert_rowid().
//   - claim_next_index_job uses FOR UPDATE SKIP LOCKED so multiple worker
//     processes can run safely against the same database.
//   - upsert_indexed_page wraps the page row + optional page_cache row in
//     a single transaction, same as SQLite.

#[async_trait]
impl IndexDb for PostgresDb {
    // --- Shelves ---

    async fn upsert_indexed_shelf(&self, shelf: &IndexedShelf) -> Result<(), String> {
        sqlx::query(
            "INSERT INTO bookstack_shelves (shelf_id, name, slug, shelf_kind, indexed_at, deleted)
             VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (shelf_id) DO UPDATE SET
                 name = EXCLUDED.name,
                 slug = EXCLUDED.slug,
                 shelf_kind = EXCLUDED.shelf_kind,
                 indexed_at = EXCLUDED.indexed_at,
                 deleted = EXCLUDED.deleted",
        )
        .bind(shelf.shelf_id)
        .bind(&shelf.name)
        .bind(&shelf.slug)
        .bind(shelf.shelf_kind.as_str())
        .bind(shelf.indexed_at)
        .bind(shelf.deleted)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("upsert_indexed_shelf: {e}"))?;
        Ok(())
    }

    async fn get_indexed_shelf(&self, shelf_id: i64) -> Result<Option<IndexedShelf>, String> {
        let row = sqlx::query(
            "SELECT shelf_id, name, slug, shelf_kind, indexed_at, deleted
             FROM bookstack_shelves WHERE shelf_id = $1",
        )
        .bind(shelf_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("get_indexed_shelf: {e}"))?;
        Ok(row.map(|r| {
            let kind_str: String = r.get("shelf_kind");
            IndexedShelf {
                shelf_id: r.get("shelf_id"),
                name: r.get("name"),
                slug: r.get("slug"),
                shelf_kind: kind_str.parse().unwrap_or(ShelfKind::Unclassified),
                indexed_at: r.get("indexed_at"),
                deleted: r.get("deleted"),
            }
        }))
    }

    async fn soft_delete_indexed_shelf(&self, shelf_id: i64) -> Result<(), String> {
        sqlx::query("UPDATE bookstack_shelves SET deleted = TRUE WHERE shelf_id = $1")
            .bind(shelf_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("soft_delete_indexed_shelf: {e}"))?;
        Ok(())
    }

    // --- Books ---

    async fn upsert_indexed_book(&self, book: &IndexedBook) -> Result<(), String> {
        sqlx::query(
            "INSERT INTO bookstack_books
                (book_id, name, slug, shelf_id, identity_ouid, book_kind, indexed_at, deleted)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             ON CONFLICT (book_id) DO UPDATE SET
                 name = EXCLUDED.name,
                 slug = EXCLUDED.slug,
                 shelf_id = EXCLUDED.shelf_id,
                 identity_ouid = EXCLUDED.identity_ouid,
                 book_kind = EXCLUDED.book_kind,
                 indexed_at = EXCLUDED.indexed_at,
                 deleted = EXCLUDED.deleted",
        )
        .bind(book.book_id)
        .bind(&book.name)
        .bind(&book.slug)
        .bind(book.shelf_id)
        .bind(book.identity_ouid.as_deref())
        .bind(book.book_kind.as_str())
        .bind(book.indexed_at)
        .bind(book.deleted)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("upsert_indexed_book: {e}"))?;
        Ok(())
    }

    async fn get_indexed_book(&self, book_id: i64) -> Result<Option<IndexedBook>, String> {
        let row = sqlx::query(
            "SELECT book_id, name, slug, shelf_id, identity_ouid, book_kind, indexed_at, deleted
             FROM bookstack_books WHERE book_id = $1",
        )
        .bind(book_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("get_indexed_book: {e}"))?;
        Ok(row.map(book_from_row))
    }

    async fn list_indexed_books_by_shelf(&self, shelf_id: i64) -> Result<Vec<IndexedBook>, String> {
        let rows = sqlx::query(
            "SELECT book_id, name, slug, shelf_id, identity_ouid, book_kind, indexed_at, deleted
             FROM bookstack_books WHERE shelf_id = $1 AND deleted = FALSE
             ORDER BY name",
        )
        .bind(shelf_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_indexed_books_by_shelf: {e}"))?;
        Ok(rows.into_iter().map(book_from_row).collect())
    }

    async fn list_indexed_books_by_identity(
        &self,
        identity_ouid: &str,
    ) -> Result<Vec<IndexedBook>, String> {
        let rows = sqlx::query(
            "SELECT book_id, name, slug, shelf_id, identity_ouid, book_kind, indexed_at, deleted
             FROM bookstack_books WHERE identity_ouid = $1 AND deleted = FALSE
             ORDER BY book_kind, name",
        )
        .bind(identity_ouid)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_indexed_books_by_identity: {e}"))?;
        Ok(rows.into_iter().map(book_from_row).collect())
    }

    async fn soft_delete_indexed_book(&self, book_id: i64) -> Result<(), String> {
        sqlx::query("UPDATE bookstack_books SET deleted = TRUE WHERE book_id = $1")
            .bind(book_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("soft_delete_indexed_book: {e}"))?;
        Ok(())
    }

    // --- Chapters ---

    async fn upsert_indexed_chapter(&self, chapter: &IndexedChapter) -> Result<(), String> {
        sqlx::query(
            "INSERT INTO bookstack_chapters
                (chapter_id, book_id, name, slug, identity_ouid, chapter_kind, archive_year, indexed_at, deleted)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
             ON CONFLICT (chapter_id) DO UPDATE SET
                 book_id = EXCLUDED.book_id,
                 name = EXCLUDED.name,
                 slug = EXCLUDED.slug,
                 identity_ouid = EXCLUDED.identity_ouid,
                 chapter_kind = EXCLUDED.chapter_kind,
                 archive_year = EXCLUDED.archive_year,
                 indexed_at = EXCLUDED.indexed_at,
                 deleted = EXCLUDED.deleted",
        )
        .bind(chapter.chapter_id)
        .bind(chapter.book_id)
        .bind(&chapter.name)
        .bind(&chapter.slug)
        .bind(chapter.identity_ouid.as_deref())
        .bind(chapter.chapter_kind.as_str())
        .bind(chapter.archive_year)
        .bind(chapter.indexed_at)
        .bind(chapter.deleted)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("upsert_indexed_chapter: {e}"))?;
        Ok(())
    }

    async fn get_indexed_chapter(&self, chapter_id: i64) -> Result<Option<IndexedChapter>, String> {
        let row = sqlx::query(
            "SELECT chapter_id, book_id, name, slug, identity_ouid, chapter_kind, archive_year, indexed_at, deleted
             FROM bookstack_chapters WHERE chapter_id = $1",
        )
        .bind(chapter_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("get_indexed_chapter: {e}"))?;
        Ok(row.map(chapter_from_row))
    }

    async fn list_indexed_chapters_by_book(
        &self,
        book_id: i64,
    ) -> Result<Vec<IndexedChapter>, String> {
        let rows = sqlx::query(
            "SELECT chapter_id, book_id, name, slug, identity_ouid, chapter_kind, archive_year, indexed_at, deleted
             FROM bookstack_chapters WHERE book_id = $1 AND deleted = FALSE
             ORDER BY name",
        )
        .bind(book_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_indexed_chapters_by_book: {e}"))?;
        Ok(rows.into_iter().map(chapter_from_row).collect())
    }

    async fn soft_delete_indexed_chapter(&self, chapter_id: i64) -> Result<(), String> {
        sqlx::query("UPDATE bookstack_chapters SET deleted = TRUE WHERE chapter_id = $1")
            .bind(chapter_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("soft_delete_indexed_chapter: {e}"))?;
        Ok(())
    }

    // --- Pages ---

    async fn upsert_indexed_page(
        &self,
        page: &IndexedPage,
        cache: Option<&PageCache>,
    ) -> Result<(), String> {
        // Single transaction: page row + optional cache row land atomically
        // so the freshness invariant (page.page_updated_at == cache.page_updated_at
        // means cache is fresh) holds even on mid-write process kills.
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| format!("upsert_indexed_page tx: {e}"))?;
        sqlx::query(
            "INSERT INTO bookstack_pages
                (page_id, book_id, chapter_id, name, slug, url, page_created_at, page_updated_at,
                 identity_ouid, page_kind, page_key, archive_year, indexed_at, deleted)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
             ON CONFLICT (page_id) DO UPDATE SET
                 book_id = EXCLUDED.book_id,
                 chapter_id = EXCLUDED.chapter_id,
                 name = EXCLUDED.name,
                 slug = EXCLUDED.slug,
                 url = EXCLUDED.url,
                 page_created_at = EXCLUDED.page_created_at,
                 page_updated_at = EXCLUDED.page_updated_at,
                 identity_ouid = EXCLUDED.identity_ouid,
                 page_kind = EXCLUDED.page_kind,
                 page_key = EXCLUDED.page_key,
                 archive_year = EXCLUDED.archive_year,
                 indexed_at = EXCLUDED.indexed_at,
                 deleted = EXCLUDED.deleted",
        )
        .bind(page.page_id)
        .bind(page.book_id)
        .bind(page.chapter_id)
        .bind(&page.name)
        .bind(&page.slug)
        .bind(page.url.as_deref())
        .bind(page.page_created_at.as_deref())
        .bind(page.page_updated_at.as_deref())
        .bind(page.identity_ouid.as_deref())
        .bind(page.page_kind.as_str())
        .bind(page.page_key.as_deref())
        .bind(page.archive_year)
        .bind(page.indexed_at)
        .bind(page.deleted)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("upsert_indexed_page page: {e}"))?;

        if let Some(cache) = cache {
            sqlx::query(
                "INSERT INTO page_cache (page_id, markdown, raw_markdown, html, cached_at, page_updated_at)
                 VALUES ($1, $2, $3, $4, $5, $6)
                 ON CONFLICT (page_id) DO UPDATE SET
                     markdown = EXCLUDED.markdown,
                     raw_markdown = EXCLUDED.raw_markdown,
                     html = EXCLUDED.html,
                     cached_at = EXCLUDED.cached_at,
                     page_updated_at = EXCLUDED.page_updated_at",
            )
            .bind(cache.page_id)
            .bind(cache.markdown.as_deref())
            .bind(cache.raw_markdown.as_deref())
            .bind(cache.html.as_deref())
            .bind(cache.cached_at)
            .bind(cache.page_updated_at.as_deref())
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("upsert_indexed_page cache: {e}"))?;
        }

        tx.commit()
            .await
            .map_err(|e| format!("upsert_indexed_page commit: {e}"))?;
        Ok(())
    }

    async fn get_indexed_page(&self, page_id: i64) -> Result<Option<IndexedPage>, String> {
        let row = sqlx::query(INDEXED_PAGE_SELECT_ONE)
            .bind(page_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| format!("get_indexed_page: {e}"))?;
        Ok(row.map(page_from_row))
    }

    async fn find_indexed_page_by_key(
        &self,
        identity_ouid: &str,
        page_kind: PageKind,
        page_key: &str,
    ) -> Result<Option<IndexedPage>, String> {
        let row = sqlx::query(
            "SELECT page_id, book_id, chapter_id, name, slug, url, page_created_at, page_updated_at,
                    identity_ouid, page_kind, page_key, archive_year, indexed_at, deleted
             FROM bookstack_pages
             WHERE identity_ouid = $1 AND page_kind = $2 AND page_key = $3 AND deleted = FALSE
             LIMIT 1",
        )
        .bind(identity_ouid)
        .bind(page_kind.as_str())
        .bind(page_key)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("find_indexed_page_by_key: {e}"))?;
        Ok(row.map(page_from_row))
    }

    async fn list_indexed_pages_by_chapter(
        &self,
        chapter_id: i64,
    ) -> Result<Vec<IndexedPage>, String> {
        let rows = sqlx::query(
            "SELECT page_id, book_id, chapter_id, name, slug, url, page_created_at, page_updated_at,
                    identity_ouid, page_kind, page_key, archive_year, indexed_at, deleted
             FROM bookstack_pages WHERE chapter_id = $1 AND deleted = FALSE
             ORDER BY name",
        )
        .bind(chapter_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_indexed_pages_by_chapter: {e}"))?;
        Ok(rows.into_iter().map(page_from_row).collect())
    }

    async fn list_indexed_pages_by_book_root(
        &self,
        book_id: i64,
    ) -> Result<Vec<IndexedPage>, String> {
        let rows = sqlx::query(
            "SELECT page_id, book_id, chapter_id, name, slug, url, page_created_at, page_updated_at,
                    identity_ouid, page_kind, page_key, archive_year, indexed_at, deleted
             FROM bookstack_pages WHERE book_id = $1 AND chapter_id IS NULL AND deleted = FALSE
             ORDER BY name",
        )
        .bind(book_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_indexed_pages_by_book_root: {e}"))?;
        Ok(rows.into_iter().map(page_from_row).collect())
    }

    async fn list_indexed_pages_recent(
        &self,
        book_id: i64,
        limit: i64,
    ) -> Result<Vec<IndexedPage>, String> {
        // page_updated_at is TEXT (ISO 8601) — string-sort gives chrono
        // order because ISO 8601 is lexicographically monotonic. NULL
        // updated_at sinks to the end via COALESCE (matches SQLite).
        let rows = sqlx::query(
            "SELECT page_id, book_id, chapter_id, name, slug, url, page_created_at, page_updated_at,
                    identity_ouid, page_kind, page_key, archive_year, indexed_at, deleted
             FROM bookstack_pages WHERE book_id = $1 AND deleted = FALSE
             ORDER BY COALESCE(page_updated_at, '') DESC
             LIMIT $2",
        )
        .bind(book_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_indexed_pages_recent: {e}"))?;
        Ok(rows.into_iter().map(page_from_row).collect())
    }

    async fn soft_delete_indexed_page(&self, page_id: i64) -> Result<(), String> {
        sqlx::query("UPDATE bookstack_pages SET deleted = TRUE WHERE page_id = $1")
            .bind(page_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("soft_delete_indexed_page: {e}"))?;
        Ok(())
    }

    // --- Page cache ---

    async fn get_page_cache(&self, page_id: i64) -> Result<Option<PageCache>, String> {
        let row = sqlx::query(
            "SELECT page_id, markdown, raw_markdown, html, cached_at, page_updated_at
             FROM page_cache WHERE page_id = $1",
        )
        .bind(page_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("get_page_cache: {e}"))?;
        Ok(row.map(|r| PageCache {
            page_id: r.get("page_id"),
            markdown: r.get("markdown"),
            raw_markdown: r.get("raw_markdown"),
            html: r.get("html"),
            cached_at: r.get("cached_at"),
            page_updated_at: r.get("page_updated_at"),
        }))
    }

    // --- Index jobs ---

    async fn create_index_job(
        &self,
        scope: &str,
        kind: &str,
        triggered_by: &str,
    ) -> Result<(i64, bool), String> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| format!("create_index_job tx: {e}"))?;

        // Dedup: pending/running collapse onto the existing job. Failed jobs
        // not yet touched by the reconciler also count as active so a webhook
        // firing mid-retry-window doesn't double-enqueue.
        let existing = sqlx::query(
            "SELECT id FROM index_jobs
             WHERE scope = $1
               AND (status IN ('pending', 'running')
                    OR (status = 'failed' AND resolved_status IS NULL))
             ORDER BY id DESC LIMIT 1
             FOR UPDATE",
        )
        .bind(scope)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| format!("create_index_job check: {e}"))?;

        if let Some(row) = existing {
            tx.commit()
                .await
                .map_err(|e| format!("create_index_job commit: {e}"))?;
            return Ok((row.get("id"), false));
        }

        let inserted = sqlx::query(
            "INSERT INTO index_jobs (scope, kind, status, triggered_by)
             VALUES ($1, $2, 'pending', $3)
             RETURNING id",
        )
        .bind(scope)
        .bind(kind)
        .bind(triggered_by)
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| format!("create_index_job insert: {e}"))?;

        tx.commit()
            .await
            .map_err(|e| format!("create_index_job commit: {e}"))?;
        Ok((inserted.get("id"), true))
    }

    async fn claim_next_index_job(&self) -> Result<Option<IndexJob>, String> {
        // FOR UPDATE SKIP LOCKED enables multiple worker processes to run
        // safely against the same database without claiming the same job.
        let row = sqlx::query(&format!(
            "UPDATE index_jobs SET status = 'running', started_at = $1 \
                 WHERE id = ( \
                     SELECT id FROM index_jobs WHERE status = 'pending' \
                     ORDER BY id ASC LIMIT 1 FOR UPDATE SKIP LOCKED \
                 ) \
                 RETURNING {INDEX_JOB_COLS}"
        ))
        .bind(Self::now_secs())
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("claim_next_index_job: {e}"))?;
        Ok(row.map(index_job_from_row))
    }

    async fn update_index_job_progress(
        &self,
        job_id: i64,
        progress: i64,
        total: i64,
    ) -> Result<(), String> {
        sqlx::query("UPDATE index_jobs SET progress = $1, total = $2 WHERE id = $3")
            .bind(progress)
            .bind(total)
            .bind(job_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("update_index_job_progress: {e}"))?;
        Ok(())
    }

    async fn complete_index_job(&self, job_id: i64, error: Option<&str>) -> Result<(), String> {
        let now = Self::now_secs();
        if let Some(reason) = error {
            // Status guard makes this idempotent — a cancel that landed
            // between the worker's last should_stop_index_job poll and this
            // write must not be silently overwritten.
            sqlx::query(
                "UPDATE index_jobs \
                 SET status = 'failed', finished_at = $1, error = $2 \
                 WHERE id = $3 AND status IN ('pending', 'running')",
            )
            .bind(now)
            .bind(reason)
            .bind(job_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("complete_index_job: {e}"))?;
        } else {
            // Same guard on the success branch — a cancel-then-success race
            // must leave the row in 'cancelled', not flip it back.
            sqlx::query(
                "UPDATE index_jobs \
                 SET status = 'succeeded', finished_at = $1, \
                     resolved_status = 'succeeded', resolved_at = $1, error = NULL \
                 WHERE id = $2 AND status IN ('pending', 'running')",
            )
            .bind(now)
            .bind(job_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("complete_index_job: {e}"))?;
        }
        Ok(())
    }

    async fn list_pending_index_jobs(&self, limit: i64) -> Result<Vec<IndexJob>, String> {
        let rows = sqlx::query(&format!(
            "SELECT {INDEX_JOB_COLS} FROM index_jobs \
                 WHERE status = 'pending' ORDER BY id ASC LIMIT $1"
        ))
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_pending_index_jobs: {e}"))?;
        Ok(rows.into_iter().map(index_job_from_row).collect())
    }

    async fn get_latest_index_job(&self) -> Result<Option<IndexJob>, String> {
        let row = sqlx::query(&format!(
            "SELECT {INDEX_JOB_COLS} FROM index_jobs ORDER BY id DESC LIMIT 1"
        ))
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("get_latest_index_job: {e}"))?;
        Ok(row.map(index_job_from_row))
    }

    async fn cancel_index_job(&self, job_id: i64) -> Result<(), String> {
        let now = Self::now_secs();
        sqlx::query(
            "UPDATE index_jobs \
             SET status = 'cancelled', resolved_status = 'cancelled', \
                 resolved_at = $1, finished_at = $1 \
             WHERE id = $2 AND status IN ('pending', 'running')",
        )
        .bind(now)
        .bind(job_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("cancel_index_job: {e}"))?;
        Ok(())
    }

    async fn should_stop_index_job(&self, job_id: i64) -> Result<bool, String> {
        let row: Option<(String,)> = sqlx::query_as("SELECT status FROM index_jobs WHERE id = $1")
            .bind(job_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| format!("should_stop_index_job: {e}"))?;
        Ok(matches!(row.as_ref().map(|r| r.0.as_str()), Some(s) if s != "running"))
    }

    async fn fail_index_job(&self, job_id: i64, reason: &str) -> Result<(), String> {
        let now = Self::now_secs();
        sqlx::query(
            "UPDATE index_jobs \
             SET status = 'failed', finished_at = $1, error = $2 \
             WHERE id = $3 AND status IN ('pending', 'running')",
        )
        .bind(now)
        .bind(reason)
        .bind(job_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("fail_index_job: {e}"))?;
        Ok(())
    }

    async fn list_failed_open_index_jobs(&self) -> Result<Vec<IndexJob>, String> {
        let rows = sqlx::query(&format!(
            "SELECT {INDEX_JOB_COLS} FROM index_jobs \
                 WHERE status = 'failed' ORDER BY id ASC"
        ))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_failed_open_index_jobs: {e}"))?;
        Ok(rows.into_iter().map(index_job_from_row).collect())
    }

    async fn has_successor_index_job(&self, scope: &str, excluded_id: i64) -> Result<bool, String> {
        let row: Option<(i64,)> = sqlx::query_as(
            "SELECT 1::BIGINT FROM index_jobs \
             WHERE scope = $1 AND id > $2 \
               AND status IN ('pending','running','succeeded','cancelled','closed') \
             LIMIT 1",
        )
        .bind(scope)
        .bind(excluded_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("has_successor_index_job: {e}"))?;
        Ok(row.is_some())
    }

    async fn index_job_retry_chain_len(&self, job_id: i64) -> Result<usize, String> {
        let mut len = 1usize;
        let mut current = job_id;
        for _ in 0..1024 {
            let parent: Option<(Option<i64>,)> =
                sqlx::query_as("SELECT retry_of FROM index_jobs WHERE id = $1")
                    .bind(current)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| format!("index_job_retry_chain_len: {e}"))?;
            match parent.and_then(|(p,)| p) {
                Some(p) => {
                    len += 1;
                    current = p;
                }
                None => break,
            }
        }
        Ok(len)
    }

    async fn close_index_job(
        &self,
        job_id: i64,
        resolved_status: Option<&str>,
    ) -> Result<(), String> {
        if let Some(rs) = resolved_status {
            sqlx::query(
                "UPDATE index_jobs \
                 SET prev_status = status, status = 'closed', resolved_status = $1 \
                 WHERE id = $2 AND status != 'closed'",
            )
            .bind(rs)
            .bind(job_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("close_index_job: {e}"))?;
        } else {
            sqlx::query(
                "UPDATE index_jobs \
                 SET prev_status = status, status = 'closed' \
                 WHERE id = $1 AND status != 'closed'",
            )
            .bind(job_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("close_index_job: {e}"))?;
        }
        Ok(())
    }

    async fn create_retry_index_job(
        &self,
        scope: &str,
        kind: &str,
        retry_of: i64,
    ) -> Result<i64, String> {
        let row: (i64,) = sqlx::query_as(
            "INSERT INTO index_jobs (scope, kind, status, triggered_by, retry_of) \
             VALUES ($1, $2, 'pending', 'reconciler', $3) RETURNING id",
        )
        .bind(scope)
        .bind(kind)
        .bind(retry_of)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| format!("create_retry_index_job: {e}"))?;
        Ok(row.0)
    }

    async fn list_archivable_index_jobs(&self, older_than_secs: i64) -> Result<Vec<i64>, String> {
        let cutoff = Self::now_secs() - older_than_secs;
        let rows: Vec<(i64,)> = sqlx::query_as(
            "SELECT id FROM index_jobs \
             WHERE status IN ('succeeded', 'cancelled') AND resolved_at IS NOT NULL \
               AND resolved_at <= $1 \
             ORDER BY id ASC",
        )
        .bind(cutoff)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_archivable_index_jobs: {e}"))?;
        Ok(rows.into_iter().map(|(id,)| id).collect())
    }

    async fn list_running_index_jobs_started_before(
        &self,
        started_before_secs: i64,
    ) -> Result<Vec<IndexJob>, String> {
        let rows = sqlx::query(&format!(
            "SELECT {INDEX_JOB_COLS} FROM index_jobs \
                 WHERE status = 'running' AND started_at IS NOT NULL AND started_at < $1"
        ))
        .bind(started_before_secs)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_running_index_jobs_started_before: {e}"))?;
        Ok(rows.into_iter().map(index_job_from_row).collect())
    }

    async fn list_index_jobs(&self, recent: usize) -> Result<Vec<IndexJob>, String> {
        let rows = sqlx::query(&format!(
            "(SELECT {INDEX_JOB_COLS} FROM index_jobs \
                  WHERE status IN ('pending', 'running', 'failed') ORDER BY id ASC) \
                 UNION ALL \
                 (SELECT {INDEX_JOB_COLS} FROM index_jobs \
                  WHERE status NOT IN ('pending', 'running', 'failed') \
                  ORDER BY id DESC LIMIT {recent})"
        ))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_index_jobs: {e}"))?;
        Ok(rows.into_iter().map(index_job_from_row).collect())
    }

    // --- Index meta ---

    async fn get_index_meta(&self, key: &str) -> Result<Option<String>, String> {
        let row = sqlx::query("SELECT value FROM index_meta WHERE key = $1")
            .bind(key)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| format!("get_index_meta: {e}"))?;
        Ok(row.map(|r| r.get("value")))
    }

    async fn set_index_meta(&self, key: &str, value: &str) -> Result<(), String> {
        sqlx::query(
            "INSERT INTO index_meta (key, value) VALUES ($1, $2)
             ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        )
        .bind(key)
        .bind(value)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("set_index_meta: {e}"))?;
        Ok(())
    }

    // --- Directory tree (issue #69) ---

    async fn list_indexed_shelves(&self) -> Result<Vec<IndexedShelf>, String> {
        let rows = sqlx::query(
            "SELECT shelf_id, name, slug, shelf_kind, indexed_at, deleted
             FROM bookstack_shelves WHERE deleted = FALSE
             ORDER BY name",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_indexed_shelves: {e}"))?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let kind_str: String = r.get("shelf_kind");
                IndexedShelf {
                    shelf_id: r.get("shelf_id"),
                    name: r.get("name"),
                    slug: r.get("slug"),
                    shelf_kind: kind_str.parse().unwrap_or(ShelfKind::Unclassified),
                    indexed_at: r.get("indexed_at"),
                    deleted: r.get("deleted"),
                }
            })
            .collect())
    }

    async fn list_indexed_books_unshelved(&self) -> Result<Vec<IndexedBook>, String> {
        let rows = sqlx::query(
            "SELECT book_id, name, slug, shelf_id, identity_ouid, book_kind, indexed_at, deleted
             FROM bookstack_books WHERE shelf_id IS NULL AND deleted = FALSE
             ORDER BY name",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("list_indexed_books_unshelved: {e}"))?;
        Ok(rows.into_iter().map(book_from_row).collect())
    }

    async fn read_directory_tree(
        &self,
        scope: DirectoryScope,
        depth: Option<u32>,
    ) -> Result<Vec<DirectoryNode>, String> {
        match scope {
            DirectoryScope::All => {
                let shelves: Vec<(i64, String, String)> = sqlx::query(
                    "SELECT shelf_id, name, slug FROM bookstack_shelves
                     WHERE deleted = FALSE ORDER BY name",
                )
                .fetch_all(&self.pool)
                .await
                .map_err(|e| format!("read_directory_tree shelves: {e}"))?
                .into_iter()
                .map(|r| {
                    (
                        r.get::<i64, _>("shelf_id"),
                        r.get::<String, _>("name"),
                        r.get::<String, _>("slug"),
                    )
                })
                .collect();

                let mut out: Vec<DirectoryNode> = Vec::with_capacity(shelves.len() + 1);
                for (shelf_id, name, slug) in shelves {
                    let children = if depth_allows_descent(depth) {
                        build_shelf_children(&self.pool, shelf_id, decrement_depth(depth)).await?
                    } else {
                        Vec::new()
                    };
                    out.push(DirectoryNode {
                        kind: DirectoryNodeKind::Shelf,
                        id: shelf_id,
                        name,
                        slug,
                        page_kind: None,
                        children,
                    });
                }

                let unshelved: Vec<(i64, String, String)> = sqlx::query(
                    "SELECT book_id, name, slug FROM bookstack_books
                     WHERE shelf_id IS NULL AND deleted = FALSE ORDER BY name",
                )
                .fetch_all(&self.pool)
                .await
                .map_err(|e| format!("read_directory_tree unshelved books: {e}"))?
                .into_iter()
                .map(|r| {
                    (
                        r.get::<i64, _>("book_id"),
                        r.get::<String, _>("name"),
                        r.get::<String, _>("slug"),
                    )
                })
                .collect();

                if !unshelved.is_empty() {
                    let mut children: Vec<DirectoryNode> = Vec::with_capacity(unshelved.len());
                    if depth_allows_descent(depth) {
                        let child_depth = decrement_depth(depth);
                        for b in unshelved {
                            children.push(build_book_node(&self.pool, b, child_depth).await?);
                        }
                    }
                    out.push(DirectoryNode {
                        kind: DirectoryNodeKind::Shelf,
                        id: 0,
                        name: SHELF_NAME_UNSHELVED.to_string(),
                        slug: "unshelved".to_string(),
                        page_kind: None,
                        children,
                    });
                }
                Ok(out)
            }
            DirectoryScope::Shelf(shelf_id) => {
                let shelf = sqlx::query(
                    "SELECT name, slug FROM bookstack_shelves
                     WHERE shelf_id = $1 AND deleted = FALSE",
                )
                .bind(shelf_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| format!("read_directory_tree shelf: {e}"))?;
                let (name, slug) = match shelf {
                    Some(r) => (r.get::<String, _>("name"), r.get::<String, _>("slug")),
                    None => return Err(format!("shelf {shelf_id} not found")),
                };
                let children = if depth_allows_descent(depth) {
                    build_shelf_children(&self.pool, shelf_id, decrement_depth(depth)).await?
                } else {
                    Vec::new()
                };
                Ok(vec![DirectoryNode {
                    kind: DirectoryNodeKind::Shelf,
                    id: shelf_id,
                    name,
                    slug,
                    page_kind: None,
                    children,
                }])
            }
            DirectoryScope::Book(book_id) => {
                let row = sqlx::query(
                    "SELECT book_id, name, slug FROM bookstack_books
                     WHERE book_id = $1 AND deleted = FALSE",
                )
                .bind(book_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| format!("read_directory_tree book: {e}"))?
                .ok_or_else(|| format!("book {book_id} not found"))?;
                let b = (
                    row.get::<i64, _>("book_id"),
                    row.get::<String, _>("name"),
                    row.get::<String, _>("slug"),
                );
                Ok(vec![build_book_node(&self.pool, b, depth).await?])
            }
            DirectoryScope::Chapter(chapter_id) => {
                let row = sqlx::query(
                    "SELECT chapter_id, name, slug FROM bookstack_chapters
                     WHERE chapter_id = $1 AND deleted = FALSE",
                )
                .bind(chapter_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| format!("read_directory_tree chapter: {e}"))?
                .ok_or_else(|| format!("chapter {chapter_id} not found"))?;
                let c = (
                    row.get::<i64, _>("chapter_id"),
                    row.get::<String, _>("name"),
                    row.get::<String, _>("slug"),
                );
                Ok(vec![build_chapter_node(&self.pool, c, depth).await?])
            }
        }
    }
}

// --- Directory tree helpers (issue #69) ---

const SHELF_NAME_UNSHELVED: &str = "Unshelved";

fn depth_allows_descent(depth: Option<u32>) -> bool {
    match depth {
        None => true,
        Some(0) => false,
        Some(_) => true,
    }
}

fn decrement_depth(depth: Option<u32>) -> Option<u32> {
    depth.map(|d| d.saturating_sub(1))
}

async fn build_shelf_children(
    pool: &PgPool,
    shelf_id: i64,
    child_depth: Option<u32>,
) -> Result<Vec<DirectoryNode>, String> {
    let books: Vec<(i64, String, String)> = sqlx::query(
        "SELECT book_id, name, slug FROM bookstack_books
         WHERE shelf_id = $1 AND deleted = FALSE ORDER BY name",
    )
    .bind(shelf_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("build_shelf_children: {e}"))?
    .into_iter()
    .map(|r| {
        (
            r.get::<i64, _>("book_id"),
            r.get::<String, _>("name"),
            r.get::<String, _>("slug"),
        )
    })
    .collect();
    let mut out: Vec<DirectoryNode> = Vec::with_capacity(books.len());
    for b in books {
        out.push(build_book_node(pool, b, child_depth).await?);
    }
    Ok(out)
}

async fn build_book_node(
    pool: &PgPool,
    book: (i64, String, String),
    depth: Option<u32>,
) -> Result<DirectoryNode, String> {
    let (book_id, name, slug) = book;
    let children = if depth_allows_descent(depth) {
        let child_depth = decrement_depth(depth);

        let chap_rows: Vec<(i64, String, String)> = sqlx::query(
            "SELECT chapter_id, name, slug FROM bookstack_chapters
             WHERE book_id = $1 AND deleted = FALSE ORDER BY name",
        )
        .bind(book_id)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("build_book_node chapters: {e}"))?
        .into_iter()
        .map(|r| {
            (
                r.get::<i64, _>("chapter_id"),
                r.get::<String, _>("name"),
                r.get::<String, _>("slug"),
            )
        })
        .collect();

        let mut children: Vec<DirectoryNode> = Vec::with_capacity(chap_rows.len());
        for c in chap_rows {
            children.push(build_chapter_node(pool, c, child_depth).await?);
        }

        // Book-root (chapter-less) pages.
        let root_pages: Vec<(i64, String, String, String)> = sqlx::query(
            "SELECT page_id, name, slug, page_kind FROM bookstack_pages
             WHERE book_id = $1 AND chapter_id IS NULL AND deleted = FALSE ORDER BY name",
        )
        .bind(book_id)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("build_book_node root pages: {e}"))?
        .into_iter()
        .map(|r| {
            (
                r.get::<i64, _>("page_id"),
                r.get::<String, _>("name"),
                r.get::<String, _>("slug"),
                r.get::<String, _>("page_kind"),
            )
        })
        .collect();
        for p in root_pages {
            children.push(page_node(p));
        }

        children
    } else {
        Vec::new()
    };
    Ok(DirectoryNode {
        kind: DirectoryNodeKind::Book,
        id: book_id,
        name,
        slug,
        page_kind: None,
        children,
    })
}

async fn build_chapter_node(
    pool: &PgPool,
    chap: (i64, String, String),
    depth: Option<u32>,
) -> Result<DirectoryNode, String> {
    let (chapter_id, name, slug) = chap;
    let children = if depth_allows_descent(depth) {
        let pages: Vec<(i64, String, String, String)> = sqlx::query(
            "SELECT page_id, name, slug, page_kind FROM bookstack_pages
             WHERE chapter_id = $1 AND deleted = FALSE ORDER BY name",
        )
        .bind(chapter_id)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("build_chapter_node pages: {e}"))?
        .into_iter()
        .map(|r| {
            (
                r.get::<i64, _>("page_id"),
                r.get::<String, _>("name"),
                r.get::<String, _>("slug"),
                r.get::<String, _>("page_kind"),
            )
        })
        .collect();
        pages.into_iter().map(page_node).collect()
    } else {
        Vec::new()
    };
    Ok(DirectoryNode {
        kind: DirectoryNodeKind::Chapter,
        id: chapter_id,
        name,
        slug,
        page_kind: None,
        children,
    })
}

fn page_node(p: (i64, String, String, String)) -> DirectoryNode {
    let (id, name, slug, page_kind) = p;
    DirectoryNode {
        kind: DirectoryNodeKind::Page,
        id,
        name,
        slug,
        page_kind: Some(page_kind),
        children: Vec::new(),
    }
}

// --- Row → struct helpers (shared across IndexDb methods) ---

const INDEXED_PAGE_SELECT_ONE: &str =
    "SELECT page_id, book_id, chapter_id, name, slug, url, page_created_at, page_updated_at,
            identity_ouid, page_kind, page_key, archive_year, indexed_at, deleted
     FROM bookstack_pages WHERE page_id = $1";

fn book_from_row(r: sqlx::postgres::PgRow) -> IndexedBook {
    let kind_str: String = r.get("book_kind");
    IndexedBook {
        book_id: r.get("book_id"),
        name: r.get("name"),
        slug: r.get("slug"),
        shelf_id: r.get("shelf_id"),
        identity_ouid: r.get("identity_ouid"),
        book_kind: kind_str.parse().unwrap_or(BookKind::Unclassified),
        indexed_at: r.get("indexed_at"),
        deleted: r.get("deleted"),
    }
}

fn chapter_from_row(r: sqlx::postgres::PgRow) -> IndexedChapter {
    let kind_str: String = r.get("chapter_kind");
    IndexedChapter {
        chapter_id: r.get("chapter_id"),
        book_id: r.get("book_id"),
        name: r.get("name"),
        slug: r.get("slug"),
        identity_ouid: r.get("identity_ouid"),
        chapter_kind: kind_str.parse().unwrap_or(ChapterKind::Unclassified),
        archive_year: r.get("archive_year"),
        indexed_at: r.get("indexed_at"),
        deleted: r.get("deleted"),
    }
}

fn page_from_row(r: sqlx::postgres::PgRow) -> IndexedPage {
    let kind_str: String = r.get("page_kind");
    IndexedPage {
        page_id: r.get("page_id"),
        book_id: r.get("book_id"),
        chapter_id: r.get("chapter_id"),
        name: r.get("name"),
        slug: r.get("slug"),
        url: r.get("url"),
        page_created_at: r.get("page_created_at"),
        page_updated_at: r.get("page_updated_at"),
        identity_ouid: r.get("identity_ouid"),
        page_kind: kind_str.parse().unwrap_or(PageKind::Unclassified),
        page_key: r.get("page_key"),
        archive_year: r.get("archive_year"),
        indexed_at: r.get("indexed_at"),
        deleted: r.get("deleted"),
    }
}

fn index_job_from_row(r: sqlx::postgres::PgRow) -> IndexJob {
    IndexJob {
        id: r.get("id"),
        scope: r.get("scope"),
        kind: r.get("kind"),
        status: r.get("status"),
        triggered_by: r.get("triggered_by"),
        started_at: r.get("started_at"),
        finished_at: r.get("finished_at"),
        progress: r.get("progress"),
        total: r.get("total"),
        error: r.get("error"),
        resolved_status: r.get("resolved_status"),
        prev_status: r.get("prev_status"),
        resolved_at: r.get("resolved_at"),
        retry_of: r.get("retry_of"),
    }
}

const INDEX_JOB_COLS: &str =
    "id, scope, kind, status, triggered_by, started_at, finished_at, progress, total, error, \
     resolved_status, prev_status, resolved_at, retry_of";

fn embed_job_from_row(r: sqlx::postgres::PgRow) -> EmbedJob {
    EmbedJob {
        id: r.get("id"),
        scope: r.get("scope"),
        status: r.get("status"),
        total_pages: r.get("total_pages"),
        done_pages: r.get("done_pages"),
        started_at: r.get("started_at"),
        finished_at: r.get("finished_at"),
        error: r.get("error"),
        worker_id: r.get("worker_id"),
        resolved_status: r.get("resolved_status"),
        prev_status: r.get("prev_status"),
        resolved_at: r.get("resolved_at"),
        retry_of: r.get("retry_of"),
    }
}

const EMBED_JOB_COLS: &str =
    "id, scope, status, total_pages, done_pages, started_at, finished_at, error, worker_id, \
     resolved_status, prev_status, resolved_at, retry_of";
