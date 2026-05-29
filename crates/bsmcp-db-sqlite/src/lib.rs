use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{AeadCore, Aes256Gcm};
use async_trait::async_trait;
use base64::Engine;
use rusqlite::{params, Connection};
use sha2::Digest;
use zeroize::Zeroizing;

use bsmcp_common::config::{access_token_ttl, refresh_token_ttl};
use bsmcp_common::db::{DbBackend, IndexDb, SemanticDb};
use bsmcp_common::index::*;
use bsmcp_common::settings::GlobalSettings;
use bsmcp_common::types::*;
use bsmcp_common::vector;

const BASE64: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::STANDARD;

pub struct SqliteDb {
    conn: Arc<Mutex<Connection>>,
    encryption_key: Zeroizing<[u8; 32]>,
}

impl SqliteDb {
    pub fn open(path: &Path, encryption_key: &str) -> Self {
        let conn = Connection::open(path).expect("Failed to open SQLite database");
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS access_tokens (
                 token TEXT PRIMARY KEY,
                 token_id TEXT NOT NULL,
                 token_secret TEXT NOT NULL,
                 created_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_tokens_created ON access_tokens(created_at);
             CREATE TABLE IF NOT EXISTS refresh_tokens (
                 token TEXT PRIMARY KEY,
                 token_id TEXT NOT NULL,
                 token_secret TEXT NOT NULL,
                 created_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_refresh_tokens_created ON refresh_tokens(created_at);
             CREATE TABLE IF NOT EXISTS global_settings (
                 id INTEGER PRIMARY KEY CHECK (id = 1),
                 hive_shelf_id INTEGER,
                 user_journals_shelf_id INTEGER,
                 set_by_token_hash TEXT,
                 updated_at INTEGER NOT NULL DEFAULT 0
             );
             INSERT OR IGNORE INTO global_settings (id, updated_at) VALUES (1, 0);
             DROP TABLE IF EXISTS registrations;
             /* v1.0.0 — DB-as-index. Mirror of every BookStack content item we
                care about. Phase 3 ships the schema only; the reconciliation
                worker (Phase 4) populates these tables. Distinct from the
                semantic-search `pages` / `chunks` tables, which track only the
                embedding state — these tables track the structural reflection
                used by the briefing/journal/migration paths. */
             CREATE TABLE IF NOT EXISTS bookstack_shelves (
                 shelf_id INTEGER PRIMARY KEY,
                 name TEXT NOT NULL,
                 slug TEXT NOT NULL,
                 shelf_kind TEXT NOT NULL,
                 indexed_at INTEGER NOT NULL,
                 deleted INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS bookstack_books (
                 book_id INTEGER PRIMARY KEY,
                 name TEXT NOT NULL,
                 slug TEXT NOT NULL,
                 shelf_id INTEGER,
                 identity_ouid TEXT,
                 book_kind TEXT NOT NULL,
                 indexed_at INTEGER NOT NULL,
                 deleted INTEGER NOT NULL DEFAULT 0
             );
             CREATE INDEX IF NOT EXISTS idx_bookstack_books_shelf ON bookstack_books(shelf_id);
             CREATE INDEX IF NOT EXISTS idx_bookstack_books_identity ON bookstack_books(identity_ouid);
             CREATE TABLE IF NOT EXISTS bookstack_chapters (
                 chapter_id INTEGER PRIMARY KEY,
                 book_id INTEGER NOT NULL,
                 name TEXT NOT NULL,
                 slug TEXT NOT NULL,
                 identity_ouid TEXT,
                 chapter_kind TEXT NOT NULL,
                 archive_year INTEGER,
                 indexed_at INTEGER NOT NULL,
                 deleted INTEGER NOT NULL DEFAULT 0
             );
             CREATE INDEX IF NOT EXISTS idx_bookstack_chapters_book ON bookstack_chapters(book_id);
             CREATE INDEX IF NOT EXISTS idx_bookstack_chapters_identity ON bookstack_chapters(identity_ouid);
             CREATE TABLE IF NOT EXISTS bookstack_pages (
                 page_id INTEGER PRIMARY KEY,
                 book_id INTEGER NOT NULL,
                 chapter_id INTEGER,
                 name TEXT NOT NULL,
                 slug TEXT NOT NULL,
                 url TEXT,
                 page_created_at TEXT,
                 page_updated_at TEXT,
                 identity_ouid TEXT,
                 page_kind TEXT NOT NULL,
                 page_key TEXT,
                 archive_year INTEGER,
                 indexed_at INTEGER NOT NULL,
                 deleted INTEGER NOT NULL DEFAULT 0
             );
             CREATE INDEX IF NOT EXISTS idx_bookstack_pages_book ON bookstack_pages(book_id);
             CREATE INDEX IF NOT EXISTS idx_bookstack_pages_chapter ON bookstack_pages(chapter_id);
             CREATE INDEX IF NOT EXISTS idx_bookstack_pages_identity_kind ON bookstack_pages(identity_ouid, page_kind);
             /* Dedup enforcement: at most one non-deleted classified page per
                (identity, kind, key). NULL identity or NULL key are excluded so
                unclassified pages never trip the constraint. */
             CREATE UNIQUE INDEX IF NOT EXISTS idx_bookstack_pages_dedup
                 ON bookstack_pages(identity_ouid, page_kind, page_key)
                 WHERE deleted = 0 AND identity_ouid IS NOT NULL AND page_key IS NOT NULL;
             /* Page-body cache. One row per page; refreshed when BookStack's
                page_updated_at advances. Cache hit when our row's
                page_updated_at equals bookstack_pages.page_updated_at. */
             CREATE TABLE IF NOT EXISTS page_cache (
                 page_id INTEGER PRIMARY KEY,
                 markdown TEXT,
                 raw_markdown TEXT,
                 html TEXT,
                 cached_at INTEGER NOT NULL,
                 page_updated_at TEXT
             );
             /* Reconciliation job queue. Mirrors `embed_jobs` in shape so the
                worker pattern stays familiar. */
             CREATE TABLE IF NOT EXISTS index_jobs (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 scope TEXT NOT NULL,
                 kind TEXT NOT NULL,
                 status TEXT NOT NULL DEFAULT 'pending',
                 triggered_by TEXT NOT NULL,
                 started_at INTEGER,
                 finished_at INTEGER,
                 progress INTEGER NOT NULL DEFAULT 0,
                 total INTEGER NOT NULL DEFAULT 0,
                 error TEXT,
                 resolved_status TEXT,
                 prev_status TEXT,
                 resolved_at INTEGER,
                 retry_of INTEGER
             );
             CREATE INDEX IF NOT EXISTS idx_index_jobs_pending ON index_jobs(status) WHERE status = 'pending';
             /* Singleton bookkeeping for the indexer (last_full_walk_at,
                last_delta_walk_at, etc.). */
             CREATE TABLE IF NOT EXISTS index_meta (
                 key TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );",
        )
        .expect("Failed to initialize database schema");

        // v0.10.0 cleanup migrations — drop the briefing/per-user-settings
        // surface stripped in #78. Idempotent on rerun: SQLite 3.35+
        // supports `DROP COLUMN`; older builds silently no-op via `.ok()`.
        for sql in [
            // user_settings table — entire surface gone (no per-user state).
            "DROP TABLE IF EXISTS user_settings",
            // user_role_cache table — fed the per-user role-level ACL filter
            // that's gone with semantic search becoming user-anonymous.
            "DROP TABLE IF EXISTS user_role_cache",
            // remember_audit + leftover v1.0.0 surface (idempotent).
            "DROP INDEX IF EXISTS idx_audit_user_time",
            "DROP INDEX IF EXISTS idx_audit_resource_time",
            "DROP TABLE IF EXISTS remember_audit",
            "DROP TABLE IF EXISTS token_bindings",
            "DROP TABLE IF EXISTS sessions",
            // Briefing-only global_settings columns dropped in v0.10.0.
            "ALTER TABLE global_settings DROP COLUMN org_required_instructions_page_ids",
            "ALTER TABLE global_settings DROP COLUMN org_ai_usage_policy_page_ids",
            "ALTER TABLE global_settings DROP COLUMN org_identity_page_id",
            "ALTER TABLE global_settings DROP COLUMN org_domains",
            "ALTER TABLE global_settings DROP COLUMN guide_page_id",
            "ALTER TABLE global_settings DROP COLUMN policies_scope",
            "ALTER TABLE global_settings DROP COLUMN sops_scope",
            "ALTER TABLE global_settings DROP COLUMN best_practices_scope",
            "ALTER TABLE global_settings DROP COLUMN friendly_structure",
            "ALTER TABLE global_settings DROP COLUMN full_content_in_briefing",
            "ALTER TABLE global_settings DROP COLUMN strict_setup",
            // v0.7.x default_ai_identity_* leftovers.
            "ALTER TABLE global_settings DROP COLUMN default_ai_identity_page_id",
            "ALTER TABLE global_settings DROP COLUMN default_ai_identity_name",
            "ALTER TABLE global_settings DROP COLUMN default_ai_identity_ouid",
        ] {
            conn.execute_batch(sql).ok();
        }

        // Issue #54 — job lifecycle columns on index_jobs. The matching
        // embed_jobs migration runs in init_semantic_tables. SQLite has no
        // ADD COLUMN IF NOT EXISTS, so the duplicate-column error is
        // swallowed via .ok() (same pattern as the v0.8.0 block above).
        for sql in [
            "ALTER TABLE index_jobs ADD COLUMN resolved_status TEXT",
            "ALTER TABLE index_jobs ADD COLUMN prev_status TEXT",
            "ALTER TABLE index_jobs ADD COLUMN resolved_at INTEGER",
            "ALTER TABLE index_jobs ADD COLUMN retry_of INTEGER",
        ] {
            conn.execute_batch(sql).ok();
        }
        // Migrate any pre-#54 rows that used `status='error'` on this queue.
        // The embed_jobs equivalent runs in init_semantic_tables. Leaves
        // resolved_status NULL so the reconciler picks them up.
        conn.execute_batch(
            "UPDATE index_jobs SET status = 'failed' \
             WHERE status = 'error' AND resolved_status IS NULL",
        )
        .ok();

        let hash = sha2::Sha256::digest(encryption_key.as_bytes());
        let mut key = Zeroizing::new([0u8; 32]);
        key.copy_from_slice(&hash);

        Self {
            conn: Arc::new(Mutex::new(conn)),
            encryption_key: key,
        }
    }

    /// SHA-256 hash a bearer token before storing as primary key.
    /// This prevents token theft from database read access.
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

    fn now_secs() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }

    fn cutoff_secs(ttl: Duration) -> i64 {
        Self::now_secs() - ttl.as_secs() as i64
    }

    fn cleanup_old_backups(backup_dir: &Path) {
        let mut backups: Vec<_> = std::fs::read_dir(backup_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("bookstack-mcp-backup-")
                    && e.file_name().to_string_lossy().ends_with(".db")
            })
            .collect();

        backups.sort_by_key(|e| e.file_name());

        if backups.len() > 3 {
            for entry in &backups[..backups.len() - 3] {
                if let Err(e) = std::fs::remove_file(entry.path()) {
                    eprintln!(
                        "Failed to remove old backup {}: {e}",
                        entry.path().display()
                    );
                } else {
                    eprintln!(
                        "Removed old backup: {}",
                        entry.file_name().to_string_lossy()
                    );
                }
            }
        }
    }
}

#[async_trait]
impl DbBackend for SqliteDb {
    async fn insert_access_token(&self, token: &str, id: &str, secret: &str) -> Result<(), String> {
        let conn = self.conn.clone();
        let token_hash = Self::hash_token(token);
        let enc_id = self.encrypt(id);
        let enc_secret = self.encrypt(secret);

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let count: i64 = conn
                .query_row("SELECT COUNT(*) FROM access_tokens", [], |row| row.get(0))
                .unwrap_or(0);
            if count >= 10000 {
                let cutoff = SqliteDb::cutoff_secs(access_token_ttl());
                conn.execute("DELETE FROM access_tokens WHERE created_at <= ?1", params![cutoff]).ok();
            }
            conn.execute(
                "INSERT OR REPLACE INTO access_tokens (token, token_id, token_secret, created_at) VALUES (?1, ?2, ?3, ?4)",
                params![token_hash, enc_id, enc_secret, SqliteDb::now_secs()],
            ).ok();
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_access_token(&self, token: &str) -> Result<Option<(String, String)>, String> {
        let conn = self.conn.clone();
        let token_hash = Self::hash_token(token);
        let token_raw = token.to_string();
        let encryption_key = *self.encryption_key;

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let cutoff = SqliteDb::cutoff_secs(access_token_ttl());

            // Try hashed token first, then fall back to raw token (pre-hash migration)
            let result: Option<(String, String)> = conn.query_row(
                "SELECT token_id, token_secret FROM access_tokens WHERE token = ?1 AND created_at > ?2",
                params![token_hash, cutoff],
                |row| Ok((row.get(0)?, row.get(1)?)),
            ).ok().or_else(|| {
                conn.query_row(
                    "SELECT token_id, token_secret FROM access_tokens WHERE token = ?1 AND created_at > ?2",
                    params![token_raw, cutoff],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                ).ok()
            });

            let Some((stored_id, stored_secret)) = result else {
                return Ok(None);
            };

            // Try decrypting
            let cipher = Aes256Gcm::new((&encryption_key).into());
            let try_decrypt = |stored: &str| -> Option<String> {
                let combined = BASE64.decode(stored).ok()?;
                if combined.len() < 12 { return None; }
                let (nonce_bytes, ciphertext) = combined.split_at(12);
                let nonce = aes_gcm::Nonce::from_slice(nonce_bytes);
                let plaintext = cipher.decrypt(nonce, ciphertext).ok()?;
                String::from_utf8(plaintext).ok()
            };

            if let (Some(tid), Some(tsec)) = (try_decrypt(&stored_id), try_decrypt(&stored_secret)) {
                return Ok(Some((tid, tsec)));
            }

            // Decryption failed — treat as plaintext (pre-encryption data)
            // Re-encrypt in place for transparent migration
            let re_encrypt = |plaintext: &str| -> String {
                let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
                let ciphertext = cipher.encrypt(&nonce, plaintext.as_bytes()).expect("AES-GCM encryption failed");
                let mut combined = nonce.to_vec();
                combined.extend_from_slice(&ciphertext);
                BASE64.encode(&combined)
            };
            let enc_id = re_encrypt(&stored_id);
            let enc_secret = re_encrypt(&stored_secret);
            conn.execute(
                "UPDATE access_tokens SET token_id = ?1, token_secret = ?2 WHERE token = ?3",
                params![enc_id, enc_secret, token_raw],
            ).ok();

            Ok(Some((stored_id, stored_secret)))
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn cleanup_expired_tokens(&self) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let cutoff = SqliteDb::cutoff_secs(access_token_ttl());
            conn.execute(
                "DELETE FROM access_tokens WHERE created_at <= ?1",
                params![cutoff],
            )
            .ok();
            let refresh_cutoff = SqliteDb::cutoff_secs(refresh_token_ttl());
            conn.execute(
                "DELETE FROM refresh_tokens WHERE created_at <= ?1",
                params![refresh_cutoff],
            )
            .ok();
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn insert_refresh_token(
        &self,
        token: &str,
        id: &str,
        secret: &str,
    ) -> Result<(), String> {
        let conn = self.conn.clone();
        let token_hash = Self::hash_token(token);
        let enc_id = self.encrypt(id);
        let enc_secret = self.encrypt(secret);

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "INSERT OR REPLACE INTO refresh_tokens (token, token_id, token_secret, created_at) VALUES (?1, ?2, ?3, ?4)",
                params![token_hash, enc_id, enc_secret, SqliteDb::now_secs()],
            ).ok();
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_refresh_token(&self, token: &str) -> Result<Option<(String, String)>, String> {
        let conn = self.conn.clone();
        let token_hash = Self::hash_token(token);
        let encryption_key = *self.encryption_key;

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let cutoff = SqliteDb::cutoff_secs(refresh_token_ttl());

            let result: Option<(String, String)> = conn.query_row(
                "SELECT token_id, token_secret FROM refresh_tokens WHERE token = ?1 AND created_at > ?2",
                params![token_hash, cutoff],
                |row| Ok((row.get(0)?, row.get(1)?)),
            ).ok();

            let Some((stored_id, stored_secret)) = result else {
                return Ok(None);
            };

            let cipher = Aes256Gcm::new((&encryption_key).into());
            let try_decrypt = |stored: &str| -> Option<String> {
                let combined = BASE64.decode(stored).ok()?;
                if combined.len() < 12 { return None; }
                let (nonce_bytes, ciphertext) = combined.split_at(12);
                let nonce = aes_gcm::Nonce::from_slice(nonce_bytes);
                let plaintext = cipher.decrypt(nonce, ciphertext).ok()?;
                String::from_utf8(plaintext).ok()
            };

            match (try_decrypt(&stored_id), try_decrypt(&stored_secret)) {
                (Some(tid), Some(tsec)) => Ok(Some((tid, tsec))),
                _ => Ok(None),
            }
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn delete_refresh_token(&self, token: &str) -> Result<(), String> {
        let conn = self.conn.clone();
        let token_hash = Self::hash_token(token);

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "DELETE FROM refresh_tokens WHERE token = ?1",
                params![token_hash],
            )
            .ok();
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_global_settings(&self) -> Result<GlobalSettings, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<GlobalSettings, String> {
            let conn = conn.lock().unwrap();
            let row = conn
                .query_row(
                    "SELECT hive_shelf_id, user_journals_shelf_id,
                        set_by_token_hash, updated_at
                 FROM global_settings WHERE id = 1",
                    [],
                    |row| {
                        Ok(GlobalSettings {
                            hive_shelf_id: row.get::<_, Option<i64>>(0)?,
                            user_journals_shelf_id: row.get::<_, Option<i64>>(1)?,
                            set_by_token_hash: row.get::<_, Option<String>>(2)?,
                            updated_at: row.get::<_, i64>(3)?,
                        })
                    },
                )
                .unwrap_or_default();
            Ok(row)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn save_global_settings(
        &self,
        settings: &GlobalSettings,
        set_by_token_hash: &str,
    ) -> Result<(), String> {
        let conn = self.conn.clone();
        let s = settings.clone();
        let setter = set_by_token_hash.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let existing_setter: Option<String> = conn
                .query_row(
                    "SELECT set_by_token_hash FROM global_settings WHERE id = 1 AND updated_at > 0",
                    [],
                    |row| row.get(0),
                )
                .ok()
                .flatten();
            let final_setter = existing_setter.unwrap_or(setter);
            conn.execute(
                "UPDATE global_settings
                 SET hive_shelf_id = ?1,
                     user_journals_shelf_id = ?2,
                     set_by_token_hash = ?3,
                     updated_at = ?4
                 WHERE id = 1",
                params![
                    s.hive_shelf_id,
                    s.user_journals_shelf_id,
                    final_setter,
                    SqliteDb::now_secs(),
                ],
            )
            .map_err(|e| format!("save_global_settings: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn backup(&self, backup_dir: &Path) -> Result<(), String> {
        let conn = self.conn.clone();
        let backup_dir = backup_dir.to_path_buf();

        tokio::task::spawn_blocking(move || {
            std::fs::create_dir_all(&backup_dir)
                .map_err(|e| format!("Failed to create backup directory: {e}"))?;

            let timestamp = {
                let secs = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let days = secs / 86400;
                let time_of_day = secs % 86400;
                let (year, month, day) = unix_days_to_ymd(days as i64);
                let hours = time_of_day / 3600;
                let minutes = (time_of_day % 3600) / 60;
                let seconds = time_of_day % 60;
                format!("{year:04}{month:02}{day:02}T{hours:02}{minutes:02}{seconds:02}")
            };

            let backup_file = backup_dir.join(format!("bookstack-mcp-backup-{timestamp}.db"));
            let backup_path_str = backup_file.to_string_lossy();

            let conn = conn.lock().unwrap();
            conn.execute_batch(&format!(
                "VACUUM INTO '{}'",
                backup_path_str.replace('\'', "''")
            ))
            .map_err(|e| format!("VACUUM INTO failed: {e}"))?;

            drop(conn);
            eprintln!("Backup created: {}", backup_file.display());

            SqliteDb::cleanup_old_backups(&backup_dir);
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }
}

#[async_trait]
impl SemanticDb for SqliteDb {
    async fn init_semantic_tables(&self) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS pages (
                    page_id INTEGER PRIMARY KEY,
                    book_id INTEGER NOT NULL,
                    chapter_id INTEGER,
                    name TEXT NOT NULL,
                    slug TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedded_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_id INTEGER NOT NULL REFERENCES pages(page_id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    heading_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    UNIQUE(page_id, chunk_index)
                );
                CREATE TABLE IF NOT EXISTS relationships (
                    source_page_id INTEGER NOT NULL,
                    target_page_id INTEGER NOT NULL,
                    link_type TEXT NOT NULL DEFAULT 'link',
                    PRIMARY KEY (source_page_id, target_page_id, link_type)
                );
                CREATE TABLE IF NOT EXISTS embed_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    total_pages INTEGER DEFAULT 0,
                    done_pages INTEGER DEFAULT 0,
                    started_at INTEGER,
                    finished_at INTEGER,
                    error TEXT,
                    worker_id TEXT,
                    resolved_status TEXT,
                    prev_status TEXT,
                    resolved_at INTEGER,
                    retry_of INTEGER
                );",
            )
            .map_err(|e| format!("Failed to initialize semantic tables: {e}"))?;

            // Migration: add worker_id column if missing (existing databases)
            conn.execute_batch(
                "ALTER TABLE embed_jobs ADD COLUMN worker_id TEXT;"
            ).ok();

            // Issue #54 — job lifecycle columns. SQLite swallows duplicate-
            // column errors via .ok(); same pattern as the v0.8.0 block.
            for sql in [
                "ALTER TABLE embed_jobs ADD COLUMN resolved_status TEXT",
                "ALTER TABLE embed_jobs ADD COLUMN prev_status TEXT",
                "ALTER TABLE embed_jobs ADD COLUMN resolved_at INTEGER",
                "ALTER TABLE embed_jobs ADD COLUMN retry_of INTEGER",
            ] {
                conn.execute_batch(sql).ok();
            }
            // Migrate v0.7.x rows that used 'error' as the failed sentinel.
            // resolved_status stays NULL so the reconciler picks them up.
            conn.execute_batch(
                "UPDATE embed_jobs SET status = 'failed' \
                 WHERE status = 'error' AND resolved_status IS NULL",
            ).ok();

            // Metadata key-value store (v0.5.0+)
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            ).ok();

            // Migration: add updated_at column if missing
            conn.execute_batch(
                "ALTER TABLE pages ADD COLUMN updated_at TEXT;"
            ).ok();

            // Permission ACL: per-page role visibility populated at embed time.
            conn.execute_batch(
                "ALTER TABLE pages ADD COLUMN acl_default_open INTEGER;"
            ).ok();
            conn.execute_batch(
                "ALTER TABLE pages ADD COLUMN acl_computed_at INTEGER;"
            ).ok();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS page_view_acl (
                     page_id INTEGER NOT NULL,
                     role_id INTEGER NOT NULL,
                     PRIMARY KEY (page_id, role_id)
                 );
                 CREATE INDEX IF NOT EXISTS idx_page_view_acl_role ON page_view_acl(role_id, page_id);
                 CREATE TABLE IF NOT EXISTS acl_reconcile_state (
                     scope TEXT PRIMARY KEY,
                     last_full_run INTEGER NOT NULL DEFAULT 0
                 );"
            ).map_err(|e| format!("Failed to create ACL tables: {e}"))?;

            eprintln!("Semantic: tables initialized");
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn upsert_page(&self, meta: &PageMeta) -> Result<(), String> {
        let conn = self.conn.clone();
        let meta = meta.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "INSERT INTO pages (page_id, book_id, chapter_id, name, slug, content_hash, embedded_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(page_id) DO UPDATE SET
                    book_id = excluded.book_id,
                    chapter_id = excluded.chapter_id,
                    name = excluded.name,
                    slug = excluded.slug,
                    content_hash = excluded.content_hash,
                    embedded_at = excluded.embedded_at,
                    updated_at = excluded.updated_at",
                params![meta.page_id, meta.book_id, meta.chapter_id, meta.name, meta.slug, meta.content_hash, SqliteDb::now_secs(), meta.updated_at],
            ).map_err(|e| format!("upsert_page failed: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn delete_page(&self, page_id: i64) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let tx = conn
                .unchecked_transaction()
                .map_err(|e| format!("Transaction failed: {e}"))?;
            tx.execute("DELETE FROM chunks WHERE page_id = ?1", params![page_id])
                .ok();
            tx.execute(
                "DELETE FROM relationships WHERE source_page_id = ?1 OR target_page_id = ?1",
                params![page_id],
            )
            .ok();
            tx.execute("DELETE FROM pages WHERE page_id = ?1", params![page_id])
                .ok();
            tx.commit().map_err(|e| format!("Commit failed: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_page_content_hash(&self, page_id: i64) -> Result<Option<String>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            Ok(conn
                .query_row(
                    "SELECT content_hash FROM pages WHERE page_id = ?1",
                    params![page_id],
                    |row| row.get(0),
                )
                .ok())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_page_meta(&self, page_id: i64) -> Result<Option<PageMeta>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            Ok(conn.query_row(
                "SELECT page_id, book_id, chapter_id, name, slug, content_hash, updated_at FROM pages WHERE page_id = ?1",
                params![page_id],
                |row| Ok(PageMeta {
                    page_id: row.get(0)?,
                    book_id: row.get(1)?,
                    chapter_id: row.get(2)?,
                    name: row.get(3)?,
                    slug: row.get(4)?,
                    content_hash: row.get(5)?,
                    updated_at: row.get(6)?,
                }),
            ).ok())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn resolve_page_slug(&self, slug: &str) -> Result<Option<i64>, String> {
        let conn = self.conn.clone();
        let slug = slug.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            Ok(conn
                .query_row(
                    "SELECT page_id FROM pages WHERE slug = ?1",
                    params![slug],
                    |row| row.get(0),
                )
                .ok())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_page_book_ids(&self, page_ids: &[i64]) -> Result<Vec<(i64, i64)>, String> {
        if page_ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.clone();
        let ids = page_ids.to_vec();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let placeholders = std::iter::repeat_n("?", ids.len())
                .collect::<Vec<_>>()
                .join(",");
            let sql =
                format!("SELECT page_id, book_id FROM pages WHERE page_id IN ({placeholders})");
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("Prepare failed: {e}"))?;
            let params_vec: Vec<&dyn rusqlite::ToSql> =
                ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();
            let rows: Vec<(i64, i64)> = stmt
                .query_map(params_vec.as_slice(), |row| Ok((row.get(0)?, row.get(1)?)))
                .map_err(|e| format!("Query failed: {e}"))?
                .filter_map(|r| r.ok())
                .collect();
            Ok(rows)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_page_metas(&self, page_ids: &[i64]) -> Result<Vec<PageMeta>, String> {
        if page_ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.clone();
        let ids = page_ids.to_vec();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let placeholders = std::iter::repeat_n("?", ids.len())
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "SELECT page_id, book_id, chapter_id, name, slug, content_hash, updated_at
                 FROM pages WHERE page_id IN ({placeholders})"
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("Prepare failed: {e}"))?;
            let params_vec: Vec<&dyn rusqlite::ToSql> =
                ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();
            let rows: Vec<PageMeta> = stmt
                .query_map(params_vec.as_slice(), |row| {
                    Ok(PageMeta {
                        page_id: row.get(0)?,
                        book_id: row.get(1)?,
                        chapter_id: row.get(2)?,
                        name: row.get(3)?,
                        slug: row.get(4)?,
                        content_hash: row.get(5)?,
                        updated_at: row.get(6)?,
                    })
                })
                .map_err(|e| format!("Query failed: {e}"))?
                .filter_map(|r| r.ok())
                .collect();
            Ok(rows)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn insert_chunks(&self, page_id: i64, chunks: &[ChunkInsert]) -> Result<(), String> {
        let conn = self.conn.clone();
        let chunks: Vec<ChunkInsert> = chunks.to_vec();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let tx = conn.unchecked_transaction().map_err(|e| format!("Transaction failed: {e}"))?;
            tx.execute("DELETE FROM chunks WHERE page_id = ?1", params![page_id]).ok();
            for chunk in &chunks {
                let blob = vector::embedding_to_blob(&chunk.embedding);
                if let Err(e) = tx.execute(
                    "INSERT INTO chunks (page_id, chunk_index, heading_path, content, content_hash, embedding)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![page_id, chunk.chunk_index as i64, chunk.heading_path, chunk.content, chunk.content_hash, blob],
                ) {
                    eprintln!("DB: insert chunk {} for page {page_id}: {e}", chunk.chunk_index);
                }
            }
            tx.commit().map_err(|e| format!("Commit failed: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_chunk_details(&self, chunk_ids: &[i64]) -> Result<Vec<ChunkDetail>, String> {
        if chunk_ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.clone();
        let chunk_ids = chunk_ids.to_vec();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let placeholders: Vec<String> = (0..chunk_ids.len())
                .map(|i| format!("?{}", i + 1))
                .collect();
            let sql = format!(
                "SELECT c.id, c.page_id, c.heading_path, c.content, p.name
                 FROM chunks c JOIN pages p ON c.page_id = p.page_id
                 WHERE c.id IN ({})",
                placeholders.join(",")
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("Prepare failed: {e}"))?;
            let params: Vec<&dyn rusqlite::types::ToSql> = chunk_ids
                .iter()
                .map(|id| id as &dyn rusqlite::types::ToSql)
                .collect();
            let rows = stmt
                .query_map(params.as_slice(), |row| {
                    Ok(ChunkDetail {
                        chunk_id: row.get(0)?,
                        page_id: row.get(1)?,
                        heading_path: row.get(2)?,
                        content: row.get(3)?,
                        page_name: row.get(4)?,
                    })
                })
                .map_err(|e| format!("Query failed: {e}"))?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn replace_relationships(
        &self,
        source: i64,
        targets: &[(i64, String)],
    ) -> Result<(), String> {
        let conn = self.conn.clone();
        let targets: Vec<(i64, String)> = targets.to_vec();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let tx = conn
                .unchecked_transaction()
                .map_err(|e| format!("Transaction failed: {e}"))?;
            // Only delete explicit link relationships; preserve inferred "similar" ones
            tx.execute(
                "DELETE FROM relationships WHERE source_page_id = ?1 AND link_type = 'link'",
                params![source],
            )
            .ok();
            for (target_id, link_type) in &targets {
                tx.execute(
                    "INSERT OR IGNORE INTO relationships (source_page_id, target_page_id, link_type)
                     VALUES (?1, ?2, ?3)",
                    params![source, target_id, link_type],
                ).ok();
            }
            tx.commit().map_err(|e| format!("Commit failed: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_markov_blanket(&self, page_id: i64) -> Result<MarkovBlanket, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();

            let query_related = |sql: &str, page_id: i64| -> Vec<RelatedPage> {
                conn.prepare(sql)
                    .and_then(|mut stmt| {
                        stmt.query_map(params![page_id], |row| Ok(RelatedPage {
                            page_id: row.get(0)?,
                            name: row.get(1)?,
                        }))
                        .map(|rows| rows.filter_map(|r| r.ok()).collect())
                    })
                    .unwrap_or_default()
            };

            let linked_from = query_related(
                "SELECT r.source_page_id, p.name FROM relationships r
                 JOIN pages p ON r.source_page_id = p.page_id
                 WHERE r.target_page_id = ?1 LIMIT 20",
                page_id,
            );

            let links_to = query_related(
                "SELECT r.target_page_id, p.name FROM relationships r
                 JOIN pages p ON r.target_page_id = p.page_id
                 WHERE r.source_page_id = ?1 LIMIT 20",
                page_id,
            );

            let co_linked = query_related(
                "SELECT DISTINCT r2.source_page_id, p.name FROM relationships r1
                 JOIN relationships r2 ON r1.target_page_id = r2.target_page_id
                 JOIN pages p ON r2.source_page_id = p.page_id
                 WHERE r1.source_page_id = ?1 AND r2.source_page_id != ?1
                 LIMIT 10",
                page_id,
            );

            // Siblings: same chapter or same book
            let siblings = {
                let chapter_id: Option<i64> = conn
                    .query_row("SELECT chapter_id FROM pages WHERE page_id = ?1", params![page_id], |row| row.get(0))
                    .ok()
                    .flatten();

                if let Some(cid) = chapter_id {
                    let result: Vec<RelatedPage> = conn
                        .prepare("SELECT page_id, name FROM pages WHERE chapter_id = ?1 AND page_id != ?2 LIMIT 20")
                        .and_then(|mut stmt| {
                            stmt.query_map(params![cid, page_id], |row| Ok(RelatedPage {
                                page_id: row.get(0)?,
                                name: row.get(1)?,
                            }))
                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                        })
                        .unwrap_or_default();
                    if !result.is_empty() {
                        result
                    } else {
                        // Fall back to book siblings
                        let book_id: Option<i64> = conn
                            .query_row("SELECT book_id FROM pages WHERE page_id = ?1", params![page_id], |row| row.get(0))
                            .ok();
                        if let Some(bid) = book_id {
                            conn.prepare("SELECT page_id, name FROM pages WHERE book_id = ?1 AND page_id != ?2 LIMIT 20")
                                .and_then(|mut stmt| {
                                    stmt.query_map(params![bid, page_id], |row| Ok(RelatedPage {
                                        page_id: row.get(0)?,
                                        name: row.get(1)?,
                                    }))
                                    .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                })
                                .unwrap_or_default()
                        } else {
                            Vec::new()
                        }
                    }
                } else {
                    let book_id: Option<i64> = conn
                        .query_row("SELECT book_id FROM pages WHERE page_id = ?1", params![page_id], |row| row.get(0))
                        .ok();
                    if let Some(bid) = book_id {
                        conn.prepare("SELECT page_id, name FROM pages WHERE book_id = ?1 AND page_id != ?2 LIMIT 20")
                            .and_then(|mut stmt| {
                                stmt.query_map(params![bid, page_id], |row| Ok(RelatedPage {
                                    page_id: row.get(0)?,
                                    name: row.get(1)?,
                                }))
                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                            })
                            .unwrap_or_default()
                    } else {
                        Vec::new()
                    }
                }
            };

            Ok(MarkovBlanket { linked_from, links_to, co_linked, siblings })
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn create_embed_job(&self, scope: &str) -> Result<(i64, bool), String> {
        let conn = self.conn.clone();
        let scope = scope.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();

            // Dedup: pending/running collapse onto the existing job. Failed
            // jobs that haven't been touched by the reconciler also count as
            // active so a webhook firing mid-retry-window doesn't double-
            // enqueue. Closed jobs never count.
            let existing: Option<i64> = conn
                .query_row(
                    "SELECT id FROM embed_jobs \
                 WHERE scope = ?1 \
                   AND (status IN ('pending', 'running') \
                        OR (status = 'failed' AND resolved_status IS NULL)) \
                 ORDER BY id DESC LIMIT 1",
                    params![scope],
                    |row| row.get(0),
                )
                .ok();
            if let Some(id) = existing {
                return Ok((id, false));
            }

            conn.execute(
                "INSERT INTO embed_jobs (scope, status, started_at) VALUES (?1, 'pending', ?2)",
                params![scope, SqliteDb::now_secs()],
            )
            .map_err(|e| format!("Failed to create embed job: {e}"))?;
            Ok((conn.last_insert_rowid(), true))
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn expire_stale_jobs(&self, stale_secs: i64) -> Result<usize, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let cutoff = SqliteDb::now_secs() - stale_secs;
            let now = SqliteDb::now_secs();

            // Stale running jobs flip to failed-open; the reconciler decides
            // whether to retry, supersede, or give up. resolved_status stays
            // NULL until the reconciler touches it.
            let failed = conn
                .execute(
                    "UPDATE embed_jobs \
                 SET status = 'failed', finished_at = ?1, error = 'timeout' \
                 WHERE status = 'running' AND started_at < ?2",
                    params![now, cutoff],
                )
                .map_err(|e| format!("expire_stale_jobs failed: {e}"))?;

            Ok(failed)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn claim_next_job(&self, worker_id: &str) -> Result<Option<EmbedJob>, String> {
        let conn = self.conn.clone();
        let worker_id = worker_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            // SQLite single-writer means no contention — simple update + query
            let id: Option<i64> = conn.query_row(
                "SELECT id FROM embed_jobs WHERE status = 'pending' ORDER BY id LIMIT 1",
                [],
                |row| row.get(0),
            ).ok();

            let Some(id) = id else { return Ok(None); };

            conn.execute(
                "UPDATE embed_jobs SET status = 'running', started_at = ?1, worker_id = ?2 WHERE id = ?3",
                params![SqliteDb::now_secs(), worker_id, id],
            ).ok();

            let job = conn.query_row(
                EMBED_JOB_SELECT_BY_ID,
                params![id],
                embed_job_from_row,
            ).ok();

            Ok(job)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn recover_worker_jobs(&self, worker_id: &str) -> Result<usize, String> {
        let conn = self.conn.clone();
        let worker_id = worker_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let now = SqliteDb::now_secs();

            // Process restart: jobs left running by this worker (or orphans
            // pre-0.3.1) flip to failed-open. resolved_status stays NULL so
            // the reconciler picks them up.
            let failed = conn
                .execute(
                    "UPDATE embed_jobs \
                 SET status = 'failed', finished_at = ?1, error = 'worker_restart' \
                 WHERE status = 'running' AND (worker_id = ?2 OR worker_id IS NULL)",
                    params![now, worker_id],
                )
                .map_err(|e| format!("recover_worker_jobs failed: {e}"))?;

            Ok(failed)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn update_job_progress(&self, job_id: i64, done: i64, total: i64) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "UPDATE embed_jobs SET done_pages = ?1, total_pages = ?2 WHERE id = ?3",
                params![done, total, job_id],
            )
            .ok();
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn complete_job(&self, job_id: i64, error: Option<&str>) -> Result<(), String> {
        let conn = self.conn.clone();
        let error = error.map(|s| s.to_string());
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let now = SqliteDb::now_secs();
            if error.is_some() {
                // Failed-open: resolved_status stays NULL until the
                // reconciler closes us with superseded/retried/gave_up.
                // Status guard makes this idempotent — if the user cancelled
                // the job mid-flight, the cancel wins and this is a no-op.
                conn.execute(
                    "UPDATE embed_jobs SET status = 'failed', finished_at = ?1, error = ?2 \
                     WHERE id = ?3 AND status IN ('pending', 'running')",
                    params![now, error, job_id],
                )
                .ok();
            } else {
                // Status guard: a cancel that arrived after the pipeline's
                // last should_stop_embed_job poll but before this write must
                // not be silently overwritten back to 'succeeded'.
                conn.execute(
                    "UPDATE embed_jobs SET status = 'succeeded', finished_at = ?1, \
                                          resolved_status = 'succeeded', resolved_at = ?1, \
                                          error = NULL \
                     WHERE id = ?2 AND status IN ('pending', 'running')",
                    params![now, job_id],
                )
                .ok();
            }
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_latest_job(&self) -> Result<Option<EmbedJob>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            Ok(conn
                .query_row(
                    concatcp_embed_select("ORDER BY id DESC LIMIT 1").as_str(),
                    [],
                    embed_job_from_row,
                )
                .ok())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_stats(&self) -> Result<EmbedStats, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let total_pages: i64 = conn
                .query_row("SELECT COUNT(*) FROM pages", [], |row| row.get(0))
                .unwrap_or(0);
            let total_chunks: i64 = conn
                .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
                .unwrap_or(0);
            let latest_job = conn
                .query_row(
                    concatcp_embed_select("ORDER BY id DESC LIMIT 1").as_str(),
                    [],
                    embed_job_from_row,
                )
                .ok();
            Ok(EmbedStats {
                total_pages,
                total_chunks,
                latest_job,
            })
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_jobs(&self, recent: usize) -> Result<Vec<EmbedJob>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut jobs = Vec::new();

            let active_sql = concatcp_embed_select(
                "WHERE status IN ('pending', 'running', 'failed') ORDER BY id ASC",
            );
            let mut stmt = conn.prepare(&active_sql)
                .map_err(|e| format!("list_jobs prepare failed: {e}"))?;
            let active = stmt.query_map([], embed_job_from_row)
                .map_err(|e| format!("list_jobs query failed: {e}"))?;
            for job in active.flatten() {
                jobs.push(job);
            }

            let completed_sql = concatcp_embed_select(
                &format!(
                    "WHERE status NOT IN ('pending', 'running', 'failed') ORDER BY id DESC LIMIT {recent}"
                ),
            );
            let mut stmt = conn.prepare(&completed_sql)
                .map_err(|e| format!("list_jobs prepare failed: {e}"))?;
            let completed = stmt.query_map([], embed_job_from_row)
                .map_err(|e| format!("list_jobs query failed: {e}"))?;
            for job in completed.flatten() {
                jobs.push(job);
            }

            Ok(jobs)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn cancel_embed_job(&self, job_id: i64) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let now = SqliteDb::now_secs();
            conn.execute(
                "UPDATE embed_jobs \
                 SET status = 'cancelled', resolved_status = 'cancelled', \
                     resolved_at = ?1, finished_at = ?1 \
                 WHERE id = ?2 AND status IN ('pending', 'running')",
                params![now, job_id],
            )
            .map_err(|e| format!("cancel_embed_job: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn should_stop_embed_job(&self, job_id: i64) -> Result<bool, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let status: Option<String> = conn
                .query_row(
                    "SELECT status FROM embed_jobs WHERE id = ?1",
                    params![job_id],
                    |row| row.get(0),
                )
                .ok();
            Ok(matches!(status.as_deref(), Some(s) if s != "running"))
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn fail_embed_job(&self, job_id: i64, reason: &str) -> Result<(), String> {
        let conn = self.conn.clone();
        let reason = reason.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let now = SqliteDb::now_secs();
            conn.execute(
                "UPDATE embed_jobs \
                 SET status = 'failed', finished_at = ?1, error = ?2 \
                 WHERE id = ?3 AND status IN ('pending', 'running')",
                params![now, reason, job_id],
            )
            .map_err(|e| format!("fail_embed_job: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_failed_open_embed_jobs(&self) -> Result<Vec<EmbedJob>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let sql = concatcp_embed_select("WHERE status = 'failed' ORDER BY id ASC");
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("list_failed_open_embed_jobs prepare: {e}"))?;
            let rows = stmt
                .query_map([], embed_job_from_row)
                .map_err(|e| format!("list_failed_open_embed_jobs query: {e}"))?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn has_successor_embed_job(&self, scope: &str, excluded_id: i64) -> Result<bool, String> {
        let conn = self.conn.clone();
        let scope = scope.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM embed_jobs \
                 WHERE scope = ?1 AND id > ?2 \
                   AND status IN ('pending','running','succeeded','cancelled','closed')",
                    params![scope, excluded_id],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            Ok(count > 0)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn embed_job_retry_chain_len(&self, job_id: i64) -> Result<usize, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            // Walk retry_of back to the root, counting links. Bound the walk
            // to 1024 so a corrupt cycle can't pin the worker.
            let mut len = 1usize;
            let mut current = job_id;
            for _ in 0..1024 {
                let parent: Option<i64> = conn
                    .query_row(
                        "SELECT retry_of FROM embed_jobs WHERE id = ?1",
                        params![current],
                        |row| row.get(0),
                    )
                    .ok()
                    .flatten();
                match parent {
                    Some(p) => {
                        len += 1;
                        current = p;
                    }
                    None => break,
                }
            }
            Ok(len)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn close_embed_job(
        &self,
        job_id: i64,
        resolved_status: Option<&str>,
    ) -> Result<(), String> {
        let conn = self.conn.clone();
        let resolved = resolved_status.map(|s| s.to_string());
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            if let Some(rs) = resolved {
                conn.execute(
                    "UPDATE embed_jobs \
                     SET prev_status = status, status = 'closed', resolved_status = ?1 \
                     WHERE id = ?2 AND status != 'closed'",
                    params![rs, job_id],
                )
                .map_err(|e| format!("close_embed_job: {e}"))?;
            } else {
                conn.execute(
                    "UPDATE embed_jobs \
                     SET prev_status = status, status = 'closed' \
                     WHERE id = ?1 AND status != 'closed'",
                    params![job_id],
                )
                .map_err(|e| format!("close_embed_job: {e}"))?;
            }
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn create_retry_embed_job(&self, scope: &str, retry_of: i64) -> Result<i64, String> {
        let conn = self.conn.clone();
        let scope = scope.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "INSERT INTO embed_jobs (scope, status, retry_of) VALUES (?1, 'pending', ?2)",
                params![scope, retry_of],
            )
            .map_err(|e| format!("create_retry_embed_job: {e}"))?;
            Ok(conn.last_insert_rowid())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_archivable_embed_jobs(&self, older_than_secs: i64) -> Result<Vec<i64>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let cutoff = SqliteDb::now_secs() - older_than_secs;
            let mut stmt = conn
                .prepare(
                    "SELECT id FROM embed_jobs \
                 WHERE status IN ('succeeded', 'cancelled') AND resolved_at IS NOT NULL \
                   AND resolved_at <= ?1 \
                 ORDER BY id ASC",
                )
                .map_err(|e| format!("list_archivable_embed_jobs prepare: {e}"))?;
            let rows: Vec<i64> = stmt
                .query_map(params![cutoff], |row| row.get(0))
                .map_err(|e| format!("list_archivable_embed_jobs query: {e}"))?
                .filter_map(|r| r.ok())
                .collect();
            Ok(rows)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_running_embed_jobs_started_before(
        &self,
        started_before_secs: i64,
    ) -> Result<Vec<EmbedJob>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let sql = concatcp_embed_select(
                "WHERE status = 'running' AND started_at IS NOT NULL AND started_at < ?1",
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("list_running_embed_jobs_started_before prepare: {e}"))?;
            let rows = stmt
                .query_map(params![started_before_secs], embed_job_from_row)
                .map_err(|e| format!("list_running_embed_jobs_started_before query: {e}"))?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn vector_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        threshold: f32,
        book_ids: Option<&[i64]>,
    ) -> Result<Vec<SearchHit>, String> {
        let conn = self.conn.clone();
        let query_embedding = query_embedding.to_vec();
        // Materialize the optional filter into a Vec the closure can own. Empty
        // slice means "no filter", same as None.
        let book_filter: Option<Vec<i64>> = match book_ids {
            Some(ids) if !ids.is_empty() => Some(ids.to_vec()),
            _ => None,
        };

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();

            let all_chunks: Vec<(i64, i64, Vec<u8>)> = if let Some(ref ids) = book_filter {
                let placeholders = std::iter::repeat_n("?", ids.len())
                    .collect::<Vec<_>>()
                    .join(",");
                let sql = format!(
                    "SELECT c.id, c.page_id, c.embedding
                     FROM chunks c JOIN pages p ON c.page_id = p.page_id
                     WHERE p.book_id IN ({placeholders})"
                );
                let mut stmt = conn
                    .prepare(&sql)
                    .map_err(|e| format!("Prepare failed: {e}"))?;
                let params_vec: Vec<&dyn rusqlite::ToSql> =
                    ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();
                let out: Vec<(i64, i64, Vec<u8>)> = stmt
                    .query_map(params_vec.as_slice(), |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                    })
                    .map_err(|e| format!("Query failed: {e}"))?
                    .filter_map(|r| r.ok())
                    .collect();
                out
            } else {
                let mut stmt = conn
                    .prepare("SELECT id, page_id, embedding FROM chunks")
                    .map_err(|e| format!("Prepare failed: {e}"))?;
                let out: Vec<(i64, i64, Vec<u8>)> = stmt
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
                    .map_err(|e| format!("Query failed: {e}"))?
                    .filter_map(|r| r.ok())
                    .collect();
                out
            };

            let hits = vector::search_embeddings(&query_embedding, &all_chunks, limit, threshold);
            Ok(hits
                .into_iter()
                .map(|(chunk_id, page_id, score)| SearchHit {
                    chunk_id,
                    page_id,
                    score,
                })
                .collect())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn clear_all_embeddings(&self) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute("DELETE FROM relationships", [])
                .map_err(|e| format!("clear relationships: {e}"))?;
            conn.execute("DELETE FROM chunks", [])
                .map_err(|e| format!("clear chunks: {e}"))?;
            conn.execute("DELETE FROM pages", [])
                .map_err(|e| format!("clear pages: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn alter_embedding_dimension(&self, _dims: usize) -> Result<(), String> {
        // SQLite uses BLOB for embeddings — dimensionless, no schema change needed
        Ok(())
    }

    async fn compute_similar_pages(&self, top_k: usize, threshold: f32) -> Result<usize, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();

            // Clear existing similar relationships
            conn.execute("DELETE FROM relationships WHERE link_type = 'similar'", [])
                .map_err(|e| format!("clear similar rels: {e}"))?;

            // Load all chunks grouped by page to compute centroids
            let mut stmt = conn.prepare("SELECT page_id, embedding FROM chunks ORDER BY page_id")
                .map_err(|e| format!("prepare: {e}"))?;
            let rows: Vec<(i64, Vec<u8>)> = stmt.query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .map_err(|e| format!("query: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

            // Group by page_id and compute centroids
            let mut page_chunks: std::collections::HashMap<i64, Vec<Vec<f32>>> = std::collections::HashMap::new();
            for (page_id, blob) in &rows {
                let emb = vector::blob_to_embedding(blob);
                page_chunks.entry(*page_id).or_default().push(emb);
            }

            let centroids: Vec<(i64, Vec<f32>)> = page_chunks.into_iter().map(|(page_id, chunks)| {
                let dims = chunks[0].len();
                let n = chunks.len() as f32;
                let mut centroid = vec![0.0f32; dims];
                for chunk in &chunks {
                    for (i, &val) in chunk.iter().enumerate() {
                        centroid[i] += val;
                    }
                }
                for val in &mut centroid {
                    *val /= n;
                }
                (page_id, centroid)
            }).collect();

            // For each page, find top-K most similar pages
            let mut total_inserted = 0usize;
            let tx = conn.unchecked_transaction().map_err(|e| format!("tx: {e}"))?;
            for (i, (page_id, centroid)) in centroids.iter().enumerate() {
                let mut similarities: Vec<(i64, f32)> = centroids.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, (other_id, other_centroid))| {
                        let sim = vector::cosine_similarity(centroid, other_centroid);
                        (*other_id, sim)
                    })
                    .filter(|(_, sim)| *sim > threshold)
                    .collect();

                similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                similarities.truncate(top_k);

                for (target_id, _sim) in &similarities {
                    tx.execute(
                        "INSERT OR IGNORE INTO relationships (source_page_id, target_page_id, link_type)
                         VALUES (?1, ?2, 'similar')",
                        params![page_id, target_id],
                    ).ok();
                    total_inserted += 1;
                }
            }
            tx.commit().map_err(|e| format!("commit: {e}"))?;

            eprintln!("Semantic: computed {total_inserted} similar-page relationships (top_k={top_k}, threshold={threshold})");
            Ok(total_inserted)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_meta(&self, key: &str) -> Result<Option<String>, String> {
        let conn = self.conn.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut stmt = conn
                .prepare("SELECT value FROM meta WHERE key = ?1")
                .map_err(|e| format!("get_meta: {e}"))?;
            let result: Option<String> = stmt.query_row([&key], |row| row.get(0)).ok();
            Ok(result)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn set_meta(&self, key: &str, value: &str) -> Result<(), String> {
        let conn = self.conn.clone();
        let key = key.to_string();
        let value = value.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
                rusqlite::params![key, value],
            )
            .map_err(|e| format!("set_meta: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn upsert_page_acl(&self, acl: &PageAcl) -> Result<(), String> {
        let conn = self.conn.clone();
        let acl = acl.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let tx = conn
                .unchecked_transaction()
                .map_err(|e| format!("upsert_page_acl tx: {e}"))?;
            tx.execute(
                "DELETE FROM page_view_acl WHERE page_id = ?1",
                params![acl.page_id],
            )
            .map_err(|e| format!("upsert_page_acl delete: {e}"))?;
            for &role_id in &acl.view_roles {
                tx.execute(
                    "INSERT OR IGNORE INTO page_view_acl (page_id, role_id) VALUES (?1, ?2)",
                    params![acl.page_id, role_id],
                )
                .map_err(|e| format!("upsert_page_acl insert: {e}"))?;
            }
            let default_open: i64 = if acl.default_open { 1 } else { 0 };
            tx.execute(
                "UPDATE pages SET acl_default_open = ?1, acl_computed_at = ?2 WHERE page_id = ?3",
                params![default_open, acl.computed_at, acl.page_id],
            )
            .map_err(|e| format!("upsert_page_acl flag: {e}"))?;
            tx.commit()
                .map_err(|e| format!("upsert_page_acl commit: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn delete_page_acl(&self, page_id: i64) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "DELETE FROM page_view_acl WHERE page_id = ?1",
                params![page_id],
            )
            .map_err(|e| format!("delete_page_acl: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn delete_role_from_acl(&self, role_id: i64) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "DELETE FROM page_view_acl WHERE role_id = ?1",
                params![role_id],
            )
            .map_err(|e| format!("delete_role_from_acl: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_acl_page_ids(&self) -> Result<Vec<i64>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut stmt = conn
                .prepare("SELECT page_id FROM pages WHERE acl_computed_at IS NOT NULL")
                .map_err(|e| format!("Prepare failed: {e}"))?;
            let out: Vec<i64> = stmt
                .query_map([], |row| row.get::<_, i64>(0))
                .map_err(|e| format!("Query failed: {e}"))?
                .filter_map(|r| r.ok())
                .collect();
            Ok(out)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }
}

// --- IndexDb impl ---
//
// Phase 4a — structural index of BookStack content + page cache + the
// reconciliation job queue. Methods follow the same spawn_blocking pattern
// the rest of the SqliteDb impl uses; rusqlite is sync, so each call hops
// onto a blocking task and acquires the connection mutex.

#[async_trait]
impl IndexDb for SqliteDb {
    // --- Shelves ---

    async fn upsert_indexed_shelf(&self, shelf: &IndexedShelf) -> Result<(), String> {
        let conn = self.conn.clone();
        let s = shelf.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "INSERT INTO bookstack_shelves (shelf_id, name, slug, shelf_kind, indexed_at, deleted)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(shelf_id) DO UPDATE SET
                     name = excluded.name,
                     slug = excluded.slug,
                     shelf_kind = excluded.shelf_kind,
                     indexed_at = excluded.indexed_at,
                     deleted = excluded.deleted",
                params![s.shelf_id, s.name, s.slug, s.shelf_kind.as_str(), s.indexed_at, s.deleted as i64],
            ).map_err(|e| format!("upsert_indexed_shelf: {e}"))?;
            Ok(())
        }).await.map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_indexed_shelf(&self, shelf_id: i64) -> Result<Option<IndexedShelf>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT shelf_id, name, slug, shelf_kind, indexed_at, deleted FROM bookstack_shelves WHERE shelf_id = ?1"
            ).map_err(|e| format!("get_indexed_shelf prepare: {e}"))?;
            let row = stmt.query_row(params![shelf_id], |r| {
                let kind_str: String = r.get(3)?;
                Ok(IndexedShelf {
                    shelf_id: r.get(0)?,
                    name: r.get(1)?,
                    slug: r.get(2)?,
                    shelf_kind: kind_str.parse().unwrap_or(ShelfKind::Unclassified),
                    indexed_at: r.get(4)?,
                    deleted: r.get::<_, i64>(5)? != 0,
                })
            });
            match row {
                Ok(s) => Ok(Some(s)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(format!("get_indexed_shelf: {e}")),
            }
        }).await.map_err(|e| format!("Task failed: {e}"))?
    }

    async fn soft_delete_indexed_shelf(&self, shelf_id: i64) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "UPDATE bookstack_shelves SET deleted = 1 WHERE shelf_id = ?1",
                params![shelf_id],
            )
            .map_err(|e| format!("soft_delete_indexed_shelf: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    // --- Books ---

    async fn upsert_indexed_book(&self, book: &IndexedBook) -> Result<(), String> {
        let conn = self.conn.clone();
        let b = book.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "INSERT INTO bookstack_books (book_id, name, slug, shelf_id, identity_ouid, book_kind, indexed_at, deleted)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(book_id) DO UPDATE SET
                     name = excluded.name,
                     slug = excluded.slug,
                     shelf_id = excluded.shelf_id,
                     identity_ouid = excluded.identity_ouid,
                     book_kind = excluded.book_kind,
                     indexed_at = excluded.indexed_at,
                     deleted = excluded.deleted",
                params![b.book_id, b.name, b.slug, b.shelf_id, b.identity_ouid, b.book_kind.as_str(), b.indexed_at, b.deleted as i64],
            ).map_err(|e| format!("upsert_indexed_book: {e}"))?;
            Ok(())
        }).await.map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_indexed_book(&self, book_id: i64) -> Result<Option<IndexedBook>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT book_id, name, slug, shelf_id, identity_ouid, book_kind, indexed_at, deleted
                 FROM bookstack_books WHERE book_id = ?1"
            ).map_err(|e| format!("get_indexed_book prepare: {e}"))?;
            let row = stmt.query_row(params![book_id], |r| {
                let kind_str: String = r.get(5)?;
                Ok(IndexedBook {
                    book_id: r.get(0)?,
                    name: r.get(1)?,
                    slug: r.get(2)?,
                    shelf_id: r.get(3)?,
                    identity_ouid: r.get(4)?,
                    book_kind: kind_str.parse().unwrap_or(BookKind::Unclassified),
                    indexed_at: r.get(6)?,
                    deleted: r.get::<_, i64>(7)? != 0,
                })
            });
            match row {
                Ok(b) => Ok(Some(b)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(format!("get_indexed_book: {e}")),
            }
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_indexed_books_by_shelf(&self, shelf_id: i64) -> Result<Vec<IndexedBook>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT book_id, name, slug, shelf_id, identity_ouid, book_kind, indexed_at, deleted
                 FROM bookstack_books WHERE shelf_id = ?1 AND deleted = 0
                 ORDER BY name"
            ).map_err(|e| format!("list_indexed_books_by_shelf prepare: {e}"))?;
            let rows = stmt
                .query_map(params![shelf_id], |r| {
                    let kind_str: String = r.get(5)?;
                    Ok(IndexedBook {
                        book_id: r.get(0)?,
                        name: r.get(1)?,
                        slug: r.get(2)?,
                        shelf_id: r.get(3)?,
                        identity_ouid: r.get(4)?,
                        book_kind: kind_str.parse().unwrap_or(BookKind::Unclassified),
                        indexed_at: r.get(6)?,
                        deleted: r.get::<_, i64>(7)? != 0,
                    })
                })
                .map_err(|e| format!("list_indexed_books_by_shelf query: {e}"))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("list_indexed_books_by_shelf collect: {e}"))
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_indexed_books_by_identity(
        &self,
        identity_ouid: &str,
    ) -> Result<Vec<IndexedBook>, String> {
        let conn = self.conn.clone();
        let ouid = identity_ouid.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT book_id, name, slug, shelf_id, identity_ouid, book_kind, indexed_at, deleted
                 FROM bookstack_books WHERE identity_ouid = ?1 AND deleted = 0
                 ORDER BY book_kind, name"
            ).map_err(|e| format!("list_indexed_books_by_identity prepare: {e}"))?;
            let rows = stmt
                .query_map(params![ouid], |r| {
                    let kind_str: String = r.get(5)?;
                    Ok(IndexedBook {
                        book_id: r.get(0)?,
                        name: r.get(1)?,
                        slug: r.get(2)?,
                        shelf_id: r.get(3)?,
                        identity_ouid: r.get(4)?,
                        book_kind: kind_str.parse().unwrap_or(BookKind::Unclassified),
                        indexed_at: r.get(6)?,
                        deleted: r.get::<_, i64>(7)? != 0,
                    })
                })
                .map_err(|e| format!("list_indexed_books_by_identity query: {e}"))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("list_indexed_books_by_identity collect: {e}"))
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn soft_delete_indexed_book(&self, book_id: i64) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "UPDATE bookstack_books SET deleted = 1 WHERE book_id = ?1",
                params![book_id],
            )
            .map_err(|e| format!("soft_delete_indexed_book: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    // --- Chapters ---

    async fn upsert_indexed_chapter(&self, chapter: &IndexedChapter) -> Result<(), String> {
        let conn = self.conn.clone();
        let c = chapter.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "INSERT INTO bookstack_chapters
                    (chapter_id, book_id, name, slug, identity_ouid, chapter_kind, archive_year, indexed_at, deleted)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(chapter_id) DO UPDATE SET
                     book_id = excluded.book_id,
                     name = excluded.name,
                     slug = excluded.slug,
                     identity_ouid = excluded.identity_ouid,
                     chapter_kind = excluded.chapter_kind,
                     archive_year = excluded.archive_year,
                     indexed_at = excluded.indexed_at,
                     deleted = excluded.deleted",
                params![
                    c.chapter_id, c.book_id, c.name, c.slug, c.identity_ouid,
                    c.chapter_kind.as_str(), c.archive_year, c.indexed_at, c.deleted as i64
                ],
            ).map_err(|e| format!("upsert_indexed_chapter: {e}"))?;
            Ok(())
        }).await.map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_indexed_chapter(&self, chapter_id: i64) -> Result<Option<IndexedChapter>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT chapter_id, book_id, name, slug, identity_ouid, chapter_kind, archive_year, indexed_at, deleted
                 FROM bookstack_chapters WHERE chapter_id = ?1"
            ).map_err(|e| format!("get_indexed_chapter prepare: {e}"))?;
            let row = stmt.query_row(params![chapter_id], |r| {
                let kind_str: String = r.get(5)?;
                Ok(IndexedChapter {
                    chapter_id: r.get(0)?,
                    book_id: r.get(1)?,
                    name: r.get(2)?,
                    slug: r.get(3)?,
                    identity_ouid: r.get(4)?,
                    chapter_kind: kind_str.parse().unwrap_or(ChapterKind::Unclassified),
                    archive_year: r.get(6)?,
                    indexed_at: r.get(7)?,
                    deleted: r.get::<_, i64>(8)? != 0,
                })
            });
            match row {
                Ok(c) => Ok(Some(c)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(format!("get_indexed_chapter: {e}")),
            }
        }).await.map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_indexed_chapters_by_book(
        &self,
        book_id: i64,
    ) -> Result<Vec<IndexedChapter>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT chapter_id, book_id, name, slug, identity_ouid, chapter_kind, archive_year, indexed_at, deleted
                 FROM bookstack_chapters WHERE book_id = ?1 AND deleted = 0
                 ORDER BY name"
            ).map_err(|e| format!("list_indexed_chapters_by_book prepare: {e}"))?;
            let rows = stmt.query_map(params![book_id], |r| {
                let kind_str: String = r.get(5)?;
                Ok(IndexedChapter {
                    chapter_id: r.get(0)?,
                    book_id: r.get(1)?,
                    name: r.get(2)?,
                    slug: r.get(3)?,
                    identity_ouid: r.get(4)?,
                    chapter_kind: kind_str.parse().unwrap_or(ChapterKind::Unclassified),
                    archive_year: r.get(6)?,
                    indexed_at: r.get(7)?,
                    deleted: r.get::<_, i64>(8)? != 0,
                })
            }).map_err(|e| format!("list_indexed_chapters_by_book query: {e}"))?;
            rows.collect::<Result<Vec<_>, _>>().map_err(|e| format!("list_indexed_chapters_by_book collect: {e}"))
        }).await.map_err(|e| format!("Task failed: {e}"))?
    }

    async fn soft_delete_indexed_chapter(&self, chapter_id: i64) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "UPDATE bookstack_chapters SET deleted = 1 WHERE chapter_id = ?1",
                params![chapter_id],
            )
            .map_err(|e| format!("soft_delete_indexed_chapter: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    // --- Pages ---

    async fn upsert_indexed_page(
        &self,
        page: &IndexedPage,
        cache: Option<&PageCache>,
    ) -> Result<(), String> {
        let conn = self.conn.clone();
        let p = page.clone();
        let c = cache.cloned();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().unwrap();
            // Single transaction so the page row + optional cache row land
            // atomically — keeps the freshness invariant intact even if the
            // process is killed mid-write.
            let tx = conn.transaction().map_err(|e| format!("upsert_indexed_page tx: {e}"))?;
            tx.execute(
                "INSERT INTO bookstack_pages
                    (page_id, book_id, chapter_id, name, slug, url, page_created_at, page_updated_at,
                     identity_ouid, page_kind, page_key, archive_year, indexed_at, deleted)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
                 ON CONFLICT(page_id) DO UPDATE SET
                     book_id = excluded.book_id,
                     chapter_id = excluded.chapter_id,
                     name = excluded.name,
                     slug = excluded.slug,
                     url = excluded.url,
                     page_created_at = excluded.page_created_at,
                     page_updated_at = excluded.page_updated_at,
                     identity_ouid = excluded.identity_ouid,
                     page_kind = excluded.page_kind,
                     page_key = excluded.page_key,
                     archive_year = excluded.archive_year,
                     indexed_at = excluded.indexed_at,
                     deleted = excluded.deleted",
                params![
                    p.page_id, p.book_id, p.chapter_id, p.name, p.slug, p.url,
                    p.page_created_at, p.page_updated_at, p.identity_ouid,
                    p.page_kind.as_str(), p.page_key, p.archive_year,
                    p.indexed_at, p.deleted as i64
                ],
            ).map_err(|e| format!("upsert_indexed_page page: {e}"))?;

            if let Some(cache) = c {
                tx.execute(
                    "INSERT INTO page_cache (page_id, markdown, raw_markdown, html, cached_at, page_updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                     ON CONFLICT(page_id) DO UPDATE SET
                         markdown = excluded.markdown,
                         raw_markdown = excluded.raw_markdown,
                         html = excluded.html,
                         cached_at = excluded.cached_at,
                         page_updated_at = excluded.page_updated_at",
                    params![
                        cache.page_id, cache.markdown, cache.raw_markdown,
                        cache.html, cache.cached_at, cache.page_updated_at
                    ],
                ).map_err(|e| format!("upsert_indexed_page cache: {e}"))?;
            }
            tx.commit().map_err(|e| format!("upsert_indexed_page commit: {e}"))?;
            Ok(())
        }).await.map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_indexed_page(&self, page_id: i64) -> Result<Option<IndexedPage>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            indexed_page_by_predicate(&conn, "page_id = ?1", params![page_id])
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn find_indexed_page_by_key(
        &self,
        identity_ouid: &str,
        page_kind: PageKind,
        page_key: &str,
    ) -> Result<Option<IndexedPage>, String> {
        let conn = self.conn.clone();
        let ouid = identity_ouid.to_string();
        let kind = page_kind.as_str().to_string();
        let key = page_key.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            indexed_page_by_predicate(
                &conn,
                "identity_ouid = ?1 AND page_kind = ?2 AND page_key = ?3 AND deleted = 0",
                params![ouid, kind, key],
            )
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_indexed_pages_by_chapter(
        &self,
        chapter_id: i64,
    ) -> Result<Vec<IndexedPage>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            indexed_pages_by_predicate(
                &conn,
                "chapter_id = ?1 AND deleted = 0 ORDER BY name",
                params![chapter_id],
            )
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_indexed_pages_by_book_root(
        &self,
        book_id: i64,
    ) -> Result<Vec<IndexedPage>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            indexed_pages_by_predicate(
                &conn,
                "book_id = ?1 AND chapter_id IS NULL AND deleted = 0 ORDER BY name",
                params![book_id],
            )
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_indexed_pages_recent(
        &self,
        book_id: i64,
        limit: i64,
    ) -> Result<Vec<IndexedPage>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            // page_updated_at is TEXT (ISO 8601). String-sort gives us
            // chronological order because ISO 8601 is lexicographically
            // monotonic. NULL updated_at sinks to the end via the COALESCE.
            indexed_pages_by_predicate(
                &conn,
                "book_id = ?1 AND deleted = 0 \
                 ORDER BY COALESCE(page_updated_at, '') DESC \
                 LIMIT ?2",
                params![book_id, limit],
            )
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn soft_delete_indexed_page(&self, page_id: i64) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "UPDATE bookstack_pages SET deleted = 1 WHERE page_id = ?1",
                params![page_id],
            )
            .map_err(|e| format!("soft_delete_indexed_page: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    // --- Page cache ---

    async fn get_page_cache(&self, page_id: i64) -> Result<Option<PageCache>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT page_id, markdown, raw_markdown, html, cached_at, page_updated_at
                 FROM page_cache WHERE page_id = ?1",
                )
                .map_err(|e| format!("get_page_cache prepare: {e}"))?;
            let row = stmt.query_row(params![page_id], |r| {
                Ok(PageCache {
                    page_id: r.get(0)?,
                    markdown: r.get(1)?,
                    raw_markdown: r.get(2)?,
                    html: r.get(3)?,
                    cached_at: r.get(4)?,
                    page_updated_at: r.get(5)?,
                })
            });
            match row {
                Ok(c) => Ok(Some(c)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(format!("get_page_cache: {e}")),
            }
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    // --- Index jobs ---

    async fn create_index_job(
        &self,
        scope: &str,
        kind: &str,
        triggered_by: &str,
    ) -> Result<(i64, bool), String> {
        let conn = self.conn.clone();
        let scope = scope.to_string();
        let kind = kind.to_string();
        let triggered_by = triggered_by.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            // Dedup: pending/running collapse onto the existing job. Failed
            // jobs not yet touched by the reconciler also count as active so
            // a webhook firing mid-retry-window doesn't double-enqueue.
            let existing: Result<i64, _> = conn.query_row(
                "SELECT id FROM index_jobs
                 WHERE scope = ?1
                   AND (status IN ('pending', 'running')
                        OR (status = 'failed' AND resolved_status IS NULL))
                 ORDER BY id DESC LIMIT 1",
                params![scope],
                |r| r.get(0),
            );
            if let Ok(id) = existing {
                return Ok((id, false));
            }
            conn.execute(
                "INSERT INTO index_jobs (scope, kind, status, triggered_by) VALUES (?1, ?2, 'pending', ?3)",
                params![scope, kind, triggered_by],
            ).map_err(|e| format!("create_index_job insert: {e}"))?;
            let id = conn.last_insert_rowid();
            Ok((id, true))
        }).await.map_err(|e| format!("Task failed: {e}"))?
    }

    async fn claim_next_index_job(&self) -> Result<Option<IndexJob>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().unwrap();
            let tx = conn
                .transaction()
                .map_err(|e| format!("claim_next_index_job tx: {e}"))?;
            let claim_sql =
                concatcp_index_select("WHERE status = 'pending' ORDER BY id ASC LIMIT 1");
            let job: Option<IndexJob> = {
                let mut stmt = tx
                    .prepare(&claim_sql)
                    .map_err(|e| format!("claim_next_index_job prepare: {e}"))?;
                let row = stmt.query_row([], index_job_from_row);
                match row {
                    Ok(j) => Some(j),
                    Err(rusqlite::Error::QueryReturnedNoRows) => None,
                    Err(e) => return Err(format!("claim_next_index_job query: {e}")),
                }
            };
            if let Some(ref j) = job {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                tx.execute(
                    "UPDATE index_jobs SET status = 'running', started_at = ?1 WHERE id = ?2",
                    params![now, j.id],
                )
                .map_err(|e| format!("claim_next_index_job update: {e}"))?;
            }
            tx.commit()
                .map_err(|e| format!("claim_next_index_job commit: {e}"))?;
            Ok(job)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn update_index_job_progress(
        &self,
        job_id: i64,
        progress: i64,
        total: i64,
    ) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "UPDATE index_jobs SET progress = ?1, total = ?2 WHERE id = ?3",
                params![progress, total, job_id],
            )
            .map_err(|e| format!("update_index_job_progress: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn complete_index_job(&self, job_id: i64, error: Option<&str>) -> Result<(), String> {
        let conn = self.conn.clone();
        let error = error.map(|s| s.to_string());
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            if error.is_some() {
                // Status guard makes this idempotent — a cancel that landed
                // between the worker's last should_stop_index_job poll and
                // this write must not be silently overwritten.
                conn.execute(
                    "UPDATE index_jobs \
                     SET status = 'failed', finished_at = ?1, error = ?2 \
                     WHERE id = ?3 AND status IN ('pending', 'running')",
                    params![now, error, job_id],
                )
                .map_err(|e| format!("complete_index_job: {e}"))?;
            } else {
                // Same guard on the success branch — a cancel-then-success
                // race must leave the row in 'cancelled', not flip it back.
                conn.execute(
                    "UPDATE index_jobs \
                     SET status = 'succeeded', finished_at = ?1, \
                         resolved_status = 'succeeded', resolved_at = ?1, error = NULL \
                     WHERE id = ?2 AND status IN ('pending', 'running')",
                    params![now, job_id],
                )
                .map_err(|e| format!("complete_index_job: {e}"))?;
            }
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_pending_index_jobs(&self, limit: i64) -> Result<Vec<IndexJob>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let sql = concatcp_index_select("WHERE status = 'pending' ORDER BY id ASC LIMIT ?1");
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("list_pending_index_jobs prepare: {e}"))?;
            let rows = stmt
                .query_map(params![limit], index_job_from_row)
                .map_err(|e| format!("list_pending_index_jobs query: {e}"))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("list_pending_index_jobs collect: {e}"))
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn get_latest_index_job(&self) -> Result<Option<IndexJob>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let sql = concatcp_index_select("ORDER BY id DESC LIMIT 1");
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("get_latest_index_job prepare: {e}"))?;
            let row = stmt.query_row([], index_job_from_row);
            match row {
                Ok(j) => Ok(Some(j)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(format!("get_latest_index_job: {e}")),
            }
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn cancel_index_job(&self, job_id: i64) -> Result<(), String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            conn.execute(
                "UPDATE index_jobs \
                 SET status = 'cancelled', resolved_status = 'cancelled', \
                     resolved_at = ?1, finished_at = ?1 \
                 WHERE id = ?2 AND status IN ('pending', 'running')",
                params![now, job_id],
            )
            .map_err(|e| format!("cancel_index_job: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn should_stop_index_job(&self, job_id: i64) -> Result<bool, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let status: Option<String> = conn
                .query_row(
                    "SELECT status FROM index_jobs WHERE id = ?1",
                    params![job_id],
                    |row| row.get(0),
                )
                .ok();
            Ok(matches!(status.as_deref(), Some(s) if s != "running"))
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn fail_index_job(&self, job_id: i64, reason: &str) -> Result<(), String> {
        let conn = self.conn.clone();
        let reason = reason.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            conn.execute(
                "UPDATE index_jobs \
                 SET status = 'failed', finished_at = ?1, error = ?2 \
                 WHERE id = ?3 AND status IN ('pending', 'running')",
                params![now, reason, job_id],
            )
            .map_err(|e| format!("fail_index_job: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_failed_open_index_jobs(&self) -> Result<Vec<IndexJob>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let sql = concatcp_index_select("WHERE status = 'failed' ORDER BY id ASC");
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("list_failed_open_index_jobs prepare: {e}"))?;
            let rows = stmt
                .query_map([], index_job_from_row)
                .map_err(|e| format!("list_failed_open_index_jobs query: {e}"))?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn has_successor_index_job(&self, scope: &str, excluded_id: i64) -> Result<bool, String> {
        let conn = self.conn.clone();
        let scope = scope.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM index_jobs \
                 WHERE scope = ?1 AND id > ?2 \
                   AND status IN ('pending','running','succeeded','cancelled','closed')",
                    params![scope, excluded_id],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            Ok(count > 0)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn index_job_retry_chain_len(&self, job_id: i64) -> Result<usize, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut len = 1usize;
            let mut current = job_id;
            for _ in 0..1024 {
                let parent: Option<i64> = conn
                    .query_row(
                        "SELECT retry_of FROM index_jobs WHERE id = ?1",
                        params![current],
                        |row| row.get(0),
                    )
                    .ok()
                    .flatten();
                match parent {
                    Some(p) => {
                        len += 1;
                        current = p;
                    }
                    None => break,
                }
            }
            Ok(len)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn close_index_job(
        &self,
        job_id: i64,
        resolved_status: Option<&str>,
    ) -> Result<(), String> {
        let conn = self.conn.clone();
        let resolved = resolved_status.map(|s| s.to_string());
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            if let Some(rs) = resolved {
                conn.execute(
                    "UPDATE index_jobs \
                     SET prev_status = status, status = 'closed', resolved_status = ?1 \
                     WHERE id = ?2 AND status != 'closed'",
                    params![rs, job_id],
                )
                .map_err(|e| format!("close_index_job: {e}"))?;
            } else {
                conn.execute(
                    "UPDATE index_jobs \
                     SET prev_status = status, status = 'closed' \
                     WHERE id = ?1 AND status != 'closed'",
                    params![job_id],
                )
                .map_err(|e| format!("close_index_job: {e}"))?;
            }
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn create_retry_index_job(
        &self,
        scope: &str,
        kind: &str,
        retry_of: i64,
    ) -> Result<i64, String> {
        let conn = self.conn.clone();
        let scope = scope.to_string();
        let kind = kind.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "INSERT INTO index_jobs (scope, kind, status, triggered_by, retry_of) \
                 VALUES (?1, ?2, 'pending', 'reconciler', ?3)",
                params![scope, kind, retry_of],
            )
            .map_err(|e| format!("create_retry_index_job: {e}"))?;
            Ok(conn.last_insert_rowid())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_archivable_index_jobs(&self, older_than_secs: i64) -> Result<Vec<i64>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            let cutoff = now - older_than_secs;
            let mut stmt = conn
                .prepare(
                    "SELECT id FROM index_jobs \
                 WHERE status IN ('succeeded', 'cancelled') AND resolved_at IS NOT NULL \
                   AND resolved_at <= ?1 \
                 ORDER BY id ASC",
                )
                .map_err(|e| format!("list_archivable_index_jobs prepare: {e}"))?;
            let rows: Vec<i64> = stmt
                .query_map(params![cutoff], |row| row.get(0))
                .map_err(|e| format!("list_archivable_index_jobs query: {e}"))?
                .filter_map(|r| r.ok())
                .collect();
            Ok(rows)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_running_index_jobs_started_before(
        &self,
        started_before_secs: i64,
    ) -> Result<Vec<IndexJob>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let sql = concatcp_index_select(
                "WHERE status = 'running' AND started_at IS NOT NULL AND started_at < ?1",
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("list_running_index_jobs prepare: {e}"))?;
            let rows = stmt
                .query_map(params![started_before_secs], index_job_from_row)
                .map_err(|e| format!("list_running_index_jobs query: {e}"))?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn list_index_jobs(&self, recent: usize) -> Result<Vec<IndexJob>, String> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut jobs = Vec::new();
            let active_sql = concatcp_index_select(
                "WHERE status IN ('pending', 'running', 'failed') ORDER BY id ASC",
            );
            let mut stmt = conn
                .prepare(&active_sql)
                .map_err(|e| format!("list_index_jobs prepare: {e}"))?;
            let active = stmt
                .query_map([], index_job_from_row)
                .map_err(|e| format!("list_index_jobs query: {e}"))?;
            for j in active.flatten() {
                jobs.push(j);
            }
            let completed_sql = concatcp_index_select(&format!(
                "WHERE status NOT IN ('pending', 'running', 'failed') \
                     ORDER BY id DESC LIMIT {recent}"
            ));
            let mut stmt = conn
                .prepare(&completed_sql)
                .map_err(|e| format!("list_index_jobs prepare: {e}"))?;
            let completed = stmt
                .query_map([], index_job_from_row)
                .map_err(|e| format!("list_index_jobs query: {e}"))?;
            for j in completed.flatten() {
                jobs.push(j);
            }
            Ok(jobs)
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    // --- Index meta ---

    async fn get_index_meta(&self, key: &str) -> Result<Option<String>, String> {
        let conn = self.conn.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let row: Result<String, _> = conn.query_row(
                "SELECT value FROM index_meta WHERE key = ?1",
                params![key],
                |r| r.get(0),
            );
            match row {
                Ok(v) => Ok(Some(v)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(format!("get_index_meta: {e}")),
            }
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    async fn set_index_meta(&self, key: &str, value: &str) -> Result<(), String> {
        let conn = self.conn.clone();
        let key = key.to_string();
        let value = value.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![key, value],
            )
            .map_err(|e| format!("set_index_meta: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }
}

// --- Helpers shared across IndexDb impl methods ---

fn indexed_page_from_row(r: &rusqlite::Row) -> rusqlite::Result<IndexedPage> {
    let kind_str: String = r.get(9)?;
    Ok(IndexedPage {
        page_id: r.get(0)?,
        book_id: r.get(1)?,
        chapter_id: r.get(2)?,
        name: r.get(3)?,
        slug: r.get(4)?,
        url: r.get(5)?,
        page_created_at: r.get(6)?,
        page_updated_at: r.get(7)?,
        identity_ouid: r.get(8)?,
        page_kind: kind_str.parse().unwrap_or(PageKind::Unclassified),
        page_key: r.get(10)?,
        archive_year: r.get(11)?,
        indexed_at: r.get(12)?,
        deleted: r.get::<_, i64>(13)? != 0,
    })
}

fn indexed_page_by_predicate(
    conn: &rusqlite::Connection,
    where_clause: &str,
    params: &[&dyn rusqlite::ToSql],
) -> Result<Option<IndexedPage>, String> {
    let sql = format!(
        "SELECT page_id, book_id, chapter_id, name, slug, url, page_created_at, page_updated_at,
                identity_ouid, page_kind, page_key, archive_year, indexed_at, deleted
         FROM bookstack_pages WHERE {where_clause} LIMIT 1"
    );
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| format!("indexed_page_by_predicate prepare: {e}"))?;
    let row = stmt.query_row(params, indexed_page_from_row);
    match row {
        Ok(p) => Ok(Some(p)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(format!("indexed_page_by_predicate: {e}")),
    }
}

fn indexed_pages_by_predicate(
    conn: &rusqlite::Connection,
    where_clause: &str,
    params: &[&dyn rusqlite::ToSql],
) -> Result<Vec<IndexedPage>, String> {
    let sql = format!(
        "SELECT page_id, book_id, chapter_id, name, slug, url, page_created_at, page_updated_at,
                identity_ouid, page_kind, page_key, archive_year, indexed_at, deleted
         FROM bookstack_pages WHERE {where_clause}"
    );
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| format!("indexed_pages_by_predicate prepare: {e}"))?;
    let rows = stmt
        .query_map(params, indexed_page_from_row)
        .map_err(|e| format!("indexed_pages_by_predicate query: {e}"))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("indexed_pages_by_predicate collect: {e}"))
}

fn index_job_from_row(r: &rusqlite::Row) -> rusqlite::Result<IndexJob> {
    Ok(IndexJob {
        id: r.get(0)?,
        scope: r.get(1)?,
        kind: r.get(2)?,
        status: r.get(3)?,
        triggered_by: r.get(4)?,
        started_at: r.get(5)?,
        finished_at: r.get(6)?,
        progress: r.get(7)?,
        total: r.get(8)?,
        error: r.get(9)?,
        resolved_status: r.get(10)?,
        prev_status: r.get(11)?,
        resolved_at: r.get(12)?,
        retry_of: r.get(13)?,
    })
}

const INDEX_JOB_COLS: &str =
    "id, scope, kind, status, triggered_by, started_at, finished_at, progress, total, error, \
     resolved_status, prev_status, resolved_at, retry_of";

fn concatcp_index_select(tail: &str) -> String {
    format!("SELECT {INDEX_JOB_COLS} FROM index_jobs {tail}")
}

fn embed_job_from_row(r: &rusqlite::Row) -> rusqlite::Result<EmbedJob> {
    Ok(EmbedJob {
        id: r.get(0)?,
        scope: r.get(1)?,
        status: r.get(2)?,
        total_pages: r.get(3)?,
        done_pages: r.get(4)?,
        started_at: r.get(5)?,
        finished_at: r.get(6)?,
        error: r.get(7)?,
        worker_id: r.get(8)?,
        resolved_status: r.get(9)?,
        prev_status: r.get(10)?,
        resolved_at: r.get(11)?,
        retry_of: r.get(12)?,
    })
}

const EMBED_JOB_COLS: &str =
    "id, scope, status, total_pages, done_pages, started_at, finished_at, error, worker_id, \
     resolved_status, prev_status, resolved_at, retry_of";

const EMBED_JOB_SELECT_BY_ID: &str =
    "SELECT id, scope, status, total_pages, done_pages, started_at, \
     finished_at, error, worker_id, resolved_status, prev_status, resolved_at, retry_of \
     FROM embed_jobs WHERE id = ?1";

fn concatcp_embed_select(tail: &str) -> String {
    format!("SELECT {EMBED_JOB_COLS} FROM embed_jobs {tail}")
}

/// Convert unix days (since epoch) to (year, month, day).
fn unix_days_to_ymd(days: i64) -> (i64, u32, u32) {
    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use bsmcp_common::db::{IndexDb, SemanticDb};
    use std::env;

    fn temp_db() -> SqliteDb {
        let dir = env::temp_dir();
        let unique = format!(
            "bsmcp-test-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = dir.join(unique);
        SqliteDb::open(&path, "test-encryption-key-thirty-two-chars-long")
    }

    /// v0.10.0 migration: opening an existing v0.9.0 database drops the
    /// briefing/per-user-settings surface (user_settings, user_role_cache,
    /// briefing-only global_settings columns) without touching the surviving
    /// tables.
    #[test]
    fn v0_10_0_migration_drops_briefing_surface() {
        let dir = env::temp_dir();
        let unique = format!(
            "bsmcp-test-mig-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = dir.join(&unique);

        // Pre-create the v0.9.0 shape (user_settings + user_role_cache +
        // briefing-era global_settings columns) so the v0.10.0 open path
        // exercises every drop branch.
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE user_settings (
                 token_id_hash TEXT PRIMARY KEY,
                 settings_json TEXT NOT NULL,
                 updated_at INTEGER NOT NULL
             );
             CREATE TABLE user_role_cache (
                 token_id_hash TEXT PRIMARY KEY,
                 bookstack_user_id INTEGER NOT NULL,
                 role_ids TEXT NOT NULL,
                 fetched_at INTEGER NOT NULL
             );
             CREATE TABLE global_settings (
                 id INTEGER PRIMARY KEY CHECK (id = 1),
                 hive_shelf_id INTEGER,
                 user_journals_shelf_id INTEGER,
                 org_required_instructions_page_ids TEXT,
                 org_ai_usage_policy_page_ids TEXT,
                 org_identity_page_id INTEGER,
                 org_domains TEXT,
                 set_by_token_hash TEXT,
                 updated_at INTEGER NOT NULL DEFAULT 0,
                 guide_page_id INTEGER,
                 policies_scope TEXT,
                 sops_scope TEXT,
                 best_practices_scope TEXT,
                 friendly_structure INTEGER NOT NULL DEFAULT 1,
                 full_content_in_briefing INTEGER NOT NULL DEFAULT 0,
                 strict_setup INTEGER NOT NULL DEFAULT 0
             );
             INSERT INTO user_settings VALUES ('hash-1', '{}', 0);
             INSERT INTO user_role_cache VALUES ('hash-1', 42, '[1,2]', 0);
             INSERT INTO global_settings (id, updated_at) VALUES (1, 0);",
        )
        .unwrap();
        drop(conn);

        // Reopen — v0.10.0 init runs the cleanup migrations.
        let _ = SqliteDb::open(&path, "test-encryption-key-thirty-two-chars-long");

        let conn = rusqlite::Connection::open(&path).unwrap();

        let user_settings_exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='user_settings'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(user_settings_exists, 0, "user_settings should be dropped");

        let user_role_cache_exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='user_role_cache'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            user_role_cache_exists, 0,
            "user_role_cache should be dropped"
        );

        // The briefing-only columns should be gone; the index-worker columns
        // should remain.
        for col in [
            "org_required_instructions_page_ids",
            "org_ai_usage_policy_page_ids",
            "org_identity_page_id",
            "org_domains",
            "guide_page_id",
            "policies_scope",
            "sops_scope",
            "best_practices_scope",
            "friendly_structure",
            "full_content_in_briefing",
            "strict_setup",
        ] {
            let exists: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM pragma_table_info('global_settings') WHERE name = ?1",
                    [col],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(exists, 0, "global_settings.{col} should be dropped");
        }
        for col in ["hive_shelf_id", "user_journals_shelf_id"] {
            let exists: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM pragma_table_info('global_settings') WHERE name = ?1",
                    [col],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(exists, 1, "global_settings.{col} should be retained");
        }

        drop(conn);
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn embed_retry_chain_walks_to_root() {
        let db = temp_db();
        db.init_semantic_tables().await.unwrap();

        // Original job → fail → retry once → fail → retry again. Chain is 3.
        let (root, _) = SemanticDb::create_embed_job(&db, "page:42").await.unwrap();
        SemanticDb::fail_embed_job(&db, root, "boom").await.unwrap();
        let r1 = SemanticDb::create_retry_embed_job(&db, "page:42", root)
            .await
            .unwrap();
        SemanticDb::fail_embed_job(&db, r1, "boom2").await.unwrap();
        let r2 = SemanticDb::create_retry_embed_job(&db, "page:42", r1)
            .await
            .unwrap();

        assert_eq!(
            SemanticDb::embed_job_retry_chain_len(&db, root)
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            SemanticDb::embed_job_retry_chain_len(&db, r1)
                .await
                .unwrap(),
            2
        );
        assert_eq!(
            SemanticDb::embed_job_retry_chain_len(&db, r2)
                .await
                .unwrap(),
            3
        );
    }

    #[tokio::test]
    async fn embed_failed_with_successor_is_supersedable() {
        let db = temp_db();
        db.init_semantic_tables().await.unwrap();

        let (j1, _) = SemanticDb::create_embed_job(&db, "page:7").await.unwrap();
        SemanticDb::fail_embed_job(&db, j1, "boom").await.unwrap();
        // Once j1 is failed-open, a fresh create with the same scope dedups
        // back onto j1. To create a successor we need to first close j1
        // (mirrors what the reconciler does on supersedence).
        SemanticDb::close_embed_job(&db, j1, Some("retried"))
            .await
            .unwrap();
        let (j2, is_new) = SemanticDb::create_embed_job(&db, "page:7").await.unwrap();
        assert!(is_new);
        assert!(j2 > j1);

        assert!(SemanticDb::has_successor_embed_job(&db, "page:7", j1)
            .await
            .unwrap());
        // j2 has no successor.
        assert!(!SemanticDb::has_successor_embed_job(&db, "page:7", j2)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn embed_dedup_widens_to_failed_open() {
        let db = temp_db();
        db.init_semantic_tables().await.unwrap();

        let (j1, is_new1) = SemanticDb::create_embed_job(&db, "page:9").await.unwrap();
        assert!(is_new1);
        SemanticDb::fail_embed_job(&db, j1, "boom").await.unwrap();
        // Failed-open should still dedup onto j1 — webhook firing
        // mid-retry-window mustn't double-enqueue.
        let (j_dup, is_new2) = SemanticDb::create_embed_job(&db, "page:9").await.unwrap();
        assert!(!is_new2);
        assert_eq!(j_dup, j1);

        // Once the reconciler closes the failed job, a fresh create succeeds.
        SemanticDb::close_embed_job(&db, j1, Some("retried"))
            .await
            .unwrap();
        let (j_new, is_new3) = SemanticDb::create_embed_job(&db, "page:9").await.unwrap();
        assert!(is_new3);
        assert!(j_new > j1);
    }

    #[tokio::test]
    async fn embed_close_preserves_or_overwrites_resolved_status() {
        let db = temp_db();
        db.init_semantic_tables().await.unwrap();

        let (j, _) = SemanticDb::create_embed_job(&db, "page:1").await.unwrap();
        SemanticDb::complete_job(&db, j, None).await.unwrap();
        // Archiver path: pass None → preserve the existing 'succeeded'.
        SemanticDb::close_embed_job(&db, j, None).await.unwrap();
        let job = {
            let jobs = SemanticDb::list_jobs(&db, 5).await.unwrap();
            jobs.into_iter().find(|j2| j2.id == j).unwrap()
        };
        assert_eq!(job.status, "closed");
        assert_eq!(job.resolved_status.as_deref(), Some("succeeded"));
        assert_eq!(job.prev_status.as_deref(), Some("succeeded"));
    }

    #[tokio::test]
    async fn embed_archiver_finds_old_succeeded_jobs() {
        let db = temp_db();
        db.init_semantic_tables().await.unwrap();

        let (j, _) = SemanticDb::create_embed_job(&db, "page:5").await.unwrap();
        SemanticDb::complete_job(&db, j, None).await.unwrap();
        // Old enough? With grace=0, anything resolved at-or-before now is archivable.
        let archivable = SemanticDb::list_archivable_embed_jobs(&db, 0)
            .await
            .unwrap();
        assert!(archivable.contains(&j), "expected {j} in {:?}", archivable);
    }

    #[tokio::test]
    async fn index_retry_chain_walks_to_root() {
        let db = temp_db();

        let (root, _) = IndexDb::create_index_job(&db, "page:42", "both", "test")
            .await
            .unwrap();
        IndexDb::fail_index_job(&db, root, "boom").await.unwrap();
        let r1 = IndexDb::create_retry_index_job(&db, "page:42", "both", root)
            .await
            .unwrap();
        IndexDb::fail_index_job(&db, r1, "boom2").await.unwrap();
        let r2 = IndexDb::create_retry_index_job(&db, "page:42", "both", r1)
            .await
            .unwrap();

        assert_eq!(
            IndexDb::index_job_retry_chain_len(&db, root).await.unwrap(),
            1
        );
        assert_eq!(
            IndexDb::index_job_retry_chain_len(&db, r1).await.unwrap(),
            2
        );
        assert_eq!(
            IndexDb::index_job_retry_chain_len(&db, r2).await.unwrap(),
            3
        );
    }

    #[tokio::test]
    async fn index_dedup_widens_to_failed_open() {
        let db = temp_db();

        let (j1, is_new1) = IndexDb::create_index_job(&db, "page:9", "both", "test")
            .await
            .unwrap();
        assert!(is_new1);
        IndexDb::fail_index_job(&db, j1, "boom").await.unwrap();
        let (j_dup, is_new2) = IndexDb::create_index_job(&db, "page:9", "both", "test")
            .await
            .unwrap();
        assert!(!is_new2);
        assert_eq!(j_dup, j1);
    }

    #[tokio::test]
    async fn index_should_stop_returns_true_for_cancelled() {
        let db = temp_db();

        let (j, _) = IndexDb::create_index_job(&db, "all", "both", "test")
            .await
            .unwrap();
        // Pending = not running, so should_stop returns true. That's fine —
        // it's the worker's check at yield points; a cancelled-while-pending
        // job is still "stop".
        assert!(IndexDb::should_stop_index_job(&db, j).await.unwrap());

        // Claim it → running → should_stop returns false.
        let claimed = IndexDb::claim_next_index_job(&db).await.unwrap().unwrap();
        assert_eq!(claimed.id, j);
        assert!(!IndexDb::should_stop_index_job(&db, j).await.unwrap());

        // Cancel → should_stop returns true.
        IndexDb::cancel_index_job(&db, j).await.unwrap();
        assert!(IndexDb::should_stop_index_job(&db, j).await.unwrap());
    }

    // Cancel race: pipeline finishes its current page after a cancel landed
    // but before the next should_stop poll. complete_job must not flip
    // 'cancelled' back to 'succeeded' (or 'failed' on the error branch).
    #[tokio::test]
    async fn complete_job_does_not_overwrite_cancelled() {
        let db = temp_db();
        db.init_semantic_tables().await.unwrap();

        let (job_id, _) = SemanticDb::create_embed_job(&db, "page:1").await.unwrap();
        // Claim → running.
        let claimed = SemanticDb::claim_next_job(&db, "worker-x")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claimed.id, job_id);
        // User cancels mid-flight.
        SemanticDb::cancel_embed_job(&db, job_id).await.unwrap();
        // Pipeline finishes its in-flight page and tries to mark complete.
        SemanticDb::complete_job(&db, job_id, None).await.unwrap();

        let job = SemanticDb::list_jobs(&db, 5)
            .await
            .unwrap()
            .into_iter()
            .find(|j| j.id == job_id)
            .expect("job should still exist");
        assert_eq!(job.status, "cancelled");
        assert_eq!(job.resolved_status.as_deref(), Some("cancelled"));
    }

    #[tokio::test]
    async fn complete_job_failure_does_not_overwrite_cancelled() {
        let db = temp_db();
        db.init_semantic_tables().await.unwrap();

        let (job_id, _) = SemanticDb::create_embed_job(&db, "page:2").await.unwrap();
        SemanticDb::claim_next_job(&db, "worker-x")
            .await
            .unwrap()
            .unwrap();
        SemanticDb::cancel_embed_job(&db, job_id).await.unwrap();
        // Pipeline's in-flight page errored — but the job has already been
        // cancelled, so the failure must not flip it back.
        SemanticDb::complete_job(&db, job_id, Some("boom"))
            .await
            .unwrap();

        let job = SemanticDb::list_jobs(&db, 5)
            .await
            .unwrap()
            .into_iter()
            .find(|j| j.id == job_id)
            .expect("job should still exist");
        assert_eq!(job.status, "cancelled");
        assert_eq!(job.resolved_status.as_deref(), Some("cancelled"));
    }

    #[tokio::test]
    async fn complete_index_job_does_not_overwrite_cancelled() {
        let db = temp_db();

        let (job_id, _) = IndexDb::create_index_job(&db, "page:3", "both", "test")
            .await
            .unwrap();
        let claimed = IndexDb::claim_next_index_job(&db).await.unwrap().unwrap();
        assert_eq!(claimed.id, job_id);
        IndexDb::cancel_index_job(&db, job_id).await.unwrap();
        IndexDb::complete_index_job(&db, job_id, None)
            .await
            .unwrap();

        let job = IndexDb::list_index_jobs(&db, 5)
            .await
            .unwrap()
            .into_iter()
            .find(|j| j.id == job_id)
            .expect("job should still exist");
        assert_eq!(job.status, "cancelled");
        assert_eq!(job.resolved_status.as_deref(), Some("cancelled"));
    }

    #[tokio::test]
    async fn complete_index_job_failure_does_not_overwrite_cancelled() {
        let db = temp_db();

        let (job_id, _) = IndexDb::create_index_job(&db, "page:4", "both", "test")
            .await
            .unwrap();
        IndexDb::claim_next_index_job(&db).await.unwrap().unwrap();
        IndexDb::cancel_index_job(&db, job_id).await.unwrap();
        IndexDb::complete_index_job(&db, job_id, Some("boom"))
            .await
            .unwrap();

        let job = IndexDb::list_index_jobs(&db, 5)
            .await
            .unwrap()
            .into_iter()
            .find(|j| j.id == job_id)
            .expect("job should still exist");
        assert_eq!(job.status, "cancelled");
        assert_eq!(job.resolved_status.as_deref(), Some("cancelled"));
    }
}
