//! SQLite → PostgreSQL migration tool.
//! Copies all data tables: access_tokens, pages, chunks, relationships, embed_jobs.
//! Encrypted token blobs are copied as-is (portable when BSMCP_ENCRYPTION_KEY matches).
//! Chunk embeddings are converted from SQLite BLOB (LE f32) to pgvector vector(1024).

use std::path::Path;

use pgvector::Vector;
use rusqlite::Connection;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

use bsmcp_common::vector;

fn table_exists(conn: &Connection, table: &str) -> bool {
    conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
        [table],
        |row| row.get::<_, i64>(0),
    )
    .unwrap_or(0)
        > 0
}

pub async fn run(sqlite_path: &Path, postgres_url: &str) -> Result<(), String> {
    eprintln!("Migration: SQLite → PostgreSQL");
    eprintln!("  Source: {}", sqlite_path.display());
    eprintln!("  Target: {}", redact_url(postgres_url));

    // Open SQLite read-only
    let conn = Connection::open_with_flags(
        sqlite_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("Failed to open SQLite database: {e}"))?;

    // Connect to PostgreSQL
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(postgres_url)
        .await
        .map_err(|e| format!("Failed to connect to PostgreSQL: {e}"))?;

    // Ensure pgvector extension exists
    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to create vector extension: {e}"))?;

    // Create tables (same as PostgresDb::new and init_semantic_tables)
    ensure_schema(&pool).await?;

    // Migrate each table (only if it exists in the source SQLite DB)
    let tokens = migrate_access_tokens(&conn, &pool).await?;
    let pages = if table_exists(&conn, "pages") {
        migrate_pages(&conn, &pool).await?
    } else {
        eprintln!("  pages: table not found in SQLite, skipping");
        0
    };
    let chunks = if table_exists(&conn, "chunks") {
        migrate_chunks(&conn, &pool).await?
    } else {
        eprintln!("  chunks: table not found in SQLite, skipping");
        0
    };
    let rels = if table_exists(&conn, "relationships") {
        migrate_relationships(&conn, &pool).await?
    } else {
        eprintln!("  relationships: table not found in SQLite, skipping");
        0
    };
    let jobs = if table_exists(&conn, "embed_jobs") {
        migrate_embed_jobs(&conn, &pool).await?
    } else {
        eprintln!("  embed_jobs: table not found in SQLite, skipping");
        0
    };

    // Fix PostgreSQL sequences so new inserts don't collide with migrated IDs
    fix_sequences(&pool).await?;

    // Validate counts
    eprintln!("\nMigration complete:");
    eprintln!("  access_tokens: {tokens}");
    eprintln!("  pages:         {pages}");
    eprintln!("  chunks:        {chunks}");
    eprintln!("  relationships: {rels}");
    eprintln!("  embed_jobs:    {jobs}");

    validate(&conn, &pool).await?;

    Ok(())
}

async fn ensure_schema(pool: &PgPool) -> Result<(), String> {
    let statements = [
        "CREATE TABLE IF NOT EXISTS access_tokens (
            token TEXT PRIMARY KEY,
            token_id TEXT NOT NULL,
            token_secret TEXT NOT NULL,
            created_at BIGINT NOT NULL
        )",
        "CREATE TABLE IF NOT EXISTS pages (
            page_id BIGINT PRIMARY KEY,
            book_id BIGINT NOT NULL,
            chapter_id BIGINT,
            name TEXT NOT NULL,
            slug TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            embedded_at BIGINT NOT NULL,
            updated_at TEXT
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
            error TEXT
        )",
    ];
    for sql in statements {
        sqlx::query(sql)
            .execute(pool)
            .await
            .map_err(|e| format!("Schema creation failed: {e}"))?;
    }

    // Create indexes (ignore errors if they exist)
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_tokens_created ON access_tokens(created_at)")
        .execute(pool)
        .await
        .ok();
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)")
        .execute(pool).await.ok();
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_embed_jobs_pending ON embed_jobs(status) WHERE status = 'pending'")
        .execute(pool).await.ok();

    eprintln!("  Schema: OK");
    Ok(())
}

async fn migrate_access_tokens(conn: &Connection, pool: &PgPool) -> Result<usize, String> {
    // Copy encrypted blobs directly — they're portable when the encryption key matches
    let mut stmt = conn
        .prepare("SELECT token, token_id, token_secret, created_at FROM access_tokens")
        .map_err(|e| format!("Failed to query access_tokens: {e}"))?;

    let rows: Vec<(String, String, String, i64)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })
        .map_err(|e| format!("Failed to read access_tokens: {e}"))?
        .filter_map(|r| r.ok())
        .collect();

    let count = rows.len();
    for (token, token_id, token_secret, created_at) in &rows {
        sqlx::query(
            "INSERT INTO access_tokens (token, token_id, token_secret, created_at)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (token) DO UPDATE SET token_id = $2, token_secret = $3, created_at = $4",
        )
        .bind(token)
        .bind(token_id)
        .bind(token_secret)
        .bind(created_at)
        .execute(pool)
        .await
        .map_err(|e| format!("Failed to insert access_token: {e}"))?;
    }

    eprintln!("  access_tokens: {count} migrated");
    Ok(count)
}

#[allow(clippy::type_complexity)]
async fn migrate_pages(conn: &Connection, pool: &PgPool) -> Result<usize, String> {
    let mut stmt = conn
        .prepare("SELECT page_id, book_id, chapter_id, name, slug, content_hash, embedded_at, updated_at FROM pages")
        .map_err(|e| format!("Failed to query pages: {e}"))?;

    let rows: Vec<(
        i64,
        i64,
        Option<i64>,
        String,
        String,
        String,
        i64,
        Option<String>,
    )> = stmt
        .query_map([], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                row.get(7)?,
            ))
        })
        .map_err(|e| format!("Failed to read pages: {e}"))?
        .filter_map(|r| r.ok())
        .collect();

    let count = rows.len();
    for (page_id, book_id, chapter_id, name, slug, content_hash, embedded_at, updated_at) in &rows {
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
                updated_at = EXCLUDED.updated_at",
        )
        .bind(page_id)
        .bind(book_id)
        .bind(chapter_id)
        .bind(name)
        .bind(slug)
        .bind(content_hash)
        .bind(embedded_at)
        .bind(updated_at)
        .execute(pool)
        .await
        .map_err(|e| format!("Failed to insert page {page_id}: {e}"))?;
    }

    eprintln!("  pages: {count} migrated");
    Ok(count)
}

#[allow(clippy::type_complexity)]
async fn migrate_chunks(conn: &Connection, pool: &PgPool) -> Result<usize, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, page_id, chunk_index, heading_path, content, content_hash, embedding FROM chunks",
        )
        .map_err(|e| format!("Failed to query chunks: {e}"))?;

    let rows: Vec<(i64, i64, i32, String, String, String, Vec<u8>)> = stmt
        .query_map([], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
            ))
        })
        .map_err(|e| format!("Failed to read chunks: {e}"))?
        .filter_map(|r| r.ok())
        .collect();

    let count = rows.len();
    let mut migrated = 0;

    for (_, page_id, chunk_index, heading_path, content, content_hash, blob) in &rows {
        // Convert SQLite BLOB (LE f32 bytes) to Vec<f32> for pgvector
        let embedding = vector::blob_to_embedding(blob);
        if embedding.is_empty() {
            eprintln!("  warning: chunk page_id={page_id} index={chunk_index} has empty embedding, skipping");
            continue;
        }
        let vec = Vector::from(embedding);

        sqlx::query(
            "INSERT INTO chunks (page_id, chunk_index, heading_path, content, content_hash, embedding)
             VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (page_id, chunk_index) DO UPDATE SET
                heading_path = EXCLUDED.heading_path,
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                embedding = EXCLUDED.embedding",
        )
        .bind(page_id)
        .bind(chunk_index)
        .bind(heading_path)
        .bind(content)
        .bind(content_hash)
        .bind(vec)
        .execute(pool)
        .await
        .map_err(|e| format!("Failed to insert chunk page_id={page_id} index={chunk_index}: {e}"))?;

        migrated += 1;
        if migrated % 500 == 0 {
            eprintln!("  chunks: {migrated}/{count}...");
        }
    }

    eprintln!("  chunks: {migrated} migrated");
    Ok(migrated)
}

async fn migrate_relationships(conn: &Connection, pool: &PgPool) -> Result<usize, String> {
    let mut stmt = conn
        .prepare("SELECT source_page_id, target_page_id, link_type FROM relationships")
        .map_err(|e| format!("Failed to query relationships: {e}"))?;

    let rows: Vec<(i64, i64, String)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .map_err(|e| format!("Failed to read relationships: {e}"))?
        .filter_map(|r| r.ok())
        .collect();

    let count = rows.len();
    for (source, target, link_type) in &rows {
        sqlx::query(
            "INSERT INTO relationships (source_page_id, target_page_id, link_type)
             VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
        )
        .bind(source)
        .bind(target)
        .bind(link_type)
        .execute(pool)
        .await
        .map_err(|e| format!("Failed to insert relationship: {e}"))?;
    }

    eprintln!("  relationships: {count} migrated");
    Ok(count)
}

#[allow(clippy::type_complexity)]
async fn migrate_embed_jobs(conn: &Connection, pool: &PgPool) -> Result<usize, String> {
    let mut stmt = conn
        .prepare(
            "SELECT scope, status, total_pages, done_pages, started_at, finished_at, error FROM embed_jobs",
        )
        .map_err(|e| format!("Failed to query embed_jobs: {e}"))?;

    let rows: Vec<(
        String,
        String,
        i64,
        i64,
        Option<i64>,
        Option<i64>,
        Option<String>,
    )> = stmt
        .query_map([], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
            ))
        })
        .map_err(|e| format!("Failed to read embed_jobs: {e}"))?
        .filter_map(|r| r.ok())
        .collect();

    let count = rows.len();
    for (scope, status, total_pages, done_pages, started_at, finished_at, error) in &rows {
        sqlx::query(
            "INSERT INTO embed_jobs (scope, status, total_pages, done_pages, started_at, finished_at, error)
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(scope)
        .bind(status)
        .bind(total_pages)
        .bind(done_pages)
        .bind(started_at)
        .bind(finished_at)
        .bind(error)
        .execute(pool)
        .await
        .map_err(|e| format!("Failed to insert embed_job: {e}"))?;
    }

    eprintln!("  embed_jobs: {count} migrated");
    Ok(count)
}

/// Fix PostgreSQL BIGSERIAL sequences after migration so new inserts don't collide with migrated IDs.
async fn fix_sequences(pool: &PgPool) -> Result<(), String> {
    let sequences = [
        ("chunks_id_seq", "chunks", "id"),
        ("embed_jobs_id_seq", "embed_jobs", "id"),
    ];
    for (seq, table, col) in sequences {
        let sql = format!(
            "SELECT setval('{seq}', COALESCE((SELECT MAX({col}) FROM {table}), 0) + 1, false)"
        );
        sqlx::query(&sql)
            .execute(pool)
            .await
            .map_err(|e| format!("Failed to fix sequence {seq}: {e}"))?;
    }
    eprintln!("  Sequences: fixed");
    Ok(())
}

async fn validate(conn: &Connection, pool: &PgPool) -> Result<(), String> {
    eprintln!("\nValidation:");
    let mut ok = true;

    let tables = [
        "access_tokens",
        "pages",
        "chunks",
        "relationships",
        "embed_jobs",
    ];
    for table in tables {
        if !table_exists(conn, table) {
            eprintln!("  {table}: not in SQLite, skipped");
            continue;
        }

        let sqlite_count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap_or(-1);

        let pg_row: (i64,) = sqlx::query_as(&format!("SELECT COUNT(*) FROM {table}"))
            .fetch_one(pool)
            .await
            .unwrap_or((-1,));

        let status = if sqlite_count == pg_row.0 {
            "OK"
        } else {
            "MISMATCH"
        };
        if sqlite_count != pg_row.0 {
            ok = false;
        }
        eprintln!(
            "  {table}: SQLite={sqlite_count} PostgreSQL={} [{status}]",
            pg_row.0
        );
    }

    if ok {
        eprintln!("\nAll counts match. Migration successful.");
    } else {
        eprintln!("\nWARNING: Some counts don't match. Check for errors above.");
    }

    Ok(())
}

pub fn redact_url(url: &str) -> String {
    // Redact password from postgres://user:password@host/db
    if let Some(at_pos) = url.find('@') {
        if let Some(colon_pos) = url[..at_pos].rfind(':') {
            // Only redact if there's a scheme separator before the user:pass
            if url[..colon_pos].contains("//") {
                return format!("{}:***@{}", &url[..colon_pos], &url[at_pos + 1..]);
            }
        }
    }
    url.to_string()
}
