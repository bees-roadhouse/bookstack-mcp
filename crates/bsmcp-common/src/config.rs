use std::time::Duration;

const DEFAULT_ACCESS_TOKEN_TTL_SECS: u64 = 30 * 86400; // 30 days
const DEFAULT_REFRESH_TOKEN_TTL_SECS: u64 = 90 * 86400; // 90 days

/// Access token TTL. Configurable via `BSMCP_ACCESS_TOKEN_TTL` (seconds).
pub fn access_token_ttl() -> Duration {
    let secs = std::env::var("BSMCP_ACCESS_TOKEN_TTL")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_ACCESS_TOKEN_TTL_SECS);
    Duration::from_secs(secs)
}

/// Refresh token TTL. Configurable via `BSMCP_REFRESH_TOKEN_TTL` (seconds).
/// As long as the stored BookStack API credentials are valid, refreshing
/// transparently issues new tokens without user re-authentication.
pub fn refresh_token_ttl() -> Duration {
    let secs = std::env::var("BSMCP_REFRESH_TOKEN_TTL")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_REFRESH_TOKEN_TTL_SECS);
    Duration::from_secs(secs)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DbBackendType {
    Sqlite,
    Postgres,
}

impl DbBackendType {
    pub fn from_env() -> Self {
        match std::env::var("BSMCP_DB_BACKEND")
            .unwrap_or_else(|_| "sqlite".into())
            .to_lowercase()
            .as_str()
        {
            "postgres" | "postgresql" => Self::Postgres,
            _ => Self::Sqlite,
        }
    }
}
