//! Pure time-block helper for MCP `_meta.time`.
//!
//! Resolves a startup-time timezone from env, then emits a JSON block
//! carrying `now_unix`, `now_utc`, `now_local`, `now_human`, `timezone`,
//! `timezone_source` on every tools/call response. Field names match the
//! pre-#79 briefing-era `meta.briefing.time` shape so any consumer reading
//! that path can adopt with a one-key change.
//!
//! No I/O. No globals. The caller resolves once at startup, stashes the
//! `TimezoneConfig`, and hands it to the request handler on every call.

use std::env;

use serde_json::{json, Value};

/// Where the timezone came from. Surfaced as `timezone_source` so a
/// consumer can tell whether the server was configured explicitly or
/// fell through to UTC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimezoneSource {
    /// `BSMCP_DEFAULT_TIMEZONE` was set.
    Env,
    /// `TZ` was set (no BSMCP override).
    Tz,
    /// Neither env var was set — fell back to UTC.
    Utc,
}

impl TimezoneSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Env => "env",
            Self::Tz => "tz",
            Self::Utc => "utc",
        }
    }
}

/// Resolved timezone state. Built once at startup, shared via
/// `Arc<TimezoneConfig>` through app state.
#[derive(Clone, Debug)]
pub struct TimezoneConfig {
    /// IANA timezone name (e.g. `America/New_York`). Always an IANA-style
    /// string — when resolution falls back, this is literally `"UTC"`.
    pub name: String,
    /// Which env var (if any) resolved this timezone.
    pub source: TimezoneSource,
}

impl TimezoneConfig {
    /// Resolution order:
    /// 1. `BSMCP_DEFAULT_TIMEZONE` env (must parse as IANA via chrono-tz)
    /// 2. `TZ` env (must parse as IANA via chrono-tz)
    /// 3. UTC fallback
    ///
    /// Unparseable values fall through to the next source rather than
    /// panicking — a misconfigured env var shouldn't take the server down.
    pub fn from_env() -> Self {
        if let Ok(name) = env::var("BSMCP_DEFAULT_TIMEZONE") {
            let name = name.trim();
            if !name.is_empty() && name.parse::<chrono_tz::Tz>().is_ok() {
                return Self {
                    name: name.to_string(),
                    source: TimezoneSource::Env,
                };
            }
        }
        if let Ok(name) = env::var("TZ") {
            let name = name.trim();
            if !name.is_empty() && name.parse::<chrono_tz::Tz>().is_ok() {
                return Self {
                    name: name.to_string(),
                    source: TimezoneSource::Tz,
                };
            }
        }
        Self {
            name: "UTC".to_string(),
            source: TimezoneSource::Utc,
        }
    }

    /// Build the `_meta.time` block for the current instant. Pure: only
    /// reads system clock and self's resolved timezone, no env lookups,
    /// no I/O.
    pub fn time_block(&self) -> Value {
        let now = chrono::Utc::now();
        let now_unix = now.timestamp();
        let now_utc = now.format("%Y-%m-%dT%H:%M:%SZ").to_string();

        let tz: chrono_tz::Tz = self.name.parse().unwrap_or(chrono_tz::UTC);
        let local = now.with_timezone(&tz);
        let now_local = local.format("%Y-%m-%dT%H:%M:%S%:z").to_string();
        let now_human = local.format("%A, %B %-d, %Y at %-I:%M %p %Z").to_string();

        json!({
            "now_unix": now_unix,
            "now_utc": now_utc,
            "now_local": now_local,
            "now_human": now_human,
            "timezone": self.name,
            "timezone_source": self.source.as_str(),
        })
    }
}

/// Free-function form for callers that already hold a `TimezoneConfig`.
pub fn time_block(cfg: &TimezoneConfig) -> Value {
    cfg.time_block()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Acceptance criterion from issue #67: `now_unix` is within ±2s of
    /// `SystemTime::now`. Catches accidental wall-clock drift bugs (e.g.
    /// somebody swaps `Utc::now` for a fixed value during a refactor).
    #[test]
    fn now_unix_is_within_2s_of_system_time() {
        let cfg = TimezoneConfig {
            name: "UTC".to_string(),
            source: TimezoneSource::Utc,
        };
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let block = cfg.time_block();
        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let now_unix = block["now_unix"].as_i64().expect("now_unix is i64");
        assert!(
            (now_unix - before).abs() <= 2 && (now_unix - after).abs() <= 2,
            "now_unix={now_unix} drifted >2s from SystemTime bounds [{before}, {after}]"
        );
    }

    #[test]
    fn block_carries_all_required_fields() {
        let cfg = TimezoneConfig {
            name: "America/New_York".to_string(),
            source: TimezoneSource::Env,
        };
        let block = cfg.time_block();
        for key in [
            "now_unix",
            "now_utc",
            "now_local",
            "now_human",
            "timezone",
            "timezone_source",
        ] {
            assert!(
                block.get(key).is_some(),
                "_meta.time missing required key `{key}`"
            );
        }
        assert_eq!(block["timezone"], "America/New_York");
        assert_eq!(block["timezone_source"], "env");
    }

    #[test]
    fn utc_fallback_block_is_well_formed() {
        let cfg = TimezoneConfig {
            name: "UTC".to_string(),
            source: TimezoneSource::Utc,
        };
        let block = cfg.time_block();
        assert_eq!(block["timezone"], "UTC");
        assert_eq!(block["timezone_source"], "utc");
        // now_local on UTC should end with `+00:00`.
        let local = block["now_local"].as_str().unwrap();
        assert!(
            local.ends_with("+00:00"),
            "UTC now_local should end with +00:00, got `{local}`"
        );
    }

    #[test]
    fn source_as_str_round_trip() {
        assert_eq!(TimezoneSource::Env.as_str(), "env");
        assert_eq!(TimezoneSource::Tz.as_str(), "tz");
        assert_eq!(TimezoneSource::Utc.as_str(), "utc");
    }
}
