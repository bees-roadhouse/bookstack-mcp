//! Server-wide settings for the BookStack MCP server.
//!
//! v0.10.0: dropped per-user `UserSettings` and the briefing-only
//! `GlobalSettings` fields. The server is now a BookStack CRUD facade plus
//! semantic search; there is no per-user state to persist.
//!
//! Issue #80 (v0.13.0): adds `kb_scopes` (named scope cuts callable by name
//! from `semantic_search`) and `cascade_multipliers` (the 4/3/2/1 stage
//! pool defaults for precision-mode). Both persist as JSON blobs on the
//! singleton `global_settings` row.
//!
//! Issue #122 (v0.13.0): drops the interim `indexed_shelves: Vec<i64>` field
//! added by #119 (which itself replaced the v0.12.x `hive_shelf_id` +
//! `user_journals_shelf_id` named-shelf fields). The indexer now walks every
//! shelf the index token can see on every full walk — unconditionally. Scope
//! is a per-call concern, expressed via `semantic_search`'s `shelf_ids` /
//! `book_ids` / `chapter_ids` / `page_ids` / `scopes` params (issue #80). The
//! DB migration drops the now-stale `indexed_shelves_json` column on startup.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::ScopeFilter;

/// Hash a token_id to a stable identifier suitable for use as a database key.
/// SHA-256 hex digest. The raw token_id never appears in storage.
pub fn hash_token_id(token_id: &str) -> String {
    use sha2::Digest;
    let hash = sha2::Sha256::digest(token_id.as_bytes());
    format!("{hash:x}")
}

/// Named scope cut, addressable by name from `semantic_search`'s `scopes`
/// array. Same explicit-ID shape as the inline scope filter; the resolver
/// just unions matching entries from `GlobalSettings::kb_scopes` into the
/// per-call [`crate::types::ScopeFilter`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScopeDef {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shelf_ids: Vec<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub book_ids: Vec<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chapter_ids: Vec<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub page_ids: Vec<i64>,
}

impl ScopeDef {
    pub fn to_filter(&self) -> ScopeFilter {
        ScopeFilter {
            shelf_ids: self.shelf_ids.clone(),
            book_ids: self.book_ids.clone(),
            chapter_ids: self.chapter_ids.clone(),
            page_ids: self.page_ids.clone(),
        }
    }
}

/// Per-stage pool-size multipliers for the precision-mode cascade. The
/// caller's `limit` argument (`N`) is multiplied at each input boundary:
///
/// | Stage | Pool size       |
/// |-------|-----------------|
/// | 1     | `N * stage1`    |
/// | 2     | `N * stage2`    |
/// | 3     | `N * stage3`    |
/// | 4     | `N * stage4` (= final result count when stage4 == 1) |
///
/// Issue #80 defaults to `4 / 3 / 2 / 1`. Env-var overrides:
/// `BSMCP_CASCADE_STAGE{1,2,3,4}_MULT`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CascadeMultipliers {
    #[serde(default = "default_stage1_mult")]
    pub stage1: u32,
    #[serde(default = "default_stage2_mult")]
    pub stage2: u32,
    #[serde(default = "default_stage3_mult")]
    pub stage3: u32,
    #[serde(default = "default_stage4_mult")]
    pub stage4: u32,
}

fn default_stage1_mult() -> u32 {
    4
}
fn default_stage2_mult() -> u32 {
    3
}
fn default_stage3_mult() -> u32 {
    2
}
fn default_stage4_mult() -> u32 {
    1
}

impl Default for CascadeMultipliers {
    fn default() -> Self {
        Self {
            stage1: default_stage1_mult(),
            stage2: default_stage2_mult(),
            stage3: default_stage3_mult(),
            stage4: default_stage4_mult(),
        }
    }
}

impl CascadeMultipliers {
    /// Apply `BSMCP_CASCADE_STAGE{1,2,3,4}_MULT` env-var overrides on top of
    /// whatever was loaded from `global_settings`. Boot-time only — called
    /// once when the semantic state is constructed.
    pub fn apply_env_overrides(&mut self) {
        for (var, slot) in [
            ("BSMCP_CASCADE_STAGE1_MULT", &mut self.stage1),
            ("BSMCP_CASCADE_STAGE2_MULT", &mut self.stage2),
            ("BSMCP_CASCADE_STAGE3_MULT", &mut self.stage3),
            ("BSMCP_CASCADE_STAGE4_MULT", &mut self.stage4),
        ] {
            if let Ok(v) = std::env::var(var) {
                if let Ok(n) = v.parse::<u32>() {
                    if n > 0 {
                        *slot = n;
                    } else {
                        tracing::warn!(
                            env_var = %var,
                            value = %v,
                            kept = *slot,
                            "settings_env_rejected_non_positive"
                        );
                    }
                } else {
                    tracing::warn!(
                        env_var = %var,
                        value = %v,
                        kept = *slot,
                        "settings_env_rejected_not_integer"
                    );
                }
            }
        }
    }

    /// Validate that the multipliers form a non-increasing cascade: stage1
    /// must be >= stage2 >= stage3 >= stage4 >= 1. Returns the original
    /// settings if valid, [`Self::default`] otherwise (with a stderr warn).
    pub fn validated(self) -> Self {
        if self.stage1 >= self.stage2
            && self.stage2 >= self.stage3
            && self.stage3 >= self.stage4
            && self.stage4 >= 1
        {
            self
        } else {
            tracing::warn!(
                multipliers = ?self,
                "settings_cascade_invalid_falling_back_to_defaults"
            );
            Self::default()
        }
    }
}

/// Server-instance settings shared by all users on the same BookStack.
///
/// Single-row table. Admin-only writes (the settings UI silently drops
/// non-admin updates).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GlobalSettings {
    /// Named scope cuts addressable from `semantic_search`'s `scopes` array.
    /// Persisted as a JSON blob on the singleton settings row (issue #80).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub kb_scopes: HashMap<String, ScopeDef>,

    /// Precision-mode cascade pool-size multipliers. Defaults to `4/3/2/1`
    /// per issue #80. Env vars `BSMCP_CASCADE_STAGE{1..4}_MULT` override at
    /// boot.
    #[serde(default)]
    pub cascade_multipliers: CascadeMultipliers,

    /// Hash of the first token_id that set these values (informational).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub set_by_token_hash: Option<String>,

    /// Unix epoch seconds of last update. 0 = never set.
    #[serde(default)]
    pub updated_at: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_multipliers_default_matches_issue_80() {
        let m = CascadeMultipliers::default();
        assert_eq!(m.stage1, 4);
        assert_eq!(m.stage2, 3);
        assert_eq!(m.stage3, 2);
        assert_eq!(m.stage4, 1);
    }

    #[test]
    fn cascade_multipliers_validated_accepts_non_increasing() {
        let m = CascadeMultipliers {
            stage1: 8,
            stage2: 4,
            stage3: 2,
            stage4: 1,
        };
        let v = m.validated();
        assert_eq!((v.stage1, v.stage2, v.stage3, v.stage4), (8, 4, 2, 1));
    }

    #[test]
    fn cascade_multipliers_validated_rejects_increasing() {
        let m = CascadeMultipliers {
            stage1: 1,
            stage2: 4,
            stage3: 2,
            stage4: 1,
        };
        let v = m.validated();
        assert_eq!((v.stage1, v.stage2, v.stage3, v.stage4), (4, 3, 2, 1));
    }

    #[test]
    fn cascade_multipliers_validated_rejects_zero_stage4() {
        let m = CascadeMultipliers {
            stage1: 4,
            stage2: 3,
            stage3: 2,
            stage4: 0,
        };
        let v = m.validated();
        assert_eq!((v.stage1, v.stage2, v.stage3, v.stage4), (4, 3, 2, 1));
    }

    #[test]
    fn scope_def_to_filter_round_trips() {
        let def = ScopeDef {
            shelf_ids: vec![1, 2],
            book_ids: vec![3],
            chapter_ids: vec![],
            page_ids: vec![4, 5],
        };
        let f = def.to_filter();
        assert_eq!(f.shelf_ids, vec![1, 2]);
        assert_eq!(f.book_ids, vec![3]);
        assert!(f.chapter_ids.is_empty());
        assert_eq!(f.page_ids, vec![4, 5]);
    }

    #[test]
    fn global_settings_serde_round_trip_with_scopes() {
        let mut scopes = HashMap::new();
        scopes.insert(
            "policies".to_string(),
            ScopeDef {
                book_ids: vec![12, 34],
                ..Default::default()
            },
        );
        let g = GlobalSettings {
            kb_scopes: scopes,
            cascade_multipliers: CascadeMultipliers {
                stage1: 6,
                stage2: 4,
                stage3: 2,
                stage4: 1,
            },
            ..Default::default()
        };
        let json = serde_json::to_string(&g).expect("serialize");
        let parsed: GlobalSettings = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.cascade_multipliers.stage1, 6);
        assert_eq!(parsed.kb_scopes.len(), 1);
        assert_eq!(
            parsed.kb_scopes.get("policies").unwrap().book_ids,
            vec![12, 34]
        );
    }

    /// Issue #122 — `GlobalSettings` no longer carries `indexed_shelves`.
    /// Default serialization must not mention the dropped field, and JSON
    /// containing the legacy key must round-trip cleanly (the key is just
    /// ignored — serde's struct deser tolerates unknown fields by default).
    #[test]
    fn global_settings_serde_default_omits_indexed_shelves() {
        let g = GlobalSettings::default();
        let json = serde_json::to_string(&g).expect("serialize");
        assert!(
            !json.contains("indexed_shelves"),
            "dropped field must not appear in serialized output, got: {json}"
        );
    }

    /// Issue #122 — JSON carrying a stale `indexed_shelves` field (e.g. from
    /// a v0.13.0-RC #120 deployment) must still deserialize cleanly. The
    /// field is silently dropped.
    #[test]
    fn global_settings_serde_ignores_legacy_indexed_shelves_field() {
        let json = r#"{"indexed_shelves":[7,42],"cascade_multipliers":{"stage1":4,"stage2":3,"stage3":2,"stage4":1},"updated_at":0}"#;
        let parsed: GlobalSettings = serde_json::from_str(json).expect("parse");
        assert_eq!(parsed.updated_at, 0);
        assert_eq!(parsed.cascade_multipliers.stage1, 4);
    }
}
