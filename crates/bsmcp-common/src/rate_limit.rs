//! Process-global token-bucket rate limiter for BookStack API requests.
//!
//! BookStack runs on Laravel's per-user throttle (default 180 req/min). When
//! the embedder + worker share a token they walk the API in lockstep and
//! cascade through 429s. The limiter caps the local issue rate to whatever
//! BookStack advertises, with backoff on 429.
//!
//! `shared()` is the single entry point used by `BookStackClient`; every call
//! site picks up the same bucket via `OnceLock` so multi-user fan-out from
//! the MCP server still respects the per-process budget.

use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

const DEFAULT_RPM: f64 = 180.0;

#[derive(Debug)]
struct Inner {
    capacity: f64,
    tokens: f64,
    refill_per_sec: f64,
    last_refill: Instant,
    observed_limit_rpm: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct RateLimiter {
    inner: Arc<Mutex<Inner>>,
}

impl RateLimiter {
    pub fn new(rpm: f64) -> Self {
        let rpm = rpm.max(1.0);
        Self {
            inner: Arc::new(Mutex::new(Inner {
                capacity: rpm,
                tokens: rpm,
                refill_per_sec: rpm / 60.0,
                last_refill: Instant::now(),
                observed_limit_rpm: None,
            })),
        }
    }

    pub fn from_env() -> Self {
        let rpm = env::var("BSMCP_BOOKSTACK_RATE_LIMIT_PER_MIN")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| *v > 0.0)
            .unwrap_or(DEFAULT_RPM);
        Self::new(rpm)
    }

    pub async fn acquire(&self) {
        loop {
            let wait = {
                let mut inner = self.inner.lock().await;
                refill(&mut inner);
                if inner.tokens >= 1.0 {
                    inner.tokens -= 1.0;
                    return;
                }
                let deficit = 1.0 - inner.tokens;
                let secs = deficit / inner.refill_per_sec.max(f64::MIN_POSITIVE);
                Duration::from_secs_f64(secs.max(0.001))
            };
            tokio::time::sleep(wait).await;
        }
    }

    /// Adjust the local cap based on `X-RateLimit-Limit` advertised by
    /// BookStack. The cap is **monotonically non-increasing** for the life
    /// of the process — once we've seen a tighter limit we stay at that
    /// floor even if BookStack later advertises a higher one. To widen the
    /// cap after BookStack relaxes its throttle, restart the process.
    pub fn observe_limit(&self, headers: &reqwest::header::HeaderMap) {
        let Some(value) = headers.get("X-RateLimit-Limit") else {
            return;
        };
        let Some(s) = value.to_str().ok() else { return };
        let Ok(rpm) = s.trim().parse::<u32>() else {
            return;
        };
        let rpm = rpm as f64;
        if rpm <= 0.0 {
            return;
        }
        if let Ok(mut inner) = self.inner.try_lock() {
            inner.observed_limit_rpm = Some(rpm);
            // Defensive: only ever lower the local cap. If BookStack advertises
            // a higher limit than env-configured, ignore it — env wins.
            if rpm < inner.capacity {
                inner.capacity = rpm;
                inner.refill_per_sec = rpm / 60.0;
                if inner.tokens > rpm {
                    inner.tokens = rpm;
                }
            }
        }
    }
}

/// Counter used to ensure successive jitter calls within the same process
/// can't return the same value when invoked back-to-back on different
/// callers (embedder + worker) — paired with wall-clock nanos this gives
/// us enough entropy to desync stampedes without pulling in `rand`.
static JITTER_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Return a uniform jitter in `[0, max_ms)` milliseconds. Uses
/// wall-clock nanoseconds + a process-local counter for entropy — no
/// external RNG dependency. Good enough to desync two workers that share
/// a `Retry-After=N` value.
pub fn jitter(max_ms: u64) -> Duration {
    if max_ms == 0 {
        return Duration::from_millis(0);
    }
    let counter = JITTER_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    // Mix nanos and counter so two calls in the same nanosecond still
    // diverge. The exact distribution doesn't matter — we just need
    // non-trivially different sleeps per caller.
    let mixed = nanos
        .wrapping_mul(0x9E37_79B9_7F4A_7C15) // golden-ratio multiplier
        .wrapping_add(counter.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    Duration::from_millis(mixed % max_ms)
}

fn refill(inner: &mut Inner) {
    let now = Instant::now();
    let elapsed = now.duration_since(inner.last_refill).as_secs_f64();
    if elapsed > 0.0 {
        inner.tokens = (inner.tokens + elapsed * inner.refill_per_sec).min(inner.capacity);
        inner.last_refill = now;
    }
}

static GLOBAL: OnceLock<RateLimiter> = OnceLock::new();

pub fn shared() -> RateLimiter {
    GLOBAL.get_or_init(RateLimiter::from_env).clone()
}

/// Parse a `Retry-After` HTTP header value as integer seconds. Returns
/// `None` for HTTP-date forms (RFC 7231 allows them but BookStack/Laravel
/// emits integer seconds in practice).
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn acquire_blocks_when_bucket_empty() {
        let limiter = RateLimiter::new(60.0);
        for _ in 0..60 {
            limiter.acquire().await;
        }
        let start = Instant::now();
        let acquire = limiter.acquire();
        tokio::time::advance(Duration::from_millis(1100)).await;
        acquire.await;
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(900),
            "expected to wait ~1s for refill, got {:?}",
            elapsed
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn refill_replenishes_over_time() {
        let limiter = RateLimiter::new(60.0);
        for _ in 0..60 {
            limiter.acquire().await;
        }
        // Advance 30 seconds: at 60 rpm = 1 token/sec, we should have ~30 tokens.
        tokio::time::advance(Duration::from_secs(30)).await;
        for _ in 0..29 {
            limiter.acquire().await;
        }
    }

    #[tokio::test]
    async fn observe_limit_lowers_capacity() {
        let limiter = RateLimiter::new(180.0);
        let mut headers = HeaderMap::new();
        headers.insert("X-RateLimit-Limit", HeaderValue::from_static("60"));
        limiter.observe_limit(&headers);
        let inner = limiter.inner.lock().await;
        assert_eq!(inner.capacity, 60.0);
        assert!((inner.refill_per_sec - 1.0).abs() < f64::EPSILON);
        assert_eq!(inner.observed_limit_rpm, Some(60.0));
    }

    #[tokio::test]
    async fn observe_limit_never_raises() {
        let limiter = RateLimiter::new(60.0);
        let mut headers = HeaderMap::new();
        headers.insert("X-RateLimit-Limit", HeaderValue::from_static("9999"));
        limiter.observe_limit(&headers);
        let inner = limiter.inner.lock().await;
        assert_eq!(inner.capacity, 60.0);
        assert_eq!(inner.observed_limit_rpm, Some(9999.0));
    }

    #[tokio::test]
    async fn observe_limit_ignores_garbage_header() {
        let limiter = RateLimiter::new(60.0);
        let mut headers = HeaderMap::new();
        headers.insert("X-RateLimit-Limit", HeaderValue::from_static("nope"));
        limiter.observe_limit(&headers);
        let inner = limiter.inner.lock().await;
        assert_eq!(inner.capacity, 60.0);
        assert_eq!(inner.observed_limit_rpm, None);
    }

    #[test]
    fn parse_retry_after_integer_seconds() {
        assert_eq!(parse_retry_after("12"), Some(Duration::from_secs(12)));
        assert_eq!(parse_retry_after(" 7 "), Some(Duration::from_secs(7)));
        assert_eq!(parse_retry_after("0"), Some(Duration::from_secs(0)));
    }

    #[test]
    fn parse_retry_after_rejects_http_date() {
        // HTTP-date form is allowed by RFC 7231 but BookStack/Laravel uses
        // integer seconds. Document the limitation by asserting we don't
        // accidentally parse a date.
        assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), None);
        assert_eq!(parse_retry_after("not-a-number"), None);
    }

    #[test]
    fn jitter_within_bound() {
        for _ in 0..100 {
            let j = jitter(500);
            assert!(j < Duration::from_millis(500), "jitter {:?} not < 500ms", j);
        }
    }

    #[test]
    fn jitter_zero_returns_zero() {
        assert_eq!(jitter(0), Duration::from_millis(0));
    }

    #[test]
    fn jitter_desyncs_concurrent_callers() {
        // Two back-to-back callers with the same Retry-After value should
        // not sleep for the exact same duration — that's the whole point.
        // Sample a batch and assert at least two distinct values appear.
        let mut samples = std::collections::HashSet::new();
        for _ in 0..32 {
            samples.insert(jitter(500).as_millis());
        }
        assert!(
            samples.len() > 1,
            "jitter degenerated to a constant — embedder + worker would still stampede"
        );
    }
}
