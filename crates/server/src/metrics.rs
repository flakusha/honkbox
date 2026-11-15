//! Minimal Prometheus text exposition (`GET /metrics`, feature-survey NEW-9).
//!
//! Hand-rolled on purpose: `AtomicU64` statics in the hot path (no mutex, no
//! per-request allocation), pre-declared histogram buckets, and a
//! vLLM-vocabulary subset — `honkbox_requests_total{route,code}`,
//! `honkbox_request_duration_seconds`,
//! `honkbox_gpu_slot_busy_rejections_total`, `honkbox_build_info{version}`.

use std::fmt::Write as _;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Known route labels (bounded cardinality); everything else counts under
/// `"other"` so arbitrary fallback paths cannot explode the series set.
const ROUTES: [&str; 10] = [
    "/health",
    "/api/version",
    "/v1/openapi.json",
    "/v1/models",
    "/v1/models/refresh",
    "/v1/chat",
    "/v1/chat/stream",
    "/v1/transcribe",
    "/v1/synthesize",
    "/metrics",
];

/// Status codes tracked verbatim; any other code counts under `"other"`.
const CODES: [&str; 7] = ["200", "400", "404", "415", "429", "500", "503"];

/// Histogram bucket edges in nanoseconds (Prometheus-client defaults).
const BUCKET_EDGES_NS: [u64; 11] = [
    5_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
    100_000_000,
    250_000_000,
    500_000_000,
    1_000_000_000,
    2_500_000_000,
    5_000_000_000,
    10_000_000_000,
];

/// Bucket `le` labels, matching `BUCKET_EDGES_NS` one for one.
const BUCKET_LES: [&str; 11] = [
    "0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1.0", "2.5", "5.0", "10.0",
];

/// Known routes + the `"other"` catch-all row.
const ROUTE_SLOTS: usize = ROUTES.len() + 1;
/// Known codes + the `"other"` catch-all column.
const CODE_SLOTS: usize = CODES.len() + 1;
/// Pre-declared histogram buckets.
const BUCKETS: usize = BUCKET_EDGES_NS.len();

/// One route row of `requests_total` cells.
type CodeCells = [AtomicU64; CODE_SLOTS];

/// `requests_total` cells: `[route][code]`, zero-initialized at first use.
static REQUESTS: LazyLock<[CodeCells; ROUTE_SLOTS]> =
    LazyLock::new(|| [const { [const { AtomicU64::new(0) }; CODE_SLOTS] }; ROUTE_SLOTS]);

/// Hand-rolled cumulative histogram over `BUCKET_EDGES_NS`.
struct Histogram {
    /// Observations total (equals the `+Inf` bucket).
    count: AtomicU64,
    /// Observed nanoseconds total.
    sum_ns: AtomicU64,
    /// Per-bucket cumulative counts (`le` edges, `+Inf` = `count`).
    buckets: [AtomicU64; BUCKETS],
}

/// Global request-duration histogram (no labels; route breakdown lives on
/// `requests_total` only, to keep the series set small).
static REQUEST_DURATION: LazyLock<Histogram> = LazyLock::new(|| Histogram {
    count: AtomicU64::new(0),
    sum_ns: AtomicU64::new(0),
    buckets: [const { AtomicU64::new(0) }; BUCKETS],
});

/// 429 GPU-slot rejections (residency `Busy`), process-wide.
static GPU_SLOT_BUSY_REJECTIONS: AtomicU64 = AtomicU64::new(0);

/// Request route label for a raw URI path: the path itself for known routes,
/// `"other"` for anything else (fallback hits, escapes, probes).
pub(crate) fn route_label(path: &str) -> &'static str {
    ROUTES
        .iter()
        .find(|route| **route == path)
        .copied()
        .unwrap_or("other")
}

fn route_slot(route: &str) -> usize {
    ROUTES
        .iter()
        .position(|known| *known == route)
        .unwrap_or(ROUTES.len())
}

fn code_slot(code: u16) -> usize {
    CODES
        .iter()
        .position(|label| label.parse::<u16>().is_ok_and(|known| known == code))
        .unwrap_or(CODES.len())
}

/// Count one request: route/code counter + one duration observation. A
/// handful of relaxed atomic adds; no allocation, no lock.
pub(crate) fn record_request(route: &'static str, code: u16, elapsed: Duration) {
    REQUESTS[route_slot(route)][code_slot(code)].fetch_add(1, Ordering::Relaxed);
    let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    let hist = &*REQUEST_DURATION;
    hist.count.fetch_add(1, Ordering::Relaxed);
    hist.sum_ns.fetch_add(nanos, Ordering::Relaxed);
    for (slot, edge) in BUCKET_EDGES_NS.iter().enumerate() {
        if nanos <= *edge {
            hist.buckets[slot].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Count one 429 GPU-slot-busy rejection (call at the rejection site).
pub(crate) fn note_gpu_slot_busy() {
    GPU_SLOT_BUSY_REJECTIONS.fetch_add(1, Ordering::Relaxed);
}

/// Render the Prometheus text exposition (version 0.0.4) for every series.
pub(crate) fn exposition() -> String {
    let mut out = String::new();

    push_family(&mut out, "honkbox_build_info", "Build metadata.", "gauge");
    let _ = writeln!(
        out,
        "honkbox_build_info{{version=\"{}\"}} 1",
        env!("CARGO_PKG_VERSION")
    );

    push_family(
        &mut out,
        "honkbox_requests_total",
        "HTTP requests by route and status code.",
        "counter",
    );
    let routes = ROUTES.iter().copied().chain(["other"]);
    let codes = CODES.iter().copied().chain(["other"]);
    for (route, row) in routes.zip(REQUESTS.iter()) {
        for (code, cell) in codes.clone().zip(row.iter()) {
            let _ = writeln!(
                out,
                "honkbox_requests_total{{route=\"{route}\",code=\"{code}\"}} {}",
                cell.load(Ordering::Relaxed)
            );
        }
    }

    push_family(
        &mut out,
        "honkbox_gpu_slot_busy_rejections_total",
        "Requests rejected with 429 because the GPU slot was busy.",
        "counter",
    );
    let _ = writeln!(
        out,
        "honkbox_gpu_slot_busy_rejections_total {}",
        GPU_SLOT_BUSY_REJECTIONS.load(Ordering::Relaxed)
    );

    push_family(
        &mut out,
        "honkbox_request_duration_seconds",
        "Request duration in seconds.",
        "histogram",
    );
    let hist = &*REQUEST_DURATION;
    let count = hist.count.load(Ordering::Relaxed);
    for (le, bucket) in BUCKET_LES.iter().zip(hist.buckets.iter()) {
        let _ = writeln!(
            out,
            "honkbox_request_duration_seconds_bucket{{le=\"{le}\"}} {}",
            bucket.load(Ordering::Relaxed)
        );
    }
    let _ = writeln!(
        out,
        "honkbox_request_duration_seconds_bucket{{le=\"+Inf\"}} {count}"
    );
    let sum_ns = hist.sum_ns.load(Ordering::Relaxed);
    // Integer math only (the repo avoids float casts): 9 fractional digits.
    let _ = writeln!(
        out,
        "honkbox_request_duration_seconds_sum {}.{:09}",
        sum_ns / 1_000_000_000,
        sum_ns % 1_000_000_000
    );
    let _ = writeln!(out, "honkbox_request_duration_seconds_count {count}");
    out
}

/// HELP + TYPE header lines for one metric family.
fn push_family(out: &mut String, name: &str, help: &str, kind: &str) {
    let _ = write!(out, "# HELP {name} {help}\n# TYPE {name} {kind}\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_map_known_and_unknown() {
        assert_eq!(route_label("/v1/chat"), "/v1/chat");
        assert_eq!(route_label("/nope/xyz"), "other");
        assert_eq!(route_slot("other"), ROUTES.len());
        assert_eq!(code_slot(429), 4);
        assert_eq!(code_slot(418), CODES.len());
    }

    /// Numeric tail of the first exposition line with the given prefix.
    fn series_value(body: &str, prefix: &str) -> u64 {
        body.lines()
            .find(|line| line.starts_with(prefix))
            .unwrap_or_else(|| panic!("missing series line: {prefix}"))
            .rsplit(' ')
            .next()
            .expect("series line has a value")
            .parse()
            .expect("series value parses as u64")
    }

    #[test]
    fn exposition_counts_requests_and_buckets() {
        // Parallel unit tests share the process-wide statics: assert >=, not ==.
        record_request("/v1/models", 200, Duration::from_millis(12));
        record_request("/v1/chat/stream", 200, Duration::from_nanos(5_000_000));
        let body = exposition();

        assert!(body.contains("# TYPE honkbox_requests_total counter"));
        let models_line = "honkbox_requests_total{route=\"/v1/models\",code=\"200\"}";
        assert!(series_value(&body, models_line) >= 1);
        let stream_line = "honkbox_requests_total{route=\"/v1/chat/stream\",code=\"200\"}";
        assert!(series_value(&body, stream_line) >= 1);
        // 12ms lands in le=0.025 (and above); 5ms exactly on the le=0.005 edge.
        let le25 = "honkbox_request_duration_seconds_bucket{le=\"0.025\"}";
        assert!(series_value(&body, le25) >= 1);
        let le005 = "honkbox_request_duration_seconds_bucket{le=\"0.005\"}";
        assert!(series_value(&body, le005) >= 1);
        assert!(body.contains("# TYPE honkbox_request_duration_seconds histogram"));
        assert!(body.contains("honkbox_request_duration_seconds_bucket{le=\"+Inf\"}"));
        assert!(body.contains("honkbox_request_duration_seconds_sum 0."));
        assert!(series_value(&body, "honkbox_request_duration_seconds_count") >= 2);

        assert!(body.contains("# TYPE honkbox_gpu_slot_busy_rejections_total counter"));
        note_gpu_slot_busy();
        assert!(series_value(&exposition(), "honkbox_gpu_slot_busy_rejections_total") >= 1);
    }

    #[test]
    fn build_info_is_present() {
        let body = exposition();
        assert!(body.starts_with("# HELP honkbox_build_info"));
        assert!(body.contains("# TYPE honkbox_build_info gauge"));
        assert!(body.contains("honkbox_build_info{version=\""));
        assert!(body.ends_with('\n'));
    }
}
