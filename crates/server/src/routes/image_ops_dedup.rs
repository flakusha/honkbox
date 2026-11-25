//! Image-op idempotency ledger: (authenticated consumer, `request_id`) →
//! settled outcome, bounded on entries AND retained bytes, so retries replay
//! byte-identically while memory stays capped. The caller supplies the
//! consumer from the shared scope layer (env NAME, or a bearer-key digest
//! for legacy unscoped keys); raw credentials never enter the ledger.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock, Mutex};

use axum::http::StatusCode;
use honkbox_api_types::{ImageOpCommand, ImageOpResponse};
use sha2::{Digest as _, Sha256};
use tokio::sync::OnceCell;

use super::image_ops_limits::{IMAGE_OP_DEDUP_MAX_BYTES, IMAGE_OP_DEDUP_MAX_ENTRIES};

/// A settled image-op attempt, replayable byte-identical: the completed
/// response, or the exact error the first attempt answered (status +
/// message + named code). Errors carry the registry job id the execute
/// path registered, so the replay still stamps `X-Job-Id`.
#[derive(Clone)]
pub(super) struct ImageOpSettled {
    pub(super) response: Result<ImageOpResponse, (StatusCode, String, Option<String>)>,
    pub(super) job_id: Option<String>,
}

/// One remembered image-op submission: the canonical command hash it
/// answered and the replayable outcome. Same-hash losers of a same-key
/// race await the cell (single execution, identical replay); a different
/// command hash under the same consumer + `request_id` is a conflict, never
/// silent execution.
pub(super) struct ImageOpDedupEntry {
    command_sha256: [u8; 32],
    /// Approximate retained size of the settled response (set only for
    /// successes; failures are forgotten and their size never counts).
    retained_bytes: std::sync::atomic::AtomicUsize,
    /// The single-flight cell: concurrent same-key retries all await the
    /// one execution (tokio `get_or_init` guarantees exactly one closure
    /// runs even under concurrency).
    pub(super) outcome: OnceCell<ImageOpSettled>,
}

/// The image-op idempotency ledger: (consumer, `request_id`) → entry,
/// bounded on entries AND retained bytes (oldest-first eviction).
#[derive(Default)]
pub(super) struct ImageOpDedup {
    map: HashMap<(String, String), Arc<ImageOpDedupEntry>>,
    /// Insertion order for oldest-first eviction (stale keys tolerated).
    order: VecDeque<(String, String)>,
}

/// Outcome of [`ImageOpDedup::reserve`].
pub(super) enum ImageOpReservation {
    /// Fresh key: this caller participates in the single-flight cell.
    Execute(Arc<ImageOpDedupEntry>),
    /// Same consumer + key + same canonical command: replay the first outcome.
    Replay(Arc<ImageOpDedupEntry>),
    /// Same consumer + key with a DIFFERENT command: a named 400, never execution.
    Conflict,
}

impl ImageOpDedup {
    /// Reserve `(consumer, request_id)` for `command_sha256`. The consumer is
    /// the bearer digest resolved at the route, never a caller-controlled
    /// string, so an identical id from another consumer executes afresh.
    pub(super) fn reserve(
        &mut self,
        consumer: &str,
        request_id: &str,
        command_sha256: [u8; 32],
    ) -> ImageOpReservation {
        let key = (consumer.to_owned(), request_id.to_owned());
        if let Some(entry) = self.map.get(&key) {
            return if entry.command_sha256 == command_sha256 {
                ImageOpReservation::Replay(Arc::clone(entry))
            } else {
                ImageOpReservation::Conflict
            };
        }
        let entry = Arc::new(ImageOpDedupEntry {
            command_sha256,
            retained_bytes: const { std::sync::atomic::AtomicUsize::new(0) },
            outcome: OnceCell::new(),
        });
        self.map.insert(key.clone(), Arc::clone(&entry));
        self.order.push_back(key);
        self.trim();
        ImageOpReservation::Execute(entry)
    }

    /// Record the retained size of a settled SUCCESS (bounded cache).
    pub(super) fn remember(&mut self, consumer: &str, request_id: &str, retained_bytes: usize) {
        let key = (consumer.to_owned(), request_id.to_owned());
        if let Some(entry) = self.map.get(&key) {
            entry
                .retained_bytes
                .store(retained_bytes, std::sync::atomic::Ordering::Relaxed);
        }
        self.trim();
    }

    /// Forget an entry (a FAILED attempt): a genuine retry re-executes.
    pub(super) fn forget(&mut self, consumer: &str, request_id: &str) {
        self.map
            .remove(&(consumer.to_owned(), request_id.to_owned()));
    }

    /// Evict oldest-first while over either budget.
    fn trim(&mut self) {
        while self.map.len() > IMAGE_OP_DEDUP_MAX_ENTRIES
            || self.retained_bytes() > IMAGE_OP_DEDUP_MAX_BYTES
        {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.map.remove(&oldest);
        }
    }

    /// Sum of retained response bytes.
    fn retained_bytes(&self) -> usize {
        self.map
            .values()
            .map(|entry| {
                entry
                    .retained_bytes
                    .load(std::sync::atomic::Ordering::Relaxed)
            })
            .sum()
    }
}

/// The image-op idempotency ledger (process-global, like the job registry).
pub(super) static IMAGE_OP_DEDUP: LazyLock<Mutex<ImageOpDedup>> =
    LazyLock::new(|| Mutex::new(ImageOpDedup::default()));

/// The op wire name for sidecars/labels (the tag of the command).
pub(super) fn op_label(command: &ImageOpCommand) -> &'static str {
    match command {
        ImageOpCommand::Resize(_) => "resize",
        ImageOpCommand::Crop(_) => "crop",
        ImageOpCommand::Rotate(_) => "rotate",
        ImageOpCommand::Grayscale => "grayscale",
        ImageOpCommand::Format => "format",
        ImageOpCommand::Matting => "matting",
    }
}

/// Lowercase-hex SHA-256 of `bytes` (sidecar + response digests).
pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        let _ = write!(out, "{byte:02x}");
    }
    out
}
