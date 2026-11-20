//! NSFW classification of rendered images (TASK-nsfw-score-signal).
//!
//! Convention: the configured classifier runs as `<binary> <image-path>` —
//! explicit argv, no shell — and must print one JSON object
//! `{"score": 0.0..=1.0, "classes": ["..."]}` to stdout before exiting 0.
//! Anything else (missing binary, bad exit, oversized or unparseable output,
//! out-of-range score) is a named 501, never a silent zero score.
//!
//! `routes::supervisor` is private to `routes`, so the same source is
//! included here under a local name: the classifier gets the identical
//! bounded stdout, wall-clock bound and process-group teardown without
//! widening that module's visibility.

use std::path::{Path, PathBuf};
use std::time::Duration;

use honkbox_api_types::NsfwAnnotation;

// `routes.rs` already loads `supervisor.rs` as `routes::supervisor`; reuse it
// instead of a second `#[path]` load (clippy::duplicate_mod in every harness).
#[cfg(unix)]
use super::supervisor as classifier_supervisor;

/// Wall-clock bound for one classifier invocation.
const CLASSIFIER_TIMEOUT: Duration = Duration::from_secs(30);

/// Runtime facts, snapshotted once per image.
pub(crate) struct NsfwFacts {
    enabled: bool,
    policy: String,
    model_version: String,
    threshold: f64,
    classifier_bin: Option<PathBuf>,
}

impl NsfwFacts {
    fn current() -> Self {
        let (enabled, policy, model_version, threshold, classifier_bin) =
            super::runtime_config::nsfw_facts();
        Self {
            enabled,
            policy: policy.trim().to_ascii_lowercase(),
            model_version,
            threshold,
            classifier_bin,
        }
    }

    /// `off` suppresses classification even when the surface is enabled.
    fn classifies(&self) -> bool {
        self.enabled && self.policy != "off"
    }

    /// Only an explicit `block` policy may refuse an image; `flag` and any
    /// unrecognized policy annotate and continue.
    fn blocks(&self, score: f64) -> bool {
        self.classifies() && self.policy == "block" && score >= self.threshold
    }
}

/// What classification decided for one rendered image.
pub(crate) enum Decision {
    /// Nothing ran (disabled, or policy `off`): no annotation to attach.
    Skipped,
    /// Classified and allowed through.
    Allowed(NsfwAnnotation),
    /// Refused by policy `block` before any bundle write.
    Blocked,
}

#[derive(serde::Deserialize)]
struct ClassifierOutput {
    score: f64,
    classes: Vec<String>,
}

/// Classify one image. `Err` is the reason for a named 501.
pub(crate) fn classify(bytes: &[u8], extension: &str) -> Result<Decision, String> {
    let facts = NsfwFacts::current();
    if !facts.classifies() {
        return Ok(Decision::Skipped);
    }
    let Some(bin) = facts.classifier_bin.as_deref() else {
        return Err("nsfw classifier binary is not configured".to_owned());
    };
    let output = run_classifier(bin, bytes, extension)?;
    if !output.score.is_finite() || !(0.0..=1.0).contains(&output.score) {
        return Err(format!(
            "nsfw classifier returned an out-of-range score: {}",
            output.score
        ));
    }
    if facts.blocks(output.score) {
        return Ok(Decision::Blocked);
    }
    Ok(Decision::Allowed(NsfwAnnotation {
        score: output.score,
        classes: output.classes,
        model_version: facts.model_version,
    }))
}

/// The named `nsfw` annotation recorded in a bundle's sidecar.
pub(crate) fn bundle_annotation(annotation: &NsfwAnnotation) -> crate::jobs::BundleAnnotation {
    crate::jobs::BundleAnnotation {
        name: "nsfw".to_owned(),
        value: serde_json::json!({
            "score": annotation.score,
            "classes": annotation.classes,
            "model_version": annotation.model_version,
        }),
    }
}

#[cfg(unix)]
fn run_classifier(
    bin: &Path,
    bytes: &[u8],
    extension: &str,
) -> Result<ClassifierOutput, String> {
    // The vendor reads images from the filesystem; stage the exact rendered
    // bytes under a unique name and remove it on every exit path.
    let image = std::env::temp_dir().join(format!("honkbox-nsfw-{}.{extension}", ulid::Ulid::new()));
    let staged = std::fs::write(&image, bytes);
    let result = match staged {
        Ok(()) => run_on_image(bin, &image),
        Err(err) => Err(format!("nsfw classifier input could not be staged: {err}")),
    };
    let _ = std::fs::remove_file(&image);
    result
}

#[cfg(unix)]
fn run_on_image(bin: &Path, image: &Path) -> Result<ClassifierOutput, String> {
    use classifier_supervisor::{Spec, Termination};

    let outcome = classifier_supervisor::run(Spec::new(
        bin,
        [image.to_string_lossy().into_owned()],
        // Keep the server cwd so a relative classifier path still resolves;
        // the staged image is passed as an absolute path.
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        CLASSIFIER_TIMEOUT,
    ))
    .map_err(|err| format!("nsfw classifier unavailable: {err}"))?;
    if outcome.stdout_truncated {
        return Err("nsfw classifier output exceeded 64 KiB".to_owned());
    }
    if outcome.termination != Termination::Exited(0) {
        return Err(format!(
            "nsfw classifier did not complete: {:?}",
            outcome.termination
        ));
    }
    serde_json::from_str(&outcome.stdout)
        .map_err(|err| format!("nsfw classifier returned invalid JSON: {err}"))
}

#[cfg(not(unix))]
fn run_classifier(
    _bin: &Path,
    _bytes: &[u8],
    _extension: &str,
) -> Result<ClassifierOutput, String> {
    Err("nsfw classifier requires a unix host".to_owned())
}
