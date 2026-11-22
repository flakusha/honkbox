//! Process-global read-only facts derived from the resolved
//! [`crate::config::LoadedConfig`].
//!
//! [`configure`] publishes one immutable snapshot at startup; every getter is
//! a pure read of that [`OnceLock`]. Only env-var NAMES and non-secret policy
//! facts live here — never a bearer token, HMAC secret, or any other secret
//! VALUE.

use std::path::PathBuf;
use std::sync::OnceLock;

use crate::config::mesh::{DEFAULT_DISCOVERY_CAP, DEFAULT_DISCOVERY_TTL_SECS};
use crate::config::{
    DEFAULT_PYTHON_TIMEOUT_SECS, DEFAULT_SIGNED_URL_SECRET_ENV, DEFAULT_SIGNED_URL_TTL_SECS,
    LoadedConfig, NsfwConfig,
};

/// Immutable snapshot of the facts the runtime reads. Deliberately free of
/// secret VALUES: only env-var names and policy knobs.
// Harnesses that pull this module in through `routes.rs` never call the mesh
// getter, so its fields look dead there even though `main.rs` reads them.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Facts {
    nsfw: NsfwConfig,
    python_enabled: bool,
    python_venv: Option<PathBuf>,
    python_timeout_secs: u64,
    python_allowlist: Vec<String>,
    mesh_enabled: bool,
    mesh_upstreams: usize,
    mesh_discovery_ttl_secs: u64,
    mesh_discovery_cap: usize,
    signed_url_enabled: bool,
    signed_url_ttl_secs: u64,
    signed_url_secret_env: String,
    scopes: Vec<(String, Vec<String>)>,
}

static FACTS: OnceLock<Facts> = OnceLock::new();

/// Publish the process-global facts snapshot. The first call wins; later
/// calls are ignored because the slot is read-only once published.
// `main.rs` is the only caller; `#[path]` harnesses compile this without it.
#[allow(dead_code)]
pub fn configure(loaded: &LoadedConfig) {
    let signed_url_secret_env = loaded.auth_facts.signed_url_secret_env.clone();
    let signed_url_ttl_secs = loaded.auth_facts.signed_url_ttl_secs;
    let signed_url_enabled = !signed_url_secret_env.trim().is_empty() && signed_url_ttl_secs > 0;
    let scopes = loaded
        .auth_facts
        .scopes
        .iter()
        .map(|scope| (scope.api_key_env.clone(), scope.scopes.clone()))
        .collect();
    let _ = FACTS.set(Facts {
        nsfw: loaded.nsfw.clone(),
        python_enabled: loaded.python.enabled,
        python_venv: Some(loaded.python.venv.clone()),
        python_timeout_secs: loaded.python.timeout_secs,
        python_allowlist: loaded.python.model_allowlist.clone(),
        mesh_enabled: loaded.mesh.enabled,
        mesh_upstreams: loaded.mesh.upstreams.len(),
        mesh_discovery_ttl_secs: loaded.mesh.discovery_ttl_secs,
        mesh_discovery_cap: loaded.mesh.discovery_cap,
        signed_url_enabled,
        signed_url_ttl_secs,
        signed_url_secret_env,
        scopes,
    });
}

/// Borrow the published snapshot. Absent before `configure` (getters then
/// return documented defaults).
fn facts() -> Option<&'static Facts> {
    FACTS.get()
}

/// `(enabled, policy, model_version, threshold, classifier_bin)`.
pub fn nsfw_facts() -> (bool, String, String, f64, Option<PathBuf>) {
    facts().map_or_else(
        || {
            let default = NsfwConfig::default();
            (
                default.enabled,
                default.policy,
                default.model_version,
                default.threshold,
                default.classifier_bin,
            )
        },
        |f| {
            (
                f.nsfw.enabled,
                f.nsfw.policy.clone(),
                f.nsfw.model_version.clone(),
                f.nsfw.threshold,
                f.nsfw.classifier_bin.clone(),
            )
        },
    )
}

/// Whether the Python sidecar is enabled.
pub fn python_enabled() -> bool {
    facts().is_some_and(|f| f.python_enabled)
}

/// The resolved sidecar venv path (owned clone); `None` before `configure`.
pub fn python_venv() -> Option<PathBuf> {
    facts().and_then(|f| f.python_venv.clone())
}

/// Sidecar request timeout in seconds.
pub fn python_timeout_secs() -> u64 {
    facts().map_or(DEFAULT_PYTHON_TIMEOUT_SECS, |f| f.python_timeout_secs)
}

/// Borrowed model allowlist (empty before `configure`).
pub fn python_allowlist() -> &'static [String] {
    static EMPTY: &[String] = &[];
    facts().map_or(EMPTY, |f| f.python_allowlist.as_slice())
}

/// Whether `/v1/rerank` is backed by a ranker model. A configured, non-empty
/// first allowlist slug is the minimal honest rule: request-time membership
/// is still enforced by the sidecar, so no catalog probe is needed here.
fn rerank_backed(enabled: bool, allowlist: &[String]) -> bool {
    enabled && allowlist.first().is_some_and(|slug| !slug.is_empty())
}

/// Shared gate for the capability flag and the rerank route. `true` means
/// the route can reach a configured ranker without answering a named 501.
pub(crate) fn rerank_supported() -> bool {
    rerank_backed(python_enabled(), python_allowlist())
}

/// `(enabled, upstream_count, discovery_ttl_secs, discovery_cap)`.
// Only `main.rs` renders mesh capability; harnesses including `routes.rs`
// compile this getter without calling it.
#[allow(dead_code)]
pub fn mesh_facts() -> (bool, usize, u64, usize) {
    facts().map_or(
        (false, 0, DEFAULT_DISCOVERY_TTL_SECS, DEFAULT_DISCOVERY_CAP),
        |f| {
            (
                f.mesh_enabled,
                f.mesh_upstreams,
                f.mesh_discovery_ttl_secs,
                f.mesh_discovery_cap,
            )
        },
    )
}

/// `(enabled, ttl_secs, secret_env_NAME)` — a name, never the secret value.
pub fn signed_url_facts() -> (bool, u64, String) {
    facts().map_or(
        (
            false,
            DEFAULT_SIGNED_URL_TTL_SECS,
            DEFAULT_SIGNED_URL_SECRET_ENV.to_owned(),
        ),
        |f| {
            (
                f.signed_url_enabled,
                f.signed_url_ttl_secs,
                f.signed_url_secret_env.clone(),
            )
        },
    )
}

/// `(scopes_configured, [(api_key_env_NAME, scopes)])` — names only.
pub fn scope_facts() -> (bool, Vec<(String, Vec<String>)>) {
    facts().map_or((false, Vec::new()), |f| {
        (!f.scopes.is_empty(), f.scopes.clone())
    })
}

#[cfg(test)]
mod tests {
    use super::rerank_backed;

    #[test]
    fn rerank_backed_needs_python_and_a_configured_slug() {
        let slug = ["ranker:stub".to_owned()];
        assert!(!rerank_backed(false, &slug));
        assert!(!rerank_backed(true, &[]));
        assert!(!rerank_backed(true, &[String::new()]));
        assert!(rerank_backed(true, &slug));
    }
}
