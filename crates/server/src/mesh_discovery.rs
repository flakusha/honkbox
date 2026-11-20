//! Bounded mesh provider discovery (TASK-mesh-provider-discovery +
//! TASK-mesh-capability-advertisement).
//!
//! `/mesh/v1/models` is an OBSERVATION, not a reservation: a discovered model
//! says an upstream claims to serve it, nothing more. This module never
//! reserves capacity, admits a job, or infers admission from a model id — the
//! per-request admission path stays authoritative. The cache is pure (no
//! global state) so it can be exercised directly; the global slot is
//! configure-once/read-mostly wiring for the capability route.

#![allow(dead_code)] // Wired by the capability lane; the pure cache is the tested surface.

use std::sync::{OnceLock, RwLock, RwLockReadGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use honkbox_api_types::MeshExternalCapability;
use reqwest::Url;
use serde::Deserialize;

use crate::config::mesh::MeshUpstream;

/// Total budget for one discovery request. A hung upstream must not stall
/// capability rendering.
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);

/// One model as advertised by an upstream's `/mesh/v1/models`. The aliases
/// tolerate the common id/provider naming variants; a shape we cannot parse
/// is a failed refresh (last-known data is kept), never a guessed model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct DiscoveredModel {
    /// Upstream model id — the only field the capability document exposes.
    #[serde(alias = "model_id", alias = "model", alias = "name")]
    id: String,
    /// Provider the upstream attributes the model to.
    #[serde(alias = "provider", default)]
    provider_id: Option<String>,
    /// Workload class (`chat`, `embed`, ...) the upstream declares.
    #[serde(alias = "workload", alias = "class", default)]
    workload_class: Option<String>,
    /// Declared preconditions/capabilities; stored verbatim, never acted on.
    #[serde(default)]
    requires: Vec<String>,
}

/// Accept both a bare array and an object wrapping the array under
/// `models`/`data`. Unknown fields are ignored by serde.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ModelsResponse {
    /// Bare array of model records.
    Bare(Vec<DiscoveredModel>),
    /// Object with the model array under `models` (or `data`).
    Wrapped {
        /// The advertised models.
        #[serde(default, alias = "data")]
        models: Vec<DiscoveredModel>,
    },
}

impl ModelsResponse {
    /// Flatten either accepted wire shape into the model records.
    fn into_models(self) -> Vec<DiscoveredModel> {
        match self {
            Self::Bare(models) | Self::Wrapped { models } => models,
        }
    }
}

/// The last-known observation for one upstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Entry {
    /// Upstream config label (the cache key).
    pub(crate) upstream: String,
    /// Models from the last successful refresh; retained across a failure.
    pub(crate) models: Vec<DiscoveredModel>,
    /// Unix milliseconds of the last successful refresh; `None` if there
    /// has never been one. A failure never advances this.
    pub(crate) observed_at_ms: Option<u64>,
    /// Error from the most recent failed refresh, cleared on success.
    pub(crate) error: Option<String>,
}

impl Entry {
    /// An entry is usable only when the last refresh succeeded and is still
    /// within the freshness bound. An error or an absent observation makes it
    /// unreachable — its models are never advertised.
    fn reachable(&self, cache: &DiscoveryCache, now_ms: u64) -> bool {
        self.error.is_none()
            && self
                .observed_at_ms
                .is_none_or(|at| now_ms.saturating_sub(at) <= cache.ttl_ms)
    }
}

/// Pure, bounded discovery cache keyed by upstream name. No global state, no
/// admission decisions — callers decide what a "reachable" observation means.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DiscoveryCache {
    /// Freshness bound in milliseconds (`ttl_secs * 1000`, saturating).
    ttl_ms: u64,
    /// Hard ceiling on retained entries; zero means retain nothing.
    cap: usize,
    /// Entries in insertion order.
    entries: Vec<Entry>,
}

impl DiscoveryCache {
    /// Build a cache with an explicit TTL (seconds) and entry cap.
    pub(crate) fn new(ttl_secs: u64, cap: usize) -> Self {
        Self {
            ttl_ms: ttl_secs.saturating_mul(1000),
            cap,
            entries: Vec::new(),
        }
    }

    /// Number of retained entries.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache retains no entries.
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All retained entries, oldest first.
    pub(crate) fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The entry for `upstream`, if one exists.
    pub(crate) fn entry(&self, upstream: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.upstream == upstream)
    }

    /// Whether `entry` is stale at `now_ms`: an error, no successful
    /// observation, or an observation older than the TTL.
    pub(crate) fn stale(&self, entry: &Entry, now_ms: u64) -> bool {
        !entry.reachable(self, now_ms)
    }

    /// Record a successful observation, replacing any previous entry.
    pub(crate) fn observe(&mut self, upstream: &str, models: Vec<DiscoveredModel>, now_ms: u64) {
        let entry = Entry {
            upstream: upstream.to_owned(),
            models,
            observed_at_ms: Some(now_ms),
            error: None,
        };
        if let Some(index) = self.position(upstream) {
            self.entries[index] = entry;
        } else {
            self.insert_capped(entry);
        }
    }

    /// Record a failed refresh: keep the last-known models and timestamp and
    /// attach an error. An upstream never observed gets an empty entry with
    /// no timestamp, so a failure can never fabricate an observation.
    pub(crate) fn fail(&mut self, upstream: &str, error: String, _now_ms: u64) {
        if let Some(index) = self.position(upstream) {
            self.entries[index].error = Some(error);
            return;
        }
        self.insert_capped(Entry {
            upstream: upstream.to_owned(),
            models: Vec::new(),
            observed_at_ms: None,
            error: Some(error),
        });
    }

    fn position(&self, upstream: &str) -> Option<usize> {
        self.entries.iter().position(|e| e.upstream == upstream)
    }

    /// ponytail: linear scan + front eviction; the cap is small (upstreams
    /// are config-bounded), so a map + LRU list would be needless weight.
    /// Swap to an ordered map if the cap ever grows past a few hundred.
    fn insert_capped(&mut self, entry: Entry) {
        if self.cap == 0 {
            return;
        }
        if self.entries.len() >= self.cap {
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }
}

/// Configure-once state behind the capability route.
struct State {
    /// Operator opt-in; off means no models are advertised.
    enabled: bool,
    /// Number of configured upstreams (the expected observation count).
    upstream_count: usize,
    /// Bounded observations.
    cache: DiscoveryCache,
}

static STATE: OnceLock<RwLock<State>> = OnceLock::new();

/// Install the discovery configuration once. The first call wins; later
/// calls are ignored rather than silently mutating a live cache.
pub(crate) fn configure(enabled: bool, upstream_count: usize, ttl_secs: u64, cap: usize) {
    let state = State {
        enabled,
        upstream_count,
        cache: DiscoveryCache::new(ttl_secs, cap),
    };
    let _ = STATE.set(RwLock::new(state));
}

/// Read the current state, tolerating a poisoned lock (a panic elsewhere
/// must not take down capability rendering).
fn read_state() -> Option<RwLockReadGuard<'static, State>> {
    STATE.get().map(|slot| {
        slot.read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    })
}

/// Refresh every configured upstream into the global slot. No lock is held
/// across the network calls; failures remain observations on the cache.
pub(crate) async fn refresh_all(upstreams: &[MeshUpstream]) {
    let mut cache = {
        let Some(state) = read_state() else {
            return;
        };
        if !state.enabled {
            return;
        }
        state.cache.clone()
    };
    refresh_into(&mut cache, upstreams).await;
    write_back(cache);
}

/// Pure refresh into a caller-owned cache: a stub server on port 0 drives the
/// real fetch path without touching the global slot. Failures are recorded
/// (last-known data retained) and never propagated.
pub(crate) async fn refresh_into(cache: &mut DiscoveryCache, upstreams: &[MeshUpstream]) {
    let client = match build_client() {
        Ok(client) => client,
        Err(e) => {
            for upstream in upstreams {
                cache.fail(
                    &upstream.name,
                    format!("client build failed: {e}"),
                    now_ms(),
                );
            }
            return;
        }
    };

    for upstream in upstreams {
        let name = upstream.name.as_str();
        let url = match discovery_url(&upstream.base_url) {
            Ok(url) => url,
            Err(e) => {
                cache.fail(name, e, now_ms());
                continue;
            }
        };
        let mut request = client.get(url);
        // The only credential sent upstream: the configured bearer, attached
        // at this layer and never logged, echoed, or placed in an error.
        if let Some(token) = upstream.bearer_token.as_deref() {
            request = request.bearer_auth(token);
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(e) => {
                cache.fail(name, format!("request failed: {e}"), now_ms());
                continue;
            }
        };
        if !response.status().is_success() {
            cache.fail(
                name,
                format!("upstream status {}", response.status()),
                now_ms(),
            );
            continue;
        }
        match response.json::<ModelsResponse>().await {
            Ok(body) => cache.observe(name, body.into_models(), now_ms()),
            Err(e) => cache.fail(name, format!("invalid models response: {e}"), now_ms()),
        }
    }
}

/// No redirects (an upstream-controlled escape hatch), bounded timeouts.
fn build_client() -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .timeout(DISCOVERY_TIMEOUT)
        .connect_timeout(DISCOVERY_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
}

/// `<base_url>/mesh/v1/models`, joined by string so a base with a path keeps
/// that prefix instead of having its last segment replaced by `Url::join`.
fn discovery_url(base: &Url) -> Result<Url, String> {
    let raw = format!("{}/mesh/v1/models", base.as_str().trim_end_matches('/'));
    Url::parse(&raw).map_err(|e| format!("bad discovery url: {e}"))
}

/// Install the refreshed cache, tolerating a poisoned lock.
fn write_back(cache: DiscoveryCache) {
    if let Some(slot) = STATE.get() {
        let mut state = slot
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.cache = cache;
    }
}

/// Derive the advertised mesh capability from config + cache. Disabled mesh
/// reports nothing; an unreachable upstream contributes neither an
/// `upstreams` count nor model ids, so clients never see a model as available
/// when its source could not be observed.
pub(crate) fn capability() -> MeshExternalCapability {
    let empty = MeshExternalCapability {
        enabled: false,
        upstreams: 0,
        models: Vec::new(),
        last_observation_ms: None,
        stale: false,
    };
    let Some(state) = read_state() else {
        return empty;
    };
    if !state.enabled {
        return empty;
    }

    let now = now_ms();
    let reachable: Vec<&Entry> = state
        .cache
        .entries()
        .iter()
        .filter(|entry| entry.reachable(&state.cache, now))
        .collect();
    let mut models: Vec<String> = reachable
        .iter()
        .flat_map(|entry| entry.models.iter().map(|model| model.id.clone()))
        .collect();
    models.sort_unstable();
    models.dedup();

    let upstreams = reachable.len().min(state.upstream_count);
    let last_observation_ms = state
        .cache
        .entries()
        .iter()
        .filter_map(|entry| entry.observed_at_ms)
        .max();
    // The view may be out of date if any configured upstream has no fresh,
    // error-free observation (including one not yet observed at all).
    let stale_upstreams = reachable.len() < state.upstream_count
        || state
            .cache
            .entries()
            .iter()
            .any(|entry| state.cache.stale(entry, now));

    MeshExternalCapability {
        enabled: true,
        upstreams,
        models,
        last_observation_ms,
        stale: stale_upstreams,
    }
}

/// Unix milliseconds for freshness labels.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}
