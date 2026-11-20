//! Short-lived HMAC-signed content URLs (TASK-capability-signed-urls).
//!
//! The signature covers exactly `job_id|content_name|expires_unix|scope`, so a
//! URL minted for one job (or one manifest name) cannot authorize another.
//! HMAC-SHA256 is built from the already-present `sha2` dependency; no new
//! crate. The secret VALUE is read at request time by the caller from the env
//! NAME reported by `config::runtime_config::signed_url_facts()` — this module
//! never stores, logs, or returns it, and never reads config state itself.
//!
//! Query params are ADDITIVE on the existing content routes: without them the
//! bearer path is unchanged. With them the route verifies here before touching
//! the filesystem. The credential stays out of response bodies and logs.

use std::fmt::Write as _;

use serde::Deserialize;
use sha2::{Digest as _, Sha256};

/// Scope carried by minted read URLs (also the route-family name in the scope
/// map); it is bound into the signature so it cannot be swapped in transit.
pub(crate) const READ_SCOPE: &str = "read";
/// Synthetic content name used for the bundle route's signature.
pub(crate) const BUNDLE_CONTENT: &str = "bundle.zip";

/// Additive query parameters accepted by the two content routes. Every field
/// is a string so a malformed value yields a named 401, not axum's generic 400.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct SignedQuery {
    pub expires: Option<String>,
    pub sig: Option<String>,
    pub scope: Option<String>,
}

impl SignedQuery {
    /// True when any signed param is present: the route then REQUIRES a valid
    /// signature (a partial or tampered set is a 401, never a bearer fallback).
    pub(crate) fn is_present(&self) -> bool {
        self.expires.is_some() || self.sig.is_some() || self.scope.is_some()
    }
}

/// Unix seconds at the moment of the call; pre-epoch clocks clamp to 0.
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Expiry for a freshly minted URL, capped by the configured TTL.
pub(crate) fn expires_after(ttl_secs: u64) -> u64 {
    now_unix().saturating_add(ttl_secs)
}

/// HMAC-SHA256(secret, `job_id|content_name|expires_unix|scope`), lowercase hex.
pub(crate) fn signature(
    secret: &[u8],
    job_id: &str,
    content_name: &str,
    expires_unix: u64,
    scope: &str,
) -> String {
    let message = format!("{job_id}|{content_name}|{expires_unix}|{scope}");
    let mut padded = [0u8; 64];
    if secret.len() > 64 {
        padded[..32].copy_from_slice(&Sha256::digest(secret));
    } else {
        padded[..secret.len()].copy_from_slice(secret);
    }
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= padded[i];
        opad[i] ^= padded[i];
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(message.as_bytes());
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner.finalize());
    let digest = outer.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Query suffix for a relative content URL. Values are numeric, lowercase hex
/// or a scope slug, so no URL escaping is needed.
pub(crate) fn signed_query(
    secret: &[u8],
    job_id: &str,
    content_name: &str,
    expires_unix: u64,
    scope: &str,
) -> String {
    format!(
        "expires={expires_unix}&sig={}&scope={scope}",
        signature(secret, job_id, content_name, expires_unix, scope)
    )
}

/// Constant-time byte comparison: length mismatch folds into the accumulator
/// and every byte is visited, so timing does not reveal where a signature
/// differed. Mirrors the bearer middleware's helper (that one is module-local
/// to `routes`; this module is declared there via `#[path]`).
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut acc = (a.len() ^ b.len()) as u64;
    for i in 0..a.len().max(b.len()) {
        let x = u64::from(a.get(i).copied().unwrap_or(0));
        let y = u64::from(b.get(i).copied().unwrap_or(0));
        acc |= x ^ y;
    }
    acc == 0
}

/// Verify a presented query against one job/content tuple: constant-time
/// signature check AND a live expiry. A missing secret fails closed.
pub(crate) fn verify_query(
    secret: &[u8],
    job_id: &str,
    content_name: &str,
    query: &SignedQuery,
) -> bool {
    let (Some(expires), Some(sig)) = (&query.expires, &query.sig) else {
        return false;
    };
    let Ok(expires) = expires.parse::<u64>() else {
        return false;
    };
    let scope = query.scope.as_deref().unwrap_or(READ_SCOPE);
    let expected = signature(secret, job_id, content_name, expires, scope);
    constant_time_eq(expected.as_bytes(), sig.as_bytes()) && now_unix() < expires
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_tamper_and_expiry() {
        let secret = b"test-secret";
        let job = "01J00000000000000000000000";
        let query = signed_query(secret, job, "out.png", 4_102_444_800, READ_SCOPE);
        let mut parts = query.split('&');
        let expires = parts.next().unwrap().trim_start_matches("expires=");
        let sig = parts.next().unwrap().trim_start_matches("sig=");
        let scope = parts.next().unwrap().trim_start_matches("scope=");
        let parsed = SignedQuery {
            expires: Some(expires.to_owned()),
            sig: Some(sig.to_owned()),
            scope: Some(scope.to_owned()),
        };
        assert!(verify_query(secret, job, "out.png", &parsed));
        // A URL for another job or another name must not authorize this one.
        assert!(!verify_query(
            secret,
            "01J00000000000000000000001",
            "out.png",
            &parsed
        ));
        assert!(!verify_query(secret, job, "other.png", &parsed));
        // An already-elapsed expiry is rejected even with a valid signature.
        let expired_query = SignedQuery {
            expires: Some("1".to_owned()),
            ..parsed
        };
        assert!(!verify_query(secret, job, "out.png", &expired_query));
        assert!(!verify_query(
            secret,
            job,
            "out.png",
            &SignedQuery::default()
        ));
    }
}
