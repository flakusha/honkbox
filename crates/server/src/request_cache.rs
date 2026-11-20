//! Bounded, TTL-based dedup cache keyed by a canonical request hash.
//!
//! # Wiring (left to the orchestrator)
//!
//! Deliberately standalone: no route or `routes.rs` wiring lives here. The
//! generation orchestrator should compute the request's canonical hash with
//! [`crate::jobs::determinism::canonical_hash`] and consult a [`RequestCache`]
//! keyed by that hash *before* acquiring a GPU lease — a hit replays the stored
//! response, so identical requests consume no GPU slot; a miss runs the
//! pipeline and then `put`s the hash → response pair for later replays.
//!
//! Instances are owned by the caller (no process-global mutable state), so
//! tests construct independent caches without leaking into one another.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// One cached response and the insertion time its TTL is measured from.
struct Entry<T> {
    value: Arc<T>,
    inserted_at: Instant,
}

struct Inner<T> {
    entries: HashMap<String, Entry<T>>,
    /// Insertion order for oldest-first eviction; stale keys are tolerated.
    order: VecDeque<String>,
}

/// A bounded, TTL-based cache of `Arc<T>` responses keyed by canonical hash.
///
/// Handles returned by [`RequestCache::get`] are cheap clones (an `Arc` bump).
/// All state sits behind one [`Mutex`]; critical sections only clone an `Arc`
/// or touch the map — no generation or IO happens while the lock is held.
pub(crate) struct RequestCache<T> {
    inner: Mutex<Inner<T>>,
    max_entries: usize,
    ttl: Duration,
}

impl<T> RequestCache<T> {
    /// Create a cache holding at most `max_entries` entries, each valid for
    /// `ttl` from insertion.
    pub(crate) fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                order: VecDeque::new(),
            }),
            max_entries,
            ttl,
        }
    }

    /// Clone the live response for `key`, or `None` on a miss or expiry.
    pub(crate) fn get(&self, key: &str) -> Option<Arc<T>> {
        let inner = self.lock();
        let entry = inner.entries.get(key)?;
        if entry.inserted_at.elapsed() >= self.ttl {
            return None;
        }
        Some(Arc::clone(&entry.value))
    }

    /// Store `value` under `key`, evicting the oldest entry over capacity.
    pub(crate) fn put(&self, key: String, value: T) {
        self.insert(key, value, Instant::now());
    }

    /// Whether any live (unexpired) response remains; also drops expired ones.
    pub(crate) fn is_empty(&self) -> bool {
        let mut inner = self.lock();
        inner.prune_expired(self.ttl);
        inner.entries.is_empty()
    }

    /// Number of live (unexpired) entries.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        let mut inner = self.lock();
        inner.prune_expired(self.ttl);
        inner.entries.len()
    }

    /// Drop all expired entries.
    #[cfg(test)]
    pub(crate) fn prune(&self) {
        self.lock().prune_expired(self.ttl);
    }

    /// Drop all entries.
    #[cfg(test)]
    pub(crate) fn clear(&self) {
        let mut inner = self.lock();
        inner.entries.clear();
        inner.order.clear();
    }

    fn insert(&self, key: String, value: T, inserted_at: Instant) {
        let mut inner = self.lock();
        // Re-put refreshes in place; the insertion-order queue stays valid.
        if let Some(entry) = inner.entries.get_mut(&key) {
            entry.value = Arc::new(value);
            entry.inserted_at = inserted_at;
            return;
        }
        inner.entries.insert(
            key.clone(),
            Entry {
                value: Arc::new(value),
                inserted_at,
            },
        );
        inner.order.push_back(key);
        while inner.entries.len() > self.max_entries {
            let Some(oldest) = inner.order.pop_front() else {
                break;
            };
            inner.entries.remove(&oldest);
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner<T>> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Backdate an entry so TTL expiry is testable without sleeps.
    #[cfg(test)]
    fn put_backdated(&self, key: String, value: T, age: Duration) {
        let inserted_at = Instant::now().checked_sub(age).unwrap_or_else(Instant::now);
        self.insert(key, value, inserted_at);
    }
}

impl<T> Inner<T> {
    fn prune_expired(&mut self, ttl: Duration) {
        self.entries
            .retain(|_, entry| entry.inserted_at.elapsed() < ttl);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(max: usize, ttl: Duration) -> RequestCache<i32> {
        RequestCache::new(max, ttl)
    }

    #[test]
    fn hit_and_miss() {
        let c = cache(4, Duration::from_secs(60));
        assert!(c.get("missing").is_none());
        c.put("a".into(), 7);
        assert_eq!(*c.get("a").unwrap(), 7);
        // A get clones the same shared Arc, not a second allocation.
        assert!(Arc::ptr_eq(&c.get("a").unwrap(), &c.get("a").unwrap()));
    }

    #[test]
    fn ttl_expiry() {
        let c = cache(4, Duration::from_secs(1));
        // Insert an entry already 2s old against a 1s TTL: expired on read.
        c.put_backdated("old".into(), 1, Duration::from_secs(2));
        assert!(c.get("old").is_none());
        assert_eq!(c.len(), 0);
        c.put("fresh".into(), 2);
        assert_eq!(c.len(), 1);
        c.prune();
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn distinct_hashes_do_not_collide() {
        let c = cache(4, Duration::from_secs(60));
        c.put("hash-a".into(), 10);
        c.put("hash-b".into(), 20);
        assert_eq!(*c.get("hash-a").unwrap(), 10);
        assert_eq!(*c.get("hash-b").unwrap(), 20);
    }

    #[test]
    fn eviction_is_oldest_first() {
        let c = cache(2, Duration::from_secs(60));
        c.put("a".into(), 1);
        c.put("b".into(), 2);
        c.put("c".into(), 3);
        assert!(c.get("a").is_none());
        assert_eq!(*c.get("b").unwrap(), 2);
        assert_eq!(*c.get("c").unwrap(), 3);
        assert_eq!(c.len(), 2);
        c.clear();
        assert_eq!(c.len(), 0);
    }
}
