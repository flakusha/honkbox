//! VRAM ledger + tier detect; pure in-memory math, no backends or HTTP.
//!
//! Init scope: a single in-process GPU slot (`try_reserve`/`release`, NO fake
//! GPU math). Tier detection, measured sizes, and eviction arrive later;
//! workers never touch this crate.

use std::fmt;
use std::sync::{LazyLock, Mutex};

/// Process-wide single GPU slot shared by every `generate` in this process.
static GLOBAL: LazyLock<Ledger> = LazyLock::new(Ledger::new);

/// The shared single-slot ledger.
pub fn global() -> &'static Ledger {
    &GLOBAL
}

/// Single-slot in-memory ledger: at most one holder at a time.
#[derive(Debug, Default)]
pub struct Ledger {
    busy: Mutex<bool>,
}

/// Rejection while another job holds the slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Busy;

impl fmt::Display for Busy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gpu slot busy: another job is running; retry later")
    }
}

impl std::error::Error for Busy {}

/// Proof of reservation; dropping (or [`Ledger::release`]) frees the slot.
#[derive(Debug)]
pub struct Lease<'a> {
    ledger: &'a Ledger,
}

impl Ledger {
    /// A free ledger.
    #[must_use]
    pub fn new() -> Self {
        Self {
            busy: Mutex::new(false),
        }
    }

    /// Reserve the slot, or `Err(Busy)` while it is held.
    pub fn try_reserve(&self) -> Result<Lease<'_>, Busy> {
        let mut busy = self
            .busy
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *busy {
            return Err(Busy);
        }
        *busy = true;
        Ok(Lease { ledger: self })
    }

    /// Free the slot (idempotent; also happens on `Lease` drop).
    pub fn release(&self) {
        if let Ok(mut busy) = self.busy.lock() {
            *busy = false;
        }
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.ledger.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_reserve_fails_until_drop() {
        let ledger = Ledger::new();
        let lease = ledger.try_reserve().expect("first reserve");
        assert_eq!(ledger.try_reserve().unwrap_err(), Busy);
        drop(lease);
        assert!(ledger.try_reserve().is_ok());
    }

    #[test]
    fn explicit_release_frees_slot() {
        let ledger = Ledger::new();
        let lease = ledger.try_reserve().expect("first reserve");
        ledger.release();
        ledger.release();
        drop(lease);
        assert!(ledger.try_reserve().is_ok());
    }

    #[test]
    fn holder_blocks_other_threads() {
        let ledger = Ledger::new();
        let _lease = ledger.try_reserve().expect("first reserve");
        let blocked =
            std::thread::scope(|s| s.spawn(|| ledger.try_reserve().is_err()).join().unwrap());
        assert!(blocked);
    }
}
