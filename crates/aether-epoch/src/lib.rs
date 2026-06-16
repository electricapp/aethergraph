//! Monotonic version clock for cross-subsystem read consistency.
//!
//! An [`EpochClock`] is a shared, lock-free counter that subsystems use to
//! coordinate "consistent reads" across independent data sources. The typical
//! pattern: a reader snapshots [`EpochClock::current`] before issuing a
//! multi-source read; each subsystem then guarantees its returned data was
//! committed at-or-before that pinned epoch.
//!
//! The clock itself is just an [`AtomicU64`]. Subsystems opt in by accepting
//! an `Arc<EpochClock>` and bumping it on commit. The dynamic graph
//! advances the clock on each writer-guard drop; future MVCC features
//! (versioned feature store, point-in-time queries) will read it on the
//! reader side.
//!
//! # Example
//! ```
//! use std::sync::Arc;
//! use aether_epoch::EpochClock;
//!
//! let clock = Arc::new(EpochClock::new());
//! let pinned = clock.current();           // reader pins
//! let _ = clock.advance();                // writer commits
//! assert!(clock.current() > pinned);
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

/// A monotonic version counter shared between writer subsystems and readers.
///
/// All operations are lock-free. `current` uses `Acquire` ordering so that
/// a reader pinning the epoch sees all writes published with the prior
/// `advance` call's `Release`.
#[derive(Debug, Default)]
pub struct EpochClock {
    counter: AtomicU64,
}

/// An opaque epoch value. Strictly ordered; advances by one per commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch(u64);

impl Epoch {
    /// The "zero" epoch — before any commits. Useful as a sentinel for
    /// "accept any committed data" reads (MVCC pinning is opt-in).
    pub const ZERO: Self = Self(0);

    /// Raw underlying counter. Exposed for telemetry / Python; production
    /// code paths should treat `Epoch` as opaque.
    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for Epoch {
    #[inline]
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl EpochClock {
    /// Create a clock at epoch zero.
    #[inline]
    pub const fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }

    /// Read the current epoch. Acquires any writes published by the most
    /// recent [`advance`](Self::advance).
    #[inline]
    pub fn current(&self) -> Epoch {
        Epoch(self.counter.load(Ordering::Acquire))
    }

    /// Increment and return the new epoch. Releases prior writes so that a
    /// reader observing the returned epoch via [`current`](Self::current)'s
    /// Acquire load is guaranteed to see them.
    #[inline]
    pub fn advance(&self) -> Epoch {
        Epoch(self.counter.fetch_add(1, Ordering::Release) + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn starts_at_zero() {
        let clock = EpochClock::new();
        assert_eq!(clock.current(), Epoch::ZERO);
    }

    #[test]
    fn advance_is_monotonic() {
        let clock = EpochClock::new();
        let e1 = clock.advance();
        let e2 = clock.advance();
        let e3 = clock.advance();
        assert!(e1 < e2 && e2 < e3);
        assert_eq!(clock.current(), e3);
    }

    #[test]
    fn concurrent_advances_unique() {
        let clock = Arc::new(EpochClock::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let c = Arc::clone(&clock);
            handles.push(thread::spawn(move || {
                (0..1000).map(|_| c.advance().as_u64()).collect::<Vec<_>>()
            }));
        }
        let mut all: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 8 * 1000);
        assert_eq!(clock.current().as_u64(), 8 * 1000);
    }
}
