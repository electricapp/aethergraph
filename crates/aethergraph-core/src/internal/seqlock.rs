//! Portable seqlock acceptance predicates shared with device readers.

/// Returns whether a head/tail pair describes a published, stable row.
#[must_use]
pub const fn cpu_seqlock_accept(head: u64, tail: u64) -> bool {
    head == tail && head != 0 && head & 1 == 0
}

#[cfg(test)]
mod tests {
    use super::cpu_seqlock_accept;

    #[test]
    fn accepts_only_published_even_versions() {
        assert!(cpu_seqlock_accept(2, 2));
        assert!(!cpu_seqlock_accept(0, 0));
        assert!(!cpu_seqlock_accept(3, 3));
        assert!(!cpu_seqlock_accept(2, 4));
    }
}
