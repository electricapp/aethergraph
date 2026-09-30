//! Merging scattered row offsets into contiguous byte spans.
//!
//! A sampled batch names rows by node ID, so the byte ranges it wants are
//! scattered and often repeated. Whatever consumes them — a UVM prefetch,
//! a DMA descriptor list — pays per range, not per byte, so the number of
//! ranges is the cost that matters.
//!
//! Lives outside the GPU module although `gpu::uvm` is
//! its caller: the arithmetic needs no CUDA, and behind the `gpudirect`
//! feature its tests would only ever be type-checked, never run.

/// Sort `starts`, drop duplicates, and cover each row of `row_bytes` with a
/// span rounded out to `granule` boundaries and clamped to `limit`; spans
/// that touch or overlap merge. Returns `(start, len)` pairs in order.
///
/// A consumer that moves memory in fixed units pays per unit however little
/// of it a row uses: two rows in one unit are one move, and so are rows in
/// neighbouring units. `granule` 1 merges only back-to-back rows.
///
/// `starts` is sorted in place — the caller has just built it and has no
/// use for the original order.
///
/// # Panics
/// Panics if `granule` is not a power of two.
pub fn coalesce_spans(
    starts: &mut Vec<usize>,
    row_bytes: usize,
    granule: usize,
    limit: usize,
) -> Vec<(usize, usize)> {
    assert!(
        granule.is_power_of_two(),
        "granule {granule} must be a power of two"
    );
    let mask = granule - 1;
    starts.sort_unstable();
    starts.dedup();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut open: Option<(usize, usize)> = None;
    for &start in starts.iter() {
        let lo = start & !mask;
        let hi = start
            .saturating_add(row_bytes)
            .checked_next_multiple_of(granule)
            .unwrap_or(usize::MAX)
            .min(limit);
        open = match open {
            Some((run_lo, run_hi)) if lo <= run_hi => Some((run_lo, run_hi.max(hi))),
            Some((run_lo, run_hi)) => {
                spans.push((run_lo, run_hi - run_lo));
                Some((lo, hi))
            }
            None => Some((lo, hi)),
        };
    }
    if let Some((run_lo, run_hi)) = open {
        spans.push((run_lo, run_hi - run_lo));
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::coalesce_spans;

    const NO_LIMIT: usize = usize::MAX;

    /// The point of coalescing is call count: a batch covering one
    /// contiguous span must cost one range, not one per row.
    #[test]
    fn contiguous_rows_become_a_single_span() {
        let mut starts = vec![0, 512, 1024, 1536];
        assert_eq!(
            coalesce_spans(&mut starts, 512, 1, NO_LIMIT),
            vec![(0, 2048)]
        );
    }

    /// Sorting is what makes neighbours adjacent, so an unsorted batch —
    /// which is what a sampler produces — must coalesce as well as a
    /// sorted one, and repeats must not extend a span past its rows.
    #[test]
    fn order_and_duplicates_do_not_affect_the_spans() {
        let mut shuffled = vec![1536, 0, 1024, 512, 1024, 0];
        let mut sorted = vec![0, 512, 1024, 1536];
        assert_eq!(
            coalesce_spans(&mut shuffled, 512, 1, NO_LIMIT),
            coalesce_spans(&mut sorted, 512, 1, NO_LIMIT)
        );
    }

    /// At row granularity gaps split spans — merging across one would
    /// cover bytes the batch never asked for.
    #[test]
    fn gaps_split_row_spans() {
        let mut starts = vec![0, 512, 4096, 4608, 8192];
        assert_eq!(
            coalesce_spans(&mut starts, 512, 1, NO_LIMIT),
            vec![(0, 1024), (4096, 1024), (8192, 512)]
        );
    }

    /// Sparse rows that share migration units collapse to one span per run
    /// of touched units, however scattered they are inside them.
    #[test]
    fn sparse_rows_collapse_to_their_granules() {
        let mut starts = vec![100, 5_000, 70_000, 200_000];
        assert_eq!(
            coalesce_spans(&mut starts, 400, 65_536, NO_LIMIT),
            vec![(0, 131_072), (196_608, 65_536)]
        );
    }

    /// A row straddling a granule boundary claims both granules.
    #[test]
    fn straddling_rows_cover_both_granules() {
        let mut starts = vec![65_400];
        assert_eq!(
            coalesce_spans(&mut starts, 400, 65_536, NO_LIMIT),
            vec![(0, 131_072)]
        );
    }

    /// Rounding up never runs past the allocation.
    #[test]
    fn spans_stop_at_the_limit() {
        let mut starts = vec![9_600];
        assert_eq!(
            coalesce_spans(&mut starts, 400, 65_536, 10_000),
            vec![(0, 10_000)]
        );
    }

    #[test]
    fn empty_input_yields_no_spans() {
        assert!(coalesce_spans(&mut Vec::new(), 512, 4096, NO_LIMIT).is_empty());
    }

    /// Spans cover every requested row and stay within the limit.
    #[test]
    fn spans_cover_every_row() {
        let limit = 64 * 256;
        let mut starts: Vec<usize> = (0..64).map(|i| (i * 7 % 64) * 256).collect();
        let rows = starts.clone();
        let spans = coalesce_spans(&mut starts, 256, 4096, limit);
        for row in rows {
            assert!(
                spans.iter().any(|&(s, l)| s <= row && row + 256 <= s + l),
                "row {row} uncovered"
            );
        }
        assert!(spans.iter().all(|&(s, l)| s + l <= limit));
    }
}
