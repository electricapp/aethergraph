//! Merging scattered row offsets into contiguous byte runs.
//!
//! A sampled batch names rows by node ID, so the byte ranges it wants are
//! scattered and often repeated. Whatever consumes them — a UVM prefetch,
//! a DMA descriptor list — pays per range, not per byte, so the number of
//! ranges is the cost that matters.
//!
//! Lives outside the GPU module although [`gpu::uvm`](crate::gpu::uvm) is
//! its caller: the arithmetic needs no CUDA, and behind the `gpudirect`
//! feature its tests would only ever be type-checked, never run.

/// Sort `starts`, drop duplicates, and merge them into `(start, len)` runs
/// of back-to-back rows of `row_bytes` each.
///
/// `starts` is sorted in place — the caller has just built it and has no
/// use for the original order.
pub fn coalesce_runs(starts: &mut Vec<usize>, row_bytes: usize) -> Vec<(usize, usize)> {
    starts.sort_unstable();
    starts.dedup();
    let mut runs = Vec::new();
    let Some(&first) = starts.first() else {
        return runs;
    };
    let mut run_start = first;
    let mut run_end = first + row_bytes;
    for &start in &starts[1..] {
        if start == run_end {
            run_end += row_bytes;
            continue;
        }
        runs.push((run_start, run_end - run_start));
        run_start = start;
        run_end = start + row_bytes;
    }
    runs.push((run_start, run_end - run_start));
    runs
}

#[cfg(test)]
mod tests {
    use super::coalesce_runs;

    /// The point of coalescing is call count: a batch covering one
    /// contiguous span must cost one range, not one per row.
    #[test]
    fn contiguous_rows_become_a_single_run() {
        let mut starts = vec![0, 512, 1024, 1536];
        assert_eq!(coalesce_runs(&mut starts, 512), vec![(0, 2048)]);
    }

    /// Sorting is what makes neighbours adjacent, so an unsorted batch —
    /// which is what a sampler produces — must coalesce as well as a
    /// sorted one, and repeats must not extend a run past its rows.
    #[test]
    fn order_and_duplicates_do_not_affect_the_runs() {
        let mut shuffled = vec![1536, 0, 1024, 512, 1024, 0];
        let mut sorted = vec![0, 512, 1024, 1536];
        assert_eq!(
            coalesce_runs(&mut shuffled, 512),
            coalesce_runs(&mut sorted, 512)
        );
    }

    /// Gaps split runs — merging across one would cover bytes the batch
    /// never asked for.
    #[test]
    fn gaps_split_runs() {
        let mut starts = vec![0, 512, 4096, 4608, 8192];
        assert_eq!(
            coalesce_runs(&mut starts, 512),
            vec![(0, 1024), (4096, 1024), (8192, 512)]
        );
    }

    #[test]
    fn empty_input_yields_no_runs() {
        assert!(coalesce_runs(&mut Vec::new(), 512).is_empty());
    }

    /// Coalescing must not lose coverage: the runs' total length equals
    /// the distinct row count times the row size.
    #[test]
    fn runs_cover_exactly_the_distinct_rows() {
        let mut starts: Vec<usize> = (0..64).map(|i| (i * 7 % 64) * 256).collect();
        let distinct = {
            let mut s = starts.clone();
            s.sort_unstable();
            s.dedup();
            s.len()
        };
        let covered: usize = coalesce_runs(&mut starts, 256)
            .iter()
            .map(|&(_, len)| len)
            .sum();
        assert_eq!(covered, distinct * 256);
    }

    /// A single row is still a run.
    #[test]
    fn one_row_is_one_run() {
        let mut starts = vec![4096];
        assert_eq!(coalesce_runs(&mut starts, 512), vec![(4096, 512)]);
    }
}
