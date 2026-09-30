use aethergraph_core::SampledSubgraph;
use arrow_array::{ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use pyo3::prelude::*;
use std::sync::Arc;

use crate::error::arrow_conversion_error;

/// Conversion result: two `RecordBatch`es — one indexed by edge, one by node.
///
/// Arrow `RecordBatch` requires all columns to have the same length, but a
/// `SampledSubgraph` mixes edge-indexed arrays (length E) with node-indexed
/// arrays (length N) and seed-indexed arrays (length S). Forcing them into a
/// single record batch is unsound (it would fail `RecordBatch::try_new`'s
/// equal-length check). We split into per-length batches and let consumers
/// pick the one they want.
#[derive(Debug)]
pub struct SubgraphRecordBatches {
    /// Length E. Columns: `edge_src`, `edge_dst`, `edge_id`.
    pub edges: RecordBatch,
    /// Length N. Columns: `nodes`.
    pub nodes: RecordBatch,
    /// Length S. Columns: `seeds`.
    pub seeds: RecordBatch,
}

/// Converts a `SampledSubgraph` to a triple of Arrow `RecordBatch`es. Pure
/// Rust — no Python interaction.
///
/// `SampledSubgraph` stores edges in **SOA** (struct-of-arrays) form — separate
/// `Vec<u32>` for src and dst rather than `Vec<(u32, u32)>` AOS pairs. We pass
/// the buffers straight through (`UInt32Array::from(Vec<u32>)` moves the
/// buffer into an Arrow `ScalarBuffer` with no element-wise copy).
///
/// The function is split from the PyO3 wrapper so unit tests can exercise the
/// conversion without initializing a Python interpreter — important because
/// this crate is built as a `cdylib` with `abi3-py38` and has no
/// auto-initialize path. The PyO3 wrapper is [`subgraph_into_arrow`].
pub fn subgraph_into_record_batches(
    subgraph: SampledSubgraph,
) -> Result<SubgraphRecordBatches, ArrowError> {
    // Edge ids are CSR positions, which pass u32 on graphs over 4B edges.
    let edges_schema = Schema::new(vec![
        Field::new("edge_src", DataType::UInt32, false),
        Field::new("edge_dst", DataType::UInt32, false),
        Field::new("edge_id", DataType::UInt64, false),
    ]);
    let edges = RecordBatch::try_new(
        Arc::new(edges_schema),
        vec![
            Arc::new(UInt32Array::from(subgraph.edge_src)) as ArrayRef,
            Arc::new(UInt32Array::from(subgraph.edge_dst)) as ArrayRef,
            Arc::new(UInt64Array::from(subgraph.edge_ids)) as ArrayRef,
        ],
    )?;

    let nodes_schema = Schema::new(vec![Field::new("nodes", DataType::UInt32, false)]);
    let nodes = RecordBatch::try_new(
        Arc::new(nodes_schema),
        vec![Arc::new(UInt32Array::from(subgraph.nodes)) as ArrayRef],
    )?;

    let seeds_schema = Schema::new(vec![Field::new("seeds", DataType::UInt32, false)]);
    let seeds = RecordBatch::try_new(
        Arc::new(seeds_schema),
        vec![Arc::new(UInt32Array::from(subgraph.seeds)) as ArrayRef],
    )?;

    Ok(SubgraphRecordBatches {
        edges,
        nodes,
        seeds,
    })
}

/// PyO3 wrapper around [`subgraph_into_record_batches`] that maps Arrow errors
/// to the project's [`crate::error::ArrowConversionError`] Python exception.
pub fn subgraph_into_arrow(subgraph: SampledSubgraph) -> PyResult<SubgraphRecordBatches> {
    subgraph_into_record_batches(subgraph)
        .map_err(|e| arrow_conversion_error(format!("Arrow conversion failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aethergraph_core::SampledSubgraph;

    #[test]
    fn record_batch_layout_matches_subgraph() {
        let subgraph = SampledSubgraph::from_parts(
            vec![0, 1, 2, 3], // nodes
            vec![0, 0, 1],    // edge_src
            vec![1, 2, 3],    // edge_dst
            vec![0, 1, 2],    // edge_ids
            vec![0],          // seeds
            vec![3],          // num_sampled_nodes
            vec![3],          // num_sampled_edges
        );

        // Pure-Rust path — no Python required.
        let batches = subgraph_into_record_batches(subgraph).unwrap();

        // Edges batch has 3 rows × 3 columns (edge_src, edge_dst, edge_id).
        assert_eq!(batches.edges.num_columns(), 3);
        assert_eq!(batches.edges.num_rows(), 3);
        assert_eq!(batches.edges.schema().field(0).name(), "edge_src");
        assert_eq!(batches.edges.schema().field(1).name(), "edge_dst");
        assert_eq!(batches.edges.schema().field(2).name(), "edge_id");

        // Nodes batch has 4 rows × 1 column.
        assert_eq!(batches.nodes.num_columns(), 1);
        assert_eq!(batches.nodes.num_rows(), 4);
        assert_eq!(batches.nodes.schema().field(0).name(), "nodes");

        // Seeds batch has 1 row × 1 column.
        assert_eq!(batches.seeds.num_columns(), 1);
        assert_eq!(batches.seeds.num_rows(), 1);
        assert_eq!(batches.seeds.schema().field(0).name(), "seeds");
    }

    #[test]
    fn empty_subgraph_produces_zero_length_batches() {
        let subgraph =
            SampledSubgraph::from_parts(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        let batches = subgraph_into_record_batches(subgraph).unwrap();
        assert_eq!(batches.edges.num_rows(), 0);
        assert_eq!(batches.nodes.num_rows(), 0);
        assert_eq!(batches.seeds.num_rows(), 0);
    }

    #[test]
    fn keeps_edge_ids_past_u32() {
        let big = u64::from(u32::MAX) + 1;
        let subgraph = SampledSubgraph::from_parts(
            vec![0, 1],
            vec![0],
            vec![1],
            vec![big],
            vec![0],
            vec![2],
            vec![1],
        );
        let batches = subgraph_into_record_batches(subgraph).unwrap();
        let ids = batches
            .edges
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("edge_id is UInt64");
        assert_eq!(ids.values().as_ref(), &[big]);
    }
}
