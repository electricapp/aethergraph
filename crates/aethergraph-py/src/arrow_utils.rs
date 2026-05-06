use aethergraph_core::SampledSubgraph;
use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use pyo3::prelude::*;
use std::sync::Arc;

use crate::error::arrow_conversion_error;

/// Converts a SampledSubgraph to an Arrow RecordBatch for zero-copy data transfer.
///
/// The RecordBatch contains three arrays:
/// - "edge_src": Source node IDs for all edges (UInt32)
/// - "edge_dst": Destination node IDs for all edges (UInt32)
/// - "nodes": All unique nodes in the subgraph (UInt32)
///
/// This enables zero-copy transfer to Python via PyArrow and then to PyTorch tensors.
pub fn subgraph_to_arrow(subgraph: &SampledSubgraph) -> PyResult<RecordBatch> {
    // Create Arrow arrays directly from SOA format (no unzip needed!)
    let src_array = UInt32Array::from(subgraph.edge_src.clone());
    let dst_array = UInt32Array::from(subgraph.edge_dst.clone());
    let nodes_array = UInt32Array::from(subgraph.nodes.clone());

    // Define schema
    let schema = Schema::new(vec![
        Field::new("edge_src", DataType::UInt32, false),
        Field::new("edge_dst", DataType::UInt32, false),
        Field::new("nodes", DataType::UInt32, false),
    ]);

    // Create RecordBatch
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(src_array) as ArrayRef,
            Arc::new(dst_array) as ArrayRef,
            Arc::new(nodes_array) as ArrayRef,
        ],
    )
    .map_err(|e| arrow_conversion_error(format!("Arrow conversion failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aethergraph_core::SampledSubgraph;

    #[test]
    fn test_subgraph_to_arrow() {
        let subgraph = SampledSubgraph::from_parts(
            vec![0, 1, 2, 3],
            vec![0, 0, 1],
            vec![1, 2, 3],
            vec![0, 1, 2],
            vec![0],
            vec![3],
            vec![3],
        );

        let record_batch = subgraph_to_arrow(&subgraph).unwrap();

        assert_eq!(record_batch.num_columns(), 3);
        assert_eq!(record_batch.num_rows(), 4); // Number of nodes

        // Verify column names
        let schema = record_batch.schema();
        assert_eq!(schema.field(0).name(), "edge_src");
        assert_eq!(schema.field(1).name(), "edge_dst");
        assert_eq!(schema.field(2).name(), "nodes");
    }
}
