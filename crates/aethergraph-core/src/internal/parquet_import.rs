//! Import graph edges from Parquet files.
//!
//! Reads `(src, dst)` columns from one or more Parquet files and builds
//! a CSR `Graph` using parallel radix scatter. Handles files with millions
//! of row groups and arbitrary column names.
//!
//! ```ignore
//! use aethergraph_core::internal::parquet_import::*;
//!
//! // Single file
//! let graph = from_parquet("edges.parquet", "src", "dst", 1_000_000)?;
//!
//! // Multiple files (partitioned dataset)
//! let graph = from_parquet_glob("edges/*.parquet", "src_id", "dst_id", 2_000_000_000)?;
//! ```

use crate::graph::{Graph, NodeId};
use anyhow::{Context, Result, bail};
use arrow_array::{RecordBatch, cast::AsArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::path::Path;
use tracing::info;

/// Read edges from a single Parquet file and build a CSR graph.
///
/// `src_col` and `dst_col` are the column names for source and destination
/// node IDs. Columns must be castable to u32 (uint32, int32, int64 all work).
///
/// `num_nodes` is the total number of nodes in the graph (determines the
/// CSR offsets array size). If you don't know it, pass a value larger than
/// the max node ID in the data.
pub fn from_parquet(
    path: impl AsRef<Path>,
    src_col: &str,
    dst_col: &str,
    num_nodes: usize,
) -> Result<Graph> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;

    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).context("read parquet metadata")?;

    let reader = builder.build().context("build parquet reader")?;

    let mut all_src: Vec<NodeId> = Vec::new();
    let mut all_dst: Vec<NodeId> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.context("read record batch")?;
        let (src, dst) = extract_edge_columns(&batch, src_col, dst_col)?;
        all_src.extend_from_slice(&src);
        all_dst.extend_from_slice(&dst);
    }

    info!(
        path = %path.display(),
        edges = all_src.len(),
        "read edges from parquet"
    );

    let edges: Vec<(NodeId, NodeId)> = all_src.into_iter().zip(all_dst).collect();

    Graph::from_edges(num_nodes, &edges, None)
}

/// Read edges from multiple Parquet files (glob pattern) and build a CSR graph.
///
/// Files are read sequentially and edges are accumulated before CSR construction.
/// The CSR construction itself is parallelized via Rayon.
pub fn from_parquet_files(
    paths: &[impl AsRef<Path>],
    src_col: &str,
    dst_col: &str,
    num_nodes: usize,
) -> Result<Graph> {
    let mut all_src: Vec<NodeId> = Vec::new();
    let mut all_dst: Vec<NodeId> = Vec::new();

    for path in paths {
        let path = path.as_ref();
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("read metadata: {}", path.display()))?;

        let reader = builder
            .build()
            .with_context(|| format!("build reader: {}", path.display()))?;

        for batch_result in reader {
            let batch = batch_result.context("read record batch")?;
            let (src, dst) = extract_edge_columns(&batch, src_col, dst_col)?;
            all_src.extend_from_slice(&src);
            all_dst.extend_from_slice(&dst);
        }

        info!(
            path = %path.display(),
            total_edges = all_src.len(),
            "accumulated edges from parquet file"
        );
    }

    info!(
        total_edges = all_src.len(),
        "building CSR from parquet data"
    );

    let edges: Vec<(NodeId, NodeId)> = all_src.into_iter().zip(all_dst).collect();

    Graph::from_edges(num_nodes, &edges, None)
}

/// Extract src and dst columns from a RecordBatch as Vec<u32>.
///
/// Handles uint32, int32, and int64 column types by casting to u32.
fn extract_edge_columns(
    batch: &RecordBatch,
    src_col: &str,
    dst_col: &str,
) -> Result<(Vec<NodeId>, Vec<NodeId>)> {
    let src_arr = batch
        .column_by_name(src_col)
        .with_context(|| format!("column '{}' not found", src_col))?;

    let dst_arr = batch
        .column_by_name(dst_col)
        .with_context(|| format!("column '{}' not found", dst_col))?;

    let src = arrow_col_to_u32(src_arr, src_col)?;
    let dst = arrow_col_to_u32(dst_arr, dst_col)?;

    Ok((src, dst))
}

/// Convert an Arrow array column to Vec<u32>.
///
/// Rejects nulls (Aethergraph NodeIds are non-nullable) and out-of-range
/// values — silent `as u32` truncation would otherwise turn `-1i64` into
/// `4_294_967_295` and `5_000_000_000i64` into a wrapped small ID, producing
/// a corrupted graph with no warning.
fn arrow_col_to_u32(col: &dyn arrow_array::Array, name: &str) -> Result<Vec<NodeId>> {
    use arrow_schema::DataType;

    if col.null_count() > 0 {
        bail!(
            "column '{}' has {} null value(s); NodeId columns must be non-nullable",
            name,
            col.null_count()
        );
    }

    match col.data_type() {
        DataType::UInt32 => {
            let arr = col.as_primitive::<arrow_array::types::UInt32Type>();
            Ok(arr.values().to_vec())
        }
        DataType::Int32 => {
            let arr = col.as_primitive::<arrow_array::types::Int32Type>();
            arr.values()
                .iter()
                .map(|&v| {
                    u32::try_from(v).map_err(|_| {
                        anyhow::anyhow!("column '{}' contains negative value {}", name, v)
                    })
                })
                .collect()
        }
        DataType::Int64 => {
            let arr = col.as_primitive::<arrow_array::types::Int64Type>();
            arr.values()
                .iter()
                .map(|&v| {
                    u32::try_from(v).map_err(|_| {
                        anyhow::anyhow!(
                            "column '{}' contains value {} outside u32 NodeId range",
                            name,
                            v
                        )
                    })
                })
                .collect()
        }
        DataType::UInt64 => {
            let arr = col.as_primitive::<arrow_array::types::UInt64Type>();
            arr.values()
                .iter()
                .map(|&v| {
                    u32::try_from(v).map_err(|_| {
                        anyhow::anyhow!(
                            "column '{}' contains value {} outside u32 NodeId range",
                            name,
                            v
                        )
                    })
                })
                .collect()
        }
        other => bail!(
            "column '{}' has unsupported type {:?} (need int32/int64/uint32/uint64)",
            name,
            other
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{RecordBatch, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    fn write_test_parquet(path: &Path, src: &[u32], dst: &[u32]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::UInt32, false),
            Field::new("dst", DataType::UInt32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(src.to_vec())),
                Arc::new(UInt32Array::from(dst.to_vec())),
            ],
        )
        .unwrap();

        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn single_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edges.parquet");

        let src = vec![0u32, 0, 1, 2, 2];
        let dst = vec![1u32, 2, 2, 0, 3];
        write_test_parquet(&path, &src, &dst);

        let graph = from_parquet(&path, "src", "dst", 4).unwrap();
        assert_eq!(graph.num_nodes(), 4);
        assert_eq!(graph.num_edges(), 5);
        assert_eq!(graph.degree(0), 2); // 0→1, 0→2
        assert_eq!(graph.degree(2), 2); // 2→0, 2→3
    }

    #[test]
    fn multiple_files() {
        let dir = tempfile::tempdir().unwrap();

        let p1 = dir.path().join("part1.parquet");
        let p2 = dir.path().join("part2.parquet");
        write_test_parquet(&p1, &[0, 0], &[1, 2]);
        write_test_parquet(&p2, &[1, 2], &[2, 0]);

        let graph = from_parquet_files(&[p1, p2], "src", "dst", 3).unwrap();
        assert_eq!(graph.num_edges(), 4);
    }

    #[test]
    fn int64_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edges_i64.parquet");

        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::Int64Array::from(vec![0i64, 1, 2])),
                Arc::new(arrow_array::Int64Array::from(vec![1i64, 2, 0])),
            ],
        )
        .unwrap();

        let file = std::fs::File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let graph = from_parquet(&path, "src", "dst", 3).unwrap();
        assert_eq!(graph.num_edges(), 3);
    }

    #[test]
    fn missing_column_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edges.parquet");
        write_test_parquet(&path, &[0], &[1]);

        let result = from_parquet(&path, "wrong_name", "dst", 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }
}
