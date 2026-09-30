//! Import graph edges from Parquet files.
//!
//! Reads `(src, dst)` columns from one or more Parquet files — only those
//! two are decoded — and builds a CSR `Graph` from the parallel arrays.
//! Handles files with millions of row groups and arbitrary column names.
//!
//! ```ignore
//! use aethergraph_core::internal::parquet_import::*;
//!
//! // Single file
//! let graph = from_parquet("edges.parquet", "src", "dst", 1_000_000)?;
//!
//! // Multiple files (partitioned dataset)
//! let graph = from_parquet_files(&paths, "src_id", "dst_id", 2_000_000_000)?;
//! ```

use crate::graph::{Graph, NodeId};
use anyhow::{Context, Result, bail};
use arrow_array::cast::AsArray;
use parquet::arrow::ProjectionMask;
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
///
/// All edges are buffered in memory before the CSR is built; there is no
/// streaming cap, so peak RAM scales with the total edge count of the file.
pub fn from_parquet(
    path: impl AsRef<Path>,
    src_col: &str,
    dst_col: &str,
    num_nodes: usize,
) -> Result<Graph> {
    from_parquet_files(&[path], src_col, dst_col, num_nodes)
}

/// Read edges from multiple Parquet files and build a CSR graph.
///
/// Files are read sequentially into parallel `src`/`dst` arrays (8 bytes
/// per edge), which feed [`Graph::from_src_dst`] directly.
///
/// All edges across every file are buffered in memory before the CSR is built;
/// there is no streaming cap, so peak RAM scales with the combined edge count.
pub fn from_parquet_files(
    paths: &[impl AsRef<Path>],
    src_col: &str,
    dst_col: &str,
    num_nodes: usize,
) -> Result<Graph> {
    let mut src: Vec<NodeId> = Vec::new();
    let mut dst: Vec<NodeId> = Vec::new();

    for path in paths {
        let path = path.as_ref();
        read_edge_columns(path, src_col, dst_col, &mut src, &mut dst)?;
        info!(
            path = %path.display(),
            total_edges = src.len(),
            "accumulated edges from parquet file"
        );
    }

    info!(total_edges = src.len(), "building CSR from parquet data");
    Graph::from_src_dst(num_nodes, &src, &dst, None)
}

/// Append one file's `src_col`/`dst_col` values to `src`/`dst`.
///
/// Only the two edge columns are decoded: the projection is resolved
/// against the file's top-level fields, so every other column (edge
/// features, timestamps, …) is never decompressed.
fn read_edge_columns(
    path: &Path,
    src_col: &str,
    dst_col: &str,
    src: &mut Vec<NodeId>,
    dst: &mut Vec<NodeId>,
) -> Result<()> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("read metadata: {}", path.display()))?;
    let field_index = |name: &str| {
        builder
            .schema()
            .index_of(name)
            .with_context(|| format!("column '{name}' not found in {}", path.display()))
    };
    let (src_root, dst_root) = (field_index(src_col)?, field_index(dst_col)?);
    let mask = ProjectionMask::roots(builder.parquet_schema(), [src_root, dst_root]);
    let reader = builder
        .with_projection(mask)
        .build()
        .with_context(|| format!("build reader: {}", path.display()))?;

    for batch in reader {
        let batch = batch.context("read record batch")?;
        let column = |name: &str| {
            batch
                .column_by_name(name)
                .with_context(|| format!("column '{name}' not found"))
        };
        src.reserve(batch.num_rows());
        dst.reserve(batch.num_rows());
        append_as_u32(column(src_col)?.as_ref(), src_col, src)?;
        append_as_u32(column(dst_col)?.as_ref(), dst_col, dst)?;
    }
    Ok(())
}

/// Append an Arrow id column to `out` as u32 NodeIds.
///
/// Rejects nulls (Aethergraph NodeIds are non-nullable) and out-of-range
/// values — silent `as u32` truncation would otherwise turn `-1i64` into
/// `4_294_967_295` and `5_000_000_000i64` into a wrapped small ID, producing
/// a corrupted graph with no warning.
fn append_as_u32(col: &dyn arrow_array::Array, name: &str, out: &mut Vec<NodeId>) -> Result<()> {
    use arrow_schema::DataType;

    if col.null_count() > 0 {
        bail!(
            "column '{}' has {} null value(s); NodeId columns must be non-nullable",
            name,
            col.null_count()
        );
    }

    fn narrow<T: Copy + std::fmt::Display>(
        values: &[T],
        name: &str,
        out: &mut Vec<NodeId>,
    ) -> Result<()>
    where
        u32: TryFrom<T>,
    {
        for &v in values {
            out.push(u32::try_from(v).map_err(|_| {
                anyhow::anyhow!("column '{name}' contains value {v} outside u32 NodeId range")
            })?);
        }
        Ok(())
    }

    match col.data_type() {
        DataType::UInt32 => {
            out.extend_from_slice(
                col.as_primitive::<arrow_array::types::UInt32Type>()
                    .values(),
            );
            Ok(())
        }
        DataType::Int32 => narrow(
            col.as_primitive::<arrow_array::types::Int32Type>().values(),
            name,
            out,
        ),
        DataType::Int64 => narrow(
            col.as_primitive::<arrow_array::types::Int64Type>().values(),
            name,
            out,
        ),
        DataType::UInt64 => narrow(
            col.as_primitive::<arrow_array::types::UInt64Type>()
                .values(),
            name,
            out,
        ),
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

    /// Extra columns — here a string column that could never parse as a
    /// node id — are projected away rather than decoded.
    #[test]
    fn projects_away_other_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("label", DataType::Utf8, false),
            Field::new("dst", DataType::Int32, false),
            Field::new("src", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::StringArray::from(vec!["a", "b", "c"])),
                Arc::new(arrow_array::Int32Array::from(vec![1, 2, 0])),
                Arc::new(UInt32Array::from(vec![0u32, 0, 2])),
            ],
        )
        .unwrap();
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let graph = from_parquet(&path, "src", "dst", 3).unwrap();
        assert_eq!(graph.neighbors(0), &[1, 2]);
        assert_eq!(graph.neighbors(2), &[0]);
    }

    #[test]
    fn rejects_negative_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("neg.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::Int64Array::from(vec![0i64, -1])),
                Arc::new(arrow_array::Int64Array::from(vec![1i64, 0])),
            ],
        )
        .unwrap();
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let err = from_parquet(&path, "src", "dst", 3)
            .unwrap_err()
            .to_string();
        assert!(err.contains("-1"), "got: {err}");
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
