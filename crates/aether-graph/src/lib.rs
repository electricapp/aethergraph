//! aether-graph — Lock-free dynamic graph with C-tree neighbor lists.
//!
//! Single-writer, multi-reader. Zero allocations on the read path.
//! Designed for GNN sampling at Reddit scale (2B nodes, 100B edges).
//!
//! Each vertex's neighbor list is a small balanced tree of sorted,
//! cache-line-sized chunks. 90% of nodes (degree < 15) are a single
//! chunk — one cache line read, identical cost to static CSR.

mod arena;
mod chunk;
mod compact;
mod ctree;
mod dirty;
mod embedding_cache;
mod graph;
mod historical;
pub mod ingest;
mod pad;
#[cfg(feature = "wal")]
pub mod wal;
mod writer;

pub use arena::{Arena, ArenaWriter, ReadGuard, RecycleStats};
pub use chunk::Chunk;
pub use compact::CompactError;
pub use ctree::{CTree, InsertResult};
pub use embedding_cache::EmbeddingCache;
pub use graph::DynamicGraph;
pub use historical::{HistoricalBatch, HistoricalSampler};
#[cfg(feature = "wal")]
pub use wal::{EdgeRecord, ReplayOutcome, WalError, WalWriter, replay as replay_wal};
pub use writer::{InsertError, Writer, WriterError};
