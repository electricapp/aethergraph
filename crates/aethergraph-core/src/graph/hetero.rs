//! Heterogeneous graph representation for multi-relational GNN sampling.
//!
//! A `HeteroGraph` holds multiple CSR matrices (one per edge type) plus type metadata.
//! Each edge type's CSR uses per-type local node IDs: source IDs are local to the
//! source node type, destination IDs are local to the destination node type.

use crate::graph::csr::MAX_NODES;
use crate::graph::{Graph, GraphValidationMode, NodeId};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Integer identifier for a node type.
pub type NodeTypeId = u8;

/// Integer identifier for an edge type.
pub type EdgeTypeId = u8;

/// Metadata for a node type.
#[derive(Debug, Clone)]
pub struct NodeTypeMeta {
    pub name: String,
    pub count: usize,
}

/// Metadata for an edge type (source_type, relation, destination_type).
#[derive(Debug, Clone)]
pub struct EdgeTypeMeta {
    pub src_type: NodeTypeId,
    pub relation: String,
    pub dst_type: NodeTypeId,
}

/// A heterogeneous graph: multiple CSR matrices indexed by edge type.
///
/// Each edge type has its own CSR graph using per-type local node IDs.
/// For example, in a ("user", "votes", "post") edge type, source IDs
/// are local to the "user" type and destination IDs are local to "post".
///
/// Construction proves, for every edge type: monotone CSR offsets, no
/// edges from rows at or past the source type's count, and every
/// destination below the destination type's count. Consumers index
/// per-type arrays by destination with no range check.
#[derive(Debug, Clone)]
pub struct HeteroGraph {
    node_types: Vec<NodeTypeMeta>,
    node_type_index: HashMap<String, NodeTypeId>,
    edge_types: Vec<EdgeTypeMeta>,
    edge_type_index: HashMap<(String, String, String), EdgeTypeId>,
    /// Per-edge-type CSR graph. csr_graphs[edge_type_id] is the CSR for that edge type.
    csr_graphs: Vec<Graph>,
    /// Per-edge-type incoming adjacency, built on first use.
    in_adjacency: OnceLock<Vec<InAdjacency>>,
}

/// Incoming adjacency of one edge type: for each destination node, the
/// source nodes with an edge into it, in stored edge order.
///
/// Built by [`HeteroGraph::in_adjacency`]. Rows past the source type's count
/// and destinations past the destination type's count are dropped while
/// building, so every stored source is in range for its node type.
#[derive(Debug, Clone, Default)]
pub struct InAdjacency {
    offsets: Vec<u64>,
    sources: Vec<NodeId>,
}

impl InAdjacency {
    /// Transpose `csr` (rows are sources) into per-destination source lists.
    fn transpose(csr: &Graph, num_src: usize, num_dst: usize) -> Self {
        let view = csr.csr_view();
        let rows = view.num_nodes().min(num_src) as NodeId;

        let mut offsets = vec![0u64; num_dst + 1];
        for s in 0..rows {
            for &d in view.neighbors(s) {
                if (d as usize) < num_dst {
                    offsets[d as usize + 1] += 1;
                }
            }
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }

        let mut cursor = offsets[..num_dst].to_vec();
        let mut sources = vec![0 as NodeId; offsets[num_dst] as usize];
        for s in 0..rows {
            for &d in view.neighbors(s) {
                if let Some(slot) = cursor.get_mut(d as usize) {
                    sources[*slot as usize] = s;
                    *slot += 1;
                }
            }
        }
        Self { offsets, sources }
    }

    /// Sources with an edge into `dst`; empty when `dst` is out of range.
    #[inline]
    pub fn sources(&self, dst: NodeId) -> &[NodeId] {
        let d = dst as usize;
        match (self.offsets.get(d), self.offsets.get(d + 1)) {
            (Some(&start), Some(&end)) => &self.sources[start as usize..end as usize],
            _ => &[],
        }
    }

    /// Offset array (`num_dst + 1` entries), for prefetching.
    #[inline]
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }
}

/// Errors returned when constructing a [`HeteroGraph`] via [`HeteroGraph::try_from_parts`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeteroBuildError {
    TooManyNodeTypes(usize),
    TooManyEdgeTypes(usize),
    UnknownNodeType(String),
    /// A node type name appeared more than once.
    DuplicateNodeType(String),
    /// An edge type (src, relation, dst) triple appeared more than once.
    DuplicateEdgeType(String, String, String),
    /// A node type's count exceeds the `u32` node-id space.
    NodeTypeTooLarge(String, usize),
    /// An edge type's CSR does not fit its endpoint types.
    InvalidEdgeType {
        src: String,
        rel: String,
        dst: String,
        reason: String,
    },
}

impl std::fmt::Display for HeteroBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyNodeTypes(n) => write!(f, "too many node types ({n}, max 255)"),
            Self::TooManyEdgeTypes(n) => write!(f, "too many edge types ({n}, max 255)"),
            Self::UnknownNodeType(name) => write!(f, "unknown node type: {name}"),
            Self::DuplicateNodeType(name) => write!(f, "duplicate node type: {name}"),
            Self::DuplicateEdgeType(src, rel, dst) => {
                write!(f, "duplicate edge type: ({src}, {rel}, {dst})")
            }
            Self::NodeTypeTooLarge(name, count) => write!(
                f,
                "node type {name} has {count} nodes, over the u32 node-id limit {MAX_NODES}"
            ),
            Self::InvalidEdgeType {
                src,
                rel,
                dst,
                reason,
            } => write!(f, "edge type ({src}, {rel}, {dst}): {reason}"),
        }
    }
}

/// Prove `csr` fits an edge type from `src_count` sources to `dst_count`
/// destinations.
///
/// Rows past `src_count` are allowed only when empty, so CSRs built over a
/// shared, larger row count stay valid. With monotone offsets that is one
/// comparison. Destinations are scanned unless the CSR's own `Full` proof
/// already bounds them.
fn check_edge_type(csr: &Graph, src_count: usize, dst_count: usize) -> Result<(), String> {
    csr.validate_with_mode(GraphValidationMode::OffsetsOnly)
        .map_err(|e| e.to_string())?;
    if csr.num_nodes() > src_count {
        let tail = csr.edge_offset(src_count as NodeId);
        if tail != csr.num_edges() as u64 {
            return Err(format!(
                "{} edges leave source ids at or past the source type's count {src_count}",
                csr.num_edges() as u64 - tail
            ));
        }
    }
    let bounded = csr.validated() == GraphValidationMode::Full && csr.num_dst_nodes() <= dst_count;
    if !bounded {
        let edges = csr.edges();
        let bad = if edges.len() > 100_000 {
            edges
                .par_iter()
                .position_any(|&d| (d as usize) >= dst_count)
        } else {
            edges.iter().position(|&d| (d as usize) >= dst_count)
        };
        if let Some(i) = bad {
            return Err(format!(
                "destination {} at edge {i} is not below the destination type's count {dst_count}",
                edges[i]
            ));
        }
    }
    Ok(())
}

impl std::error::Error for HeteroBuildError {}

impl HeteroGraph {
    /// Build a heterogeneous graph from node types and edge types.
    ///
    /// # Arguments
    /// * `node_types` - Vec of (name, count) pairs for each node type
    /// * `edge_types` - Vec of (src_type_name, relation, dst_type_name, csr_graph) tuples
    ///
    /// Returns `Err` if there are too many types, a name repeats, an edge
    /// type references an unknown node type, or an edge type's CSR does not
    /// fit its endpoint types (see [`HeteroGraph`]'s invariants). For tests
    /// / one-shot scripts, see [`Self::from_parts`].
    pub fn try_from_parts(
        node_types: Vec<(String, usize)>,
        edge_types: Vec<(String, String, String, Graph)>,
    ) -> Result<Self, HeteroBuildError> {
        if node_types.len() > 255 {
            return Err(HeteroBuildError::TooManyNodeTypes(node_types.len()));
        }
        if edge_types.len() > 255 {
            return Err(HeteroBuildError::TooManyEdgeTypes(edge_types.len()));
        }

        let mut node_type_index = HashMap::with_capacity(node_types.len());
        let mut node_metas = Vec::with_capacity(node_types.len());

        for (i, (name, count)) in node_types.into_iter().enumerate() {
            if count > MAX_NODES {
                return Err(HeteroBuildError::NodeTypeTooLarge(name, count));
            }
            if node_type_index
                .insert(name.clone(), i as NodeTypeId)
                .is_some()
            {
                return Err(HeteroBuildError::DuplicateNodeType(name));
            }
            node_metas.push(NodeTypeMeta { name, count });
        }

        let mut edge_type_index = HashMap::with_capacity(edge_types.len());
        let mut edge_metas = Vec::with_capacity(edge_types.len());
        let mut csr_graphs = Vec::with_capacity(edge_types.len());

        for (i, (src, rel, dst, graph)) in edge_types.into_iter().enumerate() {
            let src_id = *node_type_index
                .get(&src)
                .ok_or_else(|| HeteroBuildError::UnknownNodeType(src.clone()))?;
            let dst_id = *node_type_index
                .get(&dst)
                .ok_or_else(|| HeteroBuildError::UnknownNodeType(dst.clone()))?;

            if edge_type_index
                .insert((src.clone(), rel.clone(), dst.clone()), i as EdgeTypeId)
                .is_some()
            {
                return Err(HeteroBuildError::DuplicateEdgeType(src, rel, dst));
            }
            let (src_count, dst_count) = (
                node_metas[src_id as usize].count,
                node_metas[dst_id as usize].count,
            );
            if let Err(reason) = check_edge_type(&graph, src_count, dst_count) {
                return Err(HeteroBuildError::InvalidEdgeType {
                    src,
                    rel,
                    dst,
                    reason,
                });
            }
            edge_metas.push(EdgeTypeMeta {
                src_type: src_id,
                relation: rel,
                dst_type: dst_id,
            });
            csr_graphs.push(graph);
        }

        Ok(Self {
            node_types: node_metas,
            node_type_index,
            edge_types: edge_metas,
            edge_type_index,
            csr_graphs,
            in_adjacency: OnceLock::new(),
        })
    }

    /// Build a heterogeneous graph, panicking on validation errors.
    /// Convenience wrapper over [`Self::try_from_parts`] for tests and demos.
    pub fn from_parts(
        node_types: Vec<(String, usize)>,
        edge_types: Vec<(String, String, String, Graph)>,
    ) -> Self {
        Self::try_from_parts(node_types, edge_types).expect("HeteroGraph::from_parts failed")
    }

    /// Look up a node type ID by name.
    #[inline]
    pub fn node_type_id(&self, name: &str) -> Option<NodeTypeId> {
        self.node_type_index.get(name).copied()
    }

    /// Look up an edge type ID by (source_type, relation, dest_type) names.
    #[inline]
    pub fn edge_type_id(&self, src: &str, rel: &str, dst: &str) -> Option<EdgeTypeId> {
        self.edge_type_index
            .get(&(src.to_owned(), rel.to_owned(), dst.to_owned()))
            .copied()
    }

    /// Get metadata for a node type.
    #[inline]
    pub fn node_type_meta(&self, id: NodeTypeId) -> &NodeTypeMeta {
        &self.node_types[id as usize]
    }

    /// Get metadata for an edge type.
    #[inline]
    pub fn edge_type_meta(&self, id: EdgeTypeId) -> &EdgeTypeMeta {
        &self.edge_types[id as usize]
    }

    /// Get the CSR graph for an edge type.
    #[inline]
    pub fn csr(&self, edge_type: EdgeTypeId) -> &Graph {
        &self.csr_graphs[edge_type as usize]
    }

    /// Number of nodes of a given type.
    #[inline]
    pub fn num_nodes(&self, node_type: NodeTypeId) -> usize {
        self.node_types[node_type as usize].count
    }

    /// Number of edges of a given edge type.
    #[inline]
    pub fn num_edges(&self, edge_type: EdgeTypeId) -> usize {
        self.csr_graphs[edge_type as usize].num_edges()
    }

    /// Total number of nodes across all types.
    pub fn total_nodes(&self) -> usize {
        self.node_types.iter().map(|nt| nt.count).sum()
    }

    /// Total number of edges across all edge types.
    pub fn total_edges(&self) -> usize {
        self.csr_graphs
            .iter()
            .map(super::csr::Graph::num_edges)
            .sum()
    }

    /// Number of node types.
    #[inline]
    pub fn node_type_count(&self) -> usize {
        self.node_types.len()
    }

    /// Number of edge types.
    #[inline]
    pub fn edge_type_count(&self) -> usize {
        self.edge_types.len()
    }

    /// Names of all node types.
    pub fn node_type_names(&self) -> Vec<&str> {
        self.node_types.iter().map(|nt| nt.name.as_str()).collect()
    }

    /// Names of all edge types as (src, relation, dst) triples.
    pub fn edge_type_names(&self) -> Vec<(&str, &str, &str)> {
        self.edge_types
            .iter()
            .map(|et| {
                (
                    self.node_types[et.src_type as usize].name.as_str(),
                    et.relation.as_str(),
                    self.node_types[et.dst_type as usize].name.as_str(),
                )
            })
            .collect()
    }

    /// Returns all edge type IDs where the given node type is the source.
    pub fn edge_types_for_src(&self, src_type: NodeTypeId) -> Vec<EdgeTypeId> {
        self.edge_types
            .iter()
            .enumerate()
            .filter(|(_, et)| et.src_type == src_type)
            .map(|(i, _)| i as EdgeTypeId)
            .collect()
    }

    /// Returns all edge type IDs where the given node type is the destination.
    pub fn edge_types_for_dst(&self, dst_type: NodeTypeId) -> Vec<EdgeTypeId> {
        self.edge_types
            .iter()
            .enumerate()
            .filter(|(_, et)| et.dst_type == dst_type)
            .map(|(i, _)| i as EdgeTypeId)
            .collect()
    }

    /// Incoming adjacency of every edge type, indexed by edge type ID.
    ///
    /// Built on the first call — one O(edges) transpose per edge type, in
    /// parallel — and shared by every later caller of this graph.
    pub fn in_adjacency(&self) -> &[InAdjacency] {
        self.in_adjacency.get_or_init(|| {
            self.edge_types
                .par_iter()
                .zip(self.csr_graphs.par_iter())
                .map(|(meta, csr)| {
                    InAdjacency::transpose(
                        csr,
                        self.num_nodes(meta.src_type),
                        self.num_nodes(meta.dst_type),
                    )
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Reddit-like heterogeneous graph for testing.
    ///
    /// Node types: user(100), post(200), comment(300), subreddit(20)
    /// Edge types:
    ///   - (user, votes, post): users 0..50 each vote on posts 0..3
    ///   - (user, writes, comment): users 0..30 each write comments 0..2
    ///   - (post, belongs_to, subreddit): posts 0..100 each belong to subreddit i%20
    fn build_reddit_graph() -> HeteroGraph {
        // user -> votes -> post
        let mut votes_edges = Vec::new();
        for user in 0u32..50 {
            for post in 0u32..4 {
                votes_edges.push((user, post));
            }
        }
        let votes_csr = Graph::from_edges(100, &votes_edges, None).unwrap();

        // user -> writes -> comment
        let mut writes_edges = Vec::new();
        for user in 0u32..30 {
            for comment in 0u32..3 {
                writes_edges.push((user, comment));
            }
        }
        let writes_csr = Graph::from_edges(100, &writes_edges, None).unwrap();

        // post -> belongs_to -> subreddit
        let mut belongs_edges = Vec::new();
        for post in 0u32..100 {
            belongs_edges.push((post, post % 20));
        }
        let belongs_csr = Graph::from_edges(200, &belongs_edges, None).unwrap();

        HeteroGraph::from_parts(
            vec![
                ("user".into(), 100),
                ("post".into(), 200),
                ("comment".into(), 300),
                ("subreddit".into(), 20),
            ],
            vec![
                ("user".into(), "votes".into(), "post".into(), votes_csr),
                ("user".into(), "writes".into(), "comment".into(), writes_csr),
                (
                    "post".into(),
                    "belongs_to".into(),
                    "subreddit".into(),
                    belongs_csr,
                ),
            ],
        )
    }

    #[test]
    fn test_node_type_lookups() {
        let g = build_reddit_graph();

        assert_eq!(g.node_type_id("user"), Some(0));
        assert_eq!(g.node_type_id("post"), Some(1));
        assert_eq!(g.node_type_id("comment"), Some(2));
        assert_eq!(g.node_type_id("subreddit"), Some(3));
        assert_eq!(g.node_type_id("nonexistent"), None);
    }

    #[test]
    fn test_edge_type_lookups() {
        let g = build_reddit_graph();

        assert_eq!(g.edge_type_id("user", "votes", "post"), Some(0));
        assert_eq!(g.edge_type_id("user", "writes", "comment"), Some(1));
        assert_eq!(g.edge_type_id("post", "belongs_to", "subreddit"), Some(2));
        assert_eq!(g.edge_type_id("user", "votes", "comment"), None);
    }

    #[test]
    fn test_node_counts() {
        let g = build_reddit_graph();

        assert_eq!(g.num_nodes(0), 100); // user
        assert_eq!(g.num_nodes(1), 200); // post
        assert_eq!(g.num_nodes(2), 300); // comment
        assert_eq!(g.num_nodes(3), 20); // subreddit
        assert_eq!(g.total_nodes(), 620);
    }

    #[test]
    fn test_edge_counts() {
        let g = build_reddit_graph();

        // 50 users * 4 votes each = 200
        assert_eq!(g.num_edges(0), 200);
        // 30 users * 3 writes each = 90
        assert_eq!(g.num_edges(1), 90);
        // 100 posts * 1 belongs_to each = 100
        assert_eq!(g.num_edges(2), 100);
        assert_eq!(g.total_edges(), 390);
    }

    #[test]
    fn test_type_counts() {
        let g = build_reddit_graph();

        assert_eq!(g.node_type_count(), 4);
        assert_eq!(g.edge_type_count(), 3);
    }

    #[test]
    fn test_node_type_names() {
        let g = build_reddit_graph();
        let names = g.node_type_names();
        assert_eq!(names, vec!["user", "post", "comment", "subreddit"]);
    }

    #[test]
    fn test_edge_type_names() {
        let g = build_reddit_graph();
        let names = g.edge_type_names();
        assert_eq!(
            names,
            vec![
                ("user", "votes", "post"),
                ("user", "writes", "comment"),
                ("post", "belongs_to", "subreddit"),
            ]
        );
    }

    #[test]
    fn test_edge_types_for_src() {
        let g = build_reddit_graph();

        // user is src for "votes" (0) and "writes" (1)
        let user_edges = g.edge_types_for_src(0);
        assert_eq!(user_edges, vec![0, 1]);

        // post is src for "belongs_to" (2)
        let post_edges = g.edge_types_for_src(1);
        assert_eq!(post_edges, vec![2]);

        // comment is not a source for any edge type
        let comment_edges = g.edge_types_for_src(2);
        assert!(comment_edges.is_empty());

        // subreddit is not a source for any edge type
        let sub_edges = g.edge_types_for_src(3);
        assert!(sub_edges.is_empty());
    }

    #[test]
    fn test_edge_type_meta() {
        let g = build_reddit_graph();

        let votes_meta = g.edge_type_meta(0);
        assert_eq!(votes_meta.src_type, 0); // user
        assert_eq!(votes_meta.relation, "votes");
        assert_eq!(votes_meta.dst_type, 1); // post

        let writes_meta = g.edge_type_meta(1);
        assert_eq!(writes_meta.src_type, 0); // user
        assert_eq!(writes_meta.relation, "writes");
        assert_eq!(writes_meta.dst_type, 2); // comment
    }

    #[test]
    fn test_rejects_destination_past_its_type() {
        // Built over a shared 100-node id space, but "tag" has only 5 nodes.
        let csr = Graph::from_edges(100, &[(0, 3), (1, 7)], None).unwrap();
        let err = HeteroGraph::try_from_parts(
            vec![("item".into(), 100), ("tag".into(), 5)],
            vec![("item".into(), "tagged".into(), "tag".into(), csr)],
        )
        .unwrap_err();
        assert!(
            matches!(err, HeteroBuildError::InvalidEdgeType { .. }),
            "got {err}"
        );
        assert!(err.to_string().contains("destination 7"), "got {err}");
    }

    #[test]
    fn test_rejects_edges_from_rows_past_the_source_type() {
        // Rows past the source count are fine while empty...
        let empty_tail = Graph::from_edges(10, &[(0, 1)], None).unwrap();
        assert!(
            HeteroGraph::try_from_parts(
                vec![("a".into(), 2), ("b".into(), 10)],
                vec![("a".into(), "r".into(), "b".into(), empty_tail)],
            )
            .is_ok()
        );
        // ...and rejected once a row past it holds an edge.
        let live_tail = Graph::from_edges(10, &[(0, 1), (5, 2)], None).unwrap();
        let err = HeteroGraph::try_from_parts(
            vec![("a".into(), 2), ("b".into(), 10)],
            vec![("a".into(), "r".into(), "b".into(), live_tail)],
        )
        .unwrap_err();
        assert!(err.to_string().contains("source ids"), "got {err}");
    }

    #[test]
    fn test_accepts_bipartite_csr() {
        let csr =
            Graph::from_bipartite_src_dst(2, 1_000_000, &[0, 1], &[999_999, 3], None).unwrap();
        let g = HeteroGraph::try_from_parts(
            vec![("user".into(), 2), ("item".into(), 1_000_000)],
            vec![("user".into(), "buys".into(), "item".into(), csr)],
        )
        .unwrap();
        assert_eq!(g.csr(0).neighbors(0), &[999_999]);
        assert_eq!(g.csr(0).offsets().len(), 3);
    }

    #[test]
    fn test_rejects_node_type_past_id_space() {
        let err =
            HeteroGraph::try_from_parts(vec![("huge".into(), MAX_NODES + 1)], vec![]).unwrap_err();
        assert!(matches!(err, HeteroBuildError::NodeTypeTooLarge(..)));
    }

    #[test]
    fn test_csr_access() {
        let g = build_reddit_graph();

        // User 0 votes on posts 0..3
        let votes_csr = g.csr(0);
        let neighbors = votes_csr.neighbors(0);
        assert_eq!(neighbors.len(), 4);
        assert_eq!(neighbors, &[0u32, 1, 2, 3]);

        // User 50 has no votes (only users 0..49 vote)
        assert_eq!(votes_csr.neighbors(50).len(), 0);
    }
}
