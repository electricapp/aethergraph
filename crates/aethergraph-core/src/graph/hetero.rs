//! Heterogeneous graph representation for multi-relational GNN sampling.
//!
//! A `HeteroGraph` holds multiple CSR matrices (one per edge type) plus type metadata.
//! Each edge type's CSR uses per-type local node IDs: source IDs are local to the
//! source node type, destination IDs are local to the destination node type.

use crate::graph::Graph;
use std::collections::HashMap;

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
#[derive(Debug, Clone)]
pub struct HeteroGraph {
    node_types: Vec<NodeTypeMeta>,
    node_type_index: HashMap<String, NodeTypeId>,
    edge_types: Vec<EdgeTypeMeta>,
    edge_type_index: HashMap<(String, String, String), EdgeTypeId>,
    /// Per-edge-type CSR graph. csr_graphs[edge_type_id] is the CSR for that edge type.
    csr_graphs: Vec<Graph>,
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
        }
    }
}

impl std::error::Error for HeteroBuildError {}

impl HeteroGraph {
    /// Build a heterogeneous graph from node types and edge types.
    ///
    /// # Arguments
    /// * `node_types` - Vec of (name, count) pairs for each node type
    /// * `edge_types` - Vec of (src_type_name, relation, dst_type_name, csr_graph) tuples
    ///
    /// Returns `Err` if there are too many types or an edge type references an
    /// unknown node type name. For tests / one-shot scripts, see [`Self::from_parts`].
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
