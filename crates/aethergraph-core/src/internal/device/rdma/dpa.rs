//! K2.3: BlueField-3 DPA/FlexIO edge parse pipeline description.
//!
//! The DPA is a NIC-attached accelerator, not a P4 target. These types model
//! the pipeline boundary so host control-plane code can be written without
//! pretending a generic NIC exposes BlueField execution.

/// One graph-delta field extracted by a DPA/FlexIO parser stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeField {
    SourceNode,
    DestinationNode,
    EdgeType,
    EventTimestamp,
}

/// A parse-and-stage pipeline requested from a DPA program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DpaParsePipeline {
    pub fields: Vec<EdgeField>,
    pub deduplicate: bool,
    pub csr_delta_bytes: u32,
}

impl DpaParsePipeline {
    /// Build a pipeline with a non-empty extraction schema and staging area.
    pub fn new(fields: Vec<EdgeField>, deduplicate: bool, csr_delta_bytes: u32) -> Option<Self> {
        (!fields.is_empty() && csr_delta_bytes > 0).then_some(Self {
            fields,
            deduplicate,
            csr_delta_bytes,
        })
    }
}

// FlexIO path: [`super::flexio::FlexIoHost`] + `modules/aether_dpa/`.
// TODO(HARDWARE): BlueField-3 required. Compile and attach a FlexIO/DPA
// program that parses graph deltas, deduplicates them, and stages CSR updates
// while checking host-visible ordering and loss behavior.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_requires_schema_and_staging_capacity() {
        assert!(DpaParsePipeline::new(vec![], true, 4096).is_none());
        assert!(DpaParsePipeline::new(vec![EdgeField::SourceNode], true, 0).is_none());
        assert!(DpaParsePipeline::new(vec![EdgeField::SourceNode], true, 4096).is_some());
    }
}
