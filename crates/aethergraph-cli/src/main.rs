use aethergraph_core::{Graph, NodeId, load_graph, save_graph, save_graph_compressed};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, info, trace, warn};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "aethergraph")]
struct Cli {
    /// Enable verbose logging (use -v, -vv, or -vvv for more detail)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Suppress all non-error output
    #[arg(short, long, global = true)]
    quiet: bool,

    #[command(subcommand)]
    command: Commands,
}

/// Column separator of an edge list.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Delimiter {
    #[value(alias = "\t")]
    Tab,
    #[value(alias = ",")]
    Comma,
    /// Any run of whitespace, so aligned columns parse cleanly.
    #[value(alias = " ")]
    Space,
}

impl Delimiter {
    /// Pick the separator from the first data line.
    fn detect(line: &str) -> Self {
        if line.contains('\t') {
            Self::Tab
        } else if line.contains(',') {
            Self::Comma
        } else {
            Self::Space
        }
    }

    /// The first two fields of `line`.
    fn two_fields(self, line: &str) -> Option<(&str, &str)> {
        fn first_two<'a>(mut parts: impl Iterator<Item = &'a str>) -> Option<(&'a str, &'a str)> {
            Some((parts.next()?, parts.next()?))
        }
        match self {
            Self::Tab => first_two(line.split('\t')),
            Self::Comma => first_two(line.split(',')),
            Self::Space => first_two(line.split_whitespace()),
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Convert an edge list file to AetherGraph binary format
    Convert {
        /// Input edge list file (TSV or CSV format: "source dest" or "source,dest")
        #[arg(short, long)]
        input: PathBuf,

        /// Output binary graph file
        #[arg(short, long)]
        output: PathBuf,

        /// Number of nodes in the graph (required)
        #[arg(short, long)]
        num_nodes: usize,

        /// Column separator (default: detected from the first data line)
        #[arg(short, long, value_enum)]
        delimiter: Option<Delimiter>,

        /// Skip first N lines (for headers)
        #[arg(long, default_value = "0")]
        skip_lines: usize,

        /// Write the succinct-coded format (Elias-Fano offsets,
        /// StreamVByte edges) instead of flat arrays — typically 2-4x
        /// smaller at rest, and loaded into owned storage rather than mmap
        #[arg(long)]
        compressed: bool,
    },

    /// Display information about a binary graph file
    Info {
        /// Binary graph file to inspect
        path: PathBuf,
    },

    /// Display detailed statistics about a binary graph file
    Stats {
        /// Binary graph file to analyze
        path: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if !cli.quiet {
        print_splash();
    }

    // Initialize logging based on verbosity
    init_logging(cli.verbose, cli.quiet);

    match cli.command {
        Commands::Convert {
            input,
            output,
            num_nodes,
            delimiter,
            skip_lines,
            compressed,
        } => convert_edge_list(
            &input, &output, num_nodes, delimiter, skip_lines, compressed,
        ),

        Commands::Info { path } => show_info(&path),

        Commands::Stats { path } => show_stats(&path),
    }
}

fn print_splash() {
    println!(); // Add breathing room before splash
    print!(
        r#"
█████╗ ███████╗████████╗██╗  ██╗███████╗██████╗  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗
██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║
███████║█████╗     ██║   ███████║█████╗  ██████╔╝██║  ███╗██████╔╝███████║██████╔╝███████║
██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝

   High-Performance Async Neighborhood Sampling for Billions-Scale GNNs

"#
    );
    let _ = std::io::stdout().flush();
}

fn init_logging(verbose: u8, quiet: bool) {
    // Determine log level based on flags
    let level = if quiet {
        "error"
    } else {
        match verbose {
            0 => "warn",  // Default: only warnings and errors
            1 => "info",  // -v: info level
            2 => "debug", // -vv: debug level
            _ => "trace", // -vvv: trace level
        }
    };

    // Allow override via RUST_LOG environment variable
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    // Set up compact logging format for HPC use
    tracing_subscriber::registry()
        .with(filter)
        .with(
            fmt::layer()
                .compact()
                .with_target(false)
                .with_thread_ids(false)
                .without_time(),
        )
        .init();
}

fn convert_edge_list(
    input: &Path,
    output: &Path,
    num_nodes: usize,
    delimiter: Option<Delimiter>,
    skip_lines: usize,
    compressed: bool,
) -> Result<()> {
    info!("Converting edge list to AetherGraph format");
    debug!("Input: {}", input.display());
    debug!("Output: {}", output.display());
    debug!("Nodes: {}", num_nodes);

    // Open and read the edge list file
    let file = File::open(input).context("failed to open input file")?;
    let reader = BufReader::with_capacity(8 * 1024 * 1024, file); // 8MB buffer for HPC

    // Parallel arrays feed the structure-of-arrays builder directly.
    let mut src: Vec<NodeId> = Vec::new();
    let mut dst: Vec<NodeId> = Vec::new();

    // Set up progress bar only if not quiet
    let pb = if tracing::level_enabled!(tracing::Level::INFO) {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .expect("valid progress bar template"),
        );
        Some(pb)
    } else {
        None
    };

    // Resolved once from the first data line and reused for the whole file,
    // so a stray tab/comma in a later row can't switch the parser mid-stream.
    let mut delim = delimiter;

    for (idx, line) in reader.lines().enumerate() {
        let line = line.context("failed to read line")?;

        // Skip header lines
        if idx < skip_lines {
            continue;
        }

        // Skip empty lines and comments
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let delim = *delim.get_or_insert_with(|| Delimiter::detect(line));
        let (s, d) = delim
            .two_fields(line)
            .with_context(|| format!("invalid edge format at line {}: {}", idx + 1, line))?;
        src.push(
            s.trim()
                .parse()
                .with_context(|| format!("invalid source node at line {}: {s}", idx + 1))?,
        );
        dst.push(
            d.trim()
                .parse()
                .with_context(|| format!("invalid dest node at line {}: {d}", idx + 1))?,
        );

        if src.len().is_multiple_of(100_000) {
            if let Some(pb) = &pb {
                pb.set_message(format!("Read {} edges", src.len()));
            }
            trace!("Read {} edges", src.len());
        }
    }

    if let Some(pb) = pb {
        pb.finish_with_message(format!("Read {} edges total", src.len()));
    }
    info!("Read {} edges from input file", src.len());

    // The builder range-checks every endpoint; its error names the edge.
    debug!("Building CSR graph structure");
    let graph =
        Graph::from_src_dst(num_nodes, &src, &dst, None).context("failed to build CSR graph")?;
    drop((src, dst));

    // Save to binary format
    debug!("Writing binary file");
    if compressed {
        save_graph_compressed(&graph, output).context("failed to save graph")?;
    } else {
        save_graph(&graph, output).context("failed to save graph")?;
    }

    info!("Conversion complete");
    info!("  Nodes: {}", graph.num_nodes());
    info!("  Edges: {}", graph.num_edges());
    if graph.num_nodes() > 0 {
        info!(
            "  Avg degree: {:.2}",
            graph.num_edges() as f64 / graph.num_nodes() as f64
        );
    }

    let file_size = std::fs::metadata(output)?.len();
    info!("  File size: {:.2} MB", file_size as f64 / 1_000_000.0);

    Ok(())
}

fn show_info(path: &Path) -> Result<()> {
    debug!("Loading graph from {}", path.display());

    let graph = load_graph(path).context("failed to load graph")?;

    info!("Graph Information:");
    info!("  Nodes: {}", graph.num_nodes());
    info!("  Edges: {}", graph.num_edges());
    if graph.num_nodes() > 0 {
        info!(
            "  Avg degree: {:.2}",
            graph.num_edges() as f64 / graph.num_nodes() as f64
        );
    }
    info!("  Has weights: {}", graph.weights().is_some());

    let file_size = std::fs::metadata(path)?.len();
    info!("  File size: {:.2} MB", file_size as f64 / 1_000_000.0);

    Ok(())
}

/// Degree → node count, exact.
///
/// Small degrees count into a dense array; the rest go to a map. Distinct
/// degrees number at most about sqrt(2E) (their sum is bounded by E), so
/// the map stays small even for a hub of degree 10^9, where a dense
/// `max_degree + 1` table would take gigabytes.
struct DegreeHistogram {
    dense: Vec<u64>,
    sparse: BTreeMap<u64, u64>,
}

impl DegreeHistogram {
    const DENSE_DEGREES: usize = 1 << 16;

    fn new() -> Self {
        Self {
            dense: vec![0; Self::DENSE_DEGREES],
            sparse: BTreeMap::new(),
        }
    }

    fn add(&mut self, degree: u64) {
        match self.dense.get_mut(degree as usize) {
            Some(count) => *count += 1,
            None => *self.sparse.entry(degree).or_default() += 1,
        }
    }

    /// `(degree, count)` in ascending degree order.
    fn iter(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.dense
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(d, &c)| (d as u64, c))
            .chain(self.sparse.iter().map(|(&d, &c)| (d, c)))
    }
}

fn show_stats(path: &Path) -> Result<()> {
    debug!("Loading graph from {}", path.display());

    let graph = load_graph(path).context("failed to load graph")?;
    let stats = graph.stats();

    info!("Graph Statistics:");
    info!("  Nodes: {}", stats.num_nodes);
    info!("  Edges: {}", stats.num_edges);
    info!("  Max degree: {}", stats.max_degree);
    info!("  Avg degree: {:.2}", stats.avg_degree);
    info!("  Has weights: {}", stats.has_weights);

    let file_size = std::fs::metadata(path)?.len();
    info!("");
    info!("File Information:");
    info!("  Size: {:.2} MB", file_size as f64 / 1_000_000.0);
    if stats.num_nodes > 0 {
        info!(
            "  Bytes per node: {:.2}",
            file_size as f64 / stats.num_nodes as f64
        );
    }
    if stats.num_edges > 0 {
        info!(
            "  Bytes per edge: {:.2}",
            file_size as f64 / stats.num_edges as f64
        );
    }

    // Calculate degree distribution stats
    debug!("Analyzing degree distribution");
    let pb = if tracing::level_enabled!(tracing::Level::INFO) {
        let pb = ProgressBar::new(stats.num_nodes as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40.cyan/blue}] {pos}/{len} nodes")
                .expect("valid progress bar template")
                .progress_chars("=>-"),
        );
        Some(pb)
    } else {
        None
    };

    // `load_graph` proves offsets monotone, so each window is a degree.
    let mut histogram = DegreeHistogram::new();
    for (node, w) in graph.offsets().windows(2).enumerate() {
        histogram.add(w[1] - w[0]);
        if node % (1 << 20) == 0
            && let Some(pb) = &pb
        {
            pb.set_position(node as u64);
        }
    }
    if let Some(pb) = pb {
        pb.finish();
    }

    // Find some interesting degree percentiles
    let n = stats.num_nodes as u64;
    let targets = [n / 2, (n * 9) / 10, (n * 99) / 100];
    let mut percentiles = [None; 3];
    let mut cumulative = 0u64;
    for (degree, count) in histogram.iter() {
        cumulative += count;
        for (slot, &target) in percentiles.iter_mut().zip(&targets) {
            if slot.is_none() && cumulative >= target {
                *slot = Some(degree);
            }
        }
    }

    info!("");
    info!("Degree Distribution:");
    info!("  50th percentile: {}", percentiles[0].unwrap_or(0));
    info!("  90th percentile: {}", percentiles[1].unwrap_or(0));
    info!("  99th percentile: {}", percentiles[2].unwrap_or(0));

    let isolated_nodes = histogram.dense[0];
    if isolated_nodes > 0 {
        warn!(
            "  Isolated nodes (degree 0): {} ({:.2}%)",
            isolated_nodes,
            100.0 * isolated_nodes as f64 / stats.num_nodes as f64
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splash_follows_parsed_quiet_flag() {
        let cli = Cli::try_parse_from(["aethergraph", "-vq", "info", "g.bin"]).unwrap();
        assert!(cli.quiet);
        assert_eq!(cli.verbose, 1);
    }

    #[test]
    fn delimiter_accepts_names_and_literals() {
        for (arg, want) in [
            ("tab", Delimiter::Tab),
            ("\t", Delimiter::Tab),
            ("comma", Delimiter::Comma),
            (",", Delimiter::Comma),
            ("space", Delimiter::Space),
            (" ", Delimiter::Space),
        ] {
            let cli = Cli::try_parse_from([
                "aethergraph",
                "convert",
                "-i",
                "in",
                "-o",
                "out",
                "-n",
                "3",
                "-d",
                arg,
            ])
            .unwrap();
            let Commands::Convert { delimiter, .. } = cli.command else {
                panic!("expected convert");
            };
            assert!(delimiter == Some(want), "argument {arg:?}");
        }
    }

    #[test]
    fn delimiter_splits_fields() {
        assert_eq!(Delimiter::Space.two_fields("1   2  3"), Some(("1", "2")));
        assert_eq!(Delimiter::Comma.two_fields("1,2"), Some(("1", "2")));
        assert_eq!(Delimiter::Tab.two_fields("7\t8\t9"), Some(("7", "8")));
        assert_eq!(Delimiter::Comma.two_fields("7"), None);
        assert!(Delimiter::detect("1\t2") == Delimiter::Tab);
        assert!(Delimiter::detect("1,2") == Delimiter::Comma);
        assert!(Delimiter::detect("1 2") == Delimiter::Space);
    }

    #[test]
    fn histogram_is_exact_past_the_dense_range() {
        let mut h = DegreeHistogram::new();
        for d in [0, 3, 3, 1_000_000_000, 70_000] {
            h.add(d);
        }
        let got: Vec<(u64, u64)> = h.iter().collect();
        assert_eq!(got, vec![(0, 1), (3, 2), (70_000, 1), (1_000_000_000, 1)]);
    }

    #[test]
    fn convert_reports_out_of_range_edges() {
        let dir = std::env::temp_dir().join(format!("aethergraph-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("edges.tsv");
        std::fs::write(&input, "0\t1\n0\t10\n").unwrap();
        let err = convert_edge_list(&input, &dir.join("g.bin"), 3, None, 0, false).unwrap_err();
        let chain = format!("{err:#}");
        assert!(
            chain.contains("destination node 10 exceeds num_nodes 3"),
            "got: {chain}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
