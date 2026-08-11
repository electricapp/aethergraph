use aethergraph_core::{Graph, NodeId, load_graph, save_graph, save_graph_compressed};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
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

        /// Delimiter for the input file (default: auto-detect)
        #[arg(short, long)]
        delimiter: Option<String>,

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
    // Check for --quiet flag before parsing to show splash
    let is_quiet = std::env::args().any(|arg| arg == "--quiet" || arg == "-q");

    // Show splash screen unless in quiet mode
    if !is_quiet {
        print_splash();
    }

    let cli = Cli::parse();

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
    input: &PathBuf,
    output: &PathBuf,
    num_nodes: usize,
    delimiter: Option<String>,
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

    let mut edges = Vec::new();

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

    let mut line_count = 0;
    // Delimiter is resolved once from the first data line and reused for the
    // whole file, so a stray tab/comma in a later row can't switch the parser
    // mid-stream.
    let mut delim: Option<String> = delimiter;

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

        // Auto-detect delimiter from the first data line if not provided
        let delim = delim.get_or_insert_with(|| {
            if line.contains('\t') {
                "\t".to_string()
            } else if line.contains(',') {
                ",".to_string()
            } else {
                " ".to_string()
            }
        });

        // Parse edge. A space delimiter splits on any whitespace run so
        // aligned columns (multiple spaces) parse cleanly.
        let parts: Vec<&str> = if delim == " " {
            line.split_whitespace().collect()
        } else {
            line.split(delim.as_str()).collect()
        };
        if parts.len() < 2 {
            anyhow::bail!("invalid edge format at line {}: {}", idx + 1, line);
        }

        let src: NodeId = parts[0]
            .trim()
            .parse()
            .with_context(|| format!("invalid source node at line {}: {}", idx + 1, parts[0]))?;

        let dst: NodeId = parts[1]
            .trim()
            .parse()
            .with_context(|| format!("invalid dest node at line {}: {}", idx + 1, parts[1]))?;

        // Validate node IDs are within range
        if src as usize >= num_nodes {
            anyhow::bail!(
                "source node {} exceeds num_nodes {} at line {}",
                src,
                num_nodes,
                idx + 1
            );
        }
        if dst as usize >= num_nodes {
            anyhow::bail!(
                "dest node {} exceeds num_nodes {} at line {}",
                dst,
                num_nodes,
                idx + 1
            );
        }

        edges.push((src, dst));

        line_count += 1;
        if line_count % 100_000 == 0 {
            if let Some(pb) = &pb {
                pb.set_message(format!("Read {} edges", line_count));
            }
            trace!("Read {} edges", line_count);
        }
    }

    if let Some(pb) = pb {
        pb.finish_with_message(format!("Read {} edges total", edges.len()));
    }
    info!("Read {} edges from input file", edges.len());

    // Build CSR graph
    debug!("Building CSR graph structure");
    let graph = Graph::from_edges(num_nodes, &edges, None).context("failed to build CSR graph")?;

    debug!("Validating graph structure");
    graph.validate().context("graph validation failed")?;

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

fn show_info(path: &PathBuf) -> Result<()> {
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

fn show_stats(path: &PathBuf) -> Result<()> {
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

    let mut degree_counts: Vec<usize> = vec![0; stats.max_degree + 1];
    for node in 0..stats.num_nodes as NodeId {
        let degree = graph.degree(node);
        degree_counts[degree] += 1;
        if node % 10000 == 0
            && let Some(pb) = &pb
        {
            pb.set_position(node as u64);
        }
    }
    if let Some(pb) = pb {
        pb.finish();
    }

    // Find some interesting degree percentiles
    let mut cumulative = 0;
    let p50_idx = stats.num_nodes / 2;
    let p90_idx = (stats.num_nodes * 9) / 10;
    let p99_idx = (stats.num_nodes * 99) / 100;

    let mut p50 = None;
    let mut p90 = None;
    let mut p99 = None;

    for (degree, &count) in degree_counts.iter().enumerate() {
        cumulative += count;
        if p50.is_none() && cumulative >= p50_idx {
            p50 = Some(degree);
        }
        if p90.is_none() && cumulative >= p90_idx {
            p90 = Some(degree);
        }
        if p99.is_none() && cumulative >= p99_idx {
            p99 = Some(degree);
        }
    }

    info!("");
    info!("Degree Distribution:");
    info!("  50th percentile: {}", p50.unwrap_or(0));
    info!("  90th percentile: {}", p90.unwrap_or(0));
    info!("  99th percentile: {}", p99.unwrap_or(0));

    // Show top degree counts
    let isolated_nodes = degree_counts[0];
    if isolated_nodes > 0 {
        warn!(
            "  Isolated nodes (degree 0): {} ({:.2}%)",
            isolated_nodes,
            100.0 * isolated_nodes as f64 / stats.num_nodes as f64
        );
    }

    Ok(())
}
