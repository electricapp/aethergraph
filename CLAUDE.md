# AetherGraph

HPC GNN data infrastructure: Rust workspace (`crates/`) with PyO3 bindings and a
thin Python layer (`python/`).

## The boundary principle: parse at the outermost edge

The reason this project pairs a Rust core with tight Python is to enforce
invariants **structurally** — with the type system — instead of defensively at
runtime. Every value crossing into the system is parsed exactly once, at a named
choke point, into a canonical typed representation. Downstream code trusts the
type and never re-checks it.

This is "parse, don't validate" plus a placement rule: **the parse lives at the
outermost edge the value crosses** — the public constructor, the CLI argument,
the file header, the FFI ingress — never at an internal seam. Everything inside
the edge is one consistency region: internal functions and internal module
boundaries accept only canonical types, so structure makes internal parsing
_unnecessary_, not merely forbidden. When an interior layer feels like it needs
a check, the fix is to move the edge outward or share the canonical type across
the layers — not to add a second parse.

- **One parser per input kind, at the edge.** Introspection (`isinstance`,
  `hasattr`, dtype probing, `flags` checks) is legal only inside a designated
  normalizer or a `from_*` boundary constructor whose job is to consume foreign
  input (PyG objects, Ray batches, user arrays). Everywhere else values are
  already canonical; re-sniffing them is a bug.
- **Parsers return proof.** A normalizer returns the fully validated value —
  right dtype, contiguity, range, shape. Callers never follow it with their own
  validation; if a caller needs a check the parser doesn't do, the check moves
  into the parser.
- **No stringly-typed values past the boundary.** URI-ish config strings
  (`"rdma://host:port"`) are parsed at construction into typed objects. Enum-ish
  strings are `Literal[...]` in signatures so mypy rejects typos at call sites,
  with one runtime validation at the boundary for untyped callers. Internal
  protocols (thread messages, discriminators) are dataclasses or enums matched
  with `match`, never tuples of magic strings.
- **No assumptions in comments.** If a call site needs a comment asserting what
  a value already is ("this is contiguous", "Rust returns fresh arrays"), the
  contract belongs on the boundary — the `.pyi` stub, the parser's docstring, or
  the Rust signature — and the call site just uses the value.
- **The FFI surface is the contract.** `python/aethergraph/_core.pyi` is the
  single source of truth for what Rust guarantees (dtypes, ownership,
  freshness). Keep it strict and in sync; Python never compensates with
  defensive `np.asarray`/`.copy()` on values the stub already types.
- **Make illegal states unrepresentable.** Prefer a shape that cannot express
  the invalid case (unpacking a 2-tuple, frozen dataclasses, `Literal`,
  discriminated unions) over code that checks for the invalid case. `assert` to
  appease mypy is a smell — restructure so narrowing is natural.

Rust-side mirror: `PyAny` extraction and dtype dispatch live in one converter
per input kind at the `#[pymethods]` surface, not scattered per call site.
Inside the workspace the same placement rule holds between crates and modules: a
value that crossed the FFI or a file header once travels as its typed form
(`NodeId`, slot indices, validated configs) through every internal layer — an
internal API that would need to range-check or coerce its input is a sign the
boundary is drawn too far in.

## Toolchain

- Python: always `uv run` (`uv run pytest`, `uv run ruff check`,
  `uv run mypy aethergraph`); rebuild the extension with
  `uv run maturin develop --release` from `python/`.
- mypy and ruff must pass; both are CI gates on the python job, and
  `ruff format`/`ruff check` also run in the pre-commit hook. `strict = true`
  lives in `pyproject.toml`, so `mypy aethergraph` and
  `mypy aethergraph --strict` cannot disagree about what passing means.
- A gap in a third-party stub is waived at the call site with a narrow
  `# type: ignore[code]`, never by excluding a module in config — an exclusion
  would also wave through the next untyped call in that tree. `strict` implies
  `warn_unused_ignores`, so a waiver fails the run once upstream annotates it.
- `_core.pyi` stubs are hand-authored, and nothing generates them:
  `tests/test_stub_coverage.py` fails if a runtime export has no declaration.
  The stub may declare names absent at runtime — platform-gated classes are
  guarded by the `HAS_*` flags rather than by attribute probing.
- Rust: `cargo fmt --all` before committing (pre-commit enforces it); clippy
  clean across the workspace.

## Check the Linux surface before pushing

Much of the workspace is `cfg(target_os = "linux")` — the io_uring gather, NVMe
passthrough, the userfaultfd pager, the memfd shared store, perf counters, NUMA
placement. On macOS none of it compiles, so a change that breaks those paths
passes every local check and fails in CI.

```bash
scripts/check-linux.sh --clippy --tests   # what CI runs, cross-compiled
scripts/check-linux.sh --arm              # same, aarch64-linux
scripts/linux-test.sh -p aether-graph --features "wal io-uring"  # RUN tests
```

`check-linux.sh` uses zig as the cross toolchain (`brew install zig`, plus
`rustup target add x86_64-unknown-linux-gnu` / `aarch64-...` for `--arm`) — no
container, no emulation; it proves the build, not the behavior. `linux-test.sh`
executes tests in a lima VM (`brew install lima`, arm64 Linux, real kernel —
io_uring/uffd/memfd actually run; RDMA/XDP self-skip). Run one of them after
touching anything Linux-gated. Neither is a pre-commit or pre-push hook, because
a full cross-check costs more than a commit should.
