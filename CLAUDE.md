# AetherGraph

HPC GNN data infrastructure: Rust workspace (`crates/`) with PyO3 bindings and
a thin Python layer (`python/`).

## The boundary principle: parse, don't validate

The reason this project pairs a Rust core with tight Python is to enforce
invariants **structurally** — with the type system — instead of defensively at
runtime. Every value crossing a boundary is parsed exactly once, at a named
choke point, into a canonical typed representation. Downstream code trusts the
type and never re-checks it.

- **One parser per input kind.** Introspection (`isinstance`, `hasattr`, dtype
  probing, `flags` checks) is legal only inside a designated normalizer or a
  `from_*` boundary constructor whose job is to consume foreign input (PyG
  objects, Ray batches, user arrays). Everywhere else values are already
  canonical; re-sniffing them is a bug.
- **Parsers return proof.** A normalizer returns the fully validated value —
  right dtype, contiguity, range, shape. Callers never follow it with their own
  validation; if a caller needs a check the parser doesn't do, the check moves
  into the parser.
- **No stringly-typed values past the boundary.** URI-ish config strings
  (`"rdma://host:port"`) are parsed at construction into typed objects.
  Enum-ish strings are `Literal[...]` in signatures so mypy rejects typos at
  call sites, with one runtime validation at the boundary for untyped callers.
  Internal protocols (thread messages, discriminators) are dataclasses or
  enums matched with `match`, never tuples of magic strings.
- **No assumptions in comments.** If a call site needs a comment asserting
  what a value already is ("this is contiguous", "Rust returns fresh arrays"),
  the contract belongs on the boundary — the `.pyi` stub, the parser's
  docstring, or the Rust signature — and the call site just uses the value.
- **The FFI surface is the contract.** `python/aethergraph/_core.pyi` is the
  single source of truth for what Rust guarantees (dtypes, ownership,
  freshness). Keep it strict and in sync; Python never compensates with
  defensive `np.asarray`/`.copy()` on values the stub already types.
- **Make illegal states unrepresentable.** Prefer a shape that cannot express
  the invalid case (unpacking a 2-tuple, frozen dataclasses, `Literal`,
  discriminated unions) over code that checks for the invalid case. `assert`
  to appease mypy is a smell — restructure so narrowing is natural.

Rust-side mirror: `PyAny` extraction and dtype dispatch live in one converter
per input kind at the `#[pymethods]` surface, not scattered per call site.

## Toolchain

- Python: always `uv run` (`uv run pytest`, `uv run ruff check`,
  `uv run mypy aethergraph`); rebuild the extension with
  `uv run maturin develop --release` from `python/`.
- `mypy --strict` and ruff must pass; `_core.pyi` stubs are hand-authored.
- Rust: `cargo fmt --all` before committing (pre-commit enforces it); clippy
  clean across the workspace.
