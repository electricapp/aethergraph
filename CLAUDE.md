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
- `mypy --strict` and ruff must pass; `_core.pyi` stubs are hand-authored.
- Rust: `cargo fmt --all` before committing (pre-commit enforces it); clippy
  clean across the workspace.
