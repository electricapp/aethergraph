"""The hand-authored `_core.pyi` must cover everything `_core` exports.

`_core.pyi` is the single source of truth for what Rust guarantees, and
nothing generates it — a new `#[pyclass]` or `#[pyfunction]` reaches users
with no type at all unless someone remembers the stub. mypy cannot catch
that: an undeclared attribute on a stubbed module is simply an error at the
call site, in the caller's code, long after the fact.

The check runs one way only. Every runtime export needs a declaration, but
the stub is allowed to declare names that are absent here: `SharedFeatureStore`
and `PerfCounters` are Linux-only, `_dlpack_capsule_from_cuda_ptr` ships only
in `gpudirect` builds, and callers are expected to gate on the `HAS_*` flags
rather than on the attribute existing.
"""

from __future__ import annotations

import re
from pathlib import Path

from aethergraph import _core

STUB = Path(__file__).resolve().parent.parent / "aethergraph" / "_core.pyi"

# `class Foo:` / `def foo(` / `HAS_FOO: bool` — the three shapes the stub uses
# to declare a public name.
_DECLARATION = re.compile(r"^(?:class|def)\s+(\w+)|^(\w+)\s*:", re.MULTILINE)


def _declared_names() -> set[str]:
    source = STUB.read_text()
    return {name for match in _DECLARATION.finditer(source) for name in match.groups() if name}


def test_stub_declares_every_runtime_export() -> None:
    exported = {
        name
        for name in dir(_core)
        if not name.startswith("_") or name in {"__version__", "__author__"}
    }
    missing = exported - _declared_names()
    assert not missing, (
        f"exported by _core but absent from _core.pyi: {sorted(missing)}. "
        "Add them to python/aethergraph/_core.pyi."
    )
