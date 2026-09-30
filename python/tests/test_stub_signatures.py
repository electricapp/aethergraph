"""The `_core.pyi` stub names every runtime parameter exactly.

Types cannot be checked against the extension at runtime, but names can: a
keyword call that type-checks against the stub must not raise TypeError when
it runs. PyO3 publishes each callable's signature, so every stub function,
method, and constructor is compared with the one the runtime reports.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import aethergraph._core as core

STUB = Path(__file__).parent.parent / "aethergraph" / "_core.pyi"


def _stub_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = fn.args
    names = [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
    return [n for n in names if n not in ("self", "cls")]


def _is_property(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == "property")
        or (isinstance(d, ast.Attribute) and d.attr == "setter")
        for d in fn.decorator_list
    )


def _stub_callables() -> Iterator[tuple[str, Any, list[str]]]:
    """(qualified name, runtime object, stub parameter names) per stub callable."""
    tree = ast.parse(STUB.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            runtime = getattr(core, node.name, None)
            if runtime is not None:
                yield node.name, runtime, _stub_params(node)
        elif isinstance(node, ast.ClassDef):
            cls = getattr(core, node.name, None)
            if cls is None:
                continue
            for member in node.body:
                if not isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if _is_property(member):
                    continue
                if member.name == "__init__":
                    yield f"{node.name}()", cls, _stub_params(member)
                elif not member.name.startswith("__"):
                    runtime = getattr(cls, member.name, None)
                    if runtime is not None:
                        yield f"{node.name}.{member.name}", runtime, _stub_params(member)


def _runtime_params(obj: Any) -> list[str] | None:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [name for name in sig.parameters if name not in ("self", "cls")]


CALLABLES = list(_stub_callables())


def test_stub_declares_callables() -> None:
    assert len(CALLABLES) > 50


@pytest.mark.parametrize(("name", "runtime", "stub"), CALLABLES, ids=[c[0] for c in CALLABLES])
def test_stub_parameter_names_match_runtime(name: str, runtime: Any, stub: list[str]) -> None:
    actual = _runtime_params(runtime)
    if actual is None:
        pytest.skip(f"{name} publishes no signature")
    assert stub == actual, f"{name}: stub declares {stub}, runtime takes {actual}"
