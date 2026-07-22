"""Python implementation modules for Tailor."""

from __future__ import annotations

from typing import Any


__all__ = [
    "FDQRule",
    "calculate_fdq_nodes",
    "calculate_fdq_rule",
    "fornberg_weights",
    "load_fdq_rule",
    "load_or_create_fdq_rule",
    "prepare_fdq_case",
    "quintic_tensor_spline",
    "save_fdq_rule",
]

_FDQ_EXPORTS = set(__all__) - {"prepare_fdq_case", "quintic_tensor_spline"}


def __getattr__(name: str) -> Any:
    """Lazily expose FD-q helpers without burdening the main launcher."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _FDQ_EXPORTS:
        from . import fdq_nodes

        return getattr(fdq_nodes, name)
    from . import interpolate

    return getattr(interpolate, name)
