"""Standard equal-error FD-q nodes and local differentiation weights."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import sys

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, least_squares


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SCHEMA_VERSION = 2
DEFAULT_CACHE_DIRECTORY = Path(__file__).resolve().parents[2] / "fdqNodes"


@dataclass(frozen=True, slots=True)
class FDQRule:
    """One-dimensional FD-q nodes and compact local weights.

    For bounded rules, N is the number of intervals and nodes contains N + 1
    entries. For periodic rules, N is the number of unique half-open nodes.
    q is the local polynomial degree, so every stencil has q + 1 entries.
    weights[m, i, :] contains the order-m weights at node i.
    """

    N: int
    q: int
    nodes: FloatArray
    stencil_indices: IntArray
    stencil_offsets: IntArray
    weights: FloatArray
    log_error: FloatArray
    topology: str = "bounded"
    period: float = 2.0

    @property
    def node_count(self) -> int:
        """Return the number of unique nodes stored by the rule."""

        return int(self.nodes.size)


def _validate_parameters(N: int, q: int) -> tuple[int, int]:
    """Validate a bounded interval count and local polynomial degree."""

    if isinstance(N, bool) or not isinstance(N, Integral):
        raise TypeError("N must be an integer number of intervals")
    if isinstance(q, bool) or not isinstance(q, Integral):
        raise TypeError("q must be an integer polynomial degree")
    N = int(N)
    q = int(q)
    if N < 2:
        raise ValueError("N must be at least 2")
    if q < 2 or q > N:
        raise ValueError("q must satisfy 2 <= q <= N")
    return N, q


def _stencil_starts(node_count: int, degree: int) -> IntArray:
    """Return centered, boundary-shifted starts for degree+1 point stencils."""
    last_start = node_count - degree - 1
    rows = np.arange(node_count, dtype=np.int64)
    return np.clip(rows - degree // 2, 0, last_start).astype(np.int64)


def _parameters_to_auxiliary_nodes(parameters: FloatArray) -> FloatArray:
    """Map unconstrained log-gap parameters to ordered interior nodes."""

    # Fix the final log-gap to zero to remove the softmax scale null direction.
    log_gaps = np.concatenate((np.asarray(parameters, dtype=np.float64), [0.0]))
    log_gaps -= np.max(log_gaps)
    gaps = np.exp(log_gaps)
    gaps /= np.sum(gaps)
    return -1.0 + 2.0 * np.cumsum(gaps[:-1])


def _auxiliary_nodes_to_parameters(nodes: FloatArray) -> FloatArray:
    """Map ordered auxiliary nodes back to scale-fixed log gaps."""

    gaps = np.diff(np.concatenate(([-1.0], nodes, [1.0])))
    if np.any(gaps <= 0.0):
        raise ValueError("auxiliary nodes must be strictly increasing in (-1, 1)")
    return np.log(gaps[:-1] / gaps[-1])


def _extremum(stencil: FloatArray, left: float, right: float) -> float:
    """Locate the error-polynomial extremum between two adjacent roots."""

    lo = np.nextafter(left, right)
    hi = np.nextafter(right, left)

    def logarithmic_derivative(value: float) -> float:
        """Evaluate the error polynomial's logarithmic derivative."""

        # Near an endpoint the reciprocal can overflow to a correctly signed
        # infinity; brentq can use that sign, so keep it without a noisy warning.
        with np.errstate(divide="ignore", over="ignore"):
            return float(np.sum(1.0 / (value - stencil)))

    return float(
        brentq(
            logarithmic_derivative,
            lo,
            hi,
            xtol=5.0e-15,
            rtol=4.0 * np.finfo(np.float64).eps,
            maxiter=100,
        )
    )


def _equal_error_data(
    auxiliary_nodes: FloatArray, order: int
) -> tuple[FloatArray, FloatArray]:
    """Return configuration nodes and log error amplitudes for one continuation order."""

    interval_count = auxiliary_nodes.size
    interval_rows = np.arange(interval_count - 1, dtype=np.int64)
    starts = np.clip(
        interval_rows - (order - 1) // 2, 0, interval_count - order
    ).astype(np.int64)

    configuration_nodes = np.empty(interval_count + 1, dtype=np.float64)
    configuration_nodes[0] = -1.0
    configuration_nodes[-1] = 1.0
    log_error = np.empty(interval_count + 1, dtype=np.float64)

    first_stencil = auxiliary_nodes[starts[0] : starts[0] + order]
    log_error[0] = np.sum(np.log(np.abs(-1.0 - first_stencil)))

    for interval, start in enumerate(starts):
        stencil = auxiliary_nodes[start : start + order]
        location = _extremum(
            stencil,
            float(auxiliary_nodes[interval]),
            float(auxiliary_nodes[interval + 1]),
        )
        configuration_nodes[interval + 1] = location
        log_error[interval + 1] = np.sum(np.log(np.abs(location - stencil)))

    last_stencil = auxiliary_nodes[starts[-1] : starts[-1] + order]
    log_error[-1] = np.sum(np.log(np.abs(1.0 - last_stencil)))
    return configuration_nodes, log_error


def _solve_fdq_nodes(N: int, q: int) -> tuple[FloatArray, FloatArray]:
    """Solve the equal-error continuation problem through order ``q``."""

    N, q = _validate_parameters(N, q)
    auxiliary_nodes = -1.0 + (2.0 * np.arange(N, dtype=np.float64) + 1.0) / N
    parameters = _auxiliary_nodes_to_parameters(auxiliary_nodes)

    for order in range(2, q + 1):

        def residual(candidate: FloatArray) -> FloatArray:
            """Return adjacent log-error differences for least squares."""

            candidate_nodes = _parameters_to_auxiliary_nodes(candidate)
            _, log_error = _equal_error_data(candidate_nodes, order)
            return np.diff(log_error)

        result = least_squares(
            residual,
            parameters,
            method="trf",
            jac="3-point",
            x_scale="jac",
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=max(2000, 50 * N),
        )
        residual_norm = float(np.linalg.norm(result.fun, ord=np.inf))
        if not result.success or not np.isfinite(residual_norm) or residual_norm > 1.0e-9:
            raise RuntimeError(
                f"FD-q node solve failed at continuation order {order}: "
                f"{result.message}; ||residual||_inf={residual_norm:.3e}"
            )
        parameters = result.x

    auxiliary_nodes = _parameters_to_auxiliary_nodes(parameters)
    nodes, log_error = _equal_error_data(auxiliary_nodes, q)
    nodes = 0.5 * (nodes - nodes[::-1])
    nodes[0] = -1.0
    nodes[-1] = 1.0
    log_error = 0.5 * (log_error + log_error[::-1])
    _validate_nodes(nodes, log_error)
    return nodes, log_error


def _validate_nodes(nodes: FloatArray, log_error: FloatArray) -> None:
    """Check bounded FD-q node ordering, symmetry, and equal error."""

    if nodes.ndim != 1 or log_error.shape != nodes.shape:
        raise ValueError("nodes and log_error must be one-dimensional with equal size")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(log_error)):
        raise ValueError("FD-q nodes and diagnostics must be finite")
    if nodes[0] != -1.0 or nodes[-1] != 1.0:
        raise ValueError("FD-q nodes must include the exact endpoints -1 and 1")
    if np.any(np.diff(nodes) <= 0.0):
        raise ValueError("FD-q nodes must be strictly increasing")
    symmetry_error = float(np.max(np.abs(nodes + nodes[::-1])))
    if symmetry_error > 5.0e-11:
        raise ValueError(f"FD-q nodes are not symmetric: error={symmetry_error:.3e}")
    equal_error_residual = float(np.max(np.abs(np.diff(log_error))))
    if equal_error_residual > 1.0e-9:
        raise ValueError(
            "FD-q equal-error validation failed: "
            f"residual={equal_error_residual:.3e}"
        )


def calculate_fdq_nodes(N: int, q: int) -> FloatArray:
    """Calculate the N + 1 standard equal-error FD-q nodes on [-1, 1]."""

    nodes, _ = _solve_fdq_nodes(N, q)
    return nodes


def fornberg_weights(
    x0: float, stencil: FloatArray, max_derivative: int = 2
) -> FloatArray:
    """Compute finite-difference weights by the Fornberg recursion."""

    points = np.asarray(stencil, dtype=np.float64)
    if points.ndim != 1 or points.size == 0:
        raise ValueError("stencil must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(points)) or not np.isfinite(x0):
        raise ValueError("x0 and stencil entries must be finite")
    if np.unique(points).size != points.size:
        raise ValueError("stencil entries must be distinct")
    if isinstance(max_derivative, bool) or not isinstance(max_derivative, Integral):
        raise TypeError("max_derivative must be an integer")
    max_derivative = int(max_derivative)
    if max_derivative < 0 or max_derivative >= points.size:
        raise ValueError("max_derivative must satisfy 0 <= max_derivative < stencil size")

    count = points.size
    weights = np.zeros((max_derivative + 1, count), dtype=np.float64)
    weights[0, 0] = 1.0
    c1 = 1.0
    c4 = points[0] - x0

    for i in range(1, count):
        derivative_limit = min(i, max_derivative)
        c2 = 1.0
        c5 = c4
        c4 = points[i] - x0
        for j in range(i):
            c3 = points[i] - points[j]
            c2 *= c3
            if j == i - 1:
                for derivative in range(derivative_limit, 0, -1):
                    weights[derivative, i] = (
                        c1
                        * (
                            derivative * weights[derivative - 1, i - 1]
                            - c5 * weights[derivative, i - 1]
                        )
                        / c2
                    )
                weights[0, i] = -c1 * c5 * weights[0, i - 1] / c2
            for derivative in range(derivative_limit, 0, -1):
                weights[derivative, j] = (
                    c4 * weights[derivative, j]
                    - derivative * weights[derivative - 1, j]
                ) / c3
            weights[0, j] = c4 * weights[0, j] / c3
        c1 = c2

    return weights


def calculate_fdq_rule(N: int, q: int) -> FDQRule:
    """Calculate nodes, compact stencil indices, and derivative orders 0 through 2."""

    N, q = _validate_parameters(N, q)
    nodes, log_error = _solve_fdq_nodes(N, q)
    starts = _stencil_starts(N + 1, q)
    stencil_indices = starts[:, None] + np.arange(q + 1, dtype=np.int64)
    stencil_offsets = stencil_indices - np.arange(N + 1, dtype=np.int64)[:, None]
    weights = np.empty((3, N + 1, q + 1), dtype=np.float64)
    for row, columns in enumerate(stencil_indices):
        weights[:, row, :] = fornberg_weights(nodes[row], nodes[columns], 2)

    rule = FDQRule(
        N,
        q,
        nodes,
        stencil_indices,
        stencil_offsets,
        weights,
        log_error,
        "bounded",
        2.0,
    )
    _validate_rule(rule)
    return rule


def calculate_periodic_fdq_rule(
    node_count: int, q: int, period: float = 2.0
) -> FDQRule:
    """Create a centered uniform FD-q rule on a half-open periodic interval."""

    if isinstance(node_count, bool) or not isinstance(node_count, Integral):
        raise TypeError("node_count must be an integer")
    if isinstance(q, bool) or not isinstance(q, Integral):
        raise TypeError("q must be an integer polynomial degree")
    node_count = int(node_count)
    q = int(q)
    if node_count < 3:
        raise ValueError("node_count must be at least 3")
    if q < 2 or q >= node_count:
        raise ValueError("periodic q must satisfy 2 <= q < node_count")
    if q % 2 != 0:
        raise ValueError("periodic q must be even for a centered stencil")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")

    spacing = float(period) / node_count
    origin = -0.5 * float(period)
    nodes = origin + spacing * np.arange(node_count, dtype=np.float64)
    centered_offsets = np.arange(-q // 2, q // 2 + 1, dtype=np.int64)
    stencil_offsets = np.broadcast_to(
        centered_offsets, (node_count, q + 1)
    ).copy()
    rows = np.arange(node_count, dtype=np.int64)[:, None]
    stencil_indices = (rows + stencil_offsets) % node_count
    weights = np.empty((3, node_count, q + 1), dtype=np.float64)
    for row in range(node_count):
        unwrapped = nodes[row] + spacing * centered_offsets
        weights[:, row, :] = fornberg_weights(nodes[row], unwrapped, 2)

    rule = FDQRule(
        node_count,
        q,
        nodes,
        stencil_indices,
        stencil_offsets,
        weights,
        np.zeros(node_count, dtype=np.float64),
        "periodic",
        float(period),
    )
    _validate_rule(rule)
    return rule


def _validate_rule(rule: FDQRule) -> None:
    """Validate metadata, topology, stencils, weights, and moments."""

    q = int(rule.q)
    nodes = np.asarray(rule.nodes)
    indices = np.asarray(rule.stencil_indices)
    offsets = np.asarray(rule.stencil_offsets)
    weights = np.asarray(rule.weights)
    log_error = np.asarray(rule.log_error)
    if rule.topology == "bounded":
        N, q = _validate_parameters(rule.N, q)
        expected_node_count = N + 1
        _validate_nodes(nodes, log_error)
        if not np.isclose(rule.period, 2.0, rtol=0.0, atol=0.0):
            raise ValueError("bounded FD-q period must be 2")
    elif rule.topology == "periodic":
        expected_node_count = int(rule.N)
        if expected_node_count < 3 or q < 2 or q >= expected_node_count:
            raise ValueError("periodic metadata must satisfy node_count > q >= 2")
        if q % 2 != 0:
            raise ValueError("periodic q must be even")
        if not np.isfinite(rule.period) or rule.period <= 0.0:
            raise ValueError("periodic period must be finite and positive")
        if nodes.shape != (expected_node_count,) or log_error.shape != nodes.shape:
            raise ValueError("periodic nodes and diagnostics have the wrong shape")
        if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(log_error)):
            raise ValueError("periodic nodes and diagnostics must be finite")
        if np.any(np.diff(nodes) <= 0.0):
            raise ValueError("periodic nodes must be strictly increasing")
        expected_spacing = rule.period / expected_node_count
        spacing_error = float(np.max(np.abs(np.diff(nodes) - expected_spacing)))
        if spacing_error > 2.0e-13 * max(1.0, abs(expected_spacing)):
            raise ValueError(
                f"periodic nodes must be uniformly spaced: error={spacing_error:.3e}"
            )
        if not np.isclose(nodes[0], -0.5 * rule.period):
            raise ValueError("periodic nodes must start at -period/2")
        if not nodes[-1] < 0.5 * rule.period:
            raise ValueError("periodic nodes must not include the repeated endpoint")
    else:
        raise ValueError(f"unsupported FD-q topology {rule.topology!r}")

    if nodes.shape != (expected_node_count,):
        raise ValueError(f"nodes must have shape {(expected_node_count,)}")
    if indices.shape != (expected_node_count, q + 1):
        raise ValueError(
            f"stencil_indices must have shape {(expected_node_count, q + 1)}"
        )
    if offsets.shape != indices.shape:
        raise ValueError("stencil_offsets must match stencil_indices")
    if weights.shape != (3, expected_node_count, q + 1):
        raise ValueError(
            f"weights must have shape {(3, expected_node_count, q + 1)}"
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("stencil_indices must have an integer dtype")
    if not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("stencil_offsets must have an integer dtype")
    if np.any(indices < 0) or np.any(indices >= expected_node_count):
        raise ValueError("stencil_indices contain an out-of-range column")
    rows = np.arange(expected_node_count, dtype=np.int64)[:, None]
    if rule.topology == "bounded":
        if np.any(np.diff(indices, axis=1) != 1):
            raise ValueError("every bounded FD-q stencil must contain contiguous columns")
        if np.any(indices - rows != offsets):
            raise ValueError("bounded stencil offsets are inconsistent")
    else:
        if np.any((rows + offsets) % expected_node_count != indices):
            raise ValueError("periodic stencil indices and offsets are inconsistent")
        expected_offsets = np.arange(-q // 2, q // 2 + 1, dtype=np.int64)
        if np.any(offsets != expected_offsets):
            raise ValueError("periodic stencils must use centered offsets")
    if np.any(~np.any(indices == rows, axis=1)):
        raise ValueError("every FD-q stencil must contain its evaluation node")
    if not np.all(np.isfinite(weights)):
        raise ValueError("FD-q weights must be finite")

    for derivative in range(3):
        for power in range(q + 1):
            if rule.topology == "periodic":
                spacing = rule.period / expected_node_count
                sample_coordinates = nodes[:, None] + spacing * offsets
            else:
                sample_coordinates = nodes[indices]
            values = sample_coordinates**power
            approximation = np.sum(weights[derivative] * values, axis=1)
            if power < derivative:
                expected = np.zeros(expected_node_count, dtype=np.float64)
            else:
                coefficient = 1
                for factor in range(derivative):
                    coefficient *= power - factor
                expected = coefficient * nodes ** (power - derivative)
            tolerance = 2.0e-8 * max(1.0, float(np.max(np.abs(expected))))
            if not np.allclose(approximation, expected, rtol=2.0e-8, atol=tolerance):
                error = float(np.max(np.abs(approximation - expected)))
                raise ValueError(
                    f"order-{derivative} weights fail the polynomial test "
                    f"at degree {power}: error={error:.3e}"
                )


def _default_cache_path(
    N: int, q: int, directory: Path | str | None, topology: str = "bounded"
) -> Path:
    """Build the canonical bounded or periodic rule-cache filename."""

    cache_directory = (
        DEFAULT_CACHE_DIRECTORY if directory is None else Path(directory)
    )
    suffix = "p" if topology == "periodic" else ""
    return cache_directory.expanduser().resolve() / f"N{N}_q{q}{suffix}.h5"


def save_fdq_rule(rule: FDQRule, path: Path | str | None = None) -> Path:
    """Validate and atomically save an FD-q rule as HDF5."""

    _validate_rule(rule)
    destination = (
        _default_cache_path(rule.N, rule.q, None, rule.topology)
        if path is None
        else Path(path).expanduser().resolve()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["N"] = rule.N
            handle.attrs["q"] = rule.q
            handle.attrs["node_count"] = rule.node_count
            handle.attrs["stencil_size"] = rule.q + 1
            handle.attrs["topology"] = rule.topology
            handle.attrs["period"] = rule.period
            handle.create_dataset("nodes", data=rule.nodes, dtype=np.float64)
            handle.create_dataset(
                "stencil_indices", data=rule.stencil_indices, dtype=np.int64
            )
            handle.create_dataset(
                "stencil_offsets", data=rule.stencil_offsets, dtype=np.int64
            )
            weight_group = handle.create_group("weights")
            for derivative in range(3):
                weight_group.create_dataset(
                    f"d{derivative}",
                    data=rule.weights[derivative],
                    dtype=np.float64,
                )
            diagnostic_group = handle.create_group("diagnostics")
            diagnostic_group.create_dataset(
                "log_error", data=rule.log_error, dtype=np.float64
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_fdq_rule(path: Path | str) -> FDQRule:
    """Load and fully validate an FD-q HDF5 cache file."""

    source = Path(path).expanduser().resolve()
    with h5py.File(source, "r") as handle:
        schema_version = int(handle.attrs.get("schema_version", -1))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FD-q schema version {schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        N = int(handle.attrs["N"])
        q = int(handle.attrs["q"])
        node_count = int(handle.attrs["node_count"])
        topology_value = handle.attrs["topology"]
        topology = (
            topology_value.decode("utf-8")
            if isinstance(topology_value, bytes)
            else str(topology_value)
        )
        period = float(handle.attrs["period"])
        expected_node_count = N + 1 if topology == "bounded" else N
        if node_count != expected_node_count:
            raise ValueError("HDF5 node_count attribute is inconsistent")
        if int(handle.attrs["stencil_size"]) != q + 1:
            raise ValueError("HDF5 stencil_size attribute is inconsistent with q")
        nodes = np.asarray(handle["nodes"], dtype=np.float64)
        stencil_indices = np.asarray(handle["stencil_indices"], dtype=np.int64)
        stencil_offsets = np.asarray(handle["stencil_offsets"], dtype=np.int64)
        weights = np.stack(
            [
                np.asarray(handle[f"weights/d{derivative}"], dtype=np.float64)
                for derivative in range(3)
            ]
        )
        log_error = np.asarray(handle["diagnostics/log_error"], dtype=np.float64)

    rule = FDQRule(
        N,
        q,
        nodes,
        stencil_indices,
        stencil_offsets,
        weights,
        log_error,
        topology,
        period,
    )
    _validate_rule(rule)
    return rule


def load_or_create_fdq_rule(
    N: int, q: int, directory: Path | str | None = None
) -> FDQRule:
    """Load N{N}_q{q}.h5 from the cache, or calculate and save it."""

    N, q = _validate_parameters(N, q)
    path = _default_cache_path(N, q, directory, "bounded")
    if path.exists():
        try:
            rule = load_fdq_rule(path)
            if rule.N != N or rule.q != q or rule.topology != "bounded":
                raise ValueError("cached bounded rule metadata does not match")
            return rule
        except (KeyError, OSError, TypeError, ValueError) as error:
            print(f"FD-q rule cache stale, rebuilding {path}: {error}", file=sys.stderr)
    rule = calculate_fdq_rule(N, q)
    save_fdq_rule(rule, path)
    return rule


def load_or_create_periodic_fdq_rule(
    node_count: int,
    q: int,
    directory: Path | str | None = None,
    period: float = 2.0,
) -> FDQRule:
    """Load or atomically regenerate a periodic half-open FD-q rule."""

    rule_template = calculate_periodic_fdq_rule(node_count, q, period)
    path = _default_cache_path(node_count, q, directory, "periodic")
    if path.exists():
        try:
            rule = load_fdq_rule(path)
            if (
                rule.N != node_count
                or rule.q != q
                or rule.topology != "periodic"
                or not np.isclose(rule.period, period)
            ):
                raise ValueError("cached periodic rule metadata does not match")
            return rule
        except (KeyError, OSError, TypeError, ValueError) as error:
            print(f"FD-q rule cache stale, rebuilding {path}: {error}", file=sys.stderr)
    legacy = path.with_name(f"N{node_count - 1}_q{q}.h5")
    if not path.exists() and legacy.exists():
        print(f"FD-q rule cache schema is obsolete, rebuilding: {legacy}", file=sys.stderr)
    save_fdq_rule(rule_template, path)
    return rule_template


__all__ = [
    "FDQRule",
    "calculate_fdq_nodes",
    "calculate_periodic_fdq_rule",
    "calculate_fdq_rule",
    "fornberg_weights",
    "load_fdq_rule",
    "load_or_create_fdq_rule",
    "load_or_create_periodic_fdq_rule",
    "save_fdq_rule",
]
