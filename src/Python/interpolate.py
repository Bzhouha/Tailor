"""Interpolate a structured FDM slice onto a tensor-product FD-q grid."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
from typing import Sequence

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline, make_interp_spline

from .fdq_nodes import (
    FDQRule,
    load_or_create_fdq_rule,
    load_or_create_periodic_fdq_rule,
)


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
SCHEMA_VERSION = 2
INTERPOLATION_METHOD = "bounded_y_periodic_z_quintic_bspline"
INTERPOLATION_VERSION = 3


def quintic_tensor_spline(
    z_old: FloatArray,
    y_old: FloatArray,
    values: NDArray[np.generic],
    z_new: FloatArray,
    y_new: FloatArray,
) -> NDArray[np.generic]:
    """Interpolate one real or complex scalar field with an exact quintic spline."""

    z_old = np.asarray(z_old, dtype=np.float64)
    y_old = np.asarray(y_old, dtype=np.float64)
    z_new = np.asarray(z_new, dtype=np.float64)
    y_new = np.asarray(y_new, dtype=np.float64)
    field = np.asarray(values)
    if z_old.ndim != 1 or y_old.ndim != 1:
        raise ValueError("source logical coordinates must be one-dimensional")
    if z_new.ndim != 1 or y_new.ndim != 1:
        raise ValueError("target logical coordinates must be one-dimensional")
    if z_old.size < 6 or y_old.size < 6:
        raise ValueError("quintic B-spline interpolation requires at least 6 nodes per axis")
    if field.shape != (z_old.size, y_old.size):
        raise ValueError(
            "values must have shape "
            f"({z_old.size}, {y_old.size}); got {field.shape}"
        )
    if not all(
        np.all(np.isfinite(array))
        for array in (z_old, y_old, z_new, y_new, field)
    ):
        raise ValueError("spline coordinates and values must be finite")
    if np.any(np.diff(z_old) <= 0.0) or np.any(np.diff(y_old) <= 0.0):
        raise ValueError("source logical coordinates must be strictly increasing")

    def interpolate_real(real_values: NDArray[np.generic]) -> FloatArray:
        """Evaluate one real tensor-product quintic spline."""

        spline = RectBivariateSpline(
            z_old,
            y_old,
            np.asarray(real_values, dtype=np.float64),
            kx=5,
            ky=5,
            s=0.0,
        )
        return np.asarray(spline(z_new, y_new, grid=True), dtype=np.float64)

    if np.iscomplexobj(field):
        return interpolate_real(field.real) + 1j * interpolate_real(field.imag)
    return interpolate_real(field)


def periodic_quintic_tensor_spline(
    z_old: FloatArray,
    y_old: FloatArray,
    values: NDArray[np.generic],
    z_new: FloatArray,
    y_new: FloatArray,
    *,
    z_period: float = 2.0,
    value_translation: complex | float = 0.0,
) -> NDArray[np.generic]:
    """Interpolate a field that is periodic up to a constant spanwise shift."""

    z_old = np.asarray(z_old, dtype=np.float64)
    y_old = np.asarray(y_old, dtype=np.float64)
    z_new = np.asarray(z_new, dtype=np.float64)
    y_new = np.asarray(y_new, dtype=np.float64)
    field = np.asarray(values)
    if z_old.ndim != 1 or y_old.ndim != 1:
        raise ValueError("source logical coordinates must be one-dimensional")
    if z_new.ndim != 1 or y_new.ndim != 1:
        raise ValueError("target logical coordinates must be one-dimensional")
    if z_old.size < 6 or y_old.size < 6:
        raise ValueError("periodic quintic interpolation requires at least 6 nodes per axis")
    if field.shape != (z_old.size, y_old.size):
        raise ValueError(
            f"values must have shape {(z_old.size, y_old.size)}; got {field.shape}"
        )
    if not np.isfinite(z_period) or z_period <= 0.0:
        raise ValueError("z_period must be finite and positive")
    if not all(
        np.all(np.isfinite(array))
        for array in (z_old, y_old, z_new, y_new, field)
    ):
        raise ValueError("periodic spline coordinates and values must be finite")
    if np.any(np.diff(z_old) <= 0.0) or np.any(np.diff(y_old) <= 0.0):
        raise ValueError("source logical coordinates must be strictly increasing")
    expected_spacing = z_period / z_old.size
    if float(np.max(np.abs(np.diff(z_old) - expected_spacing))) > (
        5.0e-13 * max(1.0, abs(expected_spacing))
    ):
        raise ValueError("periodic source logical nodes must be uniformly spaced")
    if z_old[-1] >= z_old[0] + z_period:
        raise ValueError("periodic source nodes must not include the repeated endpoint")

    source_phase = (z_old - z_old[0]) / z_period
    target_phase = (z_new - z_old[0]) / z_period
    periodic_part = field - source_phase[:, None] * value_translation

    def interpolate_scalar(scalar_values: NDArray[np.generic]) -> FloatArray:
        """Interpolate one real component with periodic eta closure."""

        bounded_spline = make_interp_spline(
            y_old,
            np.asarray(scalar_values, dtype=np.float64),
            k=5,
            axis=1,
        )
        along_y = np.asarray(bounded_spline(y_new), dtype=np.float64)
        extended_z = np.concatenate((z_old, [z_old[0] + z_period]))
        extended_values = np.concatenate((along_y, along_y[:1]), axis=0)
        periodic_spline = make_interp_spline(
            extended_z,
            extended_values,
            k=5,
            bc_type="periodic",
            axis=0,
        )
        return np.asarray(periodic_spline(z_new), dtype=np.float64)

    if np.iscomplexobj(periodic_part):
        interpolated = interpolate_scalar(periodic_part.real) + 1j * interpolate_scalar(
            periodic_part.imag
        )
    else:
        interpolated = interpolate_scalar(periodic_part)
    return interpolated + target_phase[:, None] * value_translation


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a cache dependency."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_structured_sample(
    path: Path,
) -> tuple[ComplexArray, ComplexArray, int, int, float]:
    """Load and validate a half-open periodic structured source case."""

    with h5py.File(path, "r") as handle:
        if "grid" not in handle or "baseflow" not in handle:
            raise ValueError("sample HDF5 must contain /grid and /baseflow")
        try:
            ny = int(handle.attrs["Ny"])
            nz = int(handle.attrs["Nz"])
        except KeyError as error:
            raise ValueError("sample HDF5 must define root attributes Ny and Nz") from error
        if int(handle.attrs.get("spanwise_periodic", 0)) != 1:
            raise ValueError("sample HDF5 must declare spanwise_periodic=1")
        try:
            spanwise_period = float(handle.attrs["spanwise_period"])
        except KeyError as error:
            raise ValueError("sample HDF5 must define spanwise_period") from error
        grid_parts = np.asarray(handle["grid"], dtype=np.float64)
        baseflow_parts = np.asarray(handle["baseflow"], dtype=np.float64)

    if ny < 6 or nz < 6:
        raise ValueError("sample Ny and Nz must both be at least 6")
    if not np.isfinite(spanwise_period) or spanwise_period <= 0.0:
        raise ValueError("sample spanwise_period must be finite and positive")
    if grid_parts.shape != (nz, ny, 3, 2):
        raise ValueError(
            f"/grid must have shape {(nz, ny, 3, 2)}; got {grid_parts.shape}"
        )
    if baseflow_parts.shape != (nz, ny, 5, 2):
        raise ValueError(
            "/baseflow must have shape "
            f"{(nz, ny, 5, 2)}; got {baseflow_parts.shape}"
        )
    if not np.all(np.isfinite(grid_parts)) or not np.all(np.isfinite(baseflow_parts)):
        raise ValueError("sample grid and baseflow must contain only finite values")

    grid = np.asarray(grid_parts[..., 0] + 1j * grid_parts[..., 1], dtype=np.complex128)
    baseflow = np.asarray(
        baseflow_parts[..., 0] + 1j * baseflow_parts[..., 1],
        dtype=np.complex128,
    )
    if np.any(baseflow[..., 0].real <= 0.0):
        raise ValueError("sample density must be strictly positive")
    if np.any(baseflow[..., 4].real <= 0.0):
        raise ValueError("sample temperature must be strictly positive")
    return grid, baseflow, ny, nz, spanwise_period


def _interpolate_components(
    values: ComplexArray,
    z_old: FloatArray,
    y_old: FloatArray,
    z_new: FloatArray,
    y_new: FloatArray,
    translations: NDArray[np.generic] | None = None,
) -> ComplexArray:
    """Interpolate every component, optionally with affine translations."""

    result = np.empty((z_new.size, y_new.size, values.shape[2]), dtype=np.complex128)
    component_translations = (
        np.zeros(values.shape[2], dtype=np.complex128)
        if translations is None
        else np.asarray(translations, dtype=np.complex128)
    )
    if component_translations.shape != (values.shape[2],):
        raise ValueError("translations must contain one value per field component")
    for component in range(values.shape[2]):
        result[..., component] = periodic_quintic_tensor_spline(
            z_old,
            y_old,
            values[..., component],
            z_new,
            y_new,
            z_period=2.0,
            value_translation=component_translations[component],
        )
    return result


def _differentiate(
    values: FloatArray,
    rule: FDQRule,
    axis: int,
    periodic_translation: float = 0.0,
) -> FloatArray:
    """Apply one FD-q first-derivative rule along a tensor-product axis."""

    weights = rule.weights[1]
    if axis == 1:
        sampled = values[:, rule.stencil_indices]
        return np.einsum("zij,ij->zi", sampled, weights)
    if axis == 0:
        sampled = values[rule.stencil_indices, :]
        if rule.topology == "periodic" and periodic_translation != 0.0:
            rows = np.arange(rule.node_count, dtype=np.int64)[:, None]
            unwrapped = rows + rule.stencil_offsets
            wraps = (unwrapped - rule.stencil_indices) // rule.node_count
            sampled = sampled + wraps[:, :, None] * periodic_translation
        return np.einsum("izj,iz->ij", sampled, weights)
    raise ValueError("axis must be 0 or 1")


def _validate_interpolated_fields(
    source_grid: ComplexArray,
    source_baseflow: ComplexArray,
    grid: ComplexArray,
    baseflow: ComplexArray,
    y_rule: FDQRule,
    z_rule: FDQRule,
    spanwise_period: float,
) -> float:
    """Validate physical fields and return the minimum grid Jacobian."""

    if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(baseflow)):
        raise ValueError("interpolated grid and baseflow must be finite")
    if np.any(baseflow[..., 0].real <= 0.0):
        raise ValueError("interpolated density is not strictly positive")
    if np.any(baseflow[..., 4].real <= 0.0):
        raise ValueError("interpolated temperature is not strictly positive")

    for source, target, name in (
        (source_grid, grid, "grid"),
        (source_baseflow, baseflow, "baseflow"),
    ):
        source_corners = np.asarray(
            [source[0, 0], source[0, -1], source[-1, 0], source[-1, -1]]
        )
        target_corners = np.asarray(
            [target[0, 0], target[0, -1], target[-1, 0], target[-1, -1]]
        )
        scale = max(1.0, float(np.max(np.abs(source_corners))))
        error = float(np.max(np.abs(target_corners - source_corners)))
        if error > 2.0e-11 * scale:
            raise ValueError(f"{name} corner preservation failed: error={error:.3e}")

    physical_y = np.asarray(grid[..., 1].real, dtype=np.float64)
    physical_z = np.asarray(grid[..., 2].real, dtype=np.float64)
    y_y = _differentiate(physical_y, y_rule, axis=1)
    y_z = _differentiate(physical_y, z_rule, axis=0)
    z_y = _differentiate(physical_z, y_rule, axis=1)
    z_z = _differentiate(
        physical_z, z_rule, axis=0, periodic_translation=spanwise_period
    )
    jacobian = y_y * z_z - y_z * z_y
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("interpolated grid Jacobian contains non-finite values")
    minimum_jacobian = float(np.min(jacobian))
    if minimum_jacobian <= 0.0:
        raise ValueError(
            "interpolated grid Jacobian must be strictly positive; "
            f"minimum={minimum_jacobian:.6e}"
        )
    return minimum_jacobian


def _petsc_prefix_from_environment() -> Path | None:
    """Resolve a PETSc prefix from standard and Tailor environment variables."""

    configured = os.environ.get("TAILOR_PETSC_PREFIX")
    if configured:
        return Path(configured).expanduser().resolve()

    petsc_dir = os.environ.get("PETSC_DIR")
    if not petsc_dir:
        return None
    prefix = Path(petsc_dir).expanduser()
    petsc_arch = os.environ.get("PETSC_ARCH")
    if petsc_arch:
        prefix /= petsc_arch
    return prefix.resolve()


def _configure_petsc_bindings(prefix: Path) -> None:
    """Add an explicitly selected PETSc binding directory to ``sys.path``."""

    prefix = prefix.expanduser().resolve()
    bindings = prefix / "lib"
    if not bindings.is_dir():
        raise RuntimeError(f"PETSc Python bindings directory not found: {bindings}")
    if str(bindings) not in sys.path:
        sys.path.insert(0, str(bindings))


def _write_petsc_vectors(
    path: Path, grid: ComplexArray, baseflow: ComplexArray
) -> None:
    """Write grid and base flow using PETSc complex-Vec HDF5 layout."""

    prefix = _petsc_prefix_from_environment()
    if prefix is not None:
        _configure_petsc_bindings(prefix)
    try:
        import petsc4py
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "petsc4py is required and must match Tailor's complex PETSc build"
        ) from error

    petsc4py.init([sys.argv[0]])
    from petsc4py import PETSc

    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        raise RuntimeError("Tailor requires complex-scalar PETSc/petsc4py")

    grid_values = np.ascontiguousarray(grid, dtype=PETSc.ScalarType).reshape(-1)
    baseflow_values = np.ascontiguousarray(
        baseflow, dtype=PETSc.ScalarType
    ).reshape(-1)
    viewer = PETSc.Viewer().createHDF5(
        str(path), mode=PETSc.Viewer.Mode.WRITE, comm=PETSc.COMM_SELF
    )
    try:
        for name, block_size, values in (
            ("grid", 3, grid_values),
            ("baseflow", 5, baseflow_values),
        ):
            vector = PETSc.Vec().createWithArray(values, comm=PETSc.COMM_SELF)
            try:
                vector.setBlockSize(block_size)
                vector.setName(name)
                vector.view(viewer)
            finally:
                vector.destroy()
    finally:
        viewer.destroy()


def _write_rule(group: h5py.Group, rule: FDQRule) -> None:
    """Serialize one validated FD-q rule into an HDF5 group."""

    group.attrs["N"] = rule.N
    group.attrs["q"] = rule.q
    group.attrs["node_count"] = rule.node_count
    group.attrs["topology"] = rule.topology
    group.attrs["period"] = rule.period
    group.create_dataset("nodes", data=rule.nodes, dtype=np.float64)
    group.create_dataset(
        "stencil_indices", data=rule.stencil_indices, dtype=np.int64
    )
    group.create_dataset(
        "stencil_offsets", data=rule.stencil_offsets, dtype=np.int64
    )
    weights = group.create_group("weights")
    for derivative in range(3):
        weights.create_dataset(
            f"d{derivative}", data=rule.weights[derivative], dtype=np.float64
        )
    diagnostics = group.create_group("diagnostics")
    diagnostics.create_dataset("log_error", data=rule.log_error, dtype=np.float64)


def _append_metadata(
    path: Path,
    expected: dict[str, int | float | str],
    y_rule: FDQRule,
    z_rule: FDQRule,
) -> None:
    """Append schema metadata and both rules to a PETSc-created file."""

    with h5py.File(path, "r+") as handle:
        for name, value in expected.items():
            handle.attrs[name] = value
        discretization = handle.create_group("discretization")
        _write_rule(discretization.create_group("y"), y_rule)
        _write_rule(discretization.create_group("z"), z_rule)


def _normalize_attribute(value: object) -> object:
    """Convert HDF5 scalar and byte attributes to Python values."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_output(
    path: Path,
    expected: dict[str, int | float | str],
    y_rule: FDQRule,
    z_rule: FDQRule,
) -> None:
    """Validate a prepared schema-v2 case cache before reuse."""

    ny = y_rule.node_count
    nz = z_rule.node_count
    with h5py.File(path, "r") as handle:
        for name, value in expected.items():
            actual = _normalize_attribute(handle.attrs.get(name))
            if actual != value:
                raise ValueError(
                    f"cached attribute {name!r} is {actual!r}; expected {value!r}"
                )
        for name, block_size in (("grid", 3), ("baseflow", 5)):
            if name not in handle:
                raise ValueError(f"cached dataset /{name} is missing")
            dataset = handle[name]
            if dataset.shape != (nz * ny, block_size, 2):
                raise ValueError(f"cached dataset /{name} has shape {dataset.shape}")
            if int(dataset.attrs.get("complex", 0)) != 1:
                raise ValueError(f"cached dataset /{name} is not PETSc complex data")
            if not np.all(np.isfinite(dataset[...])):
                raise ValueError(f"cached dataset /{name} contains non-finite values")
        for direction, rule in (("y", y_rule), ("z", z_rule)):
            group = handle[f"discretization/{direction}"]
            if int(group.attrs["N"]) != rule.N or int(group.attrs["q"]) != rule.q:
                raise ValueError(f"cached {direction} FD-q metadata is inconsistent")
            if int(group.attrs["node_count"]) != rule.node_count:
                raise ValueError(f"cached {direction} node count is inconsistent")
            if _normalize_attribute(group.attrs["topology"]) != rule.topology:
                raise ValueError(f"cached {direction} topology is inconsistent")
            if group["nodes"].shape != rule.nodes.shape:
                raise ValueError(f"cached {direction} FD-q nodes have the wrong shape")
            if group["stencil_indices"].shape != rule.stencil_indices.shape:
                raise ValueError(
                    f"cached {direction} FD-q stencil indices have the wrong shape"
                )
            if group["stencil_offsets"].shape != rule.stencil_offsets.shape:
                raise ValueError(
                    f"cached {direction} stencil offsets have the wrong shape"
                )
            for derivative in range(3):
                if group[f"weights/d{derivative}"].shape != rule.weights[derivative].shape:
                    raise ValueError(
                        f"cached {direction} order-{derivative} weights have the wrong shape"
                    )


def prepare_fdq_case(
    source_h5: Path | str,
    output_h5: Path | str,
    q_y: int,
    q_z: int,
    rule_directory: Path | str,
) -> Path:
    """Load or create a PETSc-compatible FD-q case file."""

    source = Path(source_h5).expanduser().resolve()
    destination = Path(output_h5).expanduser().resolve()
    rules = Path(rule_directory).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"structured sample file does not exist: {source}")
    grid, baseflow, ny, nz, spanwise_period = _read_structured_sample(source)
    if isinstance(q_y, bool) or not isinstance(q_y, int):
        raise TypeError("q_y must be an integer polynomial degree")
    if isinstance(q_z, bool) or not isinstance(q_z, int):
        raise TypeError("q_z must be an integer polynomial degree")
    if not 2 <= q_y <= ny - 1:
        raise ValueError(f"q_y must satisfy 2 <= q_y <= {ny - 1}")
    if not 2 <= q_z <= nz - 1:
        raise ValueError(f"q_z must satisfy 2 <= q_z <= {nz - 1}")
    if q_z % 2 != 0:
        raise ValueError("periodic q_z must be even")

    y_rule = load_or_create_fdq_rule(ny - 1, q_y, rules)
    z_rule = load_or_create_periodic_fdq_rule(nz, q_z, rules)
    source_sha256 = _sha256(source)
    y_rule_sha256 = _sha256(rules / f"N{ny - 1}_q{q_y}.h5")
    z_rule_sha256 = _sha256(rules / f"N{nz}_q{q_z}p.h5")
    expected: dict[str, int | float | str] = {
        "schema_version": SCHEMA_VERSION,
        "interpolation_method": INTERPOLATION_METHOD,
        "interpolation_version": INTERPOLATION_VERSION,
        "source_file": source.name,
        "source_sha256": source_sha256,
        "y_rule_sha256": y_rule_sha256,
        "z_rule_sha256": z_rule_sha256,
        "ordering": "k_j_dof",
        "Ny": ny,
        "Nz": nz,
        "q_y": q_y,
        "q_z": q_z,
        "spanwise_periodic": 1,
        "spanwise_period": spanwise_period,
    }

    if destination.is_file():
        try:
            _validate_output(destination, expected, y_rule, z_rule)
        except (KeyError, OSError, TypeError, ValueError) as error:
            print(f"FD-q cache invalid, rebuilding: {error}", file=sys.stderr)
        else:
            print(f"FD-q cache hit: {destination}")
            return destination

    y_old = np.linspace(-1.0, 1.0, ny, dtype=np.float64)
    z_old = -1.0 + 2.0 * np.arange(nz, dtype=np.float64) / nz
    grid_translation = np.asarray(
        [0.0, 0.0, spanwise_period], dtype=np.complex128
    )
    interpolated_grid = _interpolate_components(
        grid,
        z_old,
        y_old,
        z_rule.nodes,
        y_rule.nodes,
        grid_translation,
    )
    interpolated_baseflow = _interpolate_components(
        baseflow, z_old, y_old, z_rule.nodes, y_rule.nodes
    )
    minimum_jacobian = _validate_interpolated_fields(
        grid,
        baseflow,
        interpolated_grid,
        interpolated_baseflow,
        y_rule,
        z_rule,
        spanwise_period,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    try:
        _write_petsc_vectors(temporary, interpolated_grid, interpolated_baseflow)
        _append_metadata(temporary, expected, y_rule, z_rule)
        _validate_output(temporary, expected, y_rule, z_rule)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        f"FD-q cache generated: {destination}\n"
        f"  Ny={ny}, q_y={q_y}; Nz={nz}, q_z={q_z}\n"
        f"  minimum Jacobian={minimum_jacobian:.16g}\n"
        f"  rho range=[{interpolated_baseflow[..., 0].real.min():.16g}, "
        f"{interpolated_baseflow[..., 0].real.max():.16g}]\n"
        f"  T range=[{interpolated_baseflow[..., 4].real.min():.16g}, "
        f"{interpolated_baseflow[..., 4].real.max():.16g}]"
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    """Construct the standalone preprocessing command-line parser."""

    parser = argparse.ArgumentParser(
        prog="python -m src.Python.interpolate",
        description="Generate or validate a PETSc-compatible FD-q case cache.",
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--q-y", required=True, type=int)
    parser.add_argument("--q-z", required=True, type=int)
    parser.add_argument("--rule-directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run standalone FD-q case preparation and report failures."""

    arguments = _parser().parse_args(argv)
    try:
        prepare_fdq_case(
            arguments.source,
            arguments.output,
            arguments.q_y,
            arguments.q_z,
            arguments.rule_directory,
        )
    except Exception as error:
        print(f"Tailor FD-q preprocessing failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["prepare_fdq_case", "quintic_tensor_spline"]
