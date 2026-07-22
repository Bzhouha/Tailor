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
from scipy.interpolate import RectBivariateSpline

from .fdq_nodes import FDQRule, load_or_create_fdq_rule


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
SCHEMA_VERSION = 1
INTERPOLATION_METHOD = "tensor_product_quintic_bspline"
INTERPOLATION_VERSION = 1
DEFAULT_PETSC_PREFIX = Path("/Users/becrazy/Wro/petsc/arch-complex")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_structured_sample(path: Path) -> tuple[ComplexArray, ComplexArray, int, int]:
    with h5py.File(path, "r") as handle:
        if "grid" not in handle or "baseflow" not in handle:
            raise ValueError("sample HDF5 must contain /grid and /baseflow")
        try:
            ny = int(handle.attrs["Ny"])
            nz = int(handle.attrs["Nz"])
        except KeyError as error:
            raise ValueError("sample HDF5 must define root attributes Ny and Nz") from error
        grid_parts = np.asarray(handle["grid"], dtype=np.float64)
        baseflow_parts = np.asarray(handle["baseflow"], dtype=np.float64)

    if ny < 6 or nz < 6:
        raise ValueError("sample Ny and Nz must both be at least 6")
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
    return grid, baseflow, ny, nz


def _interpolate_components(
    values: ComplexArray,
    z_old: FloatArray,
    y_old: FloatArray,
    z_new: FloatArray,
    y_new: FloatArray,
) -> ComplexArray:
    result = np.empty((z_new.size, y_new.size, values.shape[2]), dtype=np.complex128)
    for component in range(values.shape[2]):
        result[..., component] = quintic_tensor_spline(
            z_old, y_old, values[..., component], z_new, y_new
        )
    return result


def _differentiate(values: FloatArray, rule: FDQRule, axis: int) -> FloatArray:
    weights = rule.weights[1]
    if axis == 1:
        sampled = values[:, rule.stencil_indices]
        return np.einsum("zij,ij->zi", sampled, weights)
    if axis == 0:
        sampled = values[rule.stencil_indices, :]
        return np.einsum("izj,iz->ij", sampled, weights)
    raise ValueError("axis must be 0 or 1")


def _validate_interpolated_fields(
    source_grid: ComplexArray,
    source_baseflow: ComplexArray,
    grid: ComplexArray,
    baseflow: ComplexArray,
    y_rule: FDQRule,
    z_rule: FDQRule,
) -> float:
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
    z_z = _differentiate(physical_z, z_rule, axis=0)
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


def _configure_petsc_bindings(prefix: Path) -> None:
    prefix = prefix.expanduser().resolve()
    bindings = prefix / "lib"
    if not bindings.is_dir():
        raise RuntimeError(f"PETSc Python bindings directory not found: {bindings}")
    os.environ["PETSC_DIR"] = str(prefix.parent)
    os.environ["PETSC_ARCH"] = prefix.name
    sys.path.insert(0, str(bindings))


def _write_petsc_vectors(
    path: Path, grid: ComplexArray, baseflow: ComplexArray
) -> None:
    prefix = Path(os.environ.get("TAILOR_PETSC_PREFIX", str(DEFAULT_PETSC_PREFIX)))
    _configure_petsc_bindings(prefix)
    import petsc4py

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
    group.attrs["N"] = rule.N
    group.attrs["q"] = rule.q
    group.create_dataset("nodes", data=rule.nodes, dtype=np.float64)
    group.create_dataset(
        "stencil_indices", data=rule.stencil_indices, dtype=np.int64
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
    expected: dict[str, int | str],
    y_rule: FDQRule,
    z_rule: FDQRule,
) -> None:
    with h5py.File(path, "r+") as handle:
        for name, value in expected.items():
            handle.attrs[name] = value
        discretization = handle.create_group("discretization")
        _write_rule(discretization.create_group("y"), y_rule)
        _write_rule(discretization.create_group("z"), z_rule)


def _normalize_attribute(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_output(
    path: Path,
    expected: dict[str, int | str],
    y_rule: FDQRule,
    z_rule: FDQRule,
) -> None:
    ny = y_rule.N + 1
    nz = z_rule.N + 1
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
            if group["nodes"].shape != rule.nodes.shape:
                raise ValueError(f"cached {direction} FD-q nodes have the wrong shape")
            if group["stencil_indices"].shape != rule.stencil_indices.shape:
                raise ValueError(
                    f"cached {direction} FD-q stencil indices have the wrong shape"
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
    grid, baseflow, ny, nz = _read_structured_sample(source)
    if isinstance(q_y, bool) or not isinstance(q_y, int):
        raise TypeError("q_y must be an integer polynomial degree")
    if isinstance(q_z, bool) or not isinstance(q_z, int):
        raise TypeError("q_z must be an integer polynomial degree")
    if not 2 <= q_y <= ny - 1:
        raise ValueError(f"q_y must satisfy 2 <= q_y <= {ny - 1}")
    if not 2 <= q_z <= nz - 1:
        raise ValueError(f"q_z must satisfy 2 <= q_z <= {nz - 1}")

    y_rule = load_or_create_fdq_rule(ny - 1, q_y, rules)
    z_rule = load_or_create_fdq_rule(nz - 1, q_z, rules)
    source_sha256 = _sha256(source)
    y_rule_sha256 = _sha256(rules / f"N{ny - 1}_q{q_y}.h5")
    z_rule_sha256 = _sha256(rules / f"N{nz - 1}_q{q_z}.h5")
    expected: dict[str, int | str] = {
        "schema_version": SCHEMA_VERSION,
        "interpolation_method": INTERPOLATION_METHOD,
        "interpolation_version": INTERPOLATION_VERSION,
        "source_file": str(source),
        "source_sha256": source_sha256,
        "y_rule_sha256": y_rule_sha256,
        "z_rule_sha256": z_rule_sha256,
        "ordering": "k_j_dof",
        "Ny": ny,
        "Nz": nz,
        "N_y": ny - 1,
        "q_y": q_y,
        "N_z": nz - 1,
        "q_z": q_z,
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
    z_old = np.linspace(-1.0, 1.0, nz, dtype=np.float64)
    interpolated_grid = _interpolate_components(
        grid, z_old, y_old, z_rule.nodes, y_rule.nodes
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
