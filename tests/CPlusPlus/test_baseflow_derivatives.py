"""MPI integration tests for physical base-flow derivatives."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import h5py
import numpy as np
import yaml


DERIVATIVE_NAMES = ("dy", "dz", "dyy", "dzz", "dyz")
Mutation = Callable[[h5py.File], None]


def fdq_path(config_path: Path) -> Path:
    """Resolve the prepared FD-q file associated with a test config."""
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    source = Path(config["File"])
    if not source.is_absolute():
        source = config_path.parent / Path(config["Folder"]) / source
    q_y = int(config["Q-Value"]["y"])
    q_z = int(config["Q-Value"]["z"])
    return (
        source.parent
        / "FD-q"
        / f"fdq_{source.stem}_qy{q_y}_qz{q_z}.h5"
    ).resolve()


def differentiate_xi(
    values: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Apply a bounded xi differentiation rule to a tensor field."""
    return np.einsum("jis,is->ji", values[:, indices], weights)


def differentiate_eta(
    values: np.ndarray,
    indices: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    translation: float = 0.0,
) -> np.ndarray:
    """Apply a periodic eta rule, optionally unwrapping a translation."""
    sampled = values[indices, :]
    if translation != 0.0:
        rows = np.arange(indices.shape[0], dtype=np.int64)[:, None]
        wraps = (rows + offsets - indices) // indices.shape[0]
        sampled = sampled + wraps[:, :, None] * translation
    return np.einsum("jsi,js->ji", sampled, weights)


def reference_derivatives(path: Path) -> np.ndarray:
    """Compute independent physical derivative norms from HDF5 data."""
    with h5py.File(path, "r") as handle:
        ny = int(handle.attrs["Ny"])
        nz = int(handle.attrs["Nz"])
        grid = np.asarray(handle["grid"]).reshape(nz, ny, 3, 2)[..., 0]
        baseflow = np.asarray(handle["baseflow"]).reshape(nz, ny, 5, 2)[
            ..., 0
        ]
        xi_indices = np.asarray(
            handle["discretization/y/stencil_indices"], dtype=np.int64
        )
        eta_indices = np.asarray(
            handle["discretization/z/stencil_indices"], dtype=np.int64
        )
        eta_offsets = np.asarray(
            handle["discretization/z/stencil_offsets"], dtype=np.int64
        )
        spanwise_period = float(handle.attrs["spanwise_period"])
        xi_d1 = np.asarray(
            handle["discretization/y/weights/d1"], dtype=np.float64
        )
        xi_d2 = np.asarray(
            handle["discretization/y/weights/d2"], dtype=np.float64
        )
        eta_d1 = np.asarray(
            handle["discretization/z/weights/d1"], dtype=np.float64
        )
        eta_d2 = np.asarray(
            handle["discretization/z/weights/d2"], dtype=np.float64
        )

    dxi = lambda values: differentiate_xi(values, xi_indices, xi_d1)
    deta = lambda values: differentiate_eta(
        values, eta_indices, eta_offsets, eta_d1
    )
    dxixi = lambda values: differentiate_xi(values, xi_indices, xi_d2)
    detaeta = lambda values: differentiate_eta(
        values, eta_indices, eta_offsets, eta_d2
    )
    dxieta = lambda values: deta(dxi(values))

    y = grid[:, :, 1]
    z = grid[:, :, 2]
    y_xi = dxi(y)
    y_eta = deta(y)
    z_xi = dxi(z)
    z_eta = differentiate_eta(
        z,
        eta_indices,
        eta_offsets,
        eta_d1,
        translation=spanwise_period,
    )
    jacobian = y_xi * z_eta - y_eta * z_xi
    xi_y = z_eta / jacobian
    xi_z = -y_eta / jacobian
    eta_y = -z_xi / jacobian
    eta_z = y_xi / jacobian
    xi_yy = xi_y * dxi(xi_y) + eta_y * deta(xi_y)
    xi_zz = xi_z * dxi(xi_z) + eta_z * deta(xi_z)
    xi_yz = xi_z * dxi(xi_y) + eta_z * deta(xi_y)
    eta_yy = xi_y * dxi(eta_y) + eta_y * deta(eta_y)
    eta_zz = xi_z * dxi(eta_z) + eta_z * deta(eta_z)
    eta_yz = xi_z * dxi(eta_y) + eta_z * deta(eta_y)

    derivatives = np.empty((5, 5, nz, ny), dtype=np.float64)
    for field in range(5):
        value = baseflow[:, :, field]
        value_xi = dxi(value)
        value_eta = deta(value)
        value_xixi = dxixi(value)
        value_etaeta = detaeta(value)
        value_xieta = dxieta(value)
        derivatives[0, field] = xi_y * value_xi + eta_y * value_eta
        derivatives[1, field] = xi_z * value_xi + eta_z * value_eta
        derivatives[2, field] = (
            xi_y**2 * value_xixi
            + 2.0 * xi_y * eta_y * value_xieta
            + eta_y**2 * value_etaeta
            + xi_yy * value_xi
            + eta_yy * value_eta
        )
        derivatives[3, field] = (
            xi_z**2 * value_xixi
            + 2.0 * xi_z * eta_z * value_xieta
            + eta_z**2 * value_etaeta
            + xi_zz * value_xi
            + eta_zz * value_eta
        )
        derivatives[4, field] = (
            xi_y * xi_z * value_xixi
            + (xi_y * eta_z + eta_y * xi_z) * value_xieta
            + eta_y * eta_z * value_etaeta
            + xi_yz * value_xi
            + eta_yz * value_eta
        )
    return np.linalg.norm(derivatives.reshape(5, 5, -1), axis=2)


def solver_command(
    solver: Path, config: Path, mpiexec: Path | None
) -> list[str]:
    """Build an assemble-only serial or two-rank solver command."""
    command = [
        str(solver),
        "-c",
        str(config),
        "-tailor_assemble_only",
    ]
    if mpiexec is not None:
        command = [str(mpiexec), "-n", "2", *command]
    return command


def run_solver(
    solver: Path, config: Path, mpiexec: Path | None
) -> subprocess.CompletedProcess[str]:
    """Run one solver scenario and capture combined output."""
    return subprocess.run(
        solver_command(solver, config, mpiexec),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )


def parse_norms(output: str) -> np.ndarray:
    """Extract the 25 derivative norms from solver diagnostics."""
    norms = []
    for derivative in DERIVATIVE_NAMES:
        match = re.search(
            rf"base-flow derivative norm {derivative}: "
            r"([^\s]+) ([^\s]+) ([^\s]+) ([^\s]+) ([^\s]+)",
            output,
        )
        if match is None:
            raise AssertionError(
                f"Missing {derivative} derivative norms:\n{output}"
            )
        norms.append([float(value) for value in match.groups()])
    return np.asarray(norms)


def assert_success(
    solver: Path,
    config: Path,
    mpiexec: Path | None,
    scenario: str,
) -> None:
    """Compare solver derivative norms with the independent reference."""
    completed = run_solver(solver, config, mpiexec)
    if completed.returncode != 0:
        raise AssertionError(
            f"{scenario}: solver failed with {completed.returncode}\n"
            f"{completed.stdout}"
        )
    np.testing.assert_allclose(
        parse_norms(completed.stdout),
        reference_derivatives(fdq_path(config)),
        rtol=5.0e-10,
        atol=5.0e-10,
        err_msg=scenario,
    )


def create_case(
    root: Path,
    source_config: Path,
    source_fdq: Path,
    mutation: Mutation,
) -> Path:
    """Copy a case, mutate its prepared HDF5 file, and return its config."""
    case = root / "case"
    case.mkdir(parents=True)
    target_config = case / "config.yaml"
    shutil.copy2(source_config, target_config)
    target_fdq = fdq_path(target_config)
    target_fdq.parent.mkdir(parents=True)
    shutil.copy2(source_fdq, target_fdq)
    with h5py.File(target_fdq, "r+") as handle:
        mutation(handle)
    return target_config


def make_manufactured_case(handle: h5py.File) -> None:
    """Install a periodic curved grid and fully varying positive base flow."""
    xi = np.asarray(handle["discretization/y/nodes"])
    eta = np.asarray(handle["discretization/z/nodes"])
    xi_grid, eta_grid = np.meshgrid(xi, eta, indexing="xy")
    ny = xi.size
    nz = eta.size
    computational_period = float(
        handle["discretization/z"].attrs["period"]
    )
    physical_period = float(handle.attrs["spanwise_period"])
    theta = 2.0 * np.pi * (eta_grid - eta[0]) / computational_period
    phase = (eta_grid - eta[0]) / computational_period
    sine = np.sin(theta)
    cosine = np.cos(theta)
    sine2 = np.sin(2.0 * theta)

    grid = np.asarray(handle["grid"]).reshape(nz, ny, 3, 2)
    grid[:, :, 1, 0] = (
        2.0 * xi_grid
        + 0.03 * xi_grid**2
        + 0.12 * sine
        + 0.05 * xi_grid * sine
    )
    grid[:, :, 2, 0] = (
        physical_period * phase
        -0.15 * xi_grid
        + 0.08 * cosine
        + 0.04 * xi_grid * cosine
    )
    grid[:, :, 1:3, 1] = 0.0
    handle["grid"][:] = grid.reshape(ny * nz, 3, 2)

    baseflow = np.asarray(handle["baseflow"]).reshape(nz, ny, 5, 2)
    baseflow[:, :, 0, 0] = (
        1.2
        + 0.08 * xi_grid
        + 0.02 * xi_grid**2
        + 0.04 * sine
        + 0.03 * xi_grid * cosine
    )
    baseflow[:, :, 1, 0] = (
        2.4
        + 0.20 * xi_grid
        - 0.07 * cosine
        + 0.04 * xi_grid * sine
        + 0.025 * sine2
    )
    baseflow[:, :, 2, 0] = (
        0.12
        - 0.03 * xi_grid
        + 0.01 * xi_grid**2
        + 0.06 * sine
        + 0.02 * xi_grid * cosine
    )
    baseflow[:, :, 3, 0] = (
        -0.08
        + 0.05 * xi_grid
        + 0.04 * cosine
        - 0.025 * xi_grid * sine
        + 0.015 * sine2
    )
    baseflow[:, :, 4, 0] = (
        1.5
        + 0.09 * xi_grid
        + 0.025 * xi_grid**2
        + 0.05 * sine
        + 0.035 * xi_grid * cosine
        + 0.02 * sine2
    )
    baseflow[:, :, :, 1] = 0.0
    handle["baseflow"][:] = baseflow.reshape(ny * nz, 5, 2)


def make_complex_baseflow(handle: h5py.File) -> None:
    """Inject a nonzero imaginary basic-flow component."""
    baseflow = np.asarray(handle["baseflow"])
    baseflow[:, 1, 1] = 1.0e-6
    handle["baseflow"][:] = baseflow


def make_nonpositive_density(handle: h5py.File) -> None:
    """Inject an invalid zero density."""
    handle["baseflow"][0, 0, 0] = 0.0


def make_nonpositive_temperature(handle: h5py.File) -> None:
    """Inject an invalid negative temperature."""
    handle["baseflow"][0, 4, 0] = -1.0


def make_nonfinite_baseflow(handle: h5py.File) -> None:
    """Inject a NaN basic-flow component."""
    handle["baseflow"][0, 2, 0] = np.nan


def assert_failure(
    solver: Path,
    config: Path,
    mpiexec: Path | None,
    message: str,
    scenario: str,
) -> None:
    """Require an invalid case to fail collectively with a diagnostic."""
    completed = run_solver(solver, config, mpiexec)
    if completed.returncode == 0:
        raise AssertionError(f"{scenario}: invalid base flow succeeded")
    if message not in completed.stdout:
        raise AssertionError(
            f"{scenario}: expected {message!r}\n{completed.stdout}"
        )


def main() -> int:
    """Run real, manufactured, serial, MPI, and failure scenarios."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True, type=Path)
    arguments = parser.parse_args()

    solver = arguments.solver.resolve()
    config = arguments.config.resolve()
    mpiexec = arguments.mpiexec.resolve()
    source_fdq = fdq_path(config)

    for label, launcher in (("serial", None), ("mpi2", mpiexec)):
        assert_success(solver, config, launcher, f"real/{label}")

    with tempfile.TemporaryDirectory(
        prefix="tailor-manufactured-baseflow-"
    ) as temporary:
        manufactured_config = create_case(
            Path(temporary), config, source_fdq, make_manufactured_case
        )
        expected = reference_derivatives(fdq_path(manufactured_config))
        if np.any(expected < 1.0e-5):
            raise AssertionError(
                "Manufactured case does not exercise every derivative"
            )
        for label, launcher in (("serial", None), ("mpi2", mpiexec)):
            assert_success(
                solver,
                manufactured_config,
                launcher,
                f"manufactured/{label}",
            )

    failures = (
        ("complex", make_complex_baseflow, "Base-flow values must be real"),
        (
            "density",
            make_nonpositive_density,
            "density and temperature must both be positive",
        ),
        (
            "temperature",
            make_nonpositive_temperature,
            "density and temperature must both be positive",
        ),
        ("nonfinite", make_nonfinite_baseflow, "values are not finite"),
    )
    for name, mutation, message in failures:
        with tempfile.TemporaryDirectory(
            prefix=f"tailor-{name}-baseflow-"
        ) as temporary:
            invalid_config = create_case(
                Path(temporary), config, source_fdq, mutation
            )
            for label, launcher in (("serial", None), ("mpi2", mpiexec)):
                assert_failure(
                    solver,
                    invalid_config,
                    launcher,
                    message,
                    f"{name}/{label}",
                )

    print("Base-flow derivative regression and failure scenarios passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
