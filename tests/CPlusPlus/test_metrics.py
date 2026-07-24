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


METRIC_NAMES = (
    "xi_y",
    "xi_z",
    "eta_y",
    "eta_z",
    "xi_yy",
    "xi_zz",
    "xi_yz",
    "eta_yy",
    "eta_zz",
    "eta_yz",
)
Mutation = Callable[[h5py.File], None]


def fdq_path(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    folder = Path(config["Folder"])
    source = Path(config["File"])
    if not source.is_absolute():
        source = config_path.parent / folder / source
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
    return np.einsum("jis,is->ji", values[:, indices], weights)


def differentiate_eta(
    values: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    return np.einsum("jsi,js->ji", values[indices, :], weights)


def reference_metrics(path: Path) -> tuple[tuple[float, float], np.ndarray]:
    with h5py.File(path, "r") as handle:
        ny = int(handle.attrs["Ny"])
        nz = int(handle.attrs["Nz"])
        raw_grid = np.asarray(handle["grid"])
        if raw_grid.shape != (ny * nz, 3, 2):
            raise AssertionError(f"Unexpected PETSc grid shape: {raw_grid.shape}")
        grid = raw_grid.reshape(nz, ny, 3, 2)
        y = grid[:, :, 1, 0]
        z = grid[:, :, 2, 0]
        xi_indices = np.asarray(
            handle["discretization/y/stencil_indices"], dtype=np.int64
        )
        eta_indices = np.asarray(
            handle["discretization/z/stencil_indices"], dtype=np.int64
        )
        xi_weights = np.asarray(
            handle["discretization/y/weights/d1"], dtype=np.float64
        )
        eta_weights = np.asarray(
            handle["discretization/z/weights/d1"], dtype=np.float64
        )

    y_xi = differentiate_xi(y, xi_indices, xi_weights)
    y_eta = differentiate_eta(y, eta_indices, eta_weights)
    z_xi = differentiate_xi(z, xi_indices, xi_weights)
    z_eta = differentiate_eta(z, eta_indices, eta_weights)
    jacobian = y_xi * z_eta - y_eta * z_xi

    xi_y = z_eta / jacobian
    xi_z = -y_eta / jacobian
    eta_y = -z_xi / jacobian
    eta_z = y_xi / jacobian

    dxi = lambda values: differentiate_xi(values, xi_indices, xi_weights)
    deta = lambda values: differentiate_eta(values, eta_indices, eta_weights)
    metrics = (
        xi_y,
        xi_z,
        eta_y,
        eta_z,
        xi_y * dxi(xi_y) + eta_y * deta(xi_y),
        xi_z * dxi(xi_z) + eta_z * deta(xi_z),
        xi_z * dxi(xi_y) + eta_z * deta(xi_y),
        xi_y * dxi(eta_y) + eta_y * deta(eta_y),
        xi_z * dxi(eta_z) + eta_z * deta(eta_z),
        xi_z * dxi(eta_y) + eta_z * deta(eta_y),
    )
    norms = np.asarray([np.linalg.norm(value) for value in metrics])
    return (float(np.min(jacobian)), float(np.max(jacobian))), norms


def solver_command(
    solver: Path, config: Path, mpiexec: Path | None
) -> list[str]:
    command = [str(solver), "-c", str(config)]
    if mpiexec is not None:
        command = [str(mpiexec), "-n", "2", *command]
    return command


def run_solver(
    solver: Path, config: Path, mpiexec: Path | None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        solver_command(solver, config, mpiexec),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )


def parse_diagnostics(output: str) -> tuple[tuple[float, float], np.ndarray]:
    jacobian_match = re.search(
        r"Jacobian range: \[([^,]+), ([^\]]+)\]", output
    )
    if jacobian_match is None:
        raise AssertionError(f"Missing Jacobian diagnostics:\n{output}")
    jacobian = (float(jacobian_match.group(1)), float(jacobian_match.group(2)))

    norms = []
    for name in METRIC_NAMES:
        match = re.search(
            rf"metric norm {re.escape(name)}: ([^\s]+)", output
        )
        if match is None:
            raise AssertionError(f"Missing {name} norm:\n{output}")
        norms.append(float(match.group(1)))
    return jacobian, np.asarray(norms)


def assert_successful_metrics(
    solver: Path,
    config: Path,
    mpiexec: Path | None,
    scenario: str,
) -> None:
    completed = run_solver(solver, config, mpiexec)
    if completed.returncode != 0:
        raise AssertionError(
            f"{scenario}: solver failed with {completed.returncode}\n"
            f"{completed.stdout}"
        )
    actual_jacobian, actual_norms = parse_diagnostics(completed.stdout)
    expected_jacobian, expected_norms = reference_metrics(fdq_path(config))
    np.testing.assert_allclose(
        actual_jacobian,
        expected_jacobian,
        rtol=5.0e-11,
        atol=5.0e-11,
        err_msg=f"{scenario}: Jacobian range",
    )
    np.testing.assert_allclose(
        actual_norms,
        expected_norms,
        rtol=5.0e-10,
        atol=5.0e-11,
        err_msg=f"{scenario}: metric norms",
    )


def create_case(
    root: Path,
    source_config: Path,
    source_fdq: Path,
    mutation: Mutation,
) -> Path:
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


def make_skewed_curved_grid(handle: h5py.File) -> None:
    xi = np.asarray(handle["discretization/y/nodes"])
    eta = np.asarray(handle["discretization/z/nodes"])
    xi_grid, eta_grid = np.meshgrid(xi, eta, indexing="xy")
    ny = xi.size
    nz = eta.size
    grid = np.asarray(handle["grid"]).reshape(nz, ny, 3, 2)
    grid[:, :, 1, 0] = (
        2.0 * xi_grid
        + 0.25 * eta_grid
        + 0.08 * xi_grid * eta_grid
        + 0.03 * xi_grid**2
    )
    grid[:, :, 2, 0] = (
        -0.15 * xi_grid
        + 1.7 * eta_grid
        + 0.05 * xi_grid * eta_grid
        + 0.04 * eta_grid**2
    )
    grid[:, :, 1:3, 1] = 0.0
    handle["grid"][:] = grid.reshape(ny * nz, 3, 2)


def make_degenerate_grid(handle: h5py.File) -> None:
    grid = np.asarray(handle["grid"])
    grid[:, 1:3, :] = 0.0
    handle["grid"][:] = grid


def make_complex_grid(handle: h5py.File) -> None:
    grid = np.asarray(handle["grid"])
    grid[:, 1, 1] = 1.0e-6
    handle["grid"][:] = grid


def assert_collective_failure(
    solver: Path,
    config: Path,
    mpiexec: Path | None,
    expected_message: str,
    scenario: str,
) -> None:
    completed = run_solver(solver, config, mpiexec)
    if completed.returncode == 0:
        raise AssertionError(f"{scenario}: invalid grid unexpectedly succeeded")
    if expected_message not in completed.stdout:
        raise AssertionError(
            f"{scenario}: expected {expected_message!r}\n{completed.stdout}"
        )


def main() -> int:
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
        assert_successful_metrics(solver, config, launcher, f"real/{label}")

    with tempfile.TemporaryDirectory(
        prefix="tailor-skewed-metrics-"
    ) as temporary:
        skewed_config = create_case(
            Path(temporary), config, source_fdq, make_skewed_curved_grid
        )
        _, reference_norms = reference_metrics(fdq_path(skewed_config))
        if np.any(reference_norms[4:] < 1.0e-3):
            raise AssertionError(
                "Manufactured grid does not exercise every second metric"
            )
        for label, launcher in (("serial", None), ("mpi2", mpiexec)):
            assert_successful_metrics(
                solver, skewed_config, launcher, f"skewed/{label}"
            )

    failure_scenarios = (
        (
            "degenerate",
            make_degenerate_grid,
            "Grid Jacobian must be positive and nonsingular",
        ),
        (
            "complex",
            make_complex_grid,
            "Grid y/z coordinates must be real",
        ),
    )
    for name, mutation, expected_message in failure_scenarios:
        with tempfile.TemporaryDirectory(
            prefix=f"tailor-{name}-metrics-"
        ) as temporary:
            invalid_config = create_case(
                Path(temporary), config, source_fdq, mutation
            )
            for label, launcher in (("serial", None), ("mpi2", mpiexec)):
                assert_collective_failure(
                    solver,
                    invalid_config,
                    launcher,
                    expected_message,
                    f"{name}/{label}",
                )

    print("Metrics regression and collective failure scenarios passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
