"""Collective failure tests for malformed schema-v2 FD-q inputs."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import shutil
import subprocess
import tempfile

import h5py
import numpy as np
import yaml


Mutation = Callable[[h5py.File], None]


def fdq_path(config_path: Path) -> Path:
    """Resolve the prepared FD-q file used by a YAML config."""
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


def missing_weight(handle: h5py.File) -> None:
    """Remove a required derivative-weight dataset."""
    del handle["discretization/y/weights/d1"]


def wrong_q(handle: h5py.File) -> None:
    """Make a group polynomial degree disagree with root metadata."""
    handle.attrs["q_y"] = int(handle.attrs["q_y"]) - 1


def wrong_shape(handle: h5py.File) -> None:
    """Replace stencil indices with an incorrectly shaped dataset."""
    weights = handle["discretization/y/weights"]
    truncated = np.asarray(weights["d2"][:, :-1])
    del weights["d2"]
    weights.create_dataset("d2", data=truncated)


def out_of_range_index(handle: h5py.File) -> None:
    """Inject a periodic stencil index beyond the node count."""
    indices = handle["discretization/y/stencil_indices"]
    indices[0, 0] = int(handle.attrs["Ny"])


def non_finite_weight(handle: h5py.File) -> None:
    """Inject a non-finite differentiation weight."""
    handle["discretization/y/weights/d1"][0, 0] = np.nan


SCENARIOS: tuple[tuple[str, Mutation, str], ...] = (
    ("missing_weight", missing_weight, "Missing HDF5 dataset"),
    ("wrong_q", wrong_q, "do not match the YAML configuration"),
    ("wrong_shape", wrong_shape, "Unexpected shape for HDF5 dataset"),
    ("out_of_range_index", out_of_range_index, "out-of-range index"),
    ("non_finite_weight", non_finite_weight, "weight is not finite"),
)


def create_case(
    root: Path,
    source_config: Path,
    source_fdq: Path,
    mutation: Mutation,
) -> Path:
    """Copy a valid case and apply one HDF5 corruption."""
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


def assert_collective_failure(
    command: list[str], expected_message: str, scenario: str
) -> None:
    """Check serial/MPI failure text and guard against hangs."""
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    if completed.returncode == 0:
        raise AssertionError(
            f"{scenario}: invalid FD-q input unexpectedly succeeded"
        )
    if expected_message not in completed.stdout:
        raise AssertionError(
            f"{scenario}: expected {expected_message!r} in solver output\n"
            f"{completed.stdout}"
        )


def main() -> int:
    """Run every malformed-input scenario in serial and with two ranks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True, type=Path)
    arguments = parser.parse_args()

    solver = arguments.solver.resolve()
    config = arguments.config.resolve()
    mpiexec = arguments.mpiexec.resolve()
    source_fdq = fdq_path(config)

    for name, mutation, expected_message in SCENARIOS:
        with tempfile.TemporaryDirectory(prefix=f"tailor-{name}-") as temporary:
            invalid_config = create_case(
                Path(temporary), config, source_fdq, mutation
            )
            serial = [
                str(solver),
                "-c",
                str(invalid_config),
                "-tailor_assemble_only",
            ]
            parallel = [
                str(mpiexec),
                "-n",
                "2",
                str(solver),
                "-c",
                str(invalid_config),
            ]
            assert_collective_failure(serial, expected_message, f"{name}/serial")
            assert_collective_failure(parallel, expected_message, f"{name}/mpi2")

    print("All malformed FD-q loader scenarios failed collectively as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
