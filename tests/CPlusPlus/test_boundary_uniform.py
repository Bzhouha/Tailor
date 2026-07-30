"""Manufactured uniform-flow tests for characteristic far-field modes."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import h5py
import numpy as np
import yaml


def fdq_path(config_path: Path) -> Path:
    """Resolve the prepared FD-q file associated with a config."""
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


def create_uniform_case(root: Path, source_config: Path) -> Path:
    """Copy a case and replace its basic flow with a uniform state."""
    target_config = root / "config.yaml"
    shutil.copy2(source_config, target_config)
    source_fdq = fdq_path(source_config)
    target_fdq = fdq_path(target_config)
    target_fdq.parent.mkdir(parents=True)
    shutil.copy2(source_fdq, target_fdq)
    with h5py.File(target_fdq, "r+") as handle:
        baseflow = np.asarray(handle["baseflow"])
        baseflow[..., 0] = np.asarray([1.0, 1.0, 0.0, 0.0, 1.0])
        baseflow[..., 1] = 0.0
        handle["baseflow"][:] = baseflow
    return target_config


def run(
    test_binary: Path,
    config: Path,
    mpiexec: Path | None,
) -> None:
    """Check expected acoustic/neutral mode counts in serial or MPI."""
    command = [
        str(test_binary),
        "-c",
        str(config),
        "-expected_incoming_per_node",
        "1",
        "-expected_neutral_per_node",
        "3",
    ]
    if mpiexec is not None:
        command = [str(mpiexec), "-n", "2", *command]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"Uniform far-field test failed ({completed.returncode}):\n"
            f"{completed.stdout}"
        )


def main() -> int:
    """Run the uniform characteristic test in serial and on two ranks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-binary", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True, type=Path)
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="tailor-uniform-farfield-"
    ) as temporary:
        config = create_uniform_case(
            Path(temporary), arguments.config.resolve()
        )
        run(arguments.test_binary.resolve(), config, None)
        run(
            arguments.test_binary.resolve(),
            config,
            arguments.mpiexec.resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
