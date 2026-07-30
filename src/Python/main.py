"""Tailor driver for FD-q preprocessing followed by the C++ solver."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULE_DIRECTORY = PROJECT_ROOT / "fdqNodes"


@dataclass(frozen=True, slots=True)
class CaseConfiguration:
    """Resolved paths and polynomial degrees needed by preprocessing."""

    #: Absolute YAML configuration path.
    config_path: Path
    #: Absolute source-flow HDF5 path.
    source_h5: Path
    #: Absolute prepared FD-q HDF5 path.
    output_h5: Path
    #: Bounded xi-direction polynomial degree.
    q_y: int
    #: Periodic eta-direction polynomial degree.
    q_z: int


def _parser() -> argparse.ArgumentParser:
    """Construct the command-line parser for the Python driver."""

    parser = argparse.ArgumentParser(
        prog="Tailor.py",
        description=(
            "Prepare a PETSc-compatible FD-q case cache, then run the "
            "Tailor C++ solver. Arguments after '--' are passed to PETSc."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        help="YAML case configuration file",
    )
    parser.add_argument(
        "--solver",
        type=Path,
        help=(
            "path to the C++ tailor executable (default: TAILOR_EXECUTABLE "
            "or an executable found in a standard project build directory)"
        ),
    )
    parser.add_argument(
        "-n",
        "--mpi-processes",
        type=int,
        default=1,
        metavar="N",
        help="number of MPI processes used by the C++ solver (default: 1)",
    )
    parser.add_argument(
        "--mpiexec",
        type=Path,
        help=(
            "MPI launcher used when N > 1 (default: TAILOR_MPIEXEC or the "
            "mpiexec from Tailor's complex PETSc installation)"
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="prepare or validate the FD-q cache without running C++",
    )
    parser.add_argument(
        "solver_arguments",
        nargs=argparse.REMAINDER,
        help="PETSc/C++ arguments following '--'",
    )
    return parser


def _required_mapping(root: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required YAML mapping or raise a descriptive error."""

    value = root.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"YAML key {key!r} must be a mapping")
    return value


def _required_string(root: dict[str, Any], key: str) -> str:
    """Return a required non-empty YAML string."""

    value = root.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"YAML key {key!r} must be a non-empty string")
    return value


def _required_q_value(root: dict[str, Any], key: str) -> int:
    """Return and validate one polynomial degree from ``Q-Value``."""

    value = root.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Q-Value.{key} must be an integer polynomial degree")
    if value < 2:
        raise ValueError(f"Q-Value.{key} must be at least 2")
    return value


def load_case_configuration(path: Path | str) -> CaseConfiguration:
    """Parse the preprocessing fields from a Tailor YAML case file."""

    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyYAML is required; install it in the active Python environment"
        ) from error

    config = Path(path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config}")
    with config.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("the YAML document root must be a mapping")

    folder = Path(_required_string(loaded, "Folder")).expanduser()
    file = Path(_required_string(loaded, "File")).expanduser()
    q_values = _required_mapping(loaded, "Q-Value")
    q_y = _required_q_value(q_values, "y")
    q_z = _required_q_value(q_values, "z")

    source = file if file.is_absolute() else config.parent / folder / file
    source = source.resolve()
    output = source.parent / "FD-q" / f"fdq_{source.stem}_qy{q_y}_qz{q_z}.h5"
    return CaseConfiguration(config, source, output.resolve(), q_y, q_z)


def _executable(path: Path | str, description: str) -> Path:
    """Resolve and validate an executable file."""

    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{description} does not exist: {candidate}")
    if not os.access(candidate, os.X_OK):
        raise PermissionError(f"{description} is not executable: {candidate}")
    return candidate


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


def find_solver(explicit: Path | str | None = None) -> Path:
    """Resolve the C++ executable without depending on the caller's cwd."""

    if explicit is not None:
        return _executable(explicit, "Tailor C++ solver")

    configured = os.environ.get("TAILOR_EXECUTABLE")
    if configured:
        return _executable(configured, "TAILOR_EXECUTABLE")

    candidates = (
        PROJECT_ROOT / "build" / "tailor",
        PROJECT_ROOT / "cmake-build-debug" / "tailor",
        PROJECT_ROOT / "build" / "Debug" / "tailor",
        PROJECT_ROOT / "build" / "Release" / "tailor",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    installed = shutil.which("tailor")
    if installed:
        return Path(installed).resolve()
    raise FileNotFoundError(
        "Tailor C++ solver was not found. Build it with CMake, pass "
        "--solver PATH, or set TAILOR_EXECUTABLE."
    )


def find_mpiexec(explicit: Path | str | None = None) -> Path:
    """Resolve the MPI launcher from PETSc's environment or ``PATH``."""

    if explicit is not None:
        return _executable(explicit, "MPI launcher")

    configured = os.environ.get("TAILOR_MPIEXEC")
    if configured:
        return _executable(configured, "TAILOR_MPIEXEC")

    prefix = _petsc_prefix_from_environment()
    if prefix is not None:
        for name in ("mpiexec", "mpirun"):
            bundled = prefix / "bin" / name
            if bundled.is_file() and os.access(bundled, os.X_OK):
                return bundled.resolve()

    for name in ("mpiexec", "mpirun"):
        installed = shutil.which(name)
        if installed:
            return Path(installed).resolve()
    raise FileNotFoundError(
        "an MPI launcher was not found. Pass --mpiexec PATH, set "
        "TAILOR_MPIEXEC, configure PETSC_DIR/PETSC_ARCH, or add mpiexec to PATH."
    )


def solver_command(
    solver: Path,
    config: Path,
    mpi_processes: int,
    mpiexec: Path | None,
    extra_arguments: Sequence[str] = (),
) -> list[str]:
    """Build the serial or MPI C++ invocation."""

    if mpi_processes < 1:
        raise ValueError("--mpi-processes must be at least 1")
    command = [str(solver), "-c", str(config)]
    forwarded = list(extra_arguments)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    command.extend(forwarded)
    if mpi_processes > 1:
        if mpiexec is None:
            raise ValueError("an MPI launcher is required when --mpi-processes > 1")
        command = [str(mpiexec), "-n", str(mpi_processes), *command]
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Run preprocessing and, unless requested otherwise, launch C++."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        case = load_case_configuration(arguments.config)
        if arguments.mpi_processes < 1:
            raise ValueError("--mpi-processes must be at least 1")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    command = [
        sys.executable,
        "-m",
        "src.Python.interpolate",
        "--source",
        str(case.source_h5),
        "--output",
        str(case.output_h5),
        "--q-y",
        str(case.q_y),
        "--q-z",
        str(case.q_z),
        "--rule-directory",
        str(DEFAULT_RULE_DIRECTORY),
    ]
    print(
        "Tailor FD-q preprocessing\n"
        f"  config: {case.config_path}\n"
        f"  source: {case.source_h5}\n"
        f"  output: {case.output_h5}\n"
        f"  q_y: {case.q_y}\n"
        f"  q_z: {case.q_z}",
        flush=True,
    )
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        print(
            f"Tailor preprocessing subprocess exited with status {completed.returncode}",
            file=sys.stderr,
        )
        return completed.returncode
    print(f"Tailor FD-q input ready: {case.output_h5}")

    if arguments.prepare_only:
        return 0

    try:
        solver = find_solver(arguments.solver)
        mpiexec = (
            find_mpiexec(arguments.mpiexec)
            if arguments.mpi_processes > 1
            else None
        )
        command = solver_command(
            solver,
            case.config_path,
            arguments.mpi_processes,
            mpiexec,
            arguments.solver_arguments,
        )
    except (FileNotFoundError, OSError, PermissionError, ValueError) as error:
        print(f"Tailor C++ launch error: {error}", file=sys.stderr)
        return 2

    print(
        f"Starting Tailor C++ solver with {arguments.mpi_processes} MPI "
        f"process{'es' if arguments.mpi_processes != 1 else ''}: {solver}",
        flush=True,
    )
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        print(
            f"Tailor C++ solver exited with status {completed.returncode}",
            file=sys.stderr,
        )
    return completed.returncode


__all__ = [
    "CaseConfiguration",
    "find_mpiexec",
    "find_solver",
    "load_case_configuration",
    "main",
    "solver_command",
]
