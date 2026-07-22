"""Tailor Python driver for YAML parsing and isolated FD-q preprocessing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULE_DIRECTORY = PROJECT_ROOT / "fdqNodes"


@dataclass(frozen=True, slots=True)
class CaseConfiguration:
    config_path: Path
    source_h5: Path
    output_h5: Path
    q_y: int
    q_z: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Tailor.py",
        description="Prepare a PETSc-compatible FD-q case cache.",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        help="YAML case configuration file",
    )
    return parser


def _required_mapping(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"YAML key {key!r} must be a mapping")
    return value


def _required_string(root: dict[str, Any], key: str) -> str:
    value = root.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"YAML key {key!r} must be a non-empty string")
    return value


def _required_q_value(root: dict[str, Any], key: str) -> int:
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


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        case = load_case_configuration(arguments.config)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        _parser().error(str(error))

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
    return 0


__all__ = ["CaseConfiguration", "load_case_configuration", "main"]
