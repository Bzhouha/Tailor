from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import yaml


def fdq_path(config_path: Path, config: dict) -> Path:
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


def run(
    solver: Path,
    config: Path,
    mpiexec: Path | None,
) -> subprocess.CompletedProcess[str]:
    command = [str(solver), "-c", str(config)]
    if mpiexec is not None:
        command = [str(mpiexec), "-n", "2", *command]
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )


def create_case(
    root: Path,
    source_config_path: Path,
    source_config: dict,
    section: str | None = None,
    key: str | None = None,
    value: float | None = None,
) -> Path:
    case = root / "case"
    case.mkdir(parents=True)
    target_config_path = case / "config.yaml"
    target_config = source_config.copy()
    target_config["Physics"] = {
        name: values.copy()
        for name, values in source_config["Physics"].items()
    }
    target_config["Physics"]["Transport"].pop("Model", None)
    if section is not None and key is not None:
        target_config["Physics"][section][key] = value

    with target_config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(target_config, stream, sort_keys=False)

    source_fdq = fdq_path(source_config_path, source_config)
    target_fdq = fdq_path(target_config_path, target_config)
    target_fdq.parent.mkdir(parents=True)
    shutil.copy2(source_fdq, target_fdq)
    return target_config_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True, type=Path)
    arguments = parser.parse_args()

    solver = arguments.solver.resolve()
    config_path = arguments.config.resolve()
    mpiexec = arguments.mpiexec.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        source_config = yaml.safe_load(stream)

    scenarios = (
        ("Targets", "MachNumber", 0.0),
        ("Targets", "ReynoldsNumber", 0.0),
        ("Gas", "PrandtlNumber", 0.0),
        ("Gas", "RatioOfSpecificHeats", 1.0),
        ("Transport", "ReferenceTemperature", 0.0),
        ("Transport", "ReferenceMu", 0.0),
        ("Transport", "SutherlandConstant", -1.0),
    )

    with tempfile.TemporaryDirectory(
        prefix="tailor-physics-config-"
    ) as temporary:
        root = Path(temporary)
        valid_config = create_case(
            root / "valid", config_path, source_config
        )
        for label, launcher in (("serial", None), ("mpi2", mpiexec)):
            completed = run(solver, valid_config, launcher)
            if completed.returncode != 0:
                raise AssertionError(
                    f"configuration without Transport.Model failed/{label}\n"
                    f"{completed.stdout}"
                )

        for section, key, value in scenarios:
            invalid_config = create_case(
                root / f"{section}-{key}",
                config_path,
                source_config,
                section,
                key,
                value,
            )
            for label, launcher in (("serial", None), ("mpi2", mpiexec)):
                completed = run(solver, invalid_config, launcher)
                if completed.returncode == 0:
                    raise AssertionError(
                        f"{section}.{key}={value} unexpectedly succeeded/"
                        f"{label}"
                    )
                if "outside their valid ranges" not in completed.stdout:
                    raise AssertionError(
                        f"{section}.{key}={value} returned the wrong error/"
                        f"{label}\n{completed.stdout}"
                    )

    print("Physics configuration validation scenarios passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
