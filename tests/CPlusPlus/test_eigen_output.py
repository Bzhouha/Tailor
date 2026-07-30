"""HDF5 round-trip test for normalized, phase-fixed eigenmodes."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np


def complex_values(dataset: h5py.Dataset) -> np.ndarray:
    """Convert PETSc's trailing real/imaginary dimension to complex."""
    values = np.asarray(dataset)
    if values.shape[-1] != 2:
        raise AssertionError(f"Unexpected PETSc complex shape: {values.shape}")
    return values[..., 0] + 1j * values[..., 1]


def main() -> int:
    """Generate a small result file and validate its complete schema."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-binary", required=True, type=Path)
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="tailor-eigen-output-"
    ) as temporary:
        output = Path(temporary) / "nested" / "known_modes.h5"
        completed = subprocess.run(
            [str(arguments.test_binary.resolve()), "-output", str(output)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"Eigen output producer failed:\n{completed.stdout}"
            )
        if not output.is_file() or Path(f"{output}.tmp").exists():
            raise AssertionError("Atomic HDF5 replacement did not complete")

        with h5py.File(output, "r") as handle:
            if handle.attrs["ordering"].decode() != "k_j_dof":
                raise AssertionError("Natural ordering metadata is incorrect")
            if int(handle.attrs["schema_version"]) != 1:
                raise AssertionError("Unexpected output schema")
            if int(handle.attrs["requested_modes"]) != 2:
                raise AssertionError("Requested mode count is incorrect")
            converged = int(handle.attrs["converged_modes"])
            if converged < 2:
                raise AssertionError("Too few modes were written")

            grid = complex_values(handle["grid"])
            baseflow = complex_values(handle["baseflow"])
            if grid.shape != (2, 3) or baseflow.shape != (2, 5):
                raise AssertionError(
                    f"Unexpected field shapes: {grid.shape}, {baseflow.shape}"
                )
            np.testing.assert_allclose(grid.imag, 0.0, atol=0.0)
            np.testing.assert_allclose(baseflow.imag, 0.0, atol=0.0)

            eigenvalues = complex_values(handle["spectrum/lambda"])
            frequencies = complex_values(handle["spectrum/omega"])
            residuals = complex_values(handle["spectrum/residual"])
            np.testing.assert_allclose(
                frequencies, 1j * eigenvalues, rtol=1.0e-13, atol=1.0e-13
            )
            if np.max(np.abs(residuals.imag)) != 0.0:
                raise AssertionError("Residual dataset is not real-valued")
            if np.max(residuals.real) > 2.0e-10:
                raise AssertionError("Stored residual is too large")

            expected = np.asarray([-0.5 + 0.7j, -1.2 - 0.4j])
            for value in expected:
                if np.min(np.abs(eigenvalues - value)) > 2.0e-10:
                    raise AssertionError(f"Missing eigenvalue {value}")

            for index in range(converged):
                mode = complex_values(
                    handle[f"modes/mode_{index:03d}"]
                ).reshape(-1)
                np.testing.assert_allclose(
                    np.linalg.norm(mode), 1.0, rtol=2.0e-13, atol=2.0e-13
                )
                pivot = mode[np.argmax(np.abs(mode))]
                if abs(pivot.imag) > 2.0e-13 or pivot.real <= 0.0:
                    raise AssertionError(
                        "Eigenmode phase is not deterministic"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
