"""Unit tests for bounded/periodic interpolation and case preparation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from src.Python.interpolate import (
    periodic_quintic_tensor_spline,
    prepare_fdq_case,
    quintic_tensor_spline,
)
from src.Python.main import load_case_configuration


class QuinticTensorSplineTests(unittest.TestCase):
    """Verify the bounded tensor-product quintic spline."""

    def test_complex_tensor_polynomial_exactness(self) -> None:
        """Interpolate a complex tensor polynomial to roundoff."""

        z_old = np.linspace(-1.0, 1.0, 8)
        y_old = np.linspace(-1.0, 1.0, 9)
        z_grid, y_grid = np.meshgrid(z_old, y_old, indexing="ij")
        values = (
            (1.0 + 2.0j) * z_grid**5
            + (2.0 - 3.0j) * y_grid**4
            + z_grid**2 * y_grid**3
        )
        z_new = np.linspace(-1.0, 1.0, 13) ** 3
        y_new = np.linspace(-1.0, 1.0, 14) ** 3
        z_new.sort()
        y_new.sort()
        actual = quintic_tensor_spline(
            z_old, y_old, values, z_new, y_new
        )
        z_target, y_target = np.meshgrid(z_new, y_new, indexing="ij")
        expected = (
            (1.0 + 2.0j) * z_target**5
            + (2.0 - 3.0j) * y_target**4
            + z_target**2 * y_target**3
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=8.0e-14)

    def test_rejects_too_few_source_nodes(self) -> None:
        """Require enough source nodes for a quintic spline."""

        with self.assertRaisesRegex(ValueError, "at least 6"):
            quintic_tensor_spline(
                np.linspace(-1.0, 1.0, 5),
                np.linspace(-1.0, 1.0, 6),
                np.zeros((5, 6)),
                np.asarray([-1.0, 1.0]),
                np.asarray([-1.0, 1.0]),
            )

    def test_rejects_wrong_field_shape(self) -> None:
        """Reject a field whose shape does not match its coordinates."""

        with self.assertRaisesRegex(ValueError, "values must have shape"):
            quintic_tensor_spline(
                np.linspace(-1.0, 1.0, 6),
                np.linspace(-1.0, 1.0, 7),
                np.zeros((7, 6)),
                np.asarray([-1.0, 1.0]),
                np.asarray([-1.0, 1.0]),
            )


class PeriodicQuinticTensorSplineTests(unittest.TestCase):
    """Verify periodic interpolation with an optional affine translation."""

    def test_periodic_complex_field_and_affine_translation(self) -> None:
        """Preserve periodic complex content and physical-coordinate shift."""

        z_old = -1.0 + 2.0 * np.arange(24) / 24
        y_old = np.linspace(-1.0, 1.0, 9)
        z_grid, y_grid = np.meshgrid(z_old, y_old, indexing="ij")
        translation = 7.5
        values = (
            np.sin(np.pi * z_grid) * (1.0 + y_grid**3)
            + 1j * np.cos(2.0 * np.pi * z_grid) * (2.0 - y_grid)
            + 0.5 * (z_grid + 1.0) * translation
        )
        z_new = -1.0 + 2.0 * np.arange(48) / 48
        y_new = np.linspace(-1.0, 1.0, 13)
        actual = periodic_quintic_tensor_spline(
            z_old,
            y_old,
            values,
            z_new,
            y_new,
            value_translation=translation,
        )
        z_target, y_target = np.meshgrid(z_new, y_new, indexing="ij")
        expected = (
            np.sin(np.pi * z_target) * (1.0 + y_target**3)
            + 1j * np.cos(2.0 * np.pi * z_target) * (2.0 - y_target)
            + 0.5 * (z_target + 1.0) * translation
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=5.0e-6)

    def test_rejects_repeated_periodic_endpoint(self) -> None:
        """Reject a periodic coordinate array containing both endpoints."""

        z_old = np.linspace(-1.0, 1.0, 12)
        y_old = np.linspace(-1.0, 1.0, 8)
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            periodic_quintic_tensor_spline(
                z_old,
                y_old,
                np.zeros((12, 8)),
                z_old[:-1],
                y_old,
            )


class PeriodicCasePreparationTests(unittest.TestCase):
    """Verify complete schema-v2 periodic case preparation."""

    @staticmethod
    def _write_source(path: Path, ny: int = 8, nz: int = 12) -> None:
        """Create a small valid half-open periodic source file."""

        y = np.linspace(-1.0, 1.0, ny)
        eta = -1.0 + 2.0 * np.arange(nz) / nz
        eta_grid, y_grid = np.meshgrid(eta, y, indexing="ij")
        spanwise_period = 6.0
        grid = np.zeros((nz, ny, 3), dtype=np.complex128)
        grid[..., 1] = y_grid
        grid[..., 2] = 0.5 * (eta_grid + 1.0) * spanwise_period
        baseflow = np.zeros((nz, ny, 5), dtype=np.complex128)
        baseflow[..., 0] = 1.0 + 0.02 * np.cos(np.pi * eta_grid)
        baseflow[..., 1] = 0.5 + 0.1 * (1.0 + y_grid)
        baseflow[..., 2] = 0.01 * np.sin(np.pi * eta_grid)
        baseflow[..., 3] = 0.02 * np.cos(np.pi * eta_grid)
        baseflow[..., 4] = 1.0 + 0.05 * y_grid
        with h5py.File(path, "w") as handle:
            handle.attrs["Ny"] = ny
            handle.attrs["Nz"] = nz
            handle.attrs["spanwise_periodic"] = 1
            handle.attrs["spanwise_period"] = spanwise_period
            handle.create_dataset(
                "grid", data=np.stack((grid.real, grid.imag), axis=-1)
            )
            handle.create_dataset(
                "baseflow", data=np.stack((baseflow.real, baseflow.imag), axis=-1)
            )

    def test_schema_v2_and_stale_output_auto_rebuild(self) -> None:
        """Write schema v2 and automatically replace a stale prepared case."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample.h5"
            output = root / "FD-q" / "fdq_sample_qy4_qz4.h5"
            rules = root / "rules"
            self._write_source(source)
            prepare_fdq_case(source, output, 4, 4, rules)
            with h5py.File(output, "r+") as handle:
                self.assertEqual(int(handle.attrs["schema_version"]), 2)
                self.assertEqual(handle["discretization/z"].attrs["topology"], "periodic")
                self.assertEqual(
                    handle["discretization/z/stencil_offsets"].shape, (12, 5)
                )
                handle.attrs["schema_version"] = 1
            prepare_fdq_case(source, output, 4, 4, rules)
            with h5py.File(output, "r") as handle:
                self.assertEqual(int(handle.attrs["schema_version"]), 2)


class CaseConfigurationTests(unittest.TestCase):
    """Verify YAML path and Q-value preprocessing fields."""

    def test_resolves_source_and_fdq_cache(self) -> None:
        """Resolve source and prepared cache paths relative to the case."""

        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            config = case / "config.yaml"
            config.write_text(
                "CaseTitle: Test\n"
                "Folder: data\n"
                "File: sample.h5\n"
                "Q-Value:\n"
                "    y: 10\n"
                "    z: 6\n",
                encoding="utf-8",
            )
            parsed = load_case_configuration(config)
            self.assertEqual(
                parsed.source_h5, (case / "data" / "sample.h5").resolve()
            )
            self.assertEqual(
                parsed.output_h5,
                (
                    case / "data" / "FD-q" / "fdq_sample_qy10_qz6.h5"
                ).resolve(),
            )
            self.assertEqual(parsed.q_y, 10)
            self.assertEqual(parsed.q_z, 6)

    def test_rejects_legacy_scalar_q_value(self) -> None:
        """Reject the obsolete scalar Q-Value configuration."""

        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.yaml"
            config.write_text(
                "Folder: data\nFile: sample.h5\nQ-Value: 5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be a mapping"):
                load_case_configuration(config)


if __name__ == "__main__":
    unittest.main()
