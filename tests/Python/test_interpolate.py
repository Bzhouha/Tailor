from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.Python.interpolate import quintic_tensor_spline
from src.Python.main import load_case_configuration


class QuinticTensorSplineTests(unittest.TestCase):
    def test_complex_tensor_polynomial_exactness(self) -> None:
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
        with self.assertRaisesRegex(ValueError, "at least 6"):
            quintic_tensor_spline(
                np.linspace(-1.0, 1.0, 5),
                np.linspace(-1.0, 1.0, 6),
                np.zeros((5, 6)),
                np.asarray([-1.0, 1.0]),
                np.asarray([-1.0, 1.0]),
            )

    def test_rejects_wrong_field_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "values must have shape"):
            quintic_tensor_spline(
                np.linspace(-1.0, 1.0, 6),
                np.linspace(-1.0, 1.0, 7),
                np.zeros((7, 6)),
                np.asarray([-1.0, 1.0]),
                np.asarray([-1.0, 1.0]),
            )


class CaseConfigurationTests(unittest.TestCase):
    def test_resolves_source_and_fdq_cache(self) -> None:
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
