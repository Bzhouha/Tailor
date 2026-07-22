from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from src.Python.fdq_nodes import (
    calculate_fdq_nodes,
    calculate_fdq_rule,
    fornberg_weights,
    load_fdq_rule,
    load_or_create_fdq_rule,
    save_fdq_rule,
)


NOTEBOOK_N100_Q10 = np.asarray(
    [-1.0, -0.9963295146396931, -0.9856865078911036, -0.9691449371148457, -0.9483991819705881, -0.9256722077896464, -0.9045316749727464, -0.8842511398872501, -0.8639009507232964, -0.843369169893442, -0.8227395862954905, -0.802134584871565, -0.7815743033399979, -0.7610181655053783, -0.7404536148344519, -0.719883040178373, -0.6993128454735619, -0.6787448444188754, -0.6581773928484478, -0.6376096048592187, -0.61704147218173, -0.5964733096187523, -0.5759052483243713, -0.555337229320743, -0.5347691999951012, -0.5142011524330519, -0.4936331005341186, -0.4730650529703158, -0.4524970081168267, -0.4319289632625565, -0.411360917599842, -0.39079287140546276, -0.3702248253405811, -0.3496567795649425, -0.32908873385922943, -0.3085206878841234, -0.2879526420531513, -0.26738459627292793, -0.24681655039418265, -0.22624850451734616, -0.20568045862324172, -0.18511241278448107, -0.16454436689534852, -0.1439763210769019, -0.12340827513952282, -0.10284022938441131, -0.08227218345023117, -0.06170413748099069, -0.04113609166614422, -0.02056804587980431, 1.6194517991316247e-11, 0.020568045919604393, 0.041136091699814, 0.06170413747898018, 0.08227218326516939, 0.10284022936420688, 0.12340827516483133, 0.14397632092730273, 0.16454436673918948, 0.1851124126978851, 0.20568045865344609, 0.22624850427308524, 0.24681655019840887, 0.26738459609064924, 0.2879526419675683, 0.3085206879532198, 0.3290887337206182, 0.3496567794938824, 0.3702248253827162, 0.3907928712463611, 0.4113609173118839, 0.43192896324559166, 0.4524970081591641, 0.47306505296877566, 0.4936331004372826, 0.5142011521904327, 0.5347691999879286, 0.555337229239942, 0.5759052481523372, 0.5964733096018882, 0.617041472196842, 0.6376096047736364, 0.658177392797572, 0.6787448444606714, 0.6993128455493534, 0.7198830402551125, 0.740453614797957, 0.7610181655119247, 0.7815743032808853, 0.8021345848868068, 0.822739586196246, 0.8433691698579455, 0.863900950715188, 0.8842511398347661, 0.9045316750221403, 0.9256722077429592, 0.9483991819442371, 0.9691449371284784, 0.9856865078539081, 0.9963295146602356, 1.0],
    dtype=np.float64,
)


class FDQNodeTests(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        for N, q, exception in [
            (True, 2, TypeError),
            (4, False, TypeError),
            (2.5, 2, TypeError),
            (4, 2.5, TypeError),
            (1, 2, ValueError),
            (4, 1, ValueError),
            (4, 5, ValueError),
        ]:
            with self.subTest(N=N, q=q), self.assertRaises(exception):
                calculate_fdq_nodes(N, q)

    def test_even_q_rule_shapes_and_invariants(self) -> None:
        rule = calculate_fdq_rule(10, 4)
        self.assertEqual(rule.nodes.shape, (11,))
        self.assertEqual(rule.stencil_indices.shape, (11, 5))
        self.assertEqual(rule.weights.shape, (3, 11, 5))
        self.assertEqual(rule.log_error.shape, (11,))
        self.assertEqual(rule.nodes[0], -1.0)
        self.assertEqual(rule.nodes[-1], 1.0)
        self.assertTrue(np.all(np.diff(rule.nodes) > 0.0))
        np.testing.assert_allclose(rule.nodes, -rule.nodes[::-1], atol=5.0e-12)
        self.assertLess(np.max(np.abs(np.diff(rule.log_error))), 1.0e-9)

    def test_full_stencil_limit_is_cgl(self) -> None:
        N = 8
        nodes = calculate_fdq_nodes(N, N)
        expected = -np.cos(np.arange(N + 1, dtype=np.float64) * np.pi / N)
        np.testing.assert_allclose(nodes, expected, rtol=0.0, atol=2.0e-11)

    def test_notebook_n100_q10_reference(self) -> None:
        nodes = calculate_fdq_nodes(100, 10)
        np.testing.assert_allclose(
            nodes, NOTEBOOK_N100_Q10, rtol=0.0, atol=2.0e-9
        )


class FornbergTests(unittest.TestCase):
    def test_polynomial_exactness_through_stencil_degree(self) -> None:
        stencil = np.asarray([-1.0, -0.7, -0.1, 0.4, 1.0])
        x0 = -0.1
        weights = fornberg_weights(x0, stencil, 2)
        for derivative in range(3):
            for power in range(stencil.size):
                approximation = np.dot(weights[derivative], stencil**power)
                if power < derivative:
                    expected = 0.0
                else:
                    coefficient = 1
                    for factor in range(derivative):
                        coefficient *= power - factor
                    expected = coefficient * x0 ** (power - derivative)
                with self.subTest(derivative=derivative, power=power):
                    self.assertAlmostEqual(approximation, expected, places=11)

    def test_rejects_duplicate_stencil_nodes(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct"):
            fornberg_weights(0.0, np.asarray([-1.0, 0.0, 0.0, 1.0]))


class FDQCacheTests(unittest.TestCase):
    def test_hdf5_round_trip_and_schema(self) -> None:
        rule = calculate_fdq_rule(8, 4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "custom.h5"
            saved = save_fdq_rule(rule, path)
            self.assertEqual(saved, path.resolve())
            with h5py.File(path, "r") as handle:
                self.assertEqual(int(handle.attrs["schema_version"]), 1)
                self.assertEqual(int(handle.attrs["N"]), 8)
                self.assertEqual(int(handle.attrs["q"]), 4)
                self.assertIn("weights/d0", handle)
                self.assertIn("weights/d1", handle)
                self.assertIn("weights/d2", handle)
                self.assertIn("diagnostics/log_error", handle)

            loaded = load_fdq_rule(path)
            np.testing.assert_array_equal(loaded.stencil_indices, rule.stencil_indices)
            np.testing.assert_allclose(loaded.nodes, rule.nodes, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(loaded.weights, rule.weights, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(
                loaded.log_error, rule.log_error, rtol=0.0, atol=0.0
            )

    def test_load_or_create_uses_named_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            created = load_or_create_fdq_rule(6, 3, directory)
            path = directory / "N6_q3.h5"
            self.assertTrue(path.is_file())
            loaded = load_or_create_fdq_rule(6, 3, directory)
            np.testing.assert_allclose(loaded.nodes, created.nodes)

    def test_rejects_wrong_schema_version(self) -> None:
        rule = calculate_fdq_rule(6, 3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.h5"
            save_fdq_rule(rule, path)
            with h5py.File(path, "r+") as handle:
                handle.attrs["schema_version"] = 99
            with self.assertRaisesRegex(ValueError, "schema version"):
                load_fdq_rule(path)


if __name__ == "__main__":
    unittest.main()
