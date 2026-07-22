from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from src.Python.main import CaseConfiguration, main, solver_command


class SolverCommandTests(unittest.TestCase):
    def test_serial_command(self) -> None:
        self.assertEqual(
            solver_command(
                Path("/tmp/tailor"),
                Path("/tmp/config.yaml"),
                1,
                None,
                ("-eps_nev", "4"),
            ),
            [
                "/tmp/tailor",
                "-c",
                "/tmp/config.yaml",
                "-eps_nev",
                "4",
            ],
        )

    def test_mpi_command(self) -> None:
        self.assertEqual(
            solver_command(
                Path("/tmp/tailor"),
                Path("/tmp/config.yaml"),
                3,
                Path("/tmp/mpiexec"),
            ),
            [
                "/tmp/mpiexec",
                "-n",
                "3",
                "/tmp/tailor",
                "-c",
                "/tmp/config.yaml",
            ],
        )

    def test_rejects_invalid_process_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            solver_command(
                Path("/tmp/tailor"), Path("/tmp/config.yaml"), 0, None
            )


class DriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = CaseConfiguration(
            config_path=Path("/tmp/config.yaml"),
            source_h5=Path("/tmp/sample.h5"),
            output_h5=Path("/tmp/fdq_sample_qy10_qz6.h5"),
            q_y=10,
            q_z=6,
        )

    @mock.patch("src.Python.main.find_solver")
    @mock.patch("src.Python.main.subprocess.run")
    @mock.patch("src.Python.main.load_case_configuration")
    def test_preprocess_then_serial_solver(
        self,
        load_config: mock.Mock,
        run: mock.Mock,
        find_solver: mock.Mock,
    ) -> None:
        load_config.return_value = self.case
        find_solver.return_value = Path("/build/tailor")
        run.side_effect = [
            mock.Mock(returncode=0),
            mock.Mock(returncode=0),
        ]

        status = main(["-c", "/tmp/config.yaml"])

        self.assertEqual(status, 0)
        self.assertEqual(run.call_count, 2)
        solver_call = run.call_args_list[1]
        self.assertEqual(
            solver_call.args[0],
            ["/build/tailor", "-c", "/tmp/config.yaml"],
        )

    @mock.patch("src.Python.main.find_solver")
    @mock.patch("src.Python.main.subprocess.run")
    @mock.patch("src.Python.main.load_case_configuration")
    def test_preprocess_failure_does_not_start_solver(
        self,
        load_config: mock.Mock,
        run: mock.Mock,
        find_solver: mock.Mock,
    ) -> None:
        load_config.return_value = self.case
        run.return_value = mock.Mock(returncode=7)

        status = main(["-c", "/tmp/config.yaml"])

        self.assertEqual(status, 7)
        run.assert_called_once()
        find_solver.assert_not_called()

    @mock.patch("src.Python.main.find_solver")
    @mock.patch("src.Python.main.subprocess.run")
    @mock.patch("src.Python.main.load_case_configuration")
    def test_solver_failure_is_returned_to_the_shell(
        self,
        load_config: mock.Mock,
        run: mock.Mock,
        find_solver: mock.Mock,
    ) -> None:
        load_config.return_value = self.case
        find_solver.return_value = Path("/build/tailor")
        run.side_effect = [
            mock.Mock(returncode=0),
            mock.Mock(returncode=9),
        ]

        status = main(["-c", "/tmp/config.yaml"])

        self.assertEqual(status, 9)
        self.assertEqual(run.call_count, 2)

    @mock.patch("src.Python.main.subprocess.run")
    @mock.patch("src.Python.main.load_case_configuration")
    def test_prepare_only_stops_after_preprocessing(
        self,
        load_config: mock.Mock,
        run: mock.Mock,
    ) -> None:
        load_config.return_value = self.case
        run.return_value = mock.Mock(returncode=0)

        status = main(["-c", "/tmp/config.yaml", "--prepare-only"])

        self.assertEqual(status, 0)
        run.assert_called_once()

    @mock.patch("src.Python.main.find_mpiexec")
    @mock.patch("src.Python.main.find_solver")
    @mock.patch("src.Python.main.subprocess.run")
    @mock.patch("src.Python.main.load_case_configuration")
    def test_mpi_solver_and_petsc_arguments(
        self,
        load_config: mock.Mock,
        run: mock.Mock,
        find_solver: mock.Mock,
        find_mpiexec: mock.Mock,
    ) -> None:
        load_config.return_value = self.case
        find_solver.return_value = Path("/build/tailor")
        find_mpiexec.return_value = Path("/petsc/bin/mpiexec")
        run.side_effect = [
            mock.Mock(returncode=0),
            mock.Mock(returncode=0),
        ]

        status = main(
            [
                "-c",
                "/tmp/config.yaml",
                "-n",
                "2",
                "--",
                "-eps_nev",
                "8",
            ]
        )

        self.assertEqual(status, 0)
        self.assertEqual(
            run.call_args_list[1].args[0],
            [
                "/petsc/bin/mpiexec",
                "-n",
                "2",
                "/build/tailor",
                "-c",
                "/tmp/config.yaml",
                "-eps_nev",
                "8",
            ],
        )


if __name__ == "__main__":
    unittest.main()
