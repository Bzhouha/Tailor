/**
 * @file EigenOutput.hpp
 * @brief Self-contained PETSc HDF5 output for converged eigenmodes.
 */
#pragma once

#include <petscsys.h>

#include <string>

#include "EigenSolver.hpp"
#include "ProblemData.hpp"
#include "Recipe.hpp"

/** @brief Summary of an atomically written eigenmode file. */
struct EigenOutputDiagnostics {
  /** Final atomically installed HDF5 path. */
  std::string file;
  /** Number of normalized eigenmodes written. */
  PetscInt modesWritten = 0;
};

/** @brief Write grid, base flow, spectrum, and normalized modes to HDF5. */
class EigenOutput {
public:
  /** @param comm Communicator shared by the EPS and distributed vectors. */
  explicit EigenOutput(MPI_Comm comm);

  /**
   * @brief Write and validate a temporary file, then atomically replace output.
   */
  PetscErrorCode write(const Recipe &recipe, const ProblemData &data,
                       const EigenSolution &solution,
                       const EigenSolverDiagnostics &solverDiagnostics,
                       EigenOutputDiagnostics &diagnostics) const;

private:
  /** Communicator used for parallel natural-vector HDF5 output. */
  MPI_Comm comm_;
};
