/**
 * @file test_eigen_solver.cpp
 * @brief Small known GNHEP test for shift-invert solve and HDF5 output.
 */
#include <slepcsys.h>

#include <array>
#include <string>

#include "BaseFlowDerivatives.hpp"
#include "EigenOutput.hpp"
#include "EigenSolver.hpp"

namespace {

/** @brief Create a triangular generalized problem with known eigenvalues. */
PetscErrorCode createKnownProblem(Mat &eigenMatrix, Mat &massMatrix) {
  constexpr PetscInt size = 10;
  const std::array<PetscScalar, size> eigenvalues = {
      PetscCMPLX(1.0, 0.2),   PetscCMPLX(-0.5, 0.7),  PetscCMPLX(2.0, -0.3),
      PetscCMPLX(-1.2, -0.4), PetscCMPLX(3.0, 1.0),   PetscCMPLX(-3.0, -1.0),
      PetscCMPLX(4.0, 2.0),   PetscCMPLX(-4.0, -2.0), PetscCMPLX(5.0, 3.0),
      PetscCMPLX(-5.0, -3.0)};
  const std::array<PetscScalar, size> massDiagonal = {1.0, 2.0, 1.5, 1.0, 0.8,
                                                      1.2, 1.1, 0.9, 1.4, 1.3};

  PetscFunctionBeginUser;
  PetscCall(
      MatCreateSeqAIJ(PETSC_COMM_SELF, size, size, 3, nullptr, &eigenMatrix));
  PetscCall(
      MatCreateSeqAIJ(PETSC_COMM_SELF, size, size, 1, nullptr, &massMatrix));
  for (PetscInt row = 0; row < size; ++row) {
    PetscCall(MatSetValue(eigenMatrix, row, row,
                          massDiagonal[static_cast<std::size_t>(row)] *
                              eigenvalues[static_cast<std::size_t>(row)],
                          INSERT_VALUES));
    PetscCall(MatSetValue(massMatrix, row, row,
                          massDiagonal[static_cast<std::size_t>(row)],
                          INSERT_VALUES));
  }
  PetscCall(
      MatSetValue(eigenMatrix, 0, 1, PetscCMPLX(0.7, -0.2), INSERT_VALUES));
  PetscCall(
      MatSetValue(eigenMatrix, 0, 3, PetscCMPLX(-0.3, 0.5), INSERT_VALUES));
  PetscCall(
      MatSetValue(eigenMatrix, 1, 2, PetscCMPLX(1.1, 0.4), INSERT_VALUES));
  PetscCall(
      MatSetValue(eigenMatrix, 2, 3, PetscCMPLX(0.2, -0.8), INSERT_VALUES));
  for (PetscInt row = 3; row + 1 < size; ++row)
    PetscCall(MatSetValue(eigenMatrix, row, row + 1,
                          PetscCMPLX(0.05 * row, -0.03 * row), INSERT_VALUES));
  PetscCall(MatAssemblyBegin(eigenMatrix, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(massMatrix, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(eigenMatrix, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(massMatrix, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Create small DMDA grid/base-flow fields for output testing. */
PetscErrorCode createOutputFields(ProblemData &data) {
  PetscFunctionBeginUser;
  data.ny = 2;
  data.nz = 1;
  data.spanwisePeriod = 2.0;
  PetscCall(DMDACreate2d(PETSC_COMM_SELF, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX, data.ny, data.nz, PETSC_DECIDE,
                         PETSC_DECIDE, flowComponentCount, 1, nullptr, nullptr,
                         &data.fieldDM));
  PetscCall(DMSetUp(data.fieldDM));
  PetscCall(DMDACreateCompatibleDMDA(data.fieldDM, 3, &data.gridDM));
  PetscCall(DMCreateGlobalVector(data.fieldDM, &data.baseflow));
  PetscCall(DMCreateGlobalVector(data.gridDM, &data.grid));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(data.baseflow),
                               "baseflow"));
  PetscCall(
      PetscObjectSetName(reinterpret_cast<PetscObject>(data.grid), "grid"));
  for (PetscInt index = 0; index < 10; ++index)
    PetscCall(
        VecSetValue(data.baseflow, index, 1.0 + 0.1 * index, INSERT_VALUES));
  for (PetscInt index = 0; index < 6; ++index)
    PetscCall(VecSetValue(data.grid, index, 0.25 * index, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(data.baseflow));
  PetscCall(VecAssemblyBegin(data.grid));
  PetscCall(VecAssemblyEnd(data.baseflow));
  PetscCall(VecAssemblyEnd(data.grid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Solve, verify target eigenpairs, and optionally write HDF5. */
PetscErrorCode runTest() {
  Mat eigenMatrix = nullptr;
  Mat massMatrix = nullptr;
  Recipe recipe;
  EigenSolution solution;
  EigenSolverDiagnostics diagnostics;
  constexpr PetscReal tolerance = 2.0e-10;
  const std::array<PetscScalar, 2> expected = {PetscCMPLX(-0.5, 0.7),
                                               PetscCMPLX(-1.2, -0.4)};
  std::array<char, PETSC_MAX_PATH_LEN> outputPath{};
  PetscBool writeOutput = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(createKnownProblem(eigenMatrix, massMatrix));
  recipe.searchCenterOmega = {-0.65, -0.45};
  recipe.numberOfEigenvalues = 2;
  recipe.eigenTolerance = 1.0e-12;
  recipe.eigenMaximumIterations = 200;
  recipe.caseTitle = "Known generalized problem";
  recipe.alpha = {0.3, -0.1};
  recipe.qY = 2;
  recipe.qZ = 2;
  PetscCall(EigenSolver(PETSC_COMM_SELF)
                .solve(recipe, eigenMatrix, massMatrix, solution, diagnostics));
  PetscCheck(diagnostics.converged >= recipe.numberOfEigenvalues,
             PETSC_COMM_SELF, PETSC_ERR_NOT_CONVERGED,
             "The known problem returned too few eigenpairs");
  PetscCheck(PetscAbsScalar(diagnostics.targetLambda -
                            PetscCMPLX(-0.45, 0.65)) <= tolerance,
             PETSC_COMM_SELF, PETSC_ERR_PLIB,
             "omega-to-lambda target conversion is incorrect");
  PetscCheck(diagnostics.maximumRelativeError <= tolerance, PETSC_COMM_SELF,
             PETSC_ERR_PLIB, "Known eigenproblem residual is too large: %.16g",
             static_cast<double>(diagnostics.maximumRelativeError));

  std::array<PetscBool, 2> matched = {PETSC_FALSE, PETSC_FALSE};
  for (PetscInt index = 0; index < recipe.numberOfEigenvalues; ++index) {
    PetscScalar eigenvalue = 0.0;
    PetscCall(EPSGetEigenpair(solution.eps, index, &eigenvalue, nullptr,
                              nullptr, nullptr));
    for (std::size_t candidate = 0; candidate < expected.size(); ++candidate) {
      if (PetscAbsScalar(eigenvalue - expected[candidate]) <= tolerance)
        matched[candidate] = PETSC_TRUE;
    }
  }
  PetscCheck(matched[0] && matched[1], PETSC_COMM_SELF, PETSC_ERR_PLIB,
             "Shift-invert did not return the two expected local eigenvalues");
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-output",
                                  outputPath.data(), outputPath.size(),
                                  &writeOutput));
  if (writeOutput) {
    ProblemData outputData;
    EigenOutputDiagnostics outputDiagnostics;
    recipe.outputFile = outputPath.data();
    PetscCall(createOutputFields(outputData));
    PetscCall(EigenOutput(PETSC_COMM_SELF)
                  .write(recipe, outputData, solution, diagnostics,
                         outputDiagnostics));
    PetscCheck(outputDiagnostics.modesWritten == diagnostics.converged,
               PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "Output did not write every converged eigenmode");
  }
  PetscCall(solution.destroy());
  PetscCall(MatDestroy(&massMatrix));
  PetscCall(MatDestroy(&eigenMatrix));
  PetscCall(PetscPrintf(
      PETSC_COMM_SELF,
      "Small generalized non-Hermitian eigenproblem test passed.\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

/** @brief Initialize SLEPc, run the small problem, and finalize. */
int main(int argc, char **argv) {
  PetscErrorCode error = SlepcInitialize(&argc, &argv, nullptr, nullptr);
  if (error != PETSC_SUCCESS)
    return static_cast<int>(error);
  const PetscErrorCode testError = runTest();
  error = SlepcFinalize();
  if (testError != PETSC_SUCCESS)
    return static_cast<int>(testError);
  return static_cast<int>(error);
}
