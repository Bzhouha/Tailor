/**
 * @file EigenSolver.cpp
 * @brief Generalized non-Hermitian Krylov--Schur and shift-invert setup.
 */
#include "EigenSolver.hpp"

#include <algorithm>

EigenSolution::~EigenSolution() { (void)destroy(); }

PetscErrorCode EigenSolution::destroy() {
  PetscFunctionBeginUser;
  PetscCall(EPSDestroy(&eps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

EigenSolver::EigenSolver(MPI_Comm comm) : comm_(comm) {}

PetscErrorCode EigenSolver::solve(const Recipe &recipe, const ProblemData &data,
                                  EigenSolution &solution,
                                  EigenSolverDiagnostics &diagnostics) const {
  PetscFunctionBeginUser;
  PetscCheck(data.eigenMatrix && data.eigenMassMatrix, comm_,
             PETSC_ERR_ARG_WRONGSTATE,
             "Boundary-constrained matrices are required before EPSSolve");
  PetscCall(solve(recipe, data.eigenMatrix, data.eigenMassMatrix, solution,
                  diagnostics));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EigenSolver::solve(const Recipe &recipe, Mat eigenMatrix,
                                  Mat massMatrix, EigenSolution &solution,
                                  EigenSolverDiagnostics &diagnostics) const {
  ST spectralTransform = nullptr;
  KSP linearSolver = nullptr;
  PC preconditioner = nullptr;

  PetscFunctionBeginUser;
  PetscCheck(eigenMatrix && massMatrix, comm_, PETSC_ERR_ARG_NULL,
             "Both generalized eigenproblem matrices are required");
  PetscCheck(!solution.eps, comm_, PETSC_ERR_ARG_WRONGSTATE,
             "The EigenSolution already owns an EPS object");

  diagnostics = {};
  diagnostics.requested = recipe.numberOfEigenvalues;
  diagnostics.targetLambda =
      PetscCMPLX(static_cast<PetscReal>(recipe.searchCenterOmega.imag()),
                 static_cast<PetscReal>(-recipe.searchCenterOmega.real()));

  PetscCall(EPSCreate(comm_, &solution.eps));
  PetscCall(EPSSetOperators(solution.eps, eigenMatrix, massMatrix));
  PetscCall(EPSSetProblemType(solution.eps, EPS_GNHEP));
  PetscCall(EPSSetType(solution.eps, EPSKRYLOVSCHUR));
  PetscCall(EPSSetDimensions(solution.eps, recipe.numberOfEigenvalues,
                             PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(EPSSetTolerances(solution.eps, recipe.eigenTolerance,
                             recipe.eigenMaximumIterations));
  PetscCall(EPSSetTarget(solution.eps, diagnostics.targetLambda));
  PetscCall(EPSSetWhichEigenpairs(solution.eps, EPS_TARGET_MAGNITUDE));

  PetscCall(EPSGetST(solution.eps, &spectralTransform));
  PetscCall(STSetType(spectralTransform, STSINVERT));
  PetscCall(STSetShift(spectralTransform, diagnostics.targetLambda));
  PetscCall(STGetKSP(spectralTransform, &linearSolver));
  PetscCall(KSPSetType(linearSolver, KSPPREONLY));
  PetscCall(KSPGetPC(linearSolver, &preconditioner));
  PetscCall(PCSetType(preconditioner, PCLU));

  // Runtime PETSc/SLEPc options intentionally override the reproducible YAML
  // defaults, including the external sparse LU package.
  PetscCall(EPSSetFromOptions(solution.eps));
  PetscCall(EPSSolve(solution.eps));
  PetscCall(EPSGetConverged(solution.eps, &diagnostics.converged));
  PetscCall(EPSGetIterationNumber(solution.eps, &diagnostics.iterations));
  PetscCall(EPSGetConvergedReason(solution.eps, &diagnostics.reason));

  for (PetscInt index = 0; index < diagnostics.converged; ++index) {
    PetscReal error = 0.0;
    PetscCall(EPSComputeError(solution.eps, index, EPS_ERROR_RELATIVE, &error));
    diagnostics.maximumRelativeError =
        PetscMax(diagnostics.maximumRelativeError, error);
  }
  PetscCheck(diagnostics.reason > 0, comm_, PETSC_ERR_NOT_CONVERGED,
             "SLEPc failed to converge (reason %d, %" PetscInt_FMT
             " eigenpairs after %" PetscInt_FMT " iterations)",
             static_cast<int>(diagnostics.reason), diagnostics.converged,
             diagnostics.iterations);
  PetscFunctionReturn(PETSC_SUCCESS);
}
