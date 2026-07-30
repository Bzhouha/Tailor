/**
 * @file EigenSolver.hpp
 * @brief SLEPc shift-invert solution of the constrained generalized problem.
 */
#pragma once

#include <slepceps.h>

#include "ProblemData.hpp"
#include "Recipe.hpp"

/** @brief Owner of one configured and solved SLEPc EPS object. */
struct EigenSolution {
  EigenSolution() = default;
  ~EigenSolution();

  EigenSolution(const EigenSolution &) = delete;
  EigenSolution &operator=(const EigenSolution &) = delete;
  EigenSolution(EigenSolution &&) = delete;
  EigenSolution &operator=(EigenSolution &&) = delete;

  /** @brief Destroy the EPS object collectively. */
  PetscErrorCode destroy();

  /** Configured EPS object, retained for eigenpair output. */
  EPS eps = nullptr;
};

/** @brief Convergence and spectral-target diagnostics. */
struct EigenSolverDiagnostics {
  /** Search target after converting \f$\omega_s\f$ to \f$\lambda_s\f$. */
  PetscScalar targetLambda = 0.0;
  /** Number of eigenpairs requested by YAML/options. */
  PetscInt requested = 0;
  /** Number of eigenpairs that converged. */
  PetscInt converged = 0;
  /** Krylov iteration count. */
  PetscInt iterations = 0;
  /** Final SLEPc convergence or divergence reason. */
  EPSConvergedReason reason = EPS_CONVERGED_ITERATING;
  /** Largest relative residual among converged eigenpairs. */
  PetscReal maximumRelativeError = 0.0;
};

/**
 * @brief Configure Krylov--Schur with a reusable shift-invert LU factorization.
 */
class EigenSolver {
public:
  /** @param comm Communicator shared by the generalized matrices. */
  explicit EigenSolver(MPI_Comm comm);

  /**
   * @brief Solve the boundary-constrained problem stored in ProblemData.
   */
  PetscErrorCode solve(const Recipe &recipe, const ProblemData &data,
                       EigenSolution &solution,
                       EigenSolverDiagnostics &diagnostics) const;

  /**
   * @brief Solve an explicit generalized matrix pair.
   *
   * This overload supports small independently known regression problems.
   */
  PetscErrorCode solve(const Recipe &recipe, Mat eigenMatrix, Mat massMatrix,
                       EigenSolution &solution,
                       EigenSolverDiagnostics &diagnostics) const;

private:
  /** Communicator on which the EPS object is created. */
  MPI_Comm comm_;
};
