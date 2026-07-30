/**
 * @file BoundaryConditions.hpp
 * @brief Wall and characteristic far-field rows for the global eigenproblem.
 */
#pragma once

#include <petscmat.h>

#include "ProblemData.hpp"
#include "Recipe.hpp"

/** @brief Diagnostics for the constrained generalized eigenvalue matrices. */
struct BoundaryConditionDiagnostics {
  /** Number of nodes on the wall xi plane. */
  PetscInt wallNodes = 0;
  /** Number of nodes on the far-field xi plane. */
  PetscInt farfieldNodes = 0;
  /** Total number of incoming characteristic modes. */
  PetscInt incomingModes = 0;
  /** Total number of modes classified as near-zero speed. */
  PetscInt neutralModes = 0;
  /** Minimum incoming-mode count among far-field nodes. */
  PetscInt minIncomingModesPerNode = 0;
  /** Maximum incoming-mode count among far-field nodes. */
  PetscInt maxIncomingModesPerNode = 0;
  /** Largest imaginary part of a nominally real characteristic speed. */
  PetscReal maxCharacteristicImaginary = 0.0;
  /** Largest estimated infinity-norm condition number of \f$R\f$. */
  PetscReal maxEigenvectorCondition = 0.0;
  /** Frobenius norm of the constrained \f$A_{\rm bc}\f$ matrix. */
  PetscReal eigenFrobeniusNorm = 0.0;
  /** Frobenius norm of the constrained \f$B_{\rm bc}\f$ matrix. */
  PetscReal eigenMassFrobeniusNorm = 0.0;
};

/**
 * @brief Build \f$A_{\rm bc}\f$ and \f$B_{\rm bc}\f$ without changing the
 * unconstrained \f$L\f$ and \f$M_\Gamma\f$ matrices.
 *
 * The wall is the first xi plane. Its density row remains the original
 * continuity equation, while u, v, w, and T become homogeneous Dirichlet
 * rows. The last xi plane uses the inviscid normal characteristic
 * decomposition: incoming amplitudes are set to zero and outgoing/neutral
 * equations retain the projected full viscous PDE.
 */
class BoundaryConditions {
public:
  /** @param comm Communicator shared by the matrices and field DMDA. */
  explicit BoundaryConditions(MPI_Comm comm);

  /**
   * @brief Copy and constrain the assembled global operators.
   * @param recipe Physical parameters used by the inviscid characteristic
   *               matrices.
   * @param data Prepared problem; receives eigenMatrix and eigenMassMatrix.
   * @param diagnostics Global boundary counts and numerical checks.
   * @return PETSc error code.
   */
  PetscErrorCode apply(const Recipe &recipe, ProblemData &data,
                       BoundaryConditionDiagnostics &diagnostics) const;

private:
  /** Communicator used for characteristic checks and matrix row replacement. */
  MPI_Comm comm_;
};
