/**
 * @file GlobalOperator.hpp
 * @brief Distributed PETSc assembly of the BiGlobal mass and spatial matrices.
 */
#pragma once

#include <petscmat.h>

#include "ProblemData.hpp"
#include "Recipe.hpp"

/** @brief Structural and norm diagnostics for the assembled matrices. */
struct GlobalOperatorDiagnostics {
  PetscInt rows = 0;
  PetscInt columns = 0;
  PetscInt blockSize = 0;
  PetscLogDouble massUsedBlocks = 0.0;
  PetscLogDouble massAllocatedBlocks = 0.0;
  PetscLogDouble spatialUsedBlocks = 0.0;
  PetscLogDouble spatialAllocatedBlocks = 0.0;
  PetscReal massFrobeniusNorm = 0.0;
  PetscReal spatialFrobeniusNorm = 0.0;
};

/**
 * @brief Assemble the unconstrained generalized eigenvalue operators.
 *
 * The class stores \f$M_\Gamma\f$ and \f$L\f$ in ProblemData as block-size-five
 * BAIJ matrices. Boundary conditions and the later sign change
 * \f$A=-L\f$ are intentionally outside this module.
 */
class GlobalOperator {
public:
  /** @param comm Communicator used by the field DMDA and matrices. */
  explicit GlobalOperator(MPI_Comm comm);

  /**
   * @brief Assemble \f$M_\Gamma\f$ and \f$L\f$ with the loaded FD-q stencils.
   * @param recipe Physical parameters and complex streamwise wavenumber.
   * @param data Prepared fields and coefficients; receives both matrices.
   * @param diagnostics Global matrix dimensions, sparsity, and norms.
   * @return PETSc error code.
   */
  PetscErrorCode assemble(const Recipe &recipe, ProblemData &data,
                          GlobalOperatorDiagnostics &diagnostics) const;

private:
  PetscErrorCode validateCoefficients(const Recipe &recipe,
                                      const ProblemData &data) const;
  PetscErrorCode createMatrices(ProblemData &data) const;
  PetscErrorCode insertValues(const Recipe &recipe, ProblemData &data) const;
  PetscErrorCode
  computeDiagnostics(const ProblemData &data,
                     GlobalOperatorDiagnostics &diagnostics) const;

  MPI_Comm comm_;
};
