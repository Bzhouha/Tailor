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
  /** Global scalar row count. */
  PetscInt rows = 0;
  /** Global scalar column count. */
  PetscInt columns = 0;
  /** PETSc matrix block size. */
  PetscInt blockSize = 0;
  /** Number of mass-matrix blocks containing stored entries. */
  PetscLogDouble massUsedBlocks = 0.0;
  /** Number of preallocated mass-matrix blocks. */
  PetscLogDouble massAllocatedBlocks = 0.0;
  /** Number of spatial-matrix blocks containing stored entries. */
  PetscLogDouble spatialUsedBlocks = 0.0;
  /** Number of preallocated spatial-matrix blocks. */
  PetscLogDouble spatialAllocatedBlocks = 0.0;
  /** Frobenius norm of \f$M_\Gamma\f$. */
  PetscReal massFrobeniusNorm = 0.0;
  /** Frobenius norm of \f$L\f$. */
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
  /** @brief Validate pointwise coefficients before matrix creation. */
  PetscErrorCode validateCoefficients(const Recipe &recipe,
                                      const ProblemData &data) const;
  /** @brief Preallocate compatible MPIBAIJ matrices exactly. */
  PetscErrorCode createMatrices(ProblemData &data) const;
  /** @brief Insert all tensor-product stencil block contributions. */
  PetscErrorCode insertValues(const Recipe &recipe, ProblemData &data) const;
  /** @brief Collect global sparsity and norm diagnostics. */
  PetscErrorCode
  computeDiagnostics(const ProblemData &data,
                     GlobalOperatorDiagnostics &diagnostics) const;

  /** Communicator shared by all matrix assembly operations. */
  MPI_Comm comm_;
};
