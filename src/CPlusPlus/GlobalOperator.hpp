#pragma once

#include <petscmat.h>

#include "ProblemData.hpp"
#include "Recipe.hpp"

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

class GlobalOperator {
public:
  explicit GlobalOperator(MPI_Comm comm);

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
