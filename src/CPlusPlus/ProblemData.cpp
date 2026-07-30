/**
 * @file ProblemData.cpp
 * @brief Resource cleanup for persistent problem data.
 */
#include "ProblemData.hpp"

#include <algorithm>

PetscInt FDQRuleData::maxAbsOffset() const noexcept {
  PetscInt maximum = 0;
  for (const PetscInt offset : stencilOffsets)
    maximum = std::max(maximum, PetscAbsInt(offset));
  return maximum;
}

void FDQRuleData::clear() noexcept {
  nodeCountValue = 0;
  q = 0;
  topology = FDQTopology::Bounded;
  period = 2.0;
  nodes.clear();
  stencilIndices.clear();
  stencilOffsets.clear();
  for (auto &derivativeWeights : weights)
    derivativeWeights.clear();
}

ProblemData::~ProblemData() { (void)destroy(); }

PetscErrorCode ProblemData::destroy() {
  PetscFunctionBeginUser;
  PetscCall(MatDestroy(&eigenMassMatrix));
  PetscCall(MatDestroy(&eigenMatrix));
  PetscCall(MatDestroy(&spatialMatrix));
  PetscCall(MatDestroy(&massMatrix));
  PetscCall(VecDestroy(&baseflowDerivatives));
  PetscCall(VecDestroy(&metrics));
  PetscCall(VecDestroy(&grid));
  PetscCall(VecDestroy(&baseflow));
  PetscCall(DMDestroy(&baseflowDerivativeDM));
  PetscCall(DMDestroy(&metricDM));
  PetscCall(DMDestroy(&gridDM));
  PetscCall(DMDestroy(&fieldDM));
  xiRule.clear();
  etaRule.clear();
  ny = 0;
  nz = 0;
  stencilWidth = 0;
  spanwisePeriod = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
