#include "ProblemData.hpp"

void FDQRuleData::clear() noexcept {
  N = 0;
  q = 0;
  nodes.clear();
  stencilIndices.clear();
  for (auto &derivativeWeights : weights)
    derivativeWeights.clear();
}

ProblemData::~ProblemData() { (void)destroy(); }

PetscErrorCode ProblemData::destroy() {
  PetscFunctionBeginUser;
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
  PetscFunctionReturn(PETSC_SUCCESS);
}
