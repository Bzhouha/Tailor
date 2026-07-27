/**
 * @file BaseFlowDerivatives.cpp
 * @brief FD-q and metric-chain-rule implementation for base-flow derivatives.
 */
#include "BaseFlowDerivatives.hpp"

#include <petscmath.h>

#include <algorithm>
#include <array>

#include "Metrics.hpp"

namespace {

bool inHalfOpenRange(PetscInt value, PetscInt first, PetscInt count) {
  return value >= first && value < first + count;
}

PetscReal realPart(PetscScalar value) { return PetscRealPart(value); }

} // namespace

BaseFlowDerivatives::BaseFlowDerivatives(MPI_Comm comm) : comm_(comm) {}

PetscErrorCode BaseFlowDerivatives::createStorage(ProblemData &data) const {
  PetscFunctionBeginUser;
  PetscCheck(data.fieldDM, comm_, PETSC_ERR_ARG_WRONGSTATE,
             "fieldDM must be prepared before computing base-flow derivatives");
  PetscCheck(!data.baseflowDerivativeDM && !data.baseflowDerivatives, comm_,
             PETSC_ERR_ARG_WRONGSTATE,
             "Base-flow derivative storage has already been created");

  PetscCall(DMDACreateCompatibleDMDA(data.fieldDM,
                                     baseFlowDerivativeComponentCount,
                                     &data.baseflowDerivativeDM));
  PetscCall(PetscObjectSetName(
      reinterpret_cast<PetscObject>(data.baseflowDerivativeDM),
      "baseflowDerivativeDM"));
  PetscCall(DMCreateGlobalVector(data.baseflowDerivativeDM,
                                 &data.baseflowDerivatives));
  PetscCall(PetscObjectSetName(
      reinterpret_cast<PetscObject>(data.baseflowDerivatives),
      "baseflowDerivatives"));
  PetscCall(VecZeroEntries(data.baseflowDerivatives));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BaseFlowDerivatives::computeValues(
    ProblemData &data, BaseFlowDerivativeDiagnostics &diagnostics) const {
  Vec baseflowLocal = nullptr;
  const PetscScalar ***baseflowArray = nullptr;
  const PetscScalar ***metricArray = nullptr;
  PetscScalar ***derivativeArray = nullptr;
  PetscInt xs = 0;
  PetscInt ys = 0;
  PetscInt xm = 0;
  PetscInt ym = 0;
  PetscInt gxs = 0;
  PetscInt gys = 0;
  PetscInt gxm = 0;
  PetscInt gym = 0;
  std::array<PetscInt, 5> localFailures{};
  std::array<PetscInt, 5> globalFailures{};
  PetscReal localMaxImaginary = 0.0;
  PetscReal globalMaxImaginary = 0.0;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalVector(data.fieldDM, &baseflowLocal));
  PetscCall(DMGlobalToLocalBegin(data.fieldDM, data.baseflow, INSERT_VALUES,
                                 baseflowLocal));
  PetscCall(DMGlobalToLocalEnd(data.fieldDM, data.baseflow, INSERT_VALUES,
                               baseflowLocal));
  PetscCall(DMDAGetCorners(data.fieldDM, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAGetGhostCorners(data.fieldDM, &gxs, &gys, nullptr, &gxm, &gym,
                                nullptr));
  PetscCall(
      DMDAVecGetArrayDOFRead(data.fieldDM, baseflowLocal, &baseflowArray));
  PetscCall(DMDAVecGetArrayDOFRead(data.metricDM, data.metrics, &metricArray));
  PetscCall(DMDAVecGetArrayDOF(data.baseflowDerivativeDM,
                               data.baseflowDerivatives, &derivativeArray));

  for (PetscInt j = ys; j < ys + ym; ++j) {
    for (PetscInt i = xs; i < xs + xm; ++i) {
      bool stencilAvailable = true;

      const PetscScalar density =
          baseflowArray[j][i][flowIndex(FlowComponent::Density)];
      const PetscScalar temperature =
          baseflowArray[j][i][flowIndex(FlowComponent::Temperature)];
      if (PetscIsInfOrNanScalar(density) ||
          PetscIsInfOrNanScalar(temperature)) {
        localFailures[1] = 1;
      } else if (realPart(density) <= 0.0 || realPart(temperature) <= 0.0) {
        localFailures[2] = 1;
      }

      for (PetscInt field = 0; field < flowComponentCount; ++field) {
        PetscReal valueXi = 0.0;
        PetscReal valueEta = 0.0;
        PetscReal valueXiXi = 0.0;
        PetscReal valueEtaEta = 0.0;
        PetscReal valueXiEta = 0.0;

        for (PetscInt xiSlot = 0; xiSlot < data.xiRule.stencilSize();
             ++xiSlot) {
          const PetscInt column = data.xiRule.stencilIndex(i, xiSlot);
          if (!inHalfOpenRange(column, gxs, gxm)) {
            stencilAvailable = false;
            continue;
          }
          const PetscScalar value = baseflowArray[j][column][field];
          if (PetscIsInfOrNanScalar(value))
            localFailures[1] = 1;
          localMaxImaginary = std::max(localMaxImaginary,
                                       PetscAbsReal(PetscImaginaryPart(value)));
          valueXi += data.xiRule.weight(1, i, xiSlot) * realPart(value);
          valueXiXi += data.xiRule.weight(2, i, xiSlot) * realPart(value);
        }

        for (PetscInt etaSlot = 0; etaSlot < data.etaRule.stencilSize();
             ++etaSlot) {
          const PetscInt row = data.etaRule.stencilIndex(j, etaSlot);
          if (!inHalfOpenRange(row, gys, gym)) {
            stencilAvailable = false;
            continue;
          }
          const PetscScalar value = baseflowArray[row][i][field];
          if (PetscIsInfOrNanScalar(value))
            localFailures[1] = 1;
          localMaxImaginary = std::max(localMaxImaginary,
                                       PetscAbsReal(PetscImaginaryPart(value)));
          valueEta += data.etaRule.weight(1, j, etaSlot) * realPart(value);
          valueEtaEta += data.etaRule.weight(2, j, etaSlot) * realPart(value);
        }

        for (PetscInt etaSlot = 0; etaSlot < data.etaRule.stencilSize();
             ++etaSlot) {
          const PetscInt row = data.etaRule.stencilIndex(j, etaSlot);
          if (!inHalfOpenRange(row, gys, gym)) {
            stencilAvailable = false;
            continue;
          }
          const PetscReal etaWeight = data.etaRule.weight(1, j, etaSlot);
          for (PetscInt xiSlot = 0; xiSlot < data.xiRule.stencilSize();
               ++xiSlot) {
            const PetscInt column = data.xiRule.stencilIndex(i, xiSlot);
            if (!inHalfOpenRange(column, gxs, gxm)) {
              stencilAvailable = false;
              continue;
            }
            const PetscScalar value = baseflowArray[row][column][field];
            if (PetscIsInfOrNanScalar(value))
              localFailures[1] = 1;
            localMaxImaginary = std::max(
                localMaxImaginary, PetscAbsReal(PetscImaginaryPart(value)));
            valueXiEta +=
                etaWeight * data.xiRule.weight(1, i, xiSlot) * realPart(value);
          }
        }

        if (!stencilAvailable)
          continue;

        const PetscReal xiY =
            realPart(metricArray[j][i][metricIndex(MetricComponent::XiY)]);
        const PetscReal xiZ =
            realPart(metricArray[j][i][metricIndex(MetricComponent::XiZ)]);
        const PetscReal etaY =
            realPart(metricArray[j][i][metricIndex(MetricComponent::EtaY)]);
        const PetscReal etaZ =
            realPart(metricArray[j][i][metricIndex(MetricComponent::EtaZ)]);
        const PetscReal xiYY =
            realPart(metricArray[j][i][metricIndex(MetricComponent::XiYY)]);
        const PetscReal xiZZ =
            realPart(metricArray[j][i][metricIndex(MetricComponent::XiZZ)]);
        const PetscReal xiYZ =
            realPart(metricArray[j][i][metricIndex(MetricComponent::XiYZ)]);
        const PetscReal etaYY =
            realPart(metricArray[j][i][metricIndex(MetricComponent::EtaYY)]);
        const PetscReal etaZZ =
            realPart(metricArray[j][i][metricIndex(MetricComponent::EtaZZ)]);
        const PetscReal etaYZ =
            realPart(metricArray[j][i][metricIndex(MetricComponent::EtaYZ)]);

        const PetscReal valueY = xiY * valueXi + etaY * valueEta;
        const PetscReal valueZ = xiZ * valueXi + etaZ * valueEta;
        const PetscReal valueYY =
            xiY * xiY * valueXiXi + 2.0 * xiY * etaY * valueXiEta +
            etaY * etaY * valueEtaEta + xiYY * valueXi + etaYY * valueEta;
        const PetscReal valueZZ =
            xiZ * xiZ * valueXiXi + 2.0 * xiZ * etaZ * valueXiEta +
            etaZ * etaZ * valueEtaEta + xiZZ * valueXi + etaZZ * valueEta;
        const PetscReal valueYZ =
            xiY * xiZ * valueXiXi + (xiY * etaZ + etaY * xiZ) * valueXiEta +
            etaY * etaZ * valueEtaEta + xiYZ * valueXi + etaYZ * valueEta;

        const std::array<PetscReal, baseFlowDerivativeKindCount> values = {
            valueY, valueZ, valueYY, valueZZ, valueYZ};
        if (std::any_of(values.begin(), values.end(), [](PetscReal value) {
              return PetscIsInfOrNanReal(value);
            })) {
          localFailures[3] = 1;
          continue;
        }

        for (PetscInt derivative = 0; derivative < baseFlowDerivativeKindCount;
             ++derivative)
          derivativeArray[j][i][derivative * flowComponentCount + field] =
              PetscScalar(values[static_cast<std::size_t>(derivative)]);
      }

      if (!stencilAvailable)
        localFailures[0] = 1;
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(data.baseflowDerivativeDM,
                                   data.baseflowDerivatives, &derivativeArray));
  PetscCall(
      DMDAVecRestoreArrayDOFRead(data.metricDM, data.metrics, &metricArray));
  PetscCall(
      DMDAVecRestoreArrayDOFRead(data.fieldDM, baseflowLocal, &baseflowArray));
  PetscCall(DMRestoreLocalVector(data.fieldDM, &baseflowLocal));

  if (localMaxImaginary > 100.0 * PETSC_MACHINE_EPSILON)
    localFailures[4] = 1;

  PetscCallMPI(MPI_Allreduce(localFailures.data(), globalFailures.data(),
                             static_cast<int>(localFailures.size()), MPIU_INT,
                             MPI_MAX, comm_));
  PetscCallMPI(MPI_Allreduce(&localMaxImaginary, &globalMaxImaginary, 1,
                             MPIU_REAL, MPI_MAX, comm_));
  diagnostics.maxBaseFlowImaginary = globalMaxImaginary;

  PetscCheck(!globalFailures[0], comm_, PETSC_ERR_PLIB,
             "An FD-q stencil extends outside the base-flow DMDA ghost region");
  PetscCheck(!globalFailures[1], comm_, PETSC_ERR_FP,
             "Base-flow values are not finite");
  PetscCheck(!globalFailures[2], comm_, PETSC_ERR_ARG_OUTOFRANGE,
             "Base-flow density and temperature must both be positive");
  PetscCheck(!globalFailures[3], comm_, PETSC_ERR_FP,
             "Computed base-flow derivatives are not finite");
  PetscCheck(!globalFailures[4], comm_, PETSC_ERR_SUP,
             "Base-flow values must be real; maximum imaginary magnitude is "
             "%.16g",
             static_cast<double>(globalMaxImaginary));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BaseFlowDerivatives::computeNorms(
    const ProblemData &data, BaseFlowDerivativeDiagnostics &diagnostics) const {
  PetscFunctionBeginUser;
  for (PetscInt component = 0; component < baseFlowDerivativeComponentCount;
       ++component)
    PetscCall(
        VecStrideNorm(data.baseflowDerivatives, component, NORM_2,
                      &diagnostics.norms[static_cast<std::size_t>(component)]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode
BaseFlowDerivatives::compute(ProblemData &data,
                             BaseFlowDerivativeDiagnostics &diagnostics) const {
  PetscFunctionBeginUser;
  PetscCheck(data.fieldDM && data.baseflow, comm_, PETSC_ERR_ARG_WRONGSTATE,
             "Base-flow data must be prepared before computing derivatives");
  PetscCheck(data.metricDM && data.metrics, comm_, PETSC_ERR_ARG_WRONGSTATE,
             "Metrics must be computed before base-flow derivatives");
  PetscCheck(data.ny == data.xiRule.nodeCount(), comm_, PETSC_ERR_ARG_SIZ,
             "xi rule node count does not match Ny");
  PetscCheck(data.nz == data.etaRule.nodeCount(), comm_, PETSC_ERR_ARG_SIZ,
             "eta rule node count does not match Nz");

  diagnostics = {};
  PetscCall(createStorage(data));
  PetscCall(computeValues(data, diagnostics));
  PetscCall(computeNorms(data, diagnostics));
  PetscFunctionReturn(PETSC_SUCCESS);
}
