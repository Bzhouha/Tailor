/**
 * @file Metrics.cpp
 * @brief Distributed computation and validation of curvilinear metrics.
 */
#include "Metrics.hpp"

#include <petscmath.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace {

constexpr PetscInt gridY = 1;
constexpr PetscInt gridZ = 2;
constexpr PetscInt firstMetricCount = 4;

/** @brief Test whether an index belongs to a half-open local range. */
bool inHalfOpenRange(PetscInt value, PetscInt first, PetscInt count) {
  return value >= first && value < first + count;
}

/** @brief Extract the real part of a validated PETSc scalar. */
PetscReal realPart(PetscScalar value) { return PetscRealPart(value); }

/** @brief Return the integer wrap count of a periodic global index. */
PetscInt periodicWrap(PetscInt index, PetscInt nodeCount) {
  if (index >= 0)
    return index / nodeCount;
  return -((-index + nodeCount - 1) / nodeCount);
}

} // namespace

Metrics::Metrics(MPI_Comm comm) : comm_(comm) {}

PetscErrorCode Metrics::createStorage(ProblemData &data) const {
  PetscFunctionBeginUser;
  PetscCheck(data.fieldDM, comm_, PETSC_ERR_ARG_WRONGSTATE,
             "fieldDM must be prepared before computing metrics");
  PetscCheck(!data.metricDM && !data.metrics, comm_, PETSC_ERR_ARG_WRONGSTATE,
             "Metrics storage has already been created");

  PetscCall(DMDACreateCompatibleDMDA(data.fieldDM, metricComponentCount,
                                     &data.metricDM));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(data.metricDM),
                               "metricDM"));
  PetscCall(DMCreateGlobalVector(data.metricDM, &data.metrics));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(data.metrics),
                               "metrics"));
  PetscCall(VecZeroEntries(data.metrics));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode
Metrics::computeFirstOrder(ProblemData &data,
                           MetricDiagnostics &diagnostics) const {
  Vec gridLocal = nullptr;
  const PetscScalar ***gridArray = nullptr;
  PetscScalar ***metricArray = nullptr;
  PetscInt xs = 0;
  PetscInt ys = 0;
  PetscInt xm = 0;
  PetscInt ym = 0;
  PetscInt gxs = 0;
  PetscInt gys = 0;
  PetscInt gxm = 0;
  PetscInt gym = 0;
  std::array<PetscInt, 4> localFailures{};
  std::array<PetscInt, 4> globalFailures{};
  PetscReal localMinJacobian = PETSC_MAX_REAL;
  PetscReal localMaxJacobian = PETSC_MIN_REAL;
  PetscReal localMaxImaginary = 0.0;
  PetscReal globalMinJacobian = 0.0;
  PetscReal globalMaxJacobian = 0.0;
  PetscReal globalMaxImaginary = 0.0;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalVector(data.gridDM, &gridLocal));
  PetscCall(
      DMGlobalToLocalBegin(data.gridDM, data.grid, INSERT_VALUES, gridLocal));
  PetscCall(
      DMGlobalToLocalEnd(data.gridDM, data.grid, INSERT_VALUES, gridLocal));
  PetscCall(DMDAGetCorners(data.gridDM, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAGetGhostCorners(data.gridDM, &gxs, &gys, nullptr, &gxm, &gym,
                                nullptr));
  PetscCall(DMDAVecGetArrayDOFRead(data.gridDM, gridLocal, &gridArray));
  PetscCall(DMDAVecGetArrayDOF(data.metricDM, data.metrics, &metricArray));

  for (PetscInt j = ys; j < ys + ym; ++j) {
    for (PetscInt i = xs; i < xs + xm; ++i) {
      PetscReal yXi = 0.0;
      PetscReal yEta = 0.0;
      PetscReal zXi = 0.0;
      PetscReal zEta = 0.0;
      bool stencilAvailable = true;

      for (PetscInt slot = 0; slot < data.xiRule.stencilSize(); ++slot) {
        const PetscInt column = data.xiRule.localIndex(i, slot);
        if (!inHalfOpenRange(column, gxs, gxm)) {
          stencilAvailable = false;
          continue;
        }
        const PetscScalar y = gridArray[j][column][gridY];
        const PetscScalar z = gridArray[j][column][gridZ];
        if (PetscIsInfOrNanScalar(y) || PetscIsInfOrNanScalar(z))
          localFailures[1] = 1;
        localMaxImaginary =
            std::max({localMaxImaginary, PetscAbsReal(PetscImaginaryPart(y)),
                      PetscAbsReal(PetscImaginaryPart(z))});
        const PetscReal weight = data.xiRule.weight(1, i, slot);
        yXi += weight * realPart(y);
        zXi += weight * realPart(z);
      }

      for (PetscInt slot = 0; slot < data.etaRule.stencilSize(); ++slot) {
        const PetscInt row = data.etaRule.localIndex(j, slot);
        if (!inHalfOpenRange(row, gys, gym)) {
          stencilAvailable = false;
          continue;
        }
        const PetscScalar y = gridArray[row][i][gridY];
        const PetscScalar z =
            gridArray[row][i][gridZ] +
            static_cast<PetscReal>(periodicWrap(row, data.nz)) *
                data.spanwisePeriod;
        if (PetscIsInfOrNanScalar(y) || PetscIsInfOrNanScalar(z))
          localFailures[1] = 1;
        localMaxImaginary =
            std::max({localMaxImaginary, PetscAbsReal(PetscImaginaryPart(y)),
                      PetscAbsReal(PetscImaginaryPart(z))});
        const PetscReal weight = data.etaRule.weight(1, j, slot);
        yEta += weight * realPart(y);
        zEta += weight * realPart(z);
      }

      if (!stencilAvailable) {
        localFailures[0] = 1;
        continue;
      }

      if (PetscIsInfOrNanReal(yXi) || PetscIsInfOrNanReal(yEta) ||
          PetscIsInfOrNanReal(zXi) || PetscIsInfOrNanReal(zEta)) {
        localFailures[1] = 1;
        continue;
      }

      const PetscReal firstProduct = yXi * zEta;
      const PetscReal secondProduct = yEta * zXi;
      const PetscReal jacobian = firstProduct - secondProduct;
      const PetscReal jacobianScale =
          std::max({PetscReal(1.0), PetscAbsReal(firstProduct),
                    PetscAbsReal(secondProduct)});
      const PetscReal jacobianTolerance =
          100.0 * PETSC_MACHINE_EPSILON * jacobianScale;

      if (PetscIsInfOrNanReal(jacobian)) {
        localFailures[1] = 1;
        continue;
      }
      localMinJacobian = std::min(localMinJacobian, jacobian);
      localMaxJacobian = std::max(localMaxJacobian, jacobian);
      if (jacobian <= jacobianTolerance) {
        localFailures[2] = 1;
        continue;
      }

      const std::array<PetscReal, firstMetricCount> firstMetrics = {
          zEta / jacobian, -yEta / jacobian, -zXi / jacobian, yXi / jacobian};
      if (std::any_of(
              firstMetrics.begin(), firstMetrics.end(),
              [](PetscReal value) { return PetscIsInfOrNanReal(value); })) {
        localFailures[1] = 1;
        continue;
      }

      metricArray[j][i][metricIndex(MetricComponent::XiY)] =
          PetscScalar(firstMetrics[0]);
      metricArray[j][i][metricIndex(MetricComponent::XiZ)] =
          PetscScalar(firstMetrics[1]);
      metricArray[j][i][metricIndex(MetricComponent::EtaY)] =
          PetscScalar(firstMetrics[2]);
      metricArray[j][i][metricIndex(MetricComponent::EtaZ)] =
          PetscScalar(firstMetrics[3]);
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(data.metricDM, data.metrics, &metricArray));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.gridDM, gridLocal, &gridArray));
  PetscCall(DMRestoreLocalVector(data.gridDM, &gridLocal));

  const PetscReal imaginaryTolerance = 100.0 * PETSC_MACHINE_EPSILON;
  if (localMaxImaginary > imaginaryTolerance)
    localFailures[3] = 1;

  PetscCallMPI(MPI_Allreduce(localFailures.data(), globalFailures.data(),
                             static_cast<int>(localFailures.size()), MPIU_INT,
                             MPI_MAX, comm_));
  PetscCallMPI(MPI_Allreduce(&localMinJacobian, &globalMinJacobian, 1,
                             MPIU_REAL, MPI_MIN, comm_));
  PetscCallMPI(MPI_Allreduce(&localMaxJacobian, &globalMaxJacobian, 1,
                             MPIU_REAL, MPI_MAX, comm_));
  PetscCallMPI(MPI_Allreduce(&localMaxImaginary, &globalMaxImaginary, 1,
                             MPIU_REAL, MPI_MAX, comm_));

  diagnostics.minJacobian = globalMinJacobian;
  diagnostics.maxJacobian = globalMaxJacobian;
  diagnostics.maxGridImaginary = globalMaxImaginary;

  PetscCheck(!globalFailures[0], comm_, PETSC_ERR_PLIB,
             "An FD-q stencil extends outside the local DMDA ghost region");
  PetscCheck(!globalFailures[1], comm_, PETSC_ERR_FP,
             "Grid coordinates or computed first-order metrics are not finite");
  PetscCheck(!globalFailures[2], comm_, PETSC_ERR_ARG_OUTOFRANGE,
             "Grid Jacobian must be positive and nonsingular; global range is "
             "[%.16g, %.16g]",
             static_cast<double>(globalMinJacobian),
             static_cast<double>(globalMaxJacobian));
  PetscCheck(!globalFailures[3], comm_, PETSC_ERR_SUP,
             "Grid y/z coordinates must be real; maximum imaginary magnitude "
             "is %.16g",
             static_cast<double>(globalMaxImaginary));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Metrics::computeSecondOrder(ProblemData &data) const {
  Vec metricLocal = nullptr;
  const PetscScalar ***localArray = nullptr;
  PetscScalar ***globalArray = nullptr;
  PetscInt xs = 0;
  PetscInt ys = 0;
  PetscInt xm = 0;
  PetscInt ym = 0;
  PetscInt gxs = 0;
  PetscInt gys = 0;
  PetscInt gxm = 0;
  PetscInt gym = 0;
  std::array<PetscInt, 2> localFailures{};
  std::array<PetscInt, 2> globalFailures{};

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalVector(data.metricDM, &metricLocal));
  PetscCall(DMGlobalToLocalBegin(data.metricDM, data.metrics, INSERT_VALUES,
                                 metricLocal));
  PetscCall(DMGlobalToLocalEnd(data.metricDM, data.metrics, INSERT_VALUES,
                               metricLocal));
  PetscCall(
      DMDAGetCorners(data.metricDM, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAGetGhostCorners(data.metricDM, &gxs, &gys, nullptr, &gxm, &gym,
                                nullptr));
  PetscCall(DMDAVecGetArrayDOFRead(data.metricDM, metricLocal, &localArray));
  PetscCall(DMDAVecGetArrayDOF(data.metricDM, data.metrics, &globalArray));

  for (PetscInt j = ys; j < ys + ym; ++j) {
    for (PetscInt i = xs; i < xs + xm; ++i) {
      std::array<PetscReal, firstMetricCount> derivativeXi{};
      std::array<PetscReal, firstMetricCount> derivativeEta{};
      bool stencilAvailable = true;

      for (PetscInt slot = 0; slot < data.xiRule.stencilSize(); ++slot) {
        const PetscInt column = data.xiRule.localIndex(i, slot);
        if (!inHalfOpenRange(column, gxs, gxm)) {
          stencilAvailable = false;
          continue;
        }
        const PetscReal weight = data.xiRule.weight(1, i, slot);
        for (PetscInt component = 0; component < firstMetricCount; ++component)
          derivativeXi[static_cast<std::size_t>(component)] +=
              weight * realPart(localArray[j][column][component]);
      }

      for (PetscInt slot = 0; slot < data.etaRule.stencilSize(); ++slot) {
        const PetscInt row = data.etaRule.localIndex(j, slot);
        if (!inHalfOpenRange(row, gys, gym)) {
          stencilAvailable = false;
          continue;
        }
        const PetscReal weight = data.etaRule.weight(1, j, slot);
        for (PetscInt component = 0; component < firstMetricCount; ++component)
          derivativeEta[static_cast<std::size_t>(component)] +=
              weight * realPart(localArray[row][i][component]);
      }

      if (!stencilAvailable) {
        localFailures[0] = 1;
        continue;
      }

      const PetscReal xiY =
          realPart(localArray[j][i][metricIndex(MetricComponent::XiY)]);
      const PetscReal xiZ =
          realPart(localArray[j][i][metricIndex(MetricComponent::XiZ)]);
      const PetscReal etaY =
          realPart(localArray[j][i][metricIndex(MetricComponent::EtaY)]);
      const PetscReal etaZ =
          realPart(localArray[j][i][metricIndex(MetricComponent::EtaZ)]);

      const PetscReal xiYY =
          xiY * derivativeXi[metricIndex(MetricComponent::XiY)] +
          etaY * derivativeEta[metricIndex(MetricComponent::XiY)];
      const PetscReal xiZZ =
          xiZ * derivativeXi[metricIndex(MetricComponent::XiZ)] +
          etaZ * derivativeEta[metricIndex(MetricComponent::XiZ)];
      const PetscReal xiYZ =
          xiZ * derivativeXi[metricIndex(MetricComponent::XiY)] +
          etaZ * derivativeEta[metricIndex(MetricComponent::XiY)];
      const PetscReal etaYY =
          xiY * derivativeXi[metricIndex(MetricComponent::EtaY)] +
          etaY * derivativeEta[metricIndex(MetricComponent::EtaY)];
      const PetscReal etaZZ =
          xiZ * derivativeXi[metricIndex(MetricComponent::EtaZ)] +
          etaZ * derivativeEta[metricIndex(MetricComponent::EtaZ)];
      const PetscReal etaYZ =
          xiZ * derivativeXi[metricIndex(MetricComponent::EtaY)] +
          etaZ * derivativeEta[metricIndex(MetricComponent::EtaY)];

      const std::array<PetscReal, 6> secondMetrics = {xiYY,  xiZZ,  xiYZ,
                                                      etaYY, etaZZ, etaYZ};
      if (std::any_of(
              secondMetrics.begin(), secondMetrics.end(),
              [](PetscReal value) { return PetscIsInfOrNanReal(value); })) {
        localFailures[1] = 1;
        continue;
      }

      globalArray[j][i][metricIndex(MetricComponent::XiYY)] = PetscScalar(xiYY);
      globalArray[j][i][metricIndex(MetricComponent::XiZZ)] = PetscScalar(xiZZ);
      globalArray[j][i][metricIndex(MetricComponent::XiYZ)] = PetscScalar(xiYZ);
      globalArray[j][i][metricIndex(MetricComponent::EtaYY)] =
          PetscScalar(etaYY);
      globalArray[j][i][metricIndex(MetricComponent::EtaZZ)] =
          PetscScalar(etaZZ);
      globalArray[j][i][metricIndex(MetricComponent::EtaYZ)] =
          PetscScalar(etaYZ);
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(data.metricDM, data.metrics, &globalArray));
  PetscCall(
      DMDAVecRestoreArrayDOFRead(data.metricDM, metricLocal, &localArray));
  PetscCall(DMRestoreLocalVector(data.metricDM, &metricLocal));

  PetscCallMPI(MPI_Allreduce(localFailures.data(), globalFailures.data(),
                             static_cast<int>(localFailures.size()), MPIU_INT,
                             MPI_MAX, comm_));
  PetscCheck(!globalFailures[0], comm_, PETSC_ERR_PLIB,
             "An FD-q stencil extends outside the metric DMDA ghost region");
  PetscCheck(!globalFailures[1], comm_, PETSC_ERR_FP,
             "Computed second-order metrics are not finite");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Metrics::computeNorms(const ProblemData &data,
                                     MetricDiagnostics &diagnostics) const {
  PetscFunctionBeginUser;
  for (PetscInt component = 0; component < metricComponentCount; ++component)
    PetscCall(
        VecStrideNorm(data.metrics, component, NORM_2,
                      &diagnostics.norms[static_cast<std::size_t>(component)]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Metrics::compute(ProblemData &data,
                                MetricDiagnostics &diagnostics) const {
  PetscFunctionBeginUser;
  PetscCheck(data.gridDM && data.grid, comm_, PETSC_ERR_ARG_WRONGSTATE,
             "Grid data must be prepared before computing metrics");
  PetscCheck(data.ny == data.xiRule.nodeCount(), comm_, PETSC_ERR_ARG_SIZ,
             "xi rule node count does not match Ny");
  PetscCheck(data.nz == data.etaRule.nodeCount(), comm_, PETSC_ERR_ARG_SIZ,
             "eta rule node count does not match Nz");

  diagnostics = {};
  PetscCall(createStorage(data));
  PetscCall(computeFirstOrder(data, diagnostics));
  PetscCall(computeSecondOrder(data));
  PetscCall(computeNorms(data, diagnostics));
  PetscFunctionReturn(PETSC_SUCCESS);
}
