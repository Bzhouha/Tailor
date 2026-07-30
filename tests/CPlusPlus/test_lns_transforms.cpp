/**
 * @file test_lns_transforms.cpp
 * @brief Golden-data tests for Fourier and curvilinear LNS transforms.
 */
#include <slepcsys.h>

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>

#include "BaseFlowDerivatives.hpp"
#include "CurvilinearTransform.hpp"
#include "LNSCoefficients.hpp"
#include "Metrics.hpp"
#include "Parser.hpp"
#include "Prepare.hpp"
#include "ProblemData.hpp"
#include "Recipe.hpp"
#include "StreamwiseFourier.hpp"

namespace {

using BlockList = std::array<const Block5 *,
                             static_cast<std::size_t>(fourierCoefficientCount)>;

/** @brief Return Fourier blocks in canonical fixture order. */
BlockList blocks(const FourierLNSCoefficients &coefficients) {
  return {&coefficients.Gamma, &coefficients.K0,  &coefficients.Ky,
          &coefficients.Kz,    &coefficients.Vyy, &coefficients.Vyz,
          &coefficients.Vzz};
}

/** @brief Return curvilinear blocks in canonical fixture order. */
BlockList blocks(const CurvilinearLNSCoefficients &coefficients) {
  return {&coefficients.Gamma,  &coefficients.K0,    &coefficients.Kxi,
          &coefficients.Keta,   &coefficients.Vxixi, &coefficients.Vxieta,
          &coefficients.Vetaeta};
}

/** @brief Construct one physical coefficient fixture point. */
PhysicalLNSCoefficients referencePhysical(PetscInt point) {
  PhysicalLNSCoefficients physical;
  const std::array<Block5 *, physicalCoefficientCount> physicalBlocks = {
      &physical.Gamma, &physical.A,   &physical.B,   &physical.C,
      &physical.D,     &physical.Vxx, &physical.Vxy, &physical.Vxz,
      &physical.Vyy,   &physical.Vyz, &physical.Vzz,
  };
  for (PetscInt matrix = 0; matrix < physicalCoefficientCount; ++matrix) {
    for (PetscInt row = 0; row < flowComponentCount; ++row) {
      for (PetscInt column = 0; column < flowComponentCount; ++column) {
        PetscReal value = 0.13 * static_cast<PetscReal>(matrix + 1) +
                          0.017 * static_cast<PetscReal>(row + 1) +
                          0.003 * static_cast<PetscReal>(column + 1) +
                          0.0007 * static_cast<PetscReal>(point + 1);
        if ((matrix + row + column + point + 4) % 2 == 0)
          value = -value;
        (*physicalBlocks[static_cast<std::size_t>(matrix)])(row, column) =
            PetscScalar(value);
      }
    }
  }
  return physical;
}

/** @brief Return the Fourier wavenumber and metrics for a fixture point. */
void referenceParameters(PetscInt point, PetscScalar &alpha,
                         MetricPoint &metrics) {
  if (point == 0) {
    alpha = PetscCMPLX(2.43, 0.17);
    metrics = {0.81,  -0.23,  0.19,  1.07,   -0.031,
               0.024, -0.017, 0.028, -0.022, 0.013};
  } else {
    alpha = PetscCMPLX(-0.64, 0.29);
    metrics = {1.14,   0.37,  -0.26,  0.72,  0.041,
               -0.036, 0.025, -0.033, 0.019, -0.027};
  }
}

/** @brief Read one required string-valued PETSc option. */
PetscErrorCode readOption(const char *name, std::string &value) {
  std::array<char, PETSC_MAX_PATH_LEN> buffer{};
  PetscBool found = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetString(nullptr, nullptr, name, buffer.data(),
                                  buffer.size(), &found));
  PetscCheck(found && buffer[0] != '\0', PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT,
             "Required option %s was not provided", name);
  value = buffer.data();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Compare one 5-by-5 block with the text fixture stream. */
PetscErrorCode compareBlock(std::ifstream &stream, const Block5 &actual,
                            PetscInt point, const char *stage,
                            PetscInt matrix) {
  PetscFunctionBeginUser;
  for (PetscInt row = 0; row < flowComponentCount; ++row) {
    for (PetscInt column = 0; column < flowComponentCount; ++column) {
      PetscReal expectedReal = 0.0;
      PetscReal expectedImaginary = 0.0;
      stream >> expectedReal >> expectedImaginary;
      PetscCheck(stream.good(), PETSC_COMM_WORLD, PETSC_ERR_FILE_UNEXPECTED,
                 "Transform fixture ended at point %" PetscInt_FMT
                 ", %s matrix %" PetscInt_FMT ", row %" PetscInt_FMT
                 ", column %" PetscInt_FMT,
                 point, stage, matrix, row, column);
      const PetscScalar expected = PetscCMPLX(expectedReal, expectedImaginary);
      const PetscReal tolerance =
          5.0e-13 * PetscMax(1.0, PetscAbsScalar(expected));
      PetscCheck(PetscAbsScalar(actual(row, column) - expected) <= tolerance,
                 PETSC_COMM_WORLD, PETSC_ERR_PLIB,
                 "Transform mismatch at point %" PetscInt_FMT
                 ", %s matrix %" PetscInt_FMT "(%" PetscInt_FMT
                 ",%" PetscInt_FMT ")",
                 point, stage, matrix, row, column);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Compare all transformed blocks with Fortran golden data. */
PetscErrorCode compareFortranGold(const std::string &goldPath) {
  std::ifstream stream(goldPath);

  PetscFunctionBeginUser;
  PetscCheck(stream.good(), PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN,
             "Cannot open Fortran transform fixture: %s", goldPath.c_str());
  for (PetscInt point = 0; point < 2; ++point) {
    const PhysicalLNSCoefficients physical = referencePhysical(point);
    PetscScalar alpha = 0.0;
    MetricPoint metrics;
    referenceParameters(point, alpha, metrics);

    FourierLNSCoefficients fourier;
    CurvilinearLNSCoefficients curvilinear;
    PetscCheck(StreamwiseFourier(alpha).apply(physical, fourier) ==
                   CoefficientStatus::Success,
               PETSC_COMM_WORLD, PETSC_ERR_PLIB,
               "Fourier transform failed for reference point %" PetscInt_FMT,
               point);
    PetscCheck(
        CurvilinearTransform().apply(fourier, metrics, curvilinear) ==
            CoefficientStatus::Success,
        PETSC_COMM_WORLD, PETSC_ERR_PLIB,
        "Curvilinear transform failed for reference point %" PetscInt_FMT,
        point);

    const BlockList fourierBlocks = blocks(fourier);
    const BlockList curvilinearBlocks = blocks(curvilinear);
    for (PetscInt matrix = 0; matrix < fourierCoefficientCount; ++matrix)
      PetscCall(compareBlock(stream,
                             *fourierBlocks[static_cast<std::size_t>(matrix)],
                             point, "Fourier", matrix));
    for (PetscInt matrix = 0; matrix < curvilinearCoefficientCount; ++matrix)
      PetscCall(compareBlock(
          stream, *curvilinearBlocks[static_cast<std::size_t>(matrix)], point,
          "curvilinear", matrix));
  }

  PetscReal extra = 0.0;
  stream >> extra;
  PetscCheck(stream.eof(), PETSC_COMM_WORLD, PETSC_ERR_FILE_UNEXPECTED,
             "Fortran transform fixture contains extra values");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Verify identity metrics and invalid-input status paths. */
PetscErrorCode validateIdentityAndFailures() {
  const PhysicalLNSCoefficients physical = referencePhysical(0);
  FourierLNSCoefficients fourier;
  CurvilinearLNSCoefficients identity;
  const MetricPoint identityMetrics = {1.0, 0.0, 0.0, 1.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0};

  PetscFunctionBeginUser;
  PetscCheck(
      StreamwiseFourier(PetscCMPLX(1.2, -0.3)).apply(physical, fourier) ==
          CoefficientStatus::Success,
      PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Identity Fourier setup failed");
  PetscCheck(CurvilinearTransform().apply(fourier, identityMetrics, identity) ==
                 CoefficientStatus::Success,
             PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Identity curvilinear transform failed");

  const std::array<const Block5 *, 7> expected = {
      &fourier.Gamma, &fourier.K0,  &fourier.Ky,  &fourier.Kz,
      &fourier.Vyy,   &fourier.Vyz, &fourier.Vzz,
  };
  const BlockList actual = blocks(identity);
  for (PetscInt matrix = 0; matrix < curvilinearCoefficientCount; ++matrix) {
    for (PetscInt row = 0; row < flowComponentCount; ++row) {
      for (PetscInt column = 0; column < flowComponentCount; ++column) {
        PetscCheck(
            (*actual[static_cast<std::size_t>(matrix)])(row, column) ==
                (*expected[static_cast<std::size_t>(matrix)])(row, column),
            PETSC_COMM_WORLD, PETSC_ERR_PLIB,
            "Identity transform changed matrix %" PetscInt_FMT, matrix);
      }
    }
  }

  const PetscReal nan = std::numeric_limits<PetscReal>::quiet_NaN();
  FourierLNSCoefficients invalidFourier;
  PetscCheck(
      StreamwiseFourier(PetscCMPLX(nan, 0.0)).apply(physical, invalidFourier) ==
          CoefficientStatus::NonFiniteInput,
      PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Non-finite alpha was not rejected");

  std::array<PetscScalar, metricComponentCount> rawMetrics{};
  rawMetrics[metricIndex(MetricComponent::XiY)] = 1.0;
  rawMetrics[metricIndex(MetricComponent::EtaZ)] = 1.0;
  rawMetrics[metricIndex(MetricComponent::XiYZ)] = PetscCMPLX(0.0, 1.0e-6);
  MetricPoint metricPoint;
  PetscCheck(makeMetricPoint(rawMetrics.data(), metricPoint) ==
                 CoefficientStatus::NonFiniteInput,
             PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Non-real metric was not rejected");
  rawMetrics[metricIndex(MetricComponent::XiYZ)] = nan;
  PetscCheck(makeMetricPoint(rawMetrics.data(), metricPoint) ==
                 CoefficientStatus::NonFiniteInput,
             PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Non-finite metric was not rejected");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Transform coefficients at every node of the real case. */
PetscErrorCode validateDistributedField(const Recipe &recipe) {
  ProblemData data;
  MetricDiagnostics metricDiagnostics;
  BaseFlowDerivativeDiagnostics derivativeDiagnostics;
  const PetscScalar ***baseflow = nullptr;
  const PetscScalar ***derivatives = nullptr;
  const PetscScalar ***metrics = nullptr;
  PetscInt xs = 0;
  PetscInt ys = 0;
  PetscInt xm = 0;
  PetscInt ym = 0;
  PetscInt localFailure = 0;
  PetscInt globalFailure = 0;
  PetscInt localCount = 0;
  PetscInt globalCount = 0;
  const PetscScalar alpha =
      PetscCMPLX(static_cast<PetscReal>(recipe.alpha.real()),
                 static_cast<PetscReal>(recipe.alpha.imag()));
  const LNSCoefficients physicalBuilder(recipe);
  const StreamwiseFourier fourierTransform(alpha);
  const CurvilinearTransform curvilinearTransform;

  PetscFunctionBeginUser;
  PetscCall(Prepare(PETSC_COMM_WORLD).initialize(recipe, data));
  PetscCall(Metrics(PETSC_COMM_WORLD).compute(data, metricDiagnostics));
  PetscCall(BaseFlowDerivatives(PETSC_COMM_WORLD)
                .compute(data, derivativeDiagnostics));
  PetscCall(DMDAGetCorners(data.fieldDM, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAVecGetArrayDOFRead(data.fieldDM, data.baseflow, &baseflow));
  PetscCall(DMDAVecGetArrayDOFRead(data.baseflowDerivativeDM,
                                   data.baseflowDerivatives, &derivatives));
  PetscCall(DMDAVecGetArrayDOFRead(data.metricDM, data.metrics, &metrics));

  for (PetscInt j = ys; j < ys + ym; ++j) {
    for (PetscInt i = xs; i < xs + xm; ++i) {
      BaseFlowPoint point;
      PhysicalLNSCoefficients physical;
      FourierLNSCoefficients fourier;
      MetricPoint metricPoint;
      CurvilinearLNSCoefficients curvilinear;

      CoefficientStatus status =
          makeBaseFlowPoint(baseflow[j][i], derivatives[j][i], point);
      if (status == CoefficientStatus::Success)
        status = physicalBuilder.evaluate(point, physical);
      if (status == CoefficientStatus::Success)
        status = fourierTransform.apply(physical, fourier);
      if (status == CoefficientStatus::Success)
        status = makeMetricPoint(metrics[j][i], metricPoint);
      if (status == CoefficientStatus::Success)
        status = curvilinearTransform.apply(fourier, metricPoint, curvilinear);
      if (status != CoefficientStatus::Success)
        localFailure = 1;
      ++localCount;
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(data.metricDM, data.metrics, &metrics));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.baseflowDerivativeDM,
                                       data.baseflowDerivatives, &derivatives));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.fieldDM, data.baseflow, &baseflow));
  PetscCallMPI(MPI_Allreduce(&localFailure, &globalFailure, 1, MPIU_INT,
                             MPI_MAX, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localCount, &globalCount, 1, MPIU_INT, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCheck(!globalFailure, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "At least one real-case node failed the LNS transform pipeline");
  PetscCheck(globalCount == data.ny * data.nz, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Distributed transform validation visited %" PetscInt_FMT
             " nodes, expected %" PetscInt_FMT,
             globalCount, data.ny * data.nz);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Run golden, identity, failure, and distributed checks. */
PetscErrorCode runTests() {
  std::string configPath;
  std::string goldPath;
  Recipe recipe;

  PetscFunctionBeginUser;
  PetscCall(readOption("-c", configPath));
  PetscCall(readOption("-gold", goldPath));
  PetscCall(Parser(PETSC_COMM_WORLD).parse(configPath, recipe));
  PetscCall(compareFortranGold(goldPath));
  PetscCall(validateIdentityAndFailures());
  PetscCall(validateDistributedField(recipe));
  PetscCall(
      PetscPrintf(PETSC_COMM_WORLD,
                  "LNS Fourier and curvilinear transform tests passed.\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

/** @brief Initialize SLEPc, run transform tests, and finalize. */
int main(int argc, char **argv) {
  PetscErrorCode error = SlepcInitialize(&argc, &argv, nullptr, nullptr);
  if (error != PETSC_SUCCESS)
    return static_cast<int>(error);
  const PetscErrorCode testError = runTests();
  error = SlepcFinalize();
  if (testError != PETSC_SUCCESS)
    return static_cast<int>(testError);
  return static_cast<int>(error);
}
