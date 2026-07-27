#include <slepcsys.h>

#include <array>
#include <cmath>
#include <string>

#include "BaseFlowDerivatives.hpp"
#include "CurvilinearTransform.hpp"
#include "GlobalOperator.hpp"
#include "LNSCoefficients.hpp"
#include "Metrics.hpp"
#include "Parser.hpp"
#include "Prepare.hpp"
#include "ProblemData.hpp"
#include "Recipe.hpp"
#include "StreamwiseFourier.hpp"

namespace {

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

void applyBlock(const Block5 &block,
                const std::array<PetscScalar, flowComponentCount> &input,
                std::array<PetscScalar, flowComponentCount> &output,
                PetscScalar scale = PetscScalar(1.0)) {
  for (PetscInt row = 0; row < flowComponentCount; ++row) {
    PetscScalar value = 0.0;
    for (PetscInt column = 0; column < flowComponentCount; ++column)
      value += block(row, column) * input[static_cast<std::size_t>(column)];
    output[static_cast<std::size_t>(row)] += scale * value;
  }
}

CoefficientStatus
evaluateNode(const LNSCoefficients &physicalBuilder,
             const StreamwiseFourier &fourierTransform,
             const CurvilinearTransform &curvilinearTransform,
             const PetscScalar *baseflow, const PetscScalar *derivatives,
             const PetscScalar *metrics,
             CurvilinearLNSCoefficients &curvilinear) noexcept {
  BaseFlowPoint point;
  PhysicalLNSCoefficients physical;
  FourierLNSCoefficients fourier;
  MetricPoint metricPoint;

  CoefficientStatus status = makeBaseFlowPoint(baseflow, derivatives, point);
  if (status == CoefficientStatus::Success)
    status = physicalBuilder.evaluate(point, physical);
  if (status == CoefficientStatus::Success)
    status = fourierTransform.apply(physical, fourier);
  if (status == CoefficientStatus::Success)
    status = makeMetricPoint(metrics, metricPoint);
  if (status == CoefficientStatus::Success)
    status = curvilinearTransform.apply(fourier, metricPoint, curvilinear);
  return status;
}

PetscErrorCode fillDeterministicVector(DM dm, Vec vector) {
  PetscScalar ***values = nullptr;
  PetscInt xs = 0;
  PetscInt ys = 0;
  PetscInt xm = 0;
  PetscInt ym = 0;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAVecGetArrayDOF(dm, vector, &values));
  for (PetscInt j = ys; j < ys + ym; ++j) {
    for (PetscInt i = xs; i < xs + xm; ++i) {
      for (PetscInt component = 0; component < flowComponentCount;
           ++component) {
        const PetscReal phase = 0.017 * static_cast<PetscReal>(i + 1) +
                                0.031 * static_cast<PetscReal>(j + 1) +
                                0.11 * static_cast<PetscReal>(component + 1);
        values[j][i][component] =
            PetscCMPLX(std::sin(phase) + 0.07 * component,
                       0.3 * std::cos(1.7 * phase) - 0.02 * component);
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(dm, vector, &values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeReferenceAction(const Recipe &recipe,
                                      const ProblemData &data, Vec input,
                                      Vec massResult, Vec spatialResult) {
  Vec localInput = nullptr;
  const PetscScalar ***inputValues = nullptr;
  const PetscScalar ***baseflow = nullptr;
  const PetscScalar ***derivatives = nullptr;
  const PetscScalar ***metrics = nullptr;
  PetscScalar ***massValues = nullptr;
  PetscScalar ***spatialValues = nullptr;
  PetscInt xs = 0;
  PetscInt ys = 0;
  PetscInt xm = 0;
  PetscInt ym = 0;
  const PetscScalar alpha =
      PetscCMPLX(static_cast<PetscReal>(recipe.alpha.real()),
                 static_cast<PetscReal>(recipe.alpha.imag()));
  const LNSCoefficients physicalBuilder(recipe);
  const StreamwiseFourier fourierTransform(alpha);
  const CurvilinearTransform curvilinearTransform;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(massResult));
  PetscCall(VecZeroEntries(spatialResult));
  PetscCall(DMGetLocalVector(data.fieldDM, &localInput));
  PetscCall(
      DMGlobalToLocalBegin(data.fieldDM, input, INSERT_VALUES, localInput));
  PetscCall(DMGlobalToLocalEnd(data.fieldDM, input, INSERT_VALUES, localInput));
  PetscCall(DMDAGetCorners(data.fieldDM, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAVecGetArrayDOFRead(data.fieldDM, localInput, &inputValues));
  PetscCall(DMDAVecGetArrayDOFRead(data.fieldDM, data.baseflow, &baseflow));
  PetscCall(DMDAVecGetArrayDOFRead(data.baseflowDerivativeDM,
                                   data.baseflowDerivatives, &derivatives));
  PetscCall(DMDAVecGetArrayDOFRead(data.metricDM, data.metrics, &metrics));
  PetscCall(DMDAVecGetArrayDOF(data.fieldDM, massResult, &massValues));
  PetscCall(DMDAVecGetArrayDOF(data.fieldDM, spatialResult, &spatialValues));

  for (PetscInt j = ys; j < ys + ym; ++j) {
    for (PetscInt i = xs; i < xs + xm; ++i) {
      CurvilinearLNSCoefficients coefficients;
      PetscCheck(evaluateNode(physicalBuilder, fourierTransform,
                              curvilinearTransform, baseflow[j][i],
                              derivatives[j][i], metrics[j][i],
                              coefficients) == CoefficientStatus::Success,
                 PETSC_COMM_SELF, PETSC_ERR_PLIB,
                 "Could not evaluate reference coefficients at (%" PetscInt_FMT
                 ",%" PetscInt_FMT ")",
                 i, j);

      std::array<PetscScalar, flowComponentCount> center{};
      std::array<PetscScalar, flowComponentCount> dXi{};
      std::array<PetscScalar, flowComponentCount> dEta{};
      std::array<PetscScalar, flowComponentCount> dXiXi{};
      std::array<PetscScalar, flowComponentCount> dEtaEta{};
      std::array<PetscScalar, flowComponentCount> dXiEta{};
      std::array<PetscScalar, flowComponentCount> mass{};
      std::array<PetscScalar, flowComponentCount> spatial{};

      for (PetscInt component = 0; component < flowComponentCount;
           ++component) {
        center[static_cast<std::size_t>(component)] =
            inputValues[j][i][component];
        for (PetscInt xiSlot = 0; xiSlot < data.xiRule.stencilSize();
             ++xiSlot) {
          const PetscInt column = data.xiRule.stencilIndex(i, xiSlot);
          const PetscScalar value = inputValues[j][column][component];
          dXi[static_cast<std::size_t>(component)] +=
              data.xiRule.weight(1, i, xiSlot) * value;
          dXiXi[static_cast<std::size_t>(component)] +=
              data.xiRule.weight(2, i, xiSlot) * value;
        }
        for (PetscInt etaSlot = 0; etaSlot < data.etaRule.stencilSize();
             ++etaSlot) {
          const PetscInt row = data.etaRule.stencilIndex(j, etaSlot);
          const PetscScalar value = inputValues[row][i][component];
          dEta[static_cast<std::size_t>(component)] +=
              data.etaRule.weight(1, j, etaSlot) * value;
          dEtaEta[static_cast<std::size_t>(component)] +=
              data.etaRule.weight(2, j, etaSlot) * value;
          for (PetscInt xiSlot = 0; xiSlot < data.xiRule.stencilSize();
               ++xiSlot) {
            const PetscInt column = data.xiRule.stencilIndex(i, xiSlot);
            dXiEta[static_cast<std::size_t>(component)] +=
                data.etaRule.weight(1, j, etaSlot) *
                data.xiRule.weight(1, i, xiSlot) *
                inputValues[row][column][component];
          }
        }
      }

      applyBlock(coefficients.Gamma, center, mass);
      applyBlock(coefficients.K0, center, spatial);
      applyBlock(coefficients.Kxi, dXi, spatial);
      applyBlock(coefficients.Keta, dEta, spatial);
      applyBlock(coefficients.Vxixi, dXiXi, spatial, PetscScalar(-1.0));
      applyBlock(coefficients.Vetaeta, dEtaEta, spatial, PetscScalar(-1.0));
      applyBlock(coefficients.Vxieta, dXiEta, spatial, PetscScalar(-1.0));

      for (PetscInt component = 0; component < flowComponentCount;
           ++component) {
        massValues[j][i][component] = mass[static_cast<std::size_t>(component)];
        spatialValues[j][i][component] =
            spatial[static_cast<std::size_t>(component)];
      }
    }
  }

  PetscCall(
      DMDAVecRestoreArrayDOF(data.fieldDM, spatialResult, &spatialValues));
  PetscCall(DMDAVecRestoreArrayDOF(data.fieldDM, massResult, &massValues));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.metricDM, data.metrics, &metrics));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.baseflowDerivativeDM,
                                       data.baseflowDerivatives, &derivatives));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.fieldDM, data.baseflow, &baseflow));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.fieldDM, localInput, &inputValues));
  PetscCall(DMRestoreLocalVector(data.fieldDM, &localInput));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode compareAction(Mat matrix, Vec input, Vec expected,
                             const char *name, PetscReal tolerance) {
  Vec actual = nullptr;
  PetscReal expectedNorm = 0.0;
  PetscReal errorNorm = 0.0;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(expected, &actual));
  PetscCall(MatMult(matrix, input, actual));
  PetscCall(VecAXPY(actual, PetscScalar(-1.0), expected));
  PetscCall(VecNorm(expected, NORM_2, &expectedNorm));
  PetscCall(VecNorm(actual, NORM_2, &errorNorm));
  PetscCheck(errorNorm <= tolerance * PetscMax(1.0, expectedNorm),
             PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "%s MatMult mismatch: error %.16g, reference norm %.16g", name,
             static_cast<double>(errorNorm), static_cast<double>(expectedNorm));
  PetscCall(VecDestroy(&actual));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode validateStructure(const ProblemData &data,
                                 const GlobalOperatorDiagnostics &diagnostics) {
  PetscInt massBlockSize = 0;
  PetscInt spatialBlockSize = 0;
  PetscInt rows = 0;
  PetscInt columns = 0;
  PetscBool massIsBAIJ = PETSC_FALSE;
  PetscBool spatialIsBAIJ = PETSC_FALSE;
  const PetscInt nodeCount = data.ny * data.nz;
  const PetscInt expectedSpatialBlocks =
      nodeCount * data.xiRule.stencilSize() * data.etaRule.stencilSize();

  PetscFunctionBeginUser;
  PetscCall(MatGetBlockSize(data.massMatrix, &massBlockSize));
  PetscCall(MatGetBlockSize(data.spatialMatrix, &spatialBlockSize));
  PetscCall(
      PetscObjectTypeCompareAny(reinterpret_cast<PetscObject>(data.massMatrix),
                                &massIsBAIJ, MATSEQBAIJ, MATMPIBAIJ, ""));
  PetscCall(PetscObjectTypeCompareAny(
      reinterpret_cast<PetscObject>(data.spatialMatrix), &spatialIsBAIJ,
      MATSEQBAIJ, MATMPIBAIJ, ""));
  PetscCall(MatGetSize(data.spatialMatrix, &rows, &columns));
  PetscCheck(massIsBAIJ && spatialIsBAIJ, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Global matrices are not stored in BAIJ format");
  PetscCheck(massBlockSize == flowComponentCount &&
                 spatialBlockSize == flowComponentCount,
             PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Global matrices do not use 5x5 blocks");
  PetscCheck(rows == nodeCount * flowComponentCount && columns == rows,
             PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Unexpected global operator dimensions");
  PetscCheck(PetscAbsReal(static_cast<PetscReal>(diagnostics.massUsedBlocks) -
                          static_cast<PetscReal>(nodeCount)) < 0.5,
             PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Mass matrix does not contain exactly one block per node");
  PetscCheck(
      PetscAbsReal(static_cast<PetscReal>(diagnostics.spatialUsedBlocks) -
                   static_cast<PetscReal>(expectedSpatialBlocks)) < 0.5,
      PETSC_COMM_WORLD, PETSC_ERR_PLIB,
      "Spatial matrix block count does not match the tensor-product stencil");
  PetscCheck(diagnostics.massUsedBlocks == diagnostics.massAllocatedBlocks &&
                 diagnostics.spatialUsedBlocks ==
                     diagnostics.spatialAllocatedBlocks,
             PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Global matrix preallocation is not exact");
  PetscCheck(!PetscIsInfOrNanReal(diagnostics.massFrobeniusNorm) &&
                 !PetscIsInfOrNanReal(diagnostics.spatialFrobeniusNorm),
             PETSC_COMM_WORLD, PETSC_ERR_FP,
             "Global matrix norms are not finite");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode runTests() {
  std::string configPath;
  Recipe recipe;
  ProblemData data;
  MetricDiagnostics metricDiagnostics;
  BaseFlowDerivativeDiagnostics derivativeDiagnostics;
  GlobalOperatorDiagnostics operatorDiagnostics;
  Vec input = nullptr;
  Vec massReference = nullptr;
  Vec spatialReference = nullptr;

  PetscFunctionBeginUser;
  PetscCall(readOption("-c", configPath));
  PetscCall(Parser(PETSC_COMM_WORLD).parse(configPath, recipe));
  PetscCall(Prepare(PETSC_COMM_WORLD).initialize(recipe, data));
  PetscCall(Metrics(PETSC_COMM_WORLD).compute(data, metricDiagnostics));
  PetscCall(BaseFlowDerivatives(PETSC_COMM_WORLD)
                .compute(data, derivativeDiagnostics));
  PetscCall(GlobalOperator(PETSC_COMM_WORLD)
                .assemble(recipe, data, operatorDiagnostics));
  PetscCall(validateStructure(data, operatorDiagnostics));

  PetscCall(DMCreateGlobalVector(data.fieldDM, &input));
  PetscCall(VecDuplicate(input, &massReference));
  PetscCall(VecDuplicate(input, &spatialReference));
  PetscCall(fillDeterministicVector(data.fieldDM, input));
  PetscCall(computeReferenceAction(recipe, data, input, massReference,
                                   spatialReference));
  PetscCall(
      compareAction(data.massMatrix, input, massReference, "M_Gamma", 2.0e-13));
  PetscCall(
      compareAction(data.spatialMatrix, input, spatialReference, "L", 2.0e-11));

  PetscCall(VecDestroy(&spatialReference));
  PetscCall(VecDestroy(&massReference));
  PetscCall(VecDestroy(&input));
  PetscCall(
      PetscPrintf(PETSC_COMM_WORLD, "Global PETSc operator tests passed.\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

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
