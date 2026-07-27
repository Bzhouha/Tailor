#include <slepcsys.h>

#include <array>
#include <cmath>
#include <fstream>
#include <string>

#include "BaseFlowDerivatives.hpp"
#include "LNSCoefficients.hpp"
#include "Metrics.hpp"
#include "Parser.hpp"
#include "Prepare.hpp"
#include "ProblemData.hpp"
#include "Recipe.hpp"

namespace {

constexpr PetscInt matrixCount = physicalCoefficientCount;

std::array<const Block5 *, static_cast<std::size_t>(matrixCount)>
blocks(const PhysicalLNSCoefficients &coefficients) {
  return {&coefficients.Gamma, &coefficients.A,   &coefficients.B,
          &coefficients.C,     &coefficients.D,   &coefficients.Vxx,
          &coefficients.Vxy,   &coefficients.Vxz, &coefficients.Vyy,
          &coefficients.Vyz,   &coefficients.Vzz};
}

void setState(FlowState &state, std::array<PetscReal, 5> values) {
  state = values;
}

std::array<BaseFlowPoint, 2> referencePoints() {
  std::array<BaseFlowPoint, 2> points{};
  auto &first = points[0];
  setState(first.value, {1.2, 2.3, 0.14, -0.08, 1.4});
  setState(first.dx, {0.01, 0.02, -0.03, 0.04, 0.05});
  setState(first.dy, {-0.02, 0.07, 0.04, -0.05, 0.06});
  setState(first.dz, {0.03, -0.02, 0.08, 0.09, -0.04});
  setState(first.dxx, {0.004, -0.006, 0.007, -0.008, 0.009});
  setState(first.dyy, {-0.005, 0.011, -0.012, 0.013, -0.014});
  setState(first.dzz, {0.006, -0.015, 0.016, -0.017, 0.018});
  setState(first.dxy, {-0.007, 0.019, -0.020, 0.021, -0.022});
  setState(first.dxz, {0.008, -0.023, 0.024, -0.025, 0.026});
  setState(first.dyz, {-0.009, 0.027, -0.028, 0.029, -0.030});

  auto &second = points[1];
  setState(second.value, {0.95, 3.1, -0.12, 0.06, 1.15});
  setState(second.dy, {0.015, -0.045, 0.025, 0.035, -0.055});
  setState(second.dz, {-0.012, 0.038, -0.028, 0.042, 0.048});
  setState(second.dyy, {0.003, -0.009, 0.008, -0.007, 0.011});
  setState(second.dzz, {-0.004, 0.010, -0.012, 0.014, -0.013});
  setState(second.dyz, {0.005, -0.016, 0.017, -0.018, 0.019});
  return points;
}

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

PetscErrorCode compareFortranGold(const Recipe &recipe,
                                  const std::string &goldPath) {
  std::ifstream stream(goldPath);
  Recipe canonicalRecipe = recipe;
  canonicalRecipe.reynolds = 32500.0;
  canonicalRecipe.mach = 6.0;
  canonicalRecipe.prandtl = 0.72;
  canonicalRecipe.ratioOfSpecificHeats = 1.4;
  canonicalRecipe.referenceTemperature = 63.334;
  canonicalRecipe.sutherlandConstant = 110.4;
  LNSCoefficients builder(canonicalRecipe);
  const auto points = referencePoints();

  PetscFunctionBeginUser;
  PetscCheck(stream.good(), PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN,
             "Cannot open Fortran coefficient fixture: %s", goldPath.c_str());
  for (std::size_t point = 0; point < points.size(); ++point) {
    PhysicalLNSCoefficients coefficients;
    const CoefficientStatus status =
        builder.evaluate(points[point], coefficients);
    PetscCheck(status == CoefficientStatus::Success, PETSC_COMM_WORLD,
               PETSC_ERR_PLIB,
               "LNS coefficient evaluation failed for reference point %zu: %s",
               point, coefficientStatusName(status));
    const auto coefficientBlocks = blocks(coefficients);
    for (PetscInt matrix = 0; matrix < matrixCount; ++matrix) {
      for (PetscInt row = 0; row < flowComponentCount; ++row) {
        for (PetscInt column = 0; column < flowComponentCount; ++column) {
          PetscReal expected = 0.0;
          stream >> expected;
          PetscCheck(stream.good(), PETSC_COMM_WORLD, PETSC_ERR_FILE_UNEXPECTED,
                     "Fortran fixture ended at point %zu, matrix %" PetscInt_FMT
                     ", row %" PetscInt_FMT ", column %" PetscInt_FMT,
                     point, matrix, row, column);
          const PetscScalar actual =
              (*coefficientBlocks[static_cast<std::size_t>(matrix)])(row,
                                                                     column);
          const PetscReal tolerance =
              5.0e-13 * PetscMax(1.0, PetscAbsReal(expected));
          PetscCheck(
              PetscAbsReal(PetscRealPart(actual) - expected) <= tolerance,
              PETSC_COMM_WORLD, PETSC_ERR_PLIB,
              "Coefficient mismatch at point %zu, %s(%" PetscInt_FMT
              ",%" PetscInt_FMT "): expected %.17g, got %.17g",
              point, physicalCoefficientNames[static_cast<std::size_t>(matrix)],
              row, column, static_cast<double>(expected),
              static_cast<double>(PetscRealPart(actual)));
          PetscCheck(PetscImaginaryPart(actual) == 0.0, PETSC_COMM_WORLD,
                     PETSC_ERR_PLIB,
                     "Physical coefficient has a nonzero imaginary part");
        }
      }
    }
  }
  {
    PetscReal extra = 0.0;
    stream >> extra;
    PetscCheck(stream.eof(), PETSC_COMM_WORLD, PETSC_ERR_FILE_UNEXPECTED,
               "Fortran fixture contains extra coefficient values");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode validateDistributedField(const Recipe &recipe) {
  ProblemData data;
  MetricDiagnostics metricDiagnostics;
  BaseFlowDerivativeDiagnostics derivativeDiagnostics;
  const PetscScalar ***baseflow = nullptr;
  const PetscScalar ***derivatives = nullptr;
  PetscInt xs = 0;
  PetscInt ys = 0;
  PetscInt xm = 0;
  PetscInt ym = 0;
  PetscInt localFailure = 0;
  PetscInt globalFailure = 0;
  PetscInt localCount = 0;
  PetscInt globalCount = 0;
  LNSCoefficients builder(recipe);

  PetscFunctionBeginUser;
  PetscCall(Prepare(PETSC_COMM_WORLD).initialize(recipe, data));
  PetscCall(Metrics(PETSC_COMM_WORLD).compute(data, metricDiagnostics));
  PetscCall(BaseFlowDerivatives(PETSC_COMM_WORLD)
                .compute(data, derivativeDiagnostics));
  PetscCall(DMDAGetCorners(data.fieldDM, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAVecGetArrayDOFRead(data.fieldDM, data.baseflow, &baseflow));
  PetscCall(DMDAVecGetArrayDOFRead(data.baseflowDerivativeDM,
                                   data.baseflowDerivatives, &derivatives));

  for (PetscInt j = ys; j < ys + ym; ++j) {
    for (PetscInt i = xs; i < xs + xm; ++i) {
      BaseFlowPoint point;
      PhysicalLNSCoefficients coefficients;
      CoefficientStatus status =
          makeBaseFlowPoint(baseflow[j][i], derivatives[j][i], point);
      if (status == CoefficientStatus::Success)
        status = builder.evaluate(point, coefficients);
      if (status != CoefficientStatus::Success)
        localFailure = 1;
      ++localCount;
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(data.baseflowDerivativeDM,
                                       data.baseflowDerivatives, &derivatives));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.fieldDM, data.baseflow, &baseflow));
  PetscCallMPI(MPI_Allreduce(&localFailure, &globalFailure, 1, MPIU_INT,
                             MPI_MAX, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localCount, &globalCount, 1, MPIU_INT, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCheck(!globalFailure, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "At least one real-case node failed LNS coefficient evaluation");
  PetscCheck(globalCount == data.ny * data.nz, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Distributed coefficient validation visited %" PetscInt_FMT
             " nodes, expected %" PetscInt_FMT,
             globalCount, data.ny * data.nz);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode runTests() {
  std::string configPath;
  std::string goldPath;
  Recipe recipe;

  PetscFunctionBeginUser;
  PetscCall(readOption("-c", configPath));
  PetscCall(readOption("-gold", goldPath));
  PetscCall(Parser(PETSC_COMM_WORLD).parse(configPath, recipe));
  PetscCall(compareFortranGold(recipe, goldPath));
  PetscCall(validateDistributedField(recipe));
  PetscCall(
      PetscPrintf(PETSC_COMM_WORLD,
                  "LNS coefficient Fortran and distributed tests passed.\n"));
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
