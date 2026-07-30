/**
 * @file test_boundary_conditions.cpp
 * @brief Serial/MPI row-level tests for wall and characteristic boundaries.
 */
#include <slepcsys.h>

#include <array>
#include <cmath>
#include <map>
#include <string>

#include "BaseFlowDerivatives.hpp"
#include "BoundaryConditions.hpp"
#include "GlobalOperator.hpp"
#include "Metrics.hpp"
#include "Parser.hpp"
#include "Prepare.hpp"

namespace {

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

/** @brief Copy one locally owned sparse matrix row into an ordered map. */
PetscErrorCode readRow(Mat matrix, PetscInt row,
                       std::map<PetscInt, PetscScalar> &entries) {
  PetscInt count = 0;
  const PetscInt *columns = nullptr;
  const PetscScalar *values = nullptr;

  PetscFunctionBeginUser;
  entries.clear();
  PetscCall(MatGetRow(matrix, row, &count, &columns, &values));
  for (PetscInt position = 0; position < count; ++position)
    entries[columns[position]] = values[position];
  PetscCall(MatRestoreRow(matrix, row, &count, &columns, &values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Compute the Euclidean norm of sparse row values. */
PetscReal rowNorm(const std::map<PetscInt, PetscScalar> &row) {
  PetscReal norm = 0.0;
  for (const auto &[column, value] : row) {
    (void)column;
    norm += PetscAbsScalar(value) * PetscAbsScalar(value);
  }
  return PetscSqrtReal(norm);
}

/** @brief Compare two locally owned rows after applying a scale. */
PetscErrorCode compareRows(Mat first, Mat second, PetscInt row,
                           PetscScalar secondScale, PetscReal tolerance,
                           const char *description) {
  std::map<PetscInt, PetscScalar> firstEntries;
  std::map<PetscInt, PetscScalar> secondEntries;

  PetscFunctionBeginUser;
  PetscCall(readRow(first, row, firstEntries));
  PetscCall(readRow(second, row, secondEntries));
  for (const auto &[column, value] : secondEntries)
    firstEntries[column] -= secondScale * value;
  PetscCheck(rowNorm(firstEntries) <= tolerance, PETSC_COMM_SELF,
             PETSC_ERR_PLIB, "%s differs at global row %" PetscInt_FMT,
             description, row);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Validate wall, interior, and far-field scalar rows. */
PetscErrorCode
validateBoundaries(const ProblemData &data,
                   const BoundaryConditionDiagnostics &diagnostics) {
  ISLocalToGlobalMapping mapping = nullptr;
  const PetscInt *blocks = nullptr;
  PetscInt xs = 0;
  PetscInt ys = 0;
  PetscInt xm = 0;
  PetscInt ym = 0;
  PetscInt gxs = 0;
  PetscInt gys = 0;
  PetscInt gxm = 0;
  PetscInt gym = 0;
  PetscInt localIncomingRows = 0;
  PetscInt globalIncomingRows = 0;
  PetscInt localFarfieldRows = 0;
  PetscInt globalFarfieldRows = 0;
  constexpr PetscReal tolerance = 2.0e-12;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(data.fieldDM, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAGetGhostCorners(data.fieldDM, &gxs, &gys, nullptr, &gxm, &gym,
                                nullptr));
  (void)gym;
  PetscCall(DMGetLocalToGlobalMapping(data.fieldDM, &mapping));
  PetscCall(ISLocalToGlobalMappingGetBlockIndices(mapping, &blocks));
  const auto globalBlock = [&](PetscInt i, PetscInt j) {
    return blocks[(j - gys) * gxm + (i - gxs)];
  };

  if (xs == 0) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      const PetscInt block = globalBlock(0, j);
      const PetscInt densityRow = block * flowComponentCount;
      PetscCall(compareRows(data.eigenMatrix, data.spatialMatrix, densityRow,
                            -1.0, tolerance, "Wall continuity eigen row"));
      PetscCall(compareRows(data.eigenMassMatrix, data.massMatrix, densityRow,
                            1.0, tolerance, "Wall continuity mass row"));

      for (PetscInt component = 1; component < flowComponentCount;
           ++component) {
        const PetscInt row = block * flowComponentCount + component;
        std::map<PetscInt, PetscScalar> eigenEntries;
        std::map<PetscInt, PetscScalar> massEntries;
        PetscCall(readRow(data.eigenMatrix, row, eigenEntries));
        PetscCall(readRow(data.eigenMassMatrix, row, massEntries));
        eigenEntries[row] -= 1.0;
        PetscCheck(rowNorm(eigenEntries) <= tolerance &&
                       rowNorm(massEntries) <= tolerance,
                   PETSC_COMM_SELF, PETSC_ERR_PLIB,
                   "Wall Dirichlet row %" PetscInt_FMT " is incorrect", row);
      }
    }
  }

  if (xs <= data.ny - 1 && data.ny - 1 < xs + xm) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      const PetscInt block = globalBlock(data.ny - 1, j);
      for (PetscInt component = 0; component < flowComponentCount;
           ++component) {
        const PetscInt row = block * flowComponentCount + component;
        std::map<PetscInt, PetscScalar> eigenEntries;
        std::map<PetscInt, PetscScalar> massEntries;
        PetscCall(readRow(data.eigenMatrix, row, eigenEntries));
        PetscCall(readRow(data.eigenMassMatrix, row, massEntries));
        PetscCheck(rowNorm(eigenEntries) > tolerance, PETSC_COMM_SELF,
                   PETSC_ERR_PLIB,
                   "Far-field eigen row %" PetscInt_FMT " is empty", row);
        if (rowNorm(massEntries) <= tolerance) {
          ++localIncomingRows;
          for (const auto &[column, value] : eigenEntries) {
            if (PetscAbsScalar(value) > tolerance)
              PetscCheck(column / flowComponentCount == block, PETSC_COMM_SELF,
                         PETSC_ERR_PLIB,
                         "Incoming characteristic row reaches another node");
          }
        }
        ++localFarfieldRows;
      }
    }
  }

  if (xs <= 1 && 1 < xs + xm && data.ny > 2 && ym > 0) {
    const PetscInt row = globalBlock(1, ys) * flowComponentCount;
    PetscCall(compareRows(data.eigenMatrix, data.spatialMatrix, row, -1.0,
                          tolerance, "Interior eigen row"));
    PetscCall(compareRows(data.eigenMassMatrix, data.massMatrix, row, 1.0,
                          tolerance, "Interior mass row"));
  }

  PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(mapping, &blocks));
  PetscCallMPI(MPI_Allreduce(&localIncomingRows, &globalIncomingRows, 1,
                             MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localFarfieldRows, &globalFarfieldRows, 1,
                             MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCheck(globalIncomingRows == diagnostics.incomingModes, PETSC_COMM_WORLD,
             PETSC_ERR_PLIB,
             "Zero mass rows do not match the incoming-mode diagnostic");
  PetscCheck(globalFarfieldRows ==
                 diagnostics.farfieldNodes * flowComponentCount,
             PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Far-field scalar row count is incorrect");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Check that both constrained matrices have a finite MatMult action. */
PetscErrorCode validateMatMult(const ProblemData &data) {
  Vec input = nullptr;
  Vec output = nullptr;
  PetscScalar ***values = nullptr;
  PetscInt xs = 0;
  PetscInt ys = 0;
  PetscInt xm = 0;
  PetscInt ym = 0;
  PetscReal norm = 0.0;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(data.fieldDM, &input));
  PetscCall(VecDuplicate(input, &output));
  PetscCall(DMDAGetCorners(data.fieldDM, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAVecGetArrayDOF(data.fieldDM, input, &values));
  for (PetscInt j = ys; j < ys + ym; ++j)
    for (PetscInt i = xs; i < xs + xm; ++i)
      for (PetscInt component = 0; component < flowComponentCount; ++component)
        values[j][i][component] =
            PetscCMPLX(0.01 * (i + 1) + 0.03 * (j + 1), 0.07 * (component + 1));
  PetscCall(DMDAVecRestoreArrayDOF(data.fieldDM, input, &values));
  PetscCall(MatMult(data.eigenMatrix, input, output));
  PetscCall(VecNorm(output, NORM_2, &norm));
  PetscCheck(!PetscIsInfOrNanReal(norm) && norm > 0.0, PETSC_COMM_WORLD,
             PETSC_ERR_FP, "A_bc MatMult produced an invalid norm");
  PetscCall(MatMult(data.eigenMassMatrix, input, output));
  PetscCall(VecNorm(output, NORM_2, &norm));
  PetscCheck(!PetscIsInfOrNanReal(norm) && norm > 0.0, PETSC_COMM_WORLD,
             PETSC_ERR_FP, "B_bc MatMult produced an invalid norm");
  PetscCall(VecDestroy(&output));
  PetscCall(VecDestroy(&input));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Assemble a case, apply boundaries, and run all checks. */
PetscErrorCode runTests() {
  std::string configPath;
  Recipe recipe;
  ProblemData data;
  MetricDiagnostics metricDiagnostics;
  BaseFlowDerivativeDiagnostics derivativeDiagnostics;
  GlobalOperatorDiagnostics operatorDiagnostics;
  BoundaryConditionDiagnostics boundaryDiagnostics;
  PetscReal rawMassNormBefore = 0.0;
  PetscReal rawSpatialNormBefore = 0.0;
  PetscReal rawMassNormAfter = 0.0;
  PetscReal rawSpatialNormAfter = 0.0;
  PetscInt expectedIncomingPerNode = -1;
  PetscInt expectedNeutralPerNode = -1;
  PetscBool incomingProvided = PETSC_FALSE;
  PetscBool neutralProvided = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-expected_incoming_per_node",
                               &expectedIncomingPerNode, &incomingProvided));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-expected_neutral_per_node",
                               &expectedNeutralPerNode, &neutralProvided));
  PetscCall(readOption("-c", configPath));
  PetscCall(Parser(PETSC_COMM_WORLD).parse(configPath, recipe));
  PetscCall(Prepare(PETSC_COMM_WORLD).initialize(recipe, data));
  PetscCall(Metrics(PETSC_COMM_WORLD).compute(data, metricDiagnostics));
  PetscCall(BaseFlowDerivatives(PETSC_COMM_WORLD)
                .compute(data, derivativeDiagnostics));
  PetscCall(GlobalOperator(PETSC_COMM_WORLD)
                .assemble(recipe, data, operatorDiagnostics));
  PetscCall(MatNorm(data.massMatrix, NORM_FROBENIUS, &rawMassNormBefore));
  PetscCall(MatNorm(data.spatialMatrix, NORM_FROBENIUS, &rawSpatialNormBefore));
  PetscCall(BoundaryConditions(PETSC_COMM_WORLD)
                .apply(recipe, data, boundaryDiagnostics));
  PetscCall(MatNorm(data.massMatrix, NORM_FROBENIUS, &rawMassNormAfter));
  PetscCall(MatNorm(data.spatialMatrix, NORM_FROBENIUS, &rawSpatialNormAfter));
  PetscCheck(rawMassNormBefore == rawMassNormAfter &&
                 rawSpatialNormBefore == rawSpatialNormAfter,
             PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Boundary conditions modified the unconstrained operators");
  PetscCheck(boundaryDiagnostics.wallNodes == data.nz &&
                 boundaryDiagnostics.farfieldNodes == data.nz,
             PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Boundary diagnostics have incorrect node counts");
  PetscCheck(boundaryDiagnostics.incomingModes > 0 &&
                 boundaryDiagnostics.incomingModes <
                     data.nz * flowComponentCount,
             PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "The real case does not exercise incoming and outgoing modes");
  PetscCheck(
      !PetscIsInfOrNanReal(boundaryDiagnostics.maxEigenvectorCondition) &&
          boundaryDiagnostics.maxEigenvectorCondition < 1.0e12,
      PETSC_COMM_WORLD, PETSC_ERR_FP,
      "Far-field characteristic eigenvectors are ill-conditioned");
  if (incomingProvided)
    PetscCheck(boundaryDiagnostics.minIncomingModesPerNode ==
                       expectedIncomingPerNode &&
                   boundaryDiagnostics.maxIncomingModesPerNode ==
                       expectedIncomingPerNode,
               PETSC_COMM_WORLD, PETSC_ERR_PLIB,
               "Expected %" PetscInt_FMT " incoming modes per node, found "
               "%" PetscInt_FMT "..%" PetscInt_FMT,
               expectedIncomingPerNode,
               boundaryDiagnostics.minIncomingModesPerNode,
               boundaryDiagnostics.maxIncomingModesPerNode);
  if (neutralProvided)
    PetscCheck(boundaryDiagnostics.neutralModes ==
                   expectedNeutralPerNode * data.nz,
               PETSC_COMM_WORLD, PETSC_ERR_PLIB,
               "Expected %" PetscInt_FMT
               " neutral modes per node, found %" PetscInt_FMT " total",
               expectedNeutralPerNode, boundaryDiagnostics.neutralModes);
  PetscCall(validateBoundaries(data, boundaryDiagnostics));
  PetscCall(validateMatMult(data));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Boundary-condition tests passed: %" PetscInt_FMT
                        " incoming modes, %" PetscInt_FMT " neutral modes.\n",
                        boundaryDiagnostics.incomingModes,
                        boundaryDiagnostics.neutralModes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

/** @brief Initialize SLEPc, run boundary tests, and finalize collectively. */
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
