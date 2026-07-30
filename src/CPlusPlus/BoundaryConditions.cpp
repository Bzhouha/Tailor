/**
 * @file BoundaryConditions.cpp
 * @brief Row replacement for isothermal no-slip and characteristic boundaries.
 */
#include "BoundaryConditions.hpp"

#include <slepcblaslapack.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "CurvilinearTransform.hpp"
#include "LNSCoefficients.hpp"
#include "Metrics.hpp"

namespace {

constexpr PetscInt matrixSize = flowComponentCount;
using Dense5 = std::array<PetscScalar, 25>;

/** @brief One replacement scalar row in global PETSc numbering. */
struct SparseRow {
  PetscInt row = 0;
  std::vector<PetscInt> columns;
  std::vector<PetscScalar> values;
};

/** @brief Pointwise normal eigensystem and characteristic classification. */
struct CharacteristicData {
  PetscInt globalBlock = 0;
  Dense5 rightEigenvectors{};
  Dense5 inverseRightEigenvectors{};
  Dense5 projection{};
  std::array<PetscScalar, 5> speeds{};
  std::array<PetscBool, 5> incoming{};
  std::array<PetscBool, 5> neutral{};
  PetscReal maxSpeedImaginary = 0.0;
  PetscReal condition = 0.0;
};

/** @brief Non-throwing status for pointwise characteristic construction. */
enum class CharacteristicStatus {
  Success = 0,
  InvalidInput,
  SingularMetric,
  LapackFailure,
  InfiniteEigenvalue,
  ComplexSpeed,
  SingularEigenvectors,
  IllConditionedEigenvectors,
};

/** @brief Mutable row-major dense-matrix element access. */
PetscScalar &entry(Dense5 &matrix, PetscInt row, PetscInt column) {
  return matrix[static_cast<std::size_t>(row * matrixSize + column)];
}

/** @brief Constant row-major dense-matrix element access. */
const PetscScalar &entry(const Dense5 &matrix, PetscInt row, PetscInt column) {
  return matrix[static_cast<std::size_t>(row * matrixSize + column)];
}

/** @brief Multiply two small row-major dense matrices. */
Dense5 multiply(const Dense5 &left, const Dense5 &right) {
  Dense5 result{};
  for (PetscInt row = 0; row < matrixSize; ++row)
    for (PetscInt column = 0; column < matrixSize; ++column)
      for (PetscInt inner = 0; inner < matrixSize; ++inner)
        entry(result, row, column) +=
            entry(left, row, inner) * entry(right, inner, column);
  return result;
}

/** @brief Compute a small dense matrix infinity norm. */
PetscReal infinityNorm(const Dense5 &matrix) {
  PetscReal result = 0.0;
  for (PetscInt row = 0; row < matrixSize; ++row) {
    PetscReal rowSum = 0.0;
    for (PetscInt column = 0; column < matrixSize; ++column)
      rowSum += PetscAbsScalar(entry(matrix, row, column));
    result = PetscMax(result, rowSum);
  }
  return result;
}

/** @brief Invert a 5-by-5 matrix by pivoted Gauss--Jordan elimination. */
bool invert(Dense5 matrix, Dense5 &inverse) {
  inverse = {};
  for (PetscInt diagonal = 0; diagonal < matrixSize; ++diagonal)
    entry(inverse, diagonal, diagonal) = 1.0;

  for (PetscInt pivotColumn = 0; pivotColumn < matrixSize; ++pivotColumn) {
    PetscInt pivotRow = pivotColumn;
    PetscReal pivotMagnitude =
        PetscAbsScalar(entry(matrix, pivotRow, pivotColumn));
    for (PetscInt row = pivotColumn + 1; row < matrixSize; ++row) {
      const PetscReal magnitude =
          PetscAbsScalar(entry(matrix, row, pivotColumn));
      if (magnitude > pivotMagnitude) {
        pivotMagnitude = magnitude;
        pivotRow = row;
      }
    }
    if (!std::isfinite(pivotMagnitude) ||
        pivotMagnitude <= 100.0 * PETSC_MACHINE_EPSILON)
      return false;

    if (pivotRow != pivotColumn) {
      for (PetscInt column = 0; column < matrixSize; ++column) {
        std::swap(entry(matrix, pivotRow, column),
                  entry(matrix, pivotColumn, column));
        std::swap(entry(inverse, pivotRow, column),
                  entry(inverse, pivotColumn, column));
      }
    }

    const PetscScalar pivot = entry(matrix, pivotColumn, pivotColumn);
    for (PetscInt column = 0; column < matrixSize; ++column) {
      entry(matrix, pivotColumn, column) /= pivot;
      entry(inverse, pivotColumn, column) /= pivot;
    }
    for (PetscInt row = 0; row < matrixSize; ++row) {
      if (row == pivotColumn)
        continue;
      const PetscScalar factor = entry(matrix, row, pivotColumn);
      for (PetscInt column = 0; column < matrixSize; ++column) {
        entry(matrix, row, column) -=
            factor * entry(matrix, pivotColumn, column);
        entry(inverse, row, column) -=
            factor * entry(inverse, pivotColumn, column);
      }
    }
  }
  return true;
}

/** @brief Copy a row-major Block5 into helper storage. */
Dense5 blockToDense(const Block5 &block) {
  Dense5 result{};
  for (PetscInt row = 0; row < matrixSize; ++row)
    for (PetscInt column = 0; column < matrixSize; ++column)
      entry(result, row, column) = block(row, column);
  return result;
}

/** @brief Build and classify the generalized normal inviscid eigensystem. */
CharacteristicStatus computeCharacteristics(
    const LNSCoefficients &builder, const PetscScalar *baseflow,
    const PetscScalar *derivatives, const PetscScalar *metricValues,
    CharacteristicData &characteristics) {
  BaseFlowPoint flow;
  MetricPoint metrics;
  InviscidLNSCoefficients inviscid;
  if (makeBaseFlowPoint(baseflow, derivatives, flow) !=
          CoefficientStatus::Success ||
      makeMetricPoint(metricValues, metrics) != CoefficientStatus::Success ||
      builder.evaluateInviscid(flow, inviscid) != CoefficientStatus::Success)
    return CharacteristicStatus::InvalidInput;

  const PetscReal normalMagnitude =
      PetscSqrtReal(metrics.xiY * metrics.xiY + metrics.xiZ * metrics.xiZ);
  if (!std::isfinite(normalMagnitude) ||
      normalMagnitude <= 100.0 * PETSC_MACHINE_EPSILON)
    return CharacteristicStatus::SingularMetric;
  const PetscReal normalY = metrics.xiY / normalMagnitude;
  const PetscReal normalZ = metrics.xiZ / normalMagnitude;

  std::array<PetscScalar, 25> normalColumnMajor{};
  std::array<PetscScalar, 25> gammaColumnMajor{};
  std::array<PetscScalar, 25> rightColumnMajor{};
  std::array<PetscScalar, 25> unusedLeft{};
  std::array<PetscScalar, 5> alpha{};
  std::array<PetscScalar, 5> beta{};
  for (PetscInt row = 0; row < matrixSize; ++row) {
    for (PetscInt column = 0; column < matrixSize; ++column) {
      normalColumnMajor[static_cast<std::size_t>(column * matrixSize + row)] =
          normalY * inviscid.Bc(row, column) +
          normalZ * inviscid.Cc(row, column);
      gammaColumnMajor[static_cast<std::size_t>(column * matrixSize + row)] =
          inviscid.Gamma(row, column);
    }
  }

  const char noLeft = 'N';
  const char computeRight = 'V';
  const PetscBLASInt n = matrixSize;
  const PetscBLASInt leadingDimension = matrixSize;
  const PetscBLASInt unusedLeadingDimension = 1;
  PetscBLASInt workspaceSize = -1;
  PetscBLASInt info = 0;
  PetscScalar workspaceQuery = 0.0;
  std::array<PetscReal, 8 * matrixSize> realWorkspace{};
  LAPACKggev_(&noLeft, &computeRight, &n, normalColumnMajor.data(),
              &leadingDimension, gammaColumnMajor.data(), &leadingDimension,
              alpha.data(), beta.data(), unusedLeft.data(),
              &unusedLeadingDimension, rightColumnMajor.data(),
              &leadingDimension, &workspaceQuery, &workspaceSize,
              realWorkspace.data(), &info);
  if (info != 0)
    return CharacteristicStatus::LapackFailure;
  workspaceSize = std::max<PetscBLASInt>(
      1, static_cast<PetscBLASInt>(PetscRealPart(workspaceQuery)));
  std::vector<PetscScalar> workspace(static_cast<std::size_t>(workspaceSize));

  for (PetscInt row = 0; row < matrixSize; ++row) {
    for (PetscInt column = 0; column < matrixSize; ++column) {
      normalColumnMajor[static_cast<std::size_t>(column * matrixSize + row)] =
          normalY * inviscid.Bc(row, column) +
          normalZ * inviscid.Cc(row, column);
      gammaColumnMajor[static_cast<std::size_t>(column * matrixSize + row)] =
          inviscid.Gamma(row, column);
    }
  }
  LAPACKggev_(&noLeft, &computeRight, &n, normalColumnMajor.data(),
              &leadingDimension, gammaColumnMajor.data(), &leadingDimension,
              alpha.data(), beta.data(), unusedLeft.data(),
              &unusedLeadingDimension, rightColumnMajor.data(),
              &leadingDimension, workspace.data(), &workspaceSize,
              realWorkspace.data(), &info);
  if (info != 0)
    return CharacteristicStatus::LapackFailure;

  PetscReal maximumSpeed = 1.0;
  for (PetscInt mode = 0; mode < matrixSize; ++mode) {
    if (PetscAbsScalar(beta[static_cast<std::size_t>(mode)]) <=
        100.0 * PETSC_MACHINE_EPSILON)
      return CharacteristicStatus::InfiniteEigenvalue;
    characteristics.speeds[static_cast<std::size_t>(mode)] =
        alpha[static_cast<std::size_t>(mode)] /
        beta[static_cast<std::size_t>(mode)];
    maximumSpeed = PetscMax(
        maximumSpeed,
        PetscAbsScalar(characteristics.speeds[static_cast<std::size_t>(mode)]));
    characteristics.maxSpeedImaginary =
        PetscMax(characteristics.maxSpeedImaginary,
                 PetscAbsReal(PetscImaginaryPart(
                     characteristics.speeds[static_cast<std::size_t>(mode)])));
    for (PetscInt row = 0; row < matrixSize; ++row)
      entry(characteristics.rightEigenvectors, row, mode) =
          rightColumnMajor[static_cast<std::size_t>(mode * matrixSize + row)];
  }

  const PetscReal imaginaryTolerance =
      1000.0 * PETSC_MACHINE_EPSILON * maximumSpeed;
  if (characteristics.maxSpeedImaginary > imaginaryTolerance)
    return CharacteristicStatus::ComplexSpeed;
  if (!invert(characteristics.rightEigenvectors,
              characteristics.inverseRightEigenvectors))
    return CharacteristicStatus::SingularEigenvectors;

  Dense5 inverseGamma{};
  const Dense5 gamma = blockToDense(inviscid.Gamma);
  if (!invert(gamma, inverseGamma))
    return CharacteristicStatus::SingularEigenvectors;
  characteristics.projection =
      multiply(characteristics.inverseRightEigenvectors, inverseGamma);
  characteristics.condition =
      infinityNorm(characteristics.rightEigenvectors) *
      infinityNorm(characteristics.inverseRightEigenvectors);
  if (!std::isfinite(characteristics.condition) ||
      characteristics.condition > 1.0e12)
    return CharacteristicStatus::IllConditionedEigenvectors;

  const PetscReal speedTolerance =
      1000.0 * PETSC_MACHINE_EPSILON * maximumSpeed;
  for (PetscInt mode = 0; mode < matrixSize; ++mode) {
    const PetscReal speed =
        PetscRealPart(characteristics.speeds[static_cast<std::size_t>(mode)]);
    characteristics.incoming[static_cast<std::size_t>(mode)] =
        speed < -speedTolerance ? PETSC_TRUE : PETSC_FALSE;
    characteristics.neutral[static_cast<std::size_t>(mode)] =
        PetscAbsReal(speed) <= speedTolerance ? PETSC_TRUE : PETSC_FALSE;
  }
  return CharacteristicStatus::Success;
}

/** @brief Read all five scalar rows belonging to one matrix block row. */
PetscErrorCode
readNodeRows(Mat matrix, PetscInt block,
             std::array<std::map<PetscInt, PetscScalar>, 5> &rows) {
  PetscFunctionBeginUser;
  for (PetscInt component = 0; component < matrixSize; ++component) {
    const PetscInt row = block * matrixSize + component;
    PetscInt count = 0;
    const PetscInt *columns = nullptr;
    const PetscScalar *values = nullptr;
    PetscCall(MatGetRow(matrix, row, &count, &columns, &values));
    for (PetscInt position = 0; position < count; ++position)
      rows[static_cast<std::size_t>(component)][columns[position]] =
          values[position];
    PetscCall(MatRestoreRow(matrix, row, &count, &columns, &values));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Left-transform five source rows into one replacement row. */
SparseRow
combineRows(PetscInt destinationRow, const Dense5 &transform,
            PetscInt transformRow,
            const std::array<std::map<PetscInt, PetscScalar>, 5> &sourceRows) {
  SparseRow result;
  result.row = destinationRow;
  std::map<PetscInt, PetscScalar> combined;
  for (PetscInt source = 0; source < matrixSize; ++source) {
    const PetscScalar scale = entry(transform, transformRow, source);
    if (scale == PetscScalar(0.0))
      continue;
    for (const auto &[column, value] :
         sourceRows[static_cast<std::size_t>(source)])
      combined[column] += scale * value;
  }
  for (const auto &[column, value] : combined) {
    result.columns.push_back(column);
    result.values.push_back(value);
  }
  return result;
}

/** @brief Build an incoming-amplitude constraint \f$R^{-1}q=0\f$. */
SparseRow characteristicConstraint(PetscInt destinationRow, PetscInt block,
                                   const Dense5 &inverseEigenvectors,
                                   PetscInt mode) {
  SparseRow result;
  result.row = destinationRow;
  for (PetscInt component = 0; component < matrixSize; ++component) {
    result.columns.push_back(block * matrixSize + component);
    result.values.push_back(entry(inverseEigenvectors, mode, component));
  }
  return result;
}

/** @brief Insert a collection of already zeroed replacement rows. */
PetscErrorCode insertRows(Mat matrix, const std::vector<SparseRow> &rows) {
  PetscFunctionBeginUser;
  for (const SparseRow &row : rows) {
    if (!row.columns.empty())
      PetscCall(MatSetValues(
          matrix, 1, &row.row, static_cast<PetscInt>(row.columns.size()),
          row.columns.data(), row.values.data(), INSERT_VALUES));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

BoundaryConditions::BoundaryConditions(MPI_Comm comm) : comm_(comm) {}

PetscErrorCode
BoundaryConditions::apply(const Recipe &recipe, ProblemData &data,
                          BoundaryConditionDiagnostics &diagnostics) const {
  const PetscScalar ***baseflow = nullptr;
  const PetscScalar ***derivatives = nullptr;
  const PetscScalar ***metrics = nullptr;
  ISLocalToGlobalMapping mapping = nullptr;
  const PetscInt *localToGlobalBlocks = nullptr;
  PetscInt xs = 0;
  PetscInt ys = 0;
  PetscInt xm = 0;
  PetscInt ym = 0;
  PetscInt gxs = 0;
  PetscInt gys = 0;
  PetscInt gxm = 0;
  PetscInt gym = 0;
  PetscInt localFailure = 0;
  PetscInt globalFailure = 0;
  PetscInt localWallNodes = 0;
  PetscInt localFarfieldNodes = 0;
  PetscInt localIncoming = 0;
  PetscInt localNeutral = 0;
  PetscInt localMinIncoming = matrixSize;
  PetscInt localMaxIncoming = 0;
  PetscReal localMaxImaginary = 0.0;
  PetscReal localMaxCondition = 0.0;
  std::vector<CharacteristicData> farfield;
  std::vector<PetscInt> wallBlocks;
  const LNSCoefficients coefficientBuilder(recipe);

  PetscFunctionBeginUser;
  PetscCheck(data.massMatrix && data.spatialMatrix, comm_,
             PETSC_ERR_ARG_WRONGSTATE,
             "Global operators must be assembled before boundary conditions");
  PetscCheck(!data.eigenMatrix && !data.eigenMassMatrix, comm_,
             PETSC_ERR_ARG_WRONGSTATE,
             "Boundary matrices have already been created");
  PetscCheck(data.ny >= 2, comm_, PETSC_ERR_ARG_SIZ,
             "At least two xi planes are required for wall/far-field "
             "boundaries");

  diagnostics = {};
  PetscCall(DMDAGetCorners(data.fieldDM, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAGetGhostCorners(data.fieldDM, &gxs, &gys, nullptr, &gxm, &gym,
                                nullptr));
  (void)gym;
  PetscCall(DMGetLocalToGlobalMapping(data.fieldDM, &mapping));
  PetscCall(
      ISLocalToGlobalMappingGetBlockIndices(mapping, &localToGlobalBlocks));
  PetscCall(DMDAVecGetArrayDOFRead(data.fieldDM, data.baseflow, &baseflow));
  PetscCall(DMDAVecGetArrayDOFRead(data.baseflowDerivativeDM,
                                   data.baseflowDerivatives, &derivatives));
  PetscCall(DMDAVecGetArrayDOFRead(data.metricDM, data.metrics, &metrics));

  const auto globalBlock = [&](PetscInt i, PetscInt j) {
    return localToGlobalBlocks[(j - gys) * gxm + (i - gxs)];
  };
  for (PetscInt j = ys; j < ys + ym; ++j) {
    if (xs == 0) {
      ++localWallNodes;
      wallBlocks.push_back(globalBlock(0, j));
    }
    if (!(xs <= data.ny - 1 && data.ny - 1 < xs + xm))
      continue;

    CharacteristicData node;
    node.globalBlock = globalBlock(data.ny - 1, j);
    const CharacteristicStatus status = computeCharacteristics(
        coefficientBuilder, baseflow[j][data.ny - 1],
        derivatives[j][data.ny - 1], metrics[j][data.ny - 1], node);
    if (status != CharacteristicStatus::Success) {
      localFailure = PetscMax(localFailure, static_cast<PetscInt>(status));
      continue;
    }
    PetscInt incomingAtNode = 0;
    for (PetscInt mode = 0; mode < matrixSize; ++mode) {
      incomingAtNode += node.incoming[static_cast<std::size_t>(mode)] ? 1 : 0;
      localNeutral += node.neutral[static_cast<std::size_t>(mode)] ? 1 : 0;
    }
    localIncoming += incomingAtNode;
    localMinIncoming = PetscMin(localMinIncoming, incomingAtNode);
    localMaxIncoming = PetscMax(localMaxIncoming, incomingAtNode);
    localMaxImaginary = PetscMax(localMaxImaginary, node.maxSpeedImaginary);
    localMaxCondition = PetscMax(localMaxCondition, node.condition);
    ++localFarfieldNodes;
    farfield.push_back(node);
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(data.metricDM, data.metrics, &metrics));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.baseflowDerivativeDM,
                                       data.baseflowDerivatives, &derivatives));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.fieldDM, data.baseflow, &baseflow));
  PetscCall(
      ISLocalToGlobalMappingRestoreBlockIndices(mapping, &localToGlobalBlocks));

  PetscCallMPI(MPI_Allreduce(&localFailure, &globalFailure, 1, MPIU_INT,
                             MPI_MAX, comm_));
  PetscCheck(
      globalFailure == 0, comm_, PETSC_ERR_FP,
      "Far-field characteristic decomposition failed (status %" PetscInt_FMT
      ")",
      globalFailure);

  PetscCall(
      MatDuplicate(data.spatialMatrix, MAT_COPY_VALUES, &data.eigenMatrix));
  PetscCall(MatScale(data.eigenMatrix, PetscScalar(-1.0)));
  PetscCall(
      MatDuplicate(data.massMatrix, MAT_COPY_VALUES, &data.eigenMassMatrix));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(data.eigenMatrix),
                               "A_bc"));
  PetscCall(PetscObjectSetName(
      reinterpret_cast<PetscObject>(data.eigenMassMatrix), "B_bc"));
  PetscCall(
      MatSetOption(data.eigenMatrix, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCall(
      MatSetOption(data.eigenMassMatrix, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));

  std::vector<PetscInt> rowsToReplace;
  std::vector<SparseRow> eigenRows;
  std::vector<SparseRow> massRows;
  for (const PetscInt block : wallBlocks) {
    for (PetscInt component = 1; component < matrixSize; ++component) {
      const PetscInt row = block * matrixSize + component;
      rowsToReplace.push_back(row);
      SparseRow replacement;
      replacement.row = row;
      replacement.columns.push_back(row);
      replacement.values.push_back(1.0);
      eigenRows.push_back(std::move(replacement));
    }
  }

  for (const CharacteristicData &node : farfield) {
    std::array<std::map<PetscInt, PetscScalar>, 5> oldEigenRows;
    std::array<std::map<PetscInt, PetscScalar>, 5> oldMassRows;
    PetscCall(readNodeRows(data.eigenMatrix, node.globalBlock, oldEigenRows));
    PetscCall(
        readNodeRows(data.eigenMassMatrix, node.globalBlock, oldMassRows));
    for (PetscInt mode = 0; mode < matrixSize; ++mode) {
      const PetscInt row = node.globalBlock * matrixSize + mode;
      rowsToReplace.push_back(row);
      if (node.incoming[static_cast<std::size_t>(mode)]) {
        eigenRows.push_back(characteristicConstraint(
            row, node.globalBlock, node.inverseRightEigenvectors, mode));
      } else {
        eigenRows.push_back(
            combineRows(row, node.projection, mode, oldEigenRows));
        massRows.push_back(
            combineRows(row, node.projection, mode, oldMassRows));
      }
    }
  }

  PetscCall(MatZeroRows(data.eigenMatrix,
                        static_cast<PetscInt>(rowsToReplace.size()),
                        rowsToReplace.data(), 0.0, nullptr, nullptr));
  PetscCall(MatZeroRows(data.eigenMassMatrix,
                        static_cast<PetscInt>(rowsToReplace.size()),
                        rowsToReplace.data(), 0.0, nullptr, nullptr));
  PetscCall(insertRows(data.eigenMatrix, eigenRows));
  PetscCall(insertRows(data.eigenMassMatrix, massRows));
  PetscCall(MatAssemblyBegin(data.eigenMatrix, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(data.eigenMassMatrix, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(data.eigenMatrix, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(data.eigenMassMatrix, MAT_FINAL_ASSEMBLY));

  PetscCallMPI(MPI_Allreduce(&localWallNodes, &diagnostics.wallNodes, 1,
                             MPIU_INT, MPI_SUM, comm_));
  PetscCallMPI(MPI_Allreduce(&localFarfieldNodes, &diagnostics.farfieldNodes, 1,
                             MPIU_INT, MPI_SUM, comm_));
  PetscCallMPI(MPI_Allreduce(&localIncoming, &diagnostics.incomingModes, 1,
                             MPIU_INT, MPI_SUM, comm_));
  PetscCallMPI(MPI_Allreduce(&localNeutral, &diagnostics.neutralModes, 1,
                             MPIU_INT, MPI_SUM, comm_));
  PetscCallMPI(MPI_Allreduce(&localMinIncoming,
                             &diagnostics.minIncomingModesPerNode, 1, MPIU_INT,
                             MPI_MIN, comm_));
  PetscCallMPI(MPI_Allreduce(&localMaxIncoming,
                             &diagnostics.maxIncomingModesPerNode, 1, MPIU_INT,
                             MPI_MAX, comm_));
  PetscCallMPI(MPI_Allreduce(&localMaxImaginary,
                             &diagnostics.maxCharacteristicImaginary, 1,
                             MPIU_REAL, MPI_MAX, comm_));
  PetscCallMPI(MPI_Allreduce(&localMaxCondition,
                             &diagnostics.maxEigenvectorCondition, 1, MPIU_REAL,
                             MPI_MAX, comm_));
  PetscCall(MatNorm(data.eigenMatrix, NORM_FROBENIUS,
                    &diagnostics.eigenFrobeniusNorm));
  PetscCall(MatNorm(data.eigenMassMatrix, NORM_FROBENIUS,
                    &diagnostics.eigenMassFrobeniusNorm));
  PetscCheck(diagnostics.wallNodes == data.nz &&
                 diagnostics.farfieldNodes == data.nz,
             comm_, PETSC_ERR_PLIB,
             "Boundary node counts do not match the periodic eta dimension");
  PetscFunctionReturn(PETSC_SUCCESS);
}
