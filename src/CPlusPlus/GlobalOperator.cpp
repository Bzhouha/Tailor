/**
 * @file GlobalOperator.cpp
 * @brief Compact MPIBAIJ preallocation and unconstrained operator assembly.
 *
 * The spatial sparsity is the Cartesian product of the one-dimensional FD-q
 * stencils. Diagonal/off-diagonal block counts are computed in PETSc ordering
 * before any values are inserted.
 */
#include "GlobalOperator.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include "CurvilinearTransform.hpp"
#include "LNSCoefficients.hpp"
#include "StreamwiseFourier.hpp"

namespace {

constexpr PetscInt blockSize = flowComponentCount;

/** @brief Owned/ghost DMDA geometry and block-index mapping view. */
struct GridOwnership {
  PetscInt xs = 0;
  PetscInt ys = 0;
  PetscInt xm = 0;
  PetscInt ym = 0;
  PetscInt gxs = 0;
  PetscInt gys = 0;
  PetscInt gxm = 0;
  PetscInt gym = 0;
  PetscInt firstOwnedBlock = 0;
  PetscInt lastOwnedBlock = 0;
  const PetscInt *localToGlobalBlocks = nullptr;
};

/** @brief Convert one ghosted grid coordinate to a local mapping slot. */
PetscInt ghostSlot(const GridOwnership &ownership, PetscInt i, PetscInt j) {
  return (j - ownership.gys) * ownership.gxm + (i - ownership.gxs);
}

/** @brief Map one ghosted grid coordinate to a PETSc global block. */
PetscInt globalBlock(const GridOwnership &ownership, PetscInt i, PetscInt j) {
  return ownership.localToGlobalBlocks[ghostSlot(ownership, i, j)];
}

/** @brief Accumulate a scaled dense block. */
void addScaled(Block5 &destination, const Block5 &source, PetscScalar scale) {
  for (PetscInt row = 0; row < blockSize; ++row)
    for (PetscInt column = 0; column < blockSize; ++column)
      destination(row, column) += scale * source(row, column);
}

/** @brief Form a two-term dense block linear combination. */
Block5 linearCombination(const Block5 &first, PetscScalar firstScale,
                         const Block5 &second, PetscScalar secondScale) {
  Block5 result;
  addScaled(result, first, firstScale);
  addScaled(result, second, secondScale);
  return result;
}

/** @brief Borrow field ownership and the local-to-global block map. */
PetscErrorCode getOwnership(const ProblemData &data,
                            ISLocalToGlobalMapping mapping,
                            GridOwnership &ownership) {
  PetscInt firstOwnedScalar = 0;
  PetscInt lastOwnedScalar = 0;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(data.fieldDM, &ownership.xs, &ownership.ys, nullptr,
                           &ownership.xm, &ownership.ym, nullptr));
  PetscCall(DMDAGetGhostCorners(data.fieldDM, &ownership.gxs, &ownership.gys,
                                nullptr, &ownership.gxm, &ownership.gym,
                                nullptr));
  PetscCall(
      VecGetOwnershipRange(data.baseflow, &firstOwnedScalar, &lastOwnedScalar));
  PetscCheck(
      firstOwnedScalar % blockSize == 0 && lastOwnedScalar % blockSize == 0,
      PetscObjectComm(reinterpret_cast<PetscObject>(data.fieldDM)),
      PETSC_ERR_PLIB, "fieldDM ownership ranges are not aligned to 5x5 blocks");
  ownership.firstOwnedBlock = firstOwnedScalar / blockSize;
  ownership.lastOwnedBlock = lastOwnedScalar / blockSize;
  PetscCall(ISLocalToGlobalMappingGetBlockIndices(
      mapping, &ownership.localToGlobalBlocks));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Restore the borrowed local-to-global block map. */
PetscErrorCode restoreOwnership(ISLocalToGlobalMapping mapping,
                                GridOwnership &ownership) {
  PetscFunctionBeginUser;
  PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(
      mapping, &ownership.localToGlobalBlocks));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Attach DMDA stencil metadata and strict insertion options. */
PetscErrorCode configureStencilMatrix(DM dm, Mat matrix, const char *name) {
  ISLocalToGlobalMapping mapping = nullptr;
  PetscInt gxs = 0;
  PetscInt gys = 0;
  PetscInt gxm = 0;
  PetscInt gym = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(matrix), name));
  PetscCall(DMGetLocalToGlobalMapping(dm, &mapping));
  PetscCall(MatSetLocalToGlobalMapping(matrix, mapping, mapping));
  PetscCall(DMDAGetGhostCorners(dm, &gxs, &gys, nullptr, &gxm, &gym, nullptr));
  const std::array<PetscInt, 2> dimensions = {gxm, gym};
  const std::array<PetscInt, 2> starts = {gxs, gys};
  PetscCall(
      MatSetStencil(matrix, 2, dimensions.data(), starts.data(), blockSize));
  PetscCall(MatSetDM(matrix, dm));
  PetscCall(MatSetOption(matrix, MAT_ROW_ORIENTED, PETSC_TRUE));
  PetscCall(MatSetOption(matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Evaluate the complete node coefficient transformation pipeline. */
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

/** @brief Insert one row-major block by DMDA stencil coordinates. */
PetscErrorCode insertBlock(Mat matrix, const MatStencil &row,
                           const MatStencil &column, const Block5 &block,
                           InsertMode mode) {
  PetscFunctionBeginUser;
  PetscCall(MatSetValuesBlockedStencil(matrix, 1, &row, 1, &column,
                                       block.data(), mode));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

GlobalOperator::GlobalOperator(MPI_Comm comm) : comm_(comm) {}

PetscErrorCode
GlobalOperator::validateCoefficients(const Recipe &recipe,
                                     const ProblemData &data) const {
  const PetscScalar ***baseflow = nullptr;
  const PetscScalar ***derivatives = nullptr;
  const PetscScalar ***metrics = nullptr;
  PetscInt xs = 0;
  PetscInt ys = 0;
  PetscInt xm = 0;
  PetscInt ym = 0;
  PetscInt localFailure = 0;
  PetscInt globalFailure = 0;
  const PetscScalar alpha =
      PetscCMPLX(static_cast<PetscReal>(recipe.alpha.real()),
                 static_cast<PetscReal>(recipe.alpha.imag()));
  const LNSCoefficients physicalBuilder(recipe);
  const StreamwiseFourier fourierTransform(alpha);
  const CurvilinearTransform curvilinearTransform;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(data.fieldDM, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAVecGetArrayDOFRead(data.fieldDM, data.baseflow, &baseflow));
  PetscCall(DMDAVecGetArrayDOFRead(data.baseflowDerivativeDM,
                                   data.baseflowDerivatives, &derivatives));
  PetscCall(DMDAVecGetArrayDOFRead(data.metricDM, data.metrics, &metrics));

  for (PetscInt j = ys; j < ys + ym; ++j) {
    for (PetscInt i = xs; i < xs + xm; ++i) {
      CurvilinearLNSCoefficients curvilinear;
      if (evaluateNode(physicalBuilder, fourierTransform, curvilinearTransform,
                       baseflow[j][i], derivatives[j][i], metrics[j][i],
                       curvilinear) != CoefficientStatus::Success)
        localFailure = 1;
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(data.metricDM, data.metrics, &metrics));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.baseflowDerivativeDM,
                                       data.baseflowDerivatives, &derivatives));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.fieldDM, data.baseflow, &baseflow));
  PetscCallMPI(MPI_Allreduce(&localFailure, &globalFailure, 1, MPIU_INT,
                             MPI_MAX, comm_));
  PetscCheck(!globalFailure, comm_, PETSC_ERR_FP,
             "At least one node produced invalid curvilinear LNS "
             "coefficients during global operator validation");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GlobalOperator::createMatrices(ProblemData &data) const {
  ISLocalToGlobalMapping mapping = nullptr;
  GridOwnership ownership;
  PetscInt localRows = 0;
  PetscInt globalRows = 0;
  std::vector<PetscInt> massDiagonal;
  std::vector<PetscInt> massOffDiagonal;
  std::vector<PetscInt> spatialDiagonal;
  std::vector<PetscInt> spatialOffDiagonal;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(data.baseflow, &localRows));
  PetscCall(VecGetSize(data.baseflow, &globalRows));
  PetscCheck(localRows % blockSize == 0 && globalRows % blockSize == 0, comm_,
             PETSC_ERR_PLIB, "fieldDM vector sizes are not divisible by 5");

  const PetscInt localBlockRows = localRows / blockSize;
  massDiagonal.assign(static_cast<std::size_t>(localBlockRows), 1);
  massOffDiagonal.assign(static_cast<std::size_t>(localBlockRows), 0);
  spatialDiagonal.assign(static_cast<std::size_t>(localBlockRows), 0);
  spatialOffDiagonal.assign(static_cast<std::size_t>(localBlockRows), 0);

  PetscCall(DMGetLocalToGlobalMapping(data.fieldDM, &mapping));
  PetscCall(getOwnership(data, mapping, ownership));
  for (PetscInt j = ownership.ys; j < ownership.ys + ownership.ym; ++j) {
    for (PetscInt i = ownership.xs; i < ownership.xs + ownership.xm; ++i) {
      const PetscInt rowBlock = globalBlock(ownership, i, j);
      const PetscInt localRow = rowBlock - ownership.firstOwnedBlock;
      PetscCheck(localRow >= 0 && localRow < localBlockRows, comm_,
                 PETSC_ERR_PLIB,
                 "DMDA owned node maps outside the matrix ownership range");

      std::vector<PetscInt> columns;
      columns.reserve(static_cast<std::size_t>(data.xiRule.stencilSize() *
                                               data.etaRule.stencilSize()));
      for (PetscInt etaSlot = 0; etaSlot < data.etaRule.stencilSize();
           ++etaSlot) {
        const PetscInt columnJ = data.etaRule.localIndex(j, etaSlot);
        for (PetscInt xiSlot = 0; xiSlot < data.xiRule.stencilSize();
             ++xiSlot) {
          const PetscInt columnI = data.xiRule.localIndex(i, xiSlot);
          PetscCheck(
              columnI >= ownership.gxs &&
                  columnI < ownership.gxs + ownership.gxm &&
                  columnJ >= ownership.gys &&
                  columnJ < ownership.gys + ownership.gym,
              comm_, PETSC_ERR_PLIB,
              "FD-q stencil extends outside the DMDA ghost region during "
              "matrix preallocation");
          columns.push_back(globalBlock(ownership, columnI, columnJ));
        }
      }
      std::sort(columns.begin(), columns.end());
      columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
      for (const PetscInt columnBlock : columns) {
        if (columnBlock >= ownership.firstOwnedBlock &&
            columnBlock < ownership.lastOwnedBlock)
          ++spatialDiagonal[static_cast<std::size_t>(localRow)];
        else
          ++spatialOffDiagonal[static_cast<std::size_t>(localRow)];
      }
    }
  }
  PetscCall(restoreOwnership(mapping, ownership));

  PetscCall(MatCreateBAIJ(comm_, blockSize, localRows, localRows, globalRows,
                          globalRows, 0, massDiagonal.data(), 0,
                          massOffDiagonal.data(), &data.massMatrix));
  PetscCall(configureStencilMatrix(data.fieldDM, data.massMatrix, "M_Gamma"));
  PetscCall(MatCreateBAIJ(comm_, blockSize, localRows, localRows, globalRows,
                          globalRows, 0, spatialDiagonal.data(), 0,
                          spatialOffDiagonal.data(), &data.spatialMatrix));
  PetscCall(configureStencilMatrix(data.fieldDM, data.spatialMatrix, "L"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GlobalOperator::insertValues(const Recipe &recipe,
                                            ProblemData &data) const {
  const PetscScalar ***baseflow = nullptr;
  const PetscScalar ***derivatives = nullptr;
  const PetscScalar ***metrics = nullptr;
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
  PetscCall(DMDAGetCorners(data.fieldDM, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAVecGetArrayDOFRead(data.fieldDM, data.baseflow, &baseflow));
  PetscCall(DMDAVecGetArrayDOFRead(data.baseflowDerivativeDM,
                                   data.baseflowDerivatives, &derivatives));
  PetscCall(DMDAVecGetArrayDOFRead(data.metricDM, data.metrics, &metrics));

  for (PetscInt j = ys; j < ys + ym; ++j) {
    for (PetscInt i = xs; i < xs + xm; ++i) {
      CurvilinearLNSCoefficients coefficients;
      const CoefficientStatus status = evaluateNode(
          physicalBuilder, fourierTransform, curvilinearTransform,
          baseflow[j][i], derivatives[j][i], metrics[j][i], coefficients);
      PetscCheck(status == CoefficientStatus::Success, PETSC_COMM_SELF,
                 PETSC_ERR_PLIB,
                 "Validated node (%" PetscInt_FMT ",%" PetscInt_FMT
                 ") failed during matrix insertion",
                 i, j);

      MatStencil row{};
      row.i = i;
      row.j = j;
      PetscCall(insertBlock(data.massMatrix, row, row, coefficients.Gamma,
                            INSERT_VALUES));
      PetscCall(insertBlock(data.spatialMatrix, row, row, coefficients.K0,
                            ADD_VALUES));

      for (PetscInt xiSlot = 0; xiSlot < data.xiRule.stencilSize(); ++xiSlot) {
        MatStencil column{};
        column.i = data.xiRule.localIndex(i, xiSlot);
        column.j = j;
        const Block5 block = linearCombination(
            coefficients.Kxi, data.xiRule.weight(1, i, xiSlot),
            coefficients.Vxixi, -data.xiRule.weight(2, i, xiSlot));
        PetscCall(
            insertBlock(data.spatialMatrix, row, column, block, ADD_VALUES));
      }

      for (PetscInt etaSlot = 0; etaSlot < data.etaRule.stencilSize();
           ++etaSlot) {
        MatStencil column{};
        column.i = i;
        column.j = data.etaRule.localIndex(j, etaSlot);
        const Block5 block = linearCombination(
            coefficients.Keta, data.etaRule.weight(1, j, etaSlot),
            coefficients.Vetaeta, -data.etaRule.weight(2, j, etaSlot));
        PetscCall(
            insertBlock(data.spatialMatrix, row, column, block, ADD_VALUES));
      }

      for (PetscInt etaSlot = 0; etaSlot < data.etaRule.stencilSize();
           ++etaSlot) {
        const PetscScalar etaWeight = data.etaRule.weight(1, j, etaSlot);
        for (PetscInt xiSlot = 0; xiSlot < data.xiRule.stencilSize();
             ++xiSlot) {
          MatStencil column{};
          column.i = data.xiRule.localIndex(i, xiSlot);
          column.j = data.etaRule.localIndex(j, etaSlot);
          Block5 block;
          addScaled(block, coefficients.Vxieta,
                    -etaWeight * data.xiRule.weight(1, i, xiSlot));
          PetscCall(
              insertBlock(data.spatialMatrix, row, column, block, ADD_VALUES));
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(data.metricDM, data.metrics, &metrics));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.baseflowDerivativeDM,
                                       data.baseflowDerivatives, &derivatives));
  PetscCall(DMDAVecRestoreArrayDOFRead(data.fieldDM, data.baseflow, &baseflow));
  PetscCall(MatAssemblyBegin(data.massMatrix, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(data.spatialMatrix, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(data.massMatrix, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(data.spatialMatrix, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GlobalOperator::computeDiagnostics(
    const ProblemData &data, GlobalOperatorDiagnostics &diagnostics) const {
  MatInfo massInfo{};
  MatInfo spatialInfo{};

  PetscFunctionBeginUser;
  diagnostics = {};
  PetscCall(
      MatGetSize(data.massMatrix, &diagnostics.rows, &diagnostics.columns));
  PetscCall(MatGetBlockSize(data.massMatrix, &diagnostics.blockSize));
  PetscCall(MatGetInfo(data.massMatrix, MAT_GLOBAL_SUM, &massInfo));
  PetscCall(MatGetInfo(data.spatialMatrix, MAT_GLOBAL_SUM, &spatialInfo));
  const PetscLogDouble entriesPerBlock =
      static_cast<PetscLogDouble>(blockSize * blockSize);
  diagnostics.massUsedBlocks = massInfo.nz_used / entriesPerBlock;
  diagnostics.massAllocatedBlocks = massInfo.nz_allocated / entriesPerBlock;
  diagnostics.spatialUsedBlocks = spatialInfo.nz_used / entriesPerBlock;
  diagnostics.spatialAllocatedBlocks =
      spatialInfo.nz_allocated / entriesPerBlock;
  PetscCall(
      MatNorm(data.massMatrix, NORM_FROBENIUS, &diagnostics.massFrobeniusNorm));
  PetscCall(MatNorm(data.spatialMatrix, NORM_FROBENIUS,
                    &diagnostics.spatialFrobeniusNorm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode
GlobalOperator::assemble(const Recipe &recipe, ProblemData &data,
                         GlobalOperatorDiagnostics &diagnostics) const {
  PetscFunctionBeginUser;
  PetscCheck(data.fieldDM && data.baseflow, comm_, PETSC_ERR_ARG_WRONGSTATE,
             "Prepared field data are required before matrix assembly");
  PetscCheck(data.metricDM && data.metrics, comm_, PETSC_ERR_ARG_WRONGSTATE,
             "Metrics are required before matrix assembly");
  PetscCheck(data.baseflowDerivativeDM && data.baseflowDerivatives, comm_,
             PETSC_ERR_ARG_WRONGSTATE,
             "Base-flow derivatives are required before matrix assembly");
  PetscCheck(!data.massMatrix && !data.spatialMatrix, comm_,
             PETSC_ERR_ARG_WRONGSTATE,
             "Global matrices have already been created");
  PetscCheck(data.ny == data.xiRule.nodeCount() &&
                 data.nz == data.etaRule.nodeCount(),
             comm_, PETSC_ERR_ARG_SIZ,
             "FD-q rule sizes do not match the global DMDA dimensions");

  diagnostics = {};
  PetscCall(validateCoefficients(recipe, data));
  PetscCall(createMatrices(data));
  PetscCall(insertValues(recipe, data));
  PetscCall(computeDiagnostics(data, diagnostics));
  PetscFunctionReturn(PETSC_SUCCESS);
}
