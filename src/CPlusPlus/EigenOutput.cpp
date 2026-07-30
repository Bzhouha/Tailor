/**
 * @file EigenOutput.cpp
 * @brief Natural-order HDF5 serialization and deterministic mode phasing.
 */
#include "EigenOutput.hpp"

#include <hdf5.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>

#include <array>
#include <filesystem>
#include <limits>
#include <string>

namespace {

/** @brief Convert one DMDA global vector to natural order and write it. */
PetscErrorCode writeNaturalVector(PetscViewer viewer, DM dm, Vec global,
                                  const char *group, const char *name) {
  Vec natural = nullptr;

  PetscFunctionBeginUser;
  PetscCall(DMDACreateNaturalVector(dm, &natural));
  PetscCall(DMDAGlobalToNaturalBegin(dm, global, INSERT_VALUES, natural));
  PetscCall(DMDAGlobalToNaturalEnd(dm, global, INSERT_VALUES, natural));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(natural), name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, group));
  PetscCall(VecView(natural, viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(VecDestroy(&natural));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Normalize a mode and rotate its largest entry to positive real. */
PetscErrorCode fixModePhase(Vec mode) {
  const PetscScalar *values = nullptr;
  PetscInt first = 0;
  PetscInt last = 0;
  PetscInt localIndex = PETSC_MAX_INT;
  PetscInt globalIndex = PETSC_MAX_INT;
  PetscReal localMaximum = -1.0;
  PetscReal globalMaximum = 0.0;
  PetscScalar localPivot = 0.0;
  PetscScalar pivot = 0.0;
  PetscReal norm = 0.0;

  PetscFunctionBeginUser;
  PetscCall(VecNormalize(mode, &norm));
  PetscCheck(norm > 0.0 && !PetscIsInfOrNanReal(norm),
             PetscObjectComm(reinterpret_cast<PetscObject>(mode)), PETSC_ERR_FP,
             "Cannot normalize a zero or non-finite eigenmode");
  PetscCall(VecGetOwnershipRange(mode, &first, &last));
  PetscCall(VecGetArrayRead(mode, &values));
  for (PetscInt index = first; index < last; ++index) {
    const PetscReal magnitude =
        PetscAbsScalar(values[static_cast<std::size_t>(index - first)]);
    if (magnitude > localMaximum) {
      localMaximum = magnitude;
      localIndex = index;
    }
  }
  PetscCallMPI(
      MPI_Allreduce(&localMaximum, &globalMaximum, 1, MPIU_REAL, MPI_MAX,
                    PetscObjectComm(reinterpret_cast<PetscObject>(mode))));
  if (localMaximum != globalMaximum)
    localIndex = PETSC_MAX_INT;
  PetscCallMPI(
      MPI_Allreduce(&localIndex, &globalIndex, 1, MPIU_INT, MPI_MIN,
                    PetscObjectComm(reinterpret_cast<PetscObject>(mode))));
  if (first <= globalIndex && globalIndex < last)
    localPivot = values[static_cast<std::size_t>(globalIndex - first)];
  PetscCall(VecRestoreArrayRead(mode, &values));
  PetscCallMPI(
      MPI_Allreduce(&localPivot, &pivot, 1, MPIU_SCALAR, MPI_SUM,
                    PetscObjectComm(reinterpret_cast<PetscObject>(mode))));
  PetscCheck(PetscAbsScalar(pivot) > 0.0,
             PetscObjectComm(reinterpret_cast<PetscObject>(mode)), PETSC_ERR_FP,
             "Could not select a deterministic mode phase");
  PetscCall(VecScale(mode, PetscConj(pivot) / PetscAbsScalar(pivot)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Write all self-describing root attributes. */
PetscErrorCode writeMetadata(PetscViewer viewer, const Recipe &recipe,
                             const ProblemData &data,
                             const EigenSolverDiagnostics &solverDiagnostics) {
  const PetscInt schemaVersion = 1;
  const PetscInt ny = data.ny;
  const PetscInt nz = data.nz;
  const PetscInt qY = recipe.qY;
  const PetscInt qZ = recipe.qZ;
  const PetscInt requested = solverDiagnostics.requested;
  const PetscInt converged = solverDiagnostics.converged;
  const PetscReal alphaReal = recipe.alpha.real();
  const PetscReal alphaImaginary = recipe.alpha.imag();
  const PetscReal omegaTargetReal = recipe.searchCenterOmega.real();
  const PetscReal omegaTargetImaginary = recipe.searchCenterOmega.imag();
  const PetscReal lambdaTargetReal =
      PetscRealPart(solverDiagnostics.targetLambda);
  const PetscReal lambdaTargetImaginary =
      PetscImaginaryPart(solverDiagnostics.targetLambda);
  const PetscReal period = data.spanwisePeriod;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr, "schema_version",
                                          PETSC_INT, &schemaVersion));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr, "case", PETSC_STRING,
                                          recipe.caseTitle.c_str()));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr, "ordering",
                                          PETSC_STRING, "k_j_dof"));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr, "mode_dof",
                                          PETSC_STRING, "[rho,u,v,w,T]"));
  PetscCall(PetscViewerHDF5WriteAttribute(
      viewer, nullptr, "lambda_convention", PETSC_STRING,
      "lambda=-i*omega; A_bc*q=lambda*B_bc*q"));
  PetscCall(PetscViewerHDF5WriteAttribute(
      viewer, nullptr, "xi_boundary", PETSC_STRING,
      "isothermal_no_slip_wall|characteristic_farfield"));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr, "eta_boundary",
                                          PETSC_STRING, "periodic"));
  PetscCall(
      PetscViewerHDF5WriteAttribute(viewer, nullptr, "Ny", PETSC_INT, &ny));
  PetscCall(
      PetscViewerHDF5WriteAttribute(viewer, nullptr, "Nz", PETSC_INT, &nz));
  PetscCall(
      PetscViewerHDF5WriteAttribute(viewer, nullptr, "q_y", PETSC_INT, &qY));
  PetscCall(
      PetscViewerHDF5WriteAttribute(viewer, nullptr, "q_z", PETSC_INT, &qZ));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr, "alpha_real",
                                          PETSC_REAL, &alphaReal));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr, "alpha_imag",
                                          PETSC_REAL, &alphaImaginary));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr,
                                          "search_center_omega_real",
                                          PETSC_REAL, &omegaTargetReal));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr,
                                          "search_center_omega_imag",
                                          PETSC_REAL, &omegaTargetImaginary));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr,
                                          "search_center_lambda_real",
                                          PETSC_REAL, &lambdaTargetReal));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr,
                                          "search_center_lambda_imag",
                                          PETSC_REAL, &lambdaTargetImaginary));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr, "spanwise_period",
                                          PETSC_REAL, &period));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr, "requested_modes",
                                          PETSC_INT, &requested));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, nullptr, "converged_modes",
                                          PETSC_INT, &converged));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Verify required datasets and attributes before atomic replacement. */
bool validateFile(const std::string &path, PetscInt converged) {
  const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
    return false;
  bool valid = H5Lexists(file, "/grid", H5P_DEFAULT) > 0 &&
               H5Lexists(file, "/baseflow", H5P_DEFAULT) > 0 &&
               H5Lexists(file, "/spectrum/lambda", H5P_DEFAULT) > 0 &&
               H5Lexists(file, "/spectrum/omega", H5P_DEFAULT) > 0 &&
               H5Lexists(file, "/spectrum/residual", H5P_DEFAULT) > 0;
  if (converged > 0)
    valid = valid && H5Lexists(file, "/modes/mode_000", H5P_DEFAULT) > 0;
  valid = valid && H5Aexists(file, "lambda_convention") > 0 &&
          H5Aexists(file, "converged_modes") > 0;
  H5Fclose(file);
  return valid;
}

} // namespace

EigenOutput::EigenOutput(MPI_Comm comm) : comm_(comm) {}

PetscErrorCode
EigenOutput::write(const Recipe &recipe, const ProblemData &data,
                   const EigenSolution &solution,
                   const EigenSolverDiagnostics &solverDiagnostics,
                   EigenOutputDiagnostics &diagnostics) const {
  PetscViewer viewer = nullptr;
  Vec lambda = nullptr;
  Vec omega = nullptr;
  Vec residual = nullptr;
  Vec mode = nullptr;
  PetscMPIInt rank = 0;
  PetscInt first = 0;
  PetscInt last = 0;
  PetscInt localFailure = 0;
  const std::string temporaryFile = recipe.outputFile + ".tmp";

  PetscFunctionBeginUser;
  PetscCheck(solution.eps, comm_, PETSC_ERR_ARG_WRONGSTATE,
             "A solved EPS object is required for eigenmode output");
  PetscCheck(data.gridDM && data.grid && data.fieldDM && data.baseflow, comm_,
             PETSC_ERR_ARG_WRONGSTATE,
             "Prepared grid and base flow are required for output");
  PetscCheck(solverDiagnostics.converged > 0, comm_, PETSC_ERR_NOT_CONVERGED,
             "No converged modes are available for output");
  diagnostics = {};
  PetscCallMPI(MPI_Comm_rank(comm_, &rank));

  if (rank == 0) {
    std::error_code error;
    const std::filesystem::path parent =
        std::filesystem::path(recipe.outputFile).parent_path();
    if (!parent.empty())
      std::filesystem::create_directories(parent, error);
    localFailure = error ? 1 : 0;
  }
  PetscCallMPI(MPI_Bcast(&localFailure, 1, MPIU_INT, 0, comm_));
  PetscCheck(!localFailure, comm_, PETSC_ERR_FILE_OPEN,
             "Could not create the output directory for %s",
             recipe.outputFile.c_str());

  PetscCall(PetscViewerHDF5Open(comm_, temporaryFile.c_str(), FILE_MODE_WRITE,
                                &viewer));
  PetscCall(writeMetadata(viewer, recipe, data, solverDiagnostics));
  PetscCall(writeNaturalVector(viewer, data.gridDM, data.grid, "/", "grid"));
  PetscCall(
      writeNaturalVector(viewer, data.fieldDM, data.baseflow, "/", "baseflow"));

  PetscCall(
      VecCreateMPI(comm_, PETSC_DECIDE, solverDiagnostics.converged, &lambda));
  PetscCall(VecDuplicate(lambda, &omega));
  PetscCall(VecDuplicate(lambda, &residual));
  PetscCall(
      PetscObjectSetName(reinterpret_cast<PetscObject>(lambda), "lambda"));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(omega), "omega"));
  PetscCall(
      PetscObjectSetName(reinterpret_cast<PetscObject>(residual), "residual"));
  PetscCall(VecGetOwnershipRange(lambda, &first, &last));
  for (PetscInt index = 0; index < solverDiagnostics.converged; ++index) {
    PetscScalar eigenvalue = 0.0;
    PetscReal error = 0.0;
    PetscCall(EPSGetEigenpair(solution.eps, index, &eigenvalue, nullptr,
                              nullptr, nullptr));
    PetscCall(EPSComputeError(solution.eps, index, EPS_ERROR_RELATIVE, &error));
    if (first <= index && index < last) {
      PetscCall(VecSetValue(lambda, index, eigenvalue, INSERT_VALUES));
      PetscCall(VecSetValue(omega, index, PETSC_i * eigenvalue, INSERT_VALUES));
      PetscCall(VecSetValue(residual, index, error, INSERT_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(lambda));
  PetscCall(VecAssemblyBegin(omega));
  PetscCall(VecAssemblyBegin(residual));
  PetscCall(VecAssemblyEnd(lambda));
  PetscCall(VecAssemblyEnd(omega));
  PetscCall(VecAssemblyEnd(residual));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/spectrum"));
  PetscCall(VecView(lambda, viewer));
  PetscCall(VecView(omega, viewer));
  PetscCall(VecView(residual, viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  PetscCall(DMCreateGlobalVector(data.fieldDM, &mode));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/modes"));
  for (PetscInt index = 0; index < solverDiagnostics.converged; ++index) {
    std::array<char, 32> name{};
    PetscCall(EPSGetEigenvector(solution.eps, index, mode, nullptr));
    PetscCall(fixModePhase(mode));
    PetscCall(PetscSNPrintf(name.data(), name.size(), "mode_%03" PetscInt_FMT,
                            index));
    Vec natural = nullptr;
    PetscCall(DMDACreateNaturalVector(data.fieldDM, &natural));
    PetscCall(
        DMDAGlobalToNaturalBegin(data.fieldDM, mode, INSERT_VALUES, natural));
    PetscCall(
        DMDAGlobalToNaturalEnd(data.fieldDM, mode, INSERT_VALUES, natural));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(natural),
                                 name.data()));
    PetscCall(VecView(natural, viewer));
    PetscCall(VecDestroy(&natural));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  PetscCall(VecDestroy(&mode));
  PetscCall(VecDestroy(&residual));
  PetscCall(VecDestroy(&omega));
  PetscCall(VecDestroy(&lambda));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCallMPI(MPI_Barrier(comm_));

  if (rank == 0) {
    localFailure =
        validateFile(temporaryFile, solverDiagnostics.converged) ? 0 : 1;
    if (!localFailure) {
      std::error_code error;
      std::filesystem::rename(temporaryFile, recipe.outputFile, error);
      localFailure = error ? 1 : 0;
    }
  }
  PetscCallMPI(MPI_Bcast(&localFailure, 1, MPIU_INT, 0, comm_));
  PetscCheck(!localFailure, comm_, PETSC_ERR_FILE_UNEXPECTED,
             "HDF5 validation or atomic replacement failed for %s",
             recipe.outputFile.c_str());
  PetscCallMPI(MPI_Barrier(comm_));

  diagnostics.file = recipe.outputFile;
  diagnostics.modesWritten = solverDiagnostics.converged;
  PetscFunctionReturn(PETSC_SUCCESS);
}
