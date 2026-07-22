#include "Prepare.hpp"

#include <petscviewerhdf5.h>

#include <algorithm>

ProblemData::~ProblemData() { (void)destroy(); }

PetscErrorCode ProblemData::destroy() {
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&grid));
  PetscCall(VecDestroy(&baseflow));
  PetscCall(DMDestroy(&gridDM));
  PetscCall(DMDestroy(&fieldDM));
  ny = 0;
  nz = 0;
  stencilWidth = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

Prepare::Prepare(MPI_Comm comm) : comm_(comm) {}

PetscErrorCode Prepare::readDimensions(PetscViewer viewer,
                                       ProblemData &data) const {
  PetscFunctionBeginUser;
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "Ny", PETSC_INT,
                                         nullptr, &data.ny));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "Nz", PETSC_INT,
                                         nullptr, &data.nz));
  PetscCheck(data.ny > 0 && data.nz > 0, comm_, PETSC_ERR_FILE_UNEXPECTED,
             "HDF5 attributes Ny and Nz must be positive");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Prepare::createDMs(const Recipe &recipe,
                                  ProblemData &data) const {
  const PetscInt qY = static_cast<PetscInt>(recipe.qY);
  const PetscInt qZ = static_cast<PetscInt>(recipe.qZ);

  PetscFunctionBeginUser;
  PetscCheck(qY <= data.ny - 1, comm_, PETSC_ERR_ARG_OUTOFRANGE,
             "Q-Value.y (%" PetscInt_FMT
             ") must not exceed Ny-1 (%" PetscInt_FMT ")",
             qY, data.ny - 1);
  PetscCheck(qZ <= data.nz - 1, comm_, PETSC_ERR_ARG_OUTOFRANGE,
             "Q-Value.z (%" PetscInt_FMT
             ") must not exceed Nz-1 (%" PetscInt_FMT ")",
             qZ, data.nz - 1);

  // A degree-q one-sided Fornberg stencil contains q+1 points and reaches q
  // points from a boundary. DMDA uses one box width for both directions.
  data.stencilWidth = std::max(qY, qZ);
  PetscCall(DMDACreate2d(comm_, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX, data.ny, data.nz, PETSC_DECIDE,
                         PETSC_DECIDE, 5, data.stencilWidth, nullptr, nullptr,
                         &data.fieldDM));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(data.fieldDM),
                               "field_dm"));
  PetscCall(DMSetUp(data.fieldDM));

  PetscCall(DMDACreateCompatibleDMDA(data.fieldDM, 3, &data.gridDM));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(data.gridDM),
                               "grid_dm"));
  PetscCall(DMSetUp(data.gridDM));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Prepare::loadNaturalVector(PetscViewer viewer, DM dm,
                                          const char *name,
                                          Vec *global) const {
  Vec natural = nullptr;

  PetscFunctionBeginUser;
  PetscCheck(global, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "Output Vec pointer must not be null");
  PetscCall(DMDACreateNaturalVector(dm, &natural));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(natural), name));
  PetscCall(VecLoad(natural, viewer));

  PetscCall(DMCreateGlobalVector(dm, global));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(*global), name));
  PetscCall(DMDANaturalToGlobalBegin(dm, natural, INSERT_VALUES, *global));
  PetscCall(DMDANaturalToGlobalEnd(dm, natural, INSERT_VALUES, *global));
  PetscCall(VecDestroy(&natural));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Prepare::loadFields(PetscViewer viewer,
                                   ProblemData &data) const {
  PetscBool hasBaseflow = PETSC_FALSE;
  PetscBool hasGrid = PETSC_FALSE;
  PetscInt baseflowSize = 0;
  PetscInt gridSize = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerHDF5HasDataset(viewer, "baseflow", &hasBaseflow));
  PetscCall(PetscViewerHDF5HasDataset(viewer, "grid", &hasGrid));
  PetscCheck(hasBaseflow, comm_, PETSC_ERR_FILE_UNEXPECTED,
             "HDF5 dataset /baseflow is missing");
  PetscCheck(hasGrid, comm_, PETSC_ERR_FILE_UNEXPECTED,
             "HDF5 dataset /grid is missing");

  PetscCall(loadNaturalVector(viewer, data.fieldDM, "baseflow",
                              &data.baseflow));
  PetscCall(loadNaturalVector(viewer, data.gridDM, "grid", &data.grid));
  PetscCall(VecGetSize(data.baseflow, &baseflowSize));
  PetscCall(VecGetSize(data.grid, &gridSize));
  PetscCheck(baseflowSize == data.ny * data.nz * 5, comm_,
             PETSC_ERR_FILE_UNEXPECTED,
             "Unexpected baseflow size: expected %" PetscInt_FMT
             ", got %" PetscInt_FMT,
             data.ny * data.nz * 5, baseflowSize);
  PetscCheck(gridSize == data.ny * data.nz * 3, comm_,
             PETSC_ERR_FILE_UNEXPECTED,
             "Unexpected grid size: expected %" PetscInt_FMT
             ", got %" PetscInt_FMT,
             data.ny * data.nz * 3, gridSize);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Prepare::initialize(const Recipe &recipe,
                                   ProblemData &data) const {
  PetscViewer viewer = nullptr;

  PetscFunctionBeginUser;
  PetscCheck(!data.fieldDM && !data.gridDM && !data.baseflow && !data.grid,
             comm_, PETSC_ERR_ARG_WRONGSTATE,
             "ProblemData must be empty before Prepare::initialize");
  PetscCall(PetscViewerHDF5Open(comm_, recipe.inputFile.c_str(), FILE_MODE_READ,
                                &viewer));
  PetscCall(PetscViewerHDF5SetCollective(viewer, PETSC_TRUE));
  PetscCall(readDimensions(viewer, data));
  PetscCall(createDMs(recipe, data));
  PetscCall(loadFields(viewer, data));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
