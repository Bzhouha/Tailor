#include <petscdmda.h>
#include <petscviewerhdf5.h>
#include <slepcsys.h>

#include <iostream>
#include <string>

namespace {

PetscErrorCode loadNaturalVector(PetscViewer viewer, DM dm, const char *name,
                                 Vec *global) {
  Vec natural = nullptr;

  PetscFunctionBeginUser;
  PetscCall(DMDACreateNaturalVector(dm, &natural));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(natural), name));
  PetscCall(VecLoad(natural, viewer));
  PetscCall(DMCreateGlobalVector(dm, global));
  PetscCall(DMDANaturalToGlobalBegin(dm, natural, INSERT_VALUES, *global));
  PetscCall(DMDANaturalToGlobalEnd(dm, natural, INSERT_VALUES, *global));
  PetscCall(VecDestroy(&natural));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " INPUT.h5\n";
    return 2;
  }

  const std::string input_path = argv[1];
  PetscViewer viewer = nullptr;
  DM field_dm = nullptr;
  DM grid_dm = nullptr;
  Vec baseflow = nullptr;
  Vec grid = nullptr;
  PetscInt ny = 0;
  PetscInt nz = 0;
  PetscInt baseflow_size = 0;
  PetscInt grid_size = 0;
  PetscReal baseflow_norm = 0.0;
  PetscReal grid_norm = 0.0;

  PetscCall(SlepcInitialize(&argc, &argv, nullptr, nullptr));
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, input_path.c_str(), FILE_MODE_READ,
                                &viewer));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "Ny", PETSC_INT, nullptr,
                                         &ny));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "Nz", PETSC_INT, nullptr,
                                         &nz));

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX, ny, nz, PETSC_DECIDE, PETSC_DECIDE, 5,
                         4, nullptr, nullptr, &field_dm));
  PetscCall(DMSetUp(field_dm));
  PetscCall(DMDACreateCompatibleDMDA(field_dm, 3, &grid_dm));
  PetscCall(DMSetUp(grid_dm));

  PetscCall(loadNaturalVector(viewer, field_dm, "baseflow", &baseflow));
  PetscCall(loadNaturalVector(viewer, grid_dm, "grid", &grid));
  PetscCall(VecGetSize(baseflow, &baseflow_size));
  PetscCall(VecGetSize(grid, &grid_size));
  PetscCheck(baseflow_size == ny * nz * 5, PETSC_COMM_WORLD, PETSC_ERR_FILE_UNEXPECTED,
             "Unexpected baseflow size");
  PetscCheck(grid_size == ny * nz * 3, PETSC_COMM_WORLD, PETSC_ERR_FILE_UNEXPECTED,
             "Unexpected grid size");
  PetscCall(VecNorm(baseflow, NORM_2, &baseflow_norm));
  PetscCall(VecNorm(grid, NORM_2, &grid_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Loaded Ny=%" PetscInt_FMT ", Nz=%" PetscInt_FMT
                        ", ||baseflow||2=%g, ||grid||2=%g\n",
                        ny, nz, static_cast<double>(baseflow_norm),
                        static_cast<double>(grid_norm)));

  PetscCall(VecDestroy(&grid));
  PetscCall(VecDestroy(&baseflow));
  PetscCall(DMDestroy(&grid_dm));
  PetscCall(DMDestroy(&field_dm));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(SlepcFinalize());
  return 0;
}
