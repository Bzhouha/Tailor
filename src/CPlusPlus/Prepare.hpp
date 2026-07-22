#pragma once

#include <petscdmda.h>

#include "Recipe.hpp"

struct ProblemData {
  ProblemData() = default;
  ~ProblemData();

  ProblemData(const ProblemData &) = delete;
  ProblemData &operator=(const ProblemData &) = delete;

  PetscErrorCode destroy();

  PetscInt ny = 0;
  PetscInt nz = 0;
  PetscInt stencilWidth = 0;

  DM fieldDM = nullptr;
  DM gridDM = nullptr;
  Vec baseflow = nullptr;
  Vec grid = nullptr;
};

class Prepare {
public:
  explicit Prepare(MPI_Comm comm);

  PetscErrorCode initialize(const Recipe &recipe, ProblemData &data) const;

private:
  PetscErrorCode readDimensions(PetscViewer viewer, ProblemData &data) const;
  PetscErrorCode createDMs(const Recipe &recipe, ProblemData &data) const;
  PetscErrorCode loadNaturalVector(PetscViewer viewer, DM dm, const char *name,
                                   Vec *global) const;
  PetscErrorCode loadFields(PetscViewer viewer, ProblemData &data) const;

  MPI_Comm comm_;
};
