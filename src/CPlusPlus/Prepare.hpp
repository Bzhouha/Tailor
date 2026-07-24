#pragma once

#include <petscdmda.h>

#include "ProblemData.hpp"
#include "Recipe.hpp"

class Prepare {
public:
  explicit Prepare(MPI_Comm comm);

  PetscErrorCode initialize(const Recipe &recipe, ProblemData &data) const;

private:
  PetscErrorCode readMetadata(PetscViewer viewer, const Recipe &recipe,
                              ProblemData &data) const;
  PetscErrorCode createDMs(const Recipe &recipe, ProblemData &data) const;
  PetscErrorCode loadNaturalVector(PetscViewer viewer, DM dm, const char *name,
                                   Vec *global) const;
  PetscErrorCode loadFields(PetscViewer viewer, ProblemData &data) const;
  PetscErrorCode loadDiscretization(PetscViewer viewer, const Recipe &recipe,
                                    ProblemData &data) const;

  MPI_Comm comm_;
};
