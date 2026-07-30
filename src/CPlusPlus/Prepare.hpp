/**
 * @file Prepare.hpp
 * @brief Load prepared HDF5 fields and FD-q discretization into PETSc objects.
 */
#pragma once

#include <petscdmda.h>

#include "ProblemData.hpp"
#include "Recipe.hpp"

/**
 * @brief Create distributed DMDAs and load all persistent input data.
 *
 * PETSc natural vectors are converted to DMDA global ordering, while the small
 * one-dimensional FD-q rules are read on rank zero and replicated.
 */
class Prepare {
public:
  /** @param comm Communicator used for collective HDF5 and PETSc operations. */
  explicit Prepare(MPI_Comm comm);

  /**
   * @param recipe Validated case configuration.
   * @param data Empty destination that receives DMDAs, vectors, and FD-q rules.
   * @return PETSc error code.
   */
  PetscErrorCode initialize(const Recipe &recipe, ProblemData &data) const;

private:
  /** @brief Load and collectively validate root schema metadata. */
  PetscErrorCode readMetadata(PetscViewer viewer, const Recipe &recipe,
                              ProblemData &data) const;
  /** @brief Create compatible periodic/nonperiodic DMDAs. */
  PetscErrorCode createDMs(const Recipe &recipe, ProblemData &data) const;
  /** @brief Load one PETSc natural vector and map it into DMDA ordering. */
  PetscErrorCode loadNaturalVector(PetscViewer viewer, DM dm, const char *name,
                                   Vec *global) const;
  /** @brief Load persistent grid and base-flow vectors. */
  PetscErrorCode loadFields(PetscViewer viewer, ProblemData &data) const;
  /** @brief Read, broadcast, and validate both one-dimensional FD-q rules. */
  PetscErrorCode loadDiscretization(PetscViewer viewer, const Recipe &recipe,
                                    ProblemData &data) const;

  /** Communicator used for collective I/O and validation. */
  MPI_Comm comm_;
};
