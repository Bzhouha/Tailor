/**
 * @file Process.hpp
 * @brief Top-level orchestration of the Tailor preparation and assembly stages.
 */
#pragma once

#include <petscsys.h>

#include <string>

/** @brief Execute the configured solver pipeline collectively. */
class Process {
public:
  /** @param comm Communicator used by every pipeline stage. */
  explicit Process(MPI_Comm comm);

  /**
   * @brief Parse command-line configuration and run all implemented stages.
   * @return PETSc error code.
   */
  PetscErrorCode run() const;

private:
  PetscErrorCode readConfigPath(std::string &configPath) const;

  MPI_Comm comm_;
};
