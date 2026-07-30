/**
 * @file Parser.hpp
 * @brief Collective YAML configuration parsing and validation.
 */
#pragma once

#include <petscsys.h>

#include <string>

#include "Recipe.hpp"

/**
 * @brief Parse a case recipe once on rank zero and broadcast it to all ranks.
 */
class Parser {
public:
  /** @param comm Communicator across which the validated recipe is broadcast.
   */
  explicit Parser(MPI_Comm comm);

  /**
   * @param yamlConfig Path to the YAML case configuration.
   * @param recipe Destination populated identically on every MPI rank.
   * @return PETSc error code.
   */
  PetscErrorCode parse(const std::string &yamlConfig, Recipe &recipe) const;

private:
  /** @brief Parse and validate YAML on rank zero. */
  static Recipe parseOnRoot(const std::string &yamlConfig);
  /** @brief Broadcast all scalar and string recipe fields. */
  PetscErrorCode broadcast(Recipe &recipe) const;
  /** @brief Broadcast one variable-length string from rank zero. */
  PetscErrorCode broadcastString(std::string &value) const;

  /** Communicator receiving identical validated configuration data. */
  MPI_Comm comm_;
};
