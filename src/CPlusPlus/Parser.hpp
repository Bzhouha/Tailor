#pragma once

#include <petscsys.h>

#include <string>

#include "Recipe.hpp"

class Parser {
public:
  explicit Parser(MPI_Comm comm);

  PetscErrorCode parse(const std::string &yamlConfig, Recipe &recipe) const;

private:
  static Recipe parseOnRoot(const std::string &yamlConfig);
  PetscErrorCode broadcast(Recipe &recipe) const;
  PetscErrorCode broadcastString(std::string &value) const;

  MPI_Comm comm_;
};
