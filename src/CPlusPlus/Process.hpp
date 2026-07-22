#pragma once

#include <petscsys.h>

#include <string>

class Process {
public:
  explicit Process(MPI_Comm comm);

  PetscErrorCode run() const;

private:
  PetscErrorCode readConfigPath(std::string &configPath) const;

  MPI_Comm comm_;
};
