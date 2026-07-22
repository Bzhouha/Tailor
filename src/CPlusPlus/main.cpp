#include <slepcsys.h>

#include "Process.hpp"

namespace {

constexpr char help[] =
    "Tailor: a temporal BiGlobal eigenvalue solver.\n"
    "  -c <file>  YAML configuration file (required)\n";

}  // namespace

int main(int argc, char **argv) {
  PetscErrorCode run_error = PETSC_SUCCESS;

  PetscErrorCode error = SlepcInitialize(&argc, &argv, nullptr, help);
  if (error != PETSC_SUCCESS) return static_cast<int>(error);

  {
    Process process(PETSC_COMM_WORLD);
    run_error = process.run();
  }

  error = SlepcFinalize();
  if (run_error != PETSC_SUCCESS) return static_cast<int>(run_error);
  return static_cast<int>(error);
}
