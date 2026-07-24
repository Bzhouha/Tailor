#include "Process.hpp"

#include <array>

#include "Metrics.hpp"
#include "Parser.hpp"
#include "Prepare.hpp"
#include "Recipe.hpp"

Process::Process(MPI_Comm comm) : comm_(comm) {}

PetscErrorCode Process::readConfigPath(std::string &configPath) const {
  std::array<char, PETSC_MAX_PATH_LEN> buffer{};
  PetscBool found = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-c", buffer.data(),
                                  buffer.size(), &found));
  PetscCheck(found && buffer[0] != '\0', comm_, PETSC_ERR_USER_INPUT,
             "Required option -c <file> was not provided");
  configPath = buffer.data();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Process::run() const {
  std::string configPath;
  Recipe recipe;
  ProblemData data;
  MetricDiagnostics metricDiagnostics;
  PetscReal baseflowNorm = 0.0;
  PetscReal gridNorm = 0.0;

  PetscFunctionBeginUser;
  PetscCall(readConfigPath(configPath));
  PetscCall(Parser(comm_).parse(configPath, recipe));
  PetscCall(Prepare(comm_).initialize(recipe, data));
  PetscCall(Metrics(comm_).compute(data, metricDiagnostics));

  PetscCall(VecNorm(data.baseflow, NORM_2, &baseflowNorm));
  PetscCall(VecNorm(data.grid, NORM_2, &gridNorm));
  PetscCall(PetscPrintf(
      comm_,
      "Prepared case '%s'\n"
      "  input: %s\n"
      "  Ny x Nz: %" PetscInt_FMT " x %" PetscInt_FMT "\n"
      "  alpha: %.16g %+.16gi\n"
      "  FD-q degrees: y=%d, z=%d; DMDA stencil width: %" PetscInt_FMT "\n"
      "  xi rule: %" PetscInt_FMT " nodes, %" PetscInt_FMT "-point stencil\n"
      "  eta rule: %" PetscInt_FMT " nodes, %" PetscInt_FMT "-point stencil\n"
      "  ||baseflow||2: %.16g, ||grid||2: %.16g\n"
      "  Jacobian range: [%.16g, %.16g]\n"
      "  max |Im(grid y/z)|: %.16g\n",
      recipe.caseTitle.c_str(), recipe.inputFile.c_str(), data.ny, data.nz,
      recipe.alpha.real(), recipe.alpha.imag(), recipe.qY, recipe.qZ,
      data.stencilWidth, data.xiRule.nodeCount(), data.xiRule.stencilSize(),
      data.etaRule.nodeCount(), data.etaRule.stencilSize(),
      static_cast<double>(baseflowNorm), static_cast<double>(gridNorm),
      static_cast<double>(metricDiagnostics.minJacobian),
      static_cast<double>(metricDiagnostics.maxJacobian),
      static_cast<double>(metricDiagnostics.maxGridImaginary)));

  for (PetscInt component = 0; component < metricComponentCount; ++component)
    PetscCall(PetscPrintf(
        comm_, "  metric norm %s: %.16g\n",
        metricComponentNames[static_cast<std::size_t>(component)],
        static_cast<double>(
            metricDiagnostics.norms[static_cast<std::size_t>(component)])));

  // Subsequent stages will be added here in order:
  // A/B assembly -> EPS solve -> output.
  PetscFunctionReturn(PETSC_SUCCESS);
}
