/**
 * @file Process.cpp
 * @brief Collective execution order and user-facing preparation diagnostics.
 */
#include "Process.hpp"

#include <array>

#include "BaseFlowDerivatives.hpp"
#include "BoundaryConditions.hpp"
#include "EigenOutput.hpp"
#include "EigenSolver.hpp"
#include "GlobalOperator.hpp"
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
  BaseFlowDerivativeDiagnostics derivativeDiagnostics;
  GlobalOperatorDiagnostics operatorDiagnostics;
  BoundaryConditionDiagnostics boundaryDiagnostics;
  EigenSolution eigenSolution;
  EigenSolverDiagnostics eigenDiagnostics;
  EigenOutputDiagnostics outputDiagnostics;
  PetscBool assembleOnly = PETSC_FALSE;
  PetscReal baseflowNorm = 0.0;
  PetscReal gridNorm = 0.0;

  PetscFunctionBeginUser;
  PetscCall(readConfigPath(configPath));
  PetscCall(Parser(comm_).parse(configPath, recipe));
  PetscCall(Prepare(comm_).initialize(recipe, data));
  PetscCall(Metrics(comm_).compute(data, metricDiagnostics));
  PetscCall(BaseFlowDerivatives(comm_).compute(data, derivativeDiagnostics));
  PetscCall(GlobalOperator(comm_).assemble(recipe, data, operatorDiagnostics));
  PetscCall(BoundaryConditions(comm_).apply(recipe, data, boundaryDiagnostics));
  PetscCall(PetscOptionsHasName(nullptr, nullptr, "-tailor_assemble_only",
                                &assembleOnly));

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

  PetscCall(PetscPrintf(
      comm_,
      "  base-flow derivative components: %" PetscInt_FMT
      "; max |Im(baseflow)|: %.16g\n",
      baseFlowDerivativeComponentCount,
      static_cast<double>(derivativeDiagnostics.maxBaseFlowImaginary)));
  for (PetscInt derivative = 0; derivative < baseFlowDerivativeKindCount;
       ++derivative) {
    PetscCall(PetscPrintf(
        comm_,
        "  base-flow derivative norm %s: %.16g %.16g %.16g %.16g %.16g\n",
        baseFlowDerivativeNames[static_cast<std::size_t>(derivative)],
        static_cast<double>(
            derivativeDiagnostics.norms[static_cast<std::size_t>(
                derivative * flowComponentCount)]),
        static_cast<double>(
            derivativeDiagnostics.norms[static_cast<std::size_t>(
                derivative * flowComponentCount + 1)]),
        static_cast<double>(
            derivativeDiagnostics.norms[static_cast<std::size_t>(
                derivative * flowComponentCount + 2)]),
        static_cast<double>(
            derivativeDiagnostics.norms[static_cast<std::size_t>(
                derivative * flowComponentCount + 3)]),
        static_cast<double>(
            derivativeDiagnostics.norms[static_cast<std::size_t>(
                derivative * flowComponentCount + 4)])));
  }

  PetscCall(PetscPrintf(
      comm_,
      "  global matrices: %" PetscInt_FMT " x %" PetscInt_FMT
      ", block size %" PetscInt_FMT "\n"
      "  M_Gamma blocks used/allocated: %.0f / %.0f; ||M_Gamma||F: %.16g\n"
      "  L blocks used/allocated: %.0f / %.0f; ||L||F: %.16g\n"
      "  eigenvalue convention: (-L) q = lambda M_Gamma q, "
      "lambda = -i omega\n",
      operatorDiagnostics.rows, operatorDiagnostics.columns,
      operatorDiagnostics.blockSize,
      static_cast<double>(operatorDiagnostics.massUsedBlocks),
      static_cast<double>(operatorDiagnostics.massAllocatedBlocks),
      static_cast<double>(operatorDiagnostics.massFrobeniusNorm),
      static_cast<double>(operatorDiagnostics.spatialUsedBlocks),
      static_cast<double>(operatorDiagnostics.spatialAllocatedBlocks),
      static_cast<double>(operatorDiagnostics.spatialFrobeniusNorm)));

  PetscCall(PetscPrintf(
      comm_,
      "  boundaries: wall nodes=%" PetscInt_FMT
      ", far-field nodes=%" PetscInt_FMT "\n"
      "  far-field incoming modes=%" PetscInt_FMT " (per node %" PetscInt_FMT
      "..%" PetscInt_FMT "), neutral modes=%" PetscInt_FMT "\n"
      "  max characteristic |imag(c)|: %.16g; max cond(R): %.16g\n"
      "  ||A_bc||F: %.16g; ||B_bc||F: %.16g\n",
      boundaryDiagnostics.wallNodes, boundaryDiagnostics.farfieldNodes,
      boundaryDiagnostics.incomingModes,
      boundaryDiagnostics.minIncomingModesPerNode,
      boundaryDiagnostics.maxIncomingModesPerNode,
      boundaryDiagnostics.neutralModes,
      static_cast<double>(boundaryDiagnostics.maxCharacteristicImaginary),
      static_cast<double>(boundaryDiagnostics.maxEigenvectorCondition),
      static_cast<double>(boundaryDiagnostics.eigenFrobeniusNorm),
      static_cast<double>(boundaryDiagnostics.eigenMassFrobeniusNorm)));

  if (assembleOnly) {
    PetscCall(PetscPrintf(
        comm_,
        "  assemble-only: boundary matrices validated; EPSSolve skipped\n"));
  } else {
    PetscCall(EigenSolver(comm_).solve(recipe, data, eigenSolution,
                                       eigenDiagnostics));
    PetscCall(PetscPrintf(
        comm_,
        "  eigensolver: requested=%" PetscInt_FMT ", converged=%" PetscInt_FMT
        ", iterations=%" PetscInt_FMT "\n"
        "  lambda target: %.16g %+.16gi; max relative error: %.16g\n",
        eigenDiagnostics.requested, eigenDiagnostics.converged,
        eigenDiagnostics.iterations,
        static_cast<double>(PetscRealPart(eigenDiagnostics.targetLambda)),
        static_cast<double>(PetscImaginaryPart(eigenDiagnostics.targetLambda)),
        static_cast<double>(eigenDiagnostics.maximumRelativeError)));
    PetscCall(EigenOutput(comm_).write(recipe, data, eigenSolution,
                                       eigenDiagnostics, outputDiagnostics));
    PetscCall(PetscPrintf(comm_, "  output: %s (%" PetscInt_FMT " modes)\n",
                          outputDiagnostics.file.c_str(),
                          outputDiagnostics.modesWritten));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
