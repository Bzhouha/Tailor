/**
 * @file BaseFlowDerivatives.hpp
 * @brief Physical-space derivatives of the five-component base flow.
 *
 * The module differentiates the distributed base flow with the loaded FD-q
 * rules and converts computational derivatives to physical y/z derivatives
 * using the curvilinear metrics.
 */
#pragma once

#include <petscdmda.h>

#include <array>
#include <cstddef>

#include "ProblemData.hpp"

/** @brief Component ordering used by every five-variable flow state. */
enum class FlowComponent : PetscInt {
  Density = 0,
  U,
  V,
  W,
  Temperature,
};

inline constexpr PetscInt flowComponentCount = 5;

/** @brief Physical derivative groups stored for each base-flow component. */
enum class BaseFlowDerivative : PetscInt {
  Dy = 0,
  Dz,
  Dyy,
  Dzz,
  Dyz,
};

inline constexpr PetscInt baseFlowDerivativeKindCount = 5;
inline constexpr PetscInt baseFlowDerivativeComponentCount =
    flowComponentCount * baseFlowDerivativeKindCount;

inline constexpr std::array<const char *, static_cast<std::size_t>(
                                              baseFlowDerivativeKindCount)>
    baseFlowDerivativeNames = {"dy", "dz", "dyy", "dzz", "dyz"};

inline constexpr std::array<const char *,
                            static_cast<std::size_t>(flowComponentCount)>
    flowComponentNames = {"rho", "u", "v", "w", "T"};

/** @brief Return the zero-based storage index of a flow component. */
[[nodiscard]] constexpr PetscInt flowIndex(FlowComponent component) {
  return static_cast<PetscInt>(component);
}

/**
 * @brief Return the flattened index in the 25-component derivative vector.
 * @param derivative Physical derivative group.
 * @param component Flow variable within the group.
 */
[[nodiscard]] constexpr PetscInt
baseFlowDerivativeIndex(BaseFlowDerivative derivative,
                        FlowComponent component) {
  return static_cast<PetscInt>(derivative) * flowComponentCount +
         flowIndex(component);
}

/** @brief Global diagnostics produced while differentiating the base flow. */
struct BaseFlowDerivativeDiagnostics {
  /** Largest imaginary magnitude found in the nominally real base flow. */
  PetscReal maxBaseFlowImaginary = 0.0;
  /** Two-norm of each component in the fixed derivative-major layout. */
  std::array<PetscReal,
             static_cast<std::size_t>(baseFlowDerivativeComponentCount)>
      norms{};
};

/**
 * @brief Compute and retain distributed physical derivatives of the base flow.
 *
 * The output layout is `[Dy, Dz, Dyy, Dzz, Dyz]`, with each group ordered as
 * `[\f$\rho\f$, U, V, W, T]`.
 */
class BaseFlowDerivatives {
public:
  /** @param comm Communicator shared by the input DMDA and output vectors. */
  explicit BaseFlowDerivatives(MPI_Comm comm);

  /**
   * @brief Create derivative storage and populate all 25 components.
   * @param data Prepared grid, base flow, metrics, and FD-q rules; receives the
   *             derivative DMDA and global vector.
   * @param diagnostics Global validation data and component norms.
   * @return PETSc error code.
   */
  PetscErrorCode compute(ProblemData &data,
                         BaseFlowDerivativeDiagnostics &diagnostics) const;

private:
  PetscErrorCode createStorage(ProblemData &data) const;
  PetscErrorCode
  computeValues(ProblemData &data,
                BaseFlowDerivativeDiagnostics &diagnostics) const;
  PetscErrorCode computeNorms(const ProblemData &data,
                              BaseFlowDerivativeDiagnostics &diagnostics) const;

  MPI_Comm comm_;
};
