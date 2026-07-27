#pragma once

#include <petscdmda.h>

#include <array>
#include <cstddef>

#include "ProblemData.hpp"

enum class FlowComponent : PetscInt {
  Density = 0,
  U,
  V,
  W,
  Temperature,
};

inline constexpr PetscInt flowComponentCount = 5;

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

[[nodiscard]] constexpr PetscInt flowIndex(FlowComponent component) {
  return static_cast<PetscInt>(component);
}

[[nodiscard]] constexpr PetscInt
baseFlowDerivativeIndex(BaseFlowDerivative derivative,
                        FlowComponent component) {
  return static_cast<PetscInt>(derivative) * flowComponentCount +
         flowIndex(component);
}

struct BaseFlowDerivativeDiagnostics {
  PetscReal maxBaseFlowImaginary = 0.0;
  std::array<PetscReal,
             static_cast<std::size_t>(baseFlowDerivativeComponentCount)>
      norms{};
};

class BaseFlowDerivatives {
public:
  explicit BaseFlowDerivatives(MPI_Comm comm);

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
