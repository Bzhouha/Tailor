#pragma once

#include <petscsys.h>

#include <array>
#include <cstddef>

#include "BaseFlowDerivatives.hpp"
#include "Recipe.hpp"

using FlowState =
    std::array<PetscReal, static_cast<std::size_t>(flowComponentCount)>;

struct BaseFlowPoint {
  FlowState value{};
  FlowState dx{};
  FlowState dy{};
  FlowState dz{};
  FlowState dxx{};
  FlowState dyy{};
  FlowState dzz{};
  FlowState dxy{};
  FlowState dxz{};
  FlowState dyz{};
};

class Block5 {
public:
  [[nodiscard]] PetscScalar &operator()(PetscInt row, PetscInt column);
  [[nodiscard]] const PetscScalar &operator()(PetscInt row,
                                              PetscInt column) const;
  [[nodiscard]] const PetscScalar *data() const noexcept {
    return values_.data();
  }
  [[nodiscard]] PetscScalar *data() noexcept { return values_.data(); }
  [[nodiscard]] bool finite() const noexcept;

private:
  std::array<PetscScalar, 25> values_{};
};

struct PhysicalLNSCoefficients {
  Block5 Gamma;
  Block5 A;
  Block5 B;
  Block5 C;
  Block5 D;
  Block5 Vxx;
  Block5 Vxy;
  Block5 Vxz;
  Block5 Vyy;
  Block5 Vyz;
  Block5 Vzz;

  [[nodiscard]] bool finite() const noexcept;
};

inline constexpr PetscInt physicalCoefficientCount = 11;
inline constexpr std::array<const char *,
                            static_cast<std::size_t>(physicalCoefficientCount)>
    physicalCoefficientNames = {"Gamma", "A",   "B",   "C",   "D",  "Vxx",
                                "Vxy",   "Vxz", "Vyy", "Vyz", "Vzz"};

enum class CoefficientStatus {
  Success = 0,
  NonFiniteInput,
  NonPositiveDensity,
  NonPositiveTemperature,
  NonFiniteOutput,
};

[[nodiscard]] const char *
coefficientStatusName(CoefficientStatus status) noexcept;

[[nodiscard]] CoefficientStatus
makeBaseFlowPoint(const PetscScalar *state, const PetscScalar *derivatives,
                  BaseFlowPoint &point) noexcept;

class LNSCoefficients {
public:
  explicit LNSCoefficients(const Recipe &recipe);

  [[nodiscard]] CoefficientStatus
  evaluate(const BaseFlowPoint &flow,
           PhysicalLNSCoefficients &coefficients) const noexcept;

private:
  PetscReal reynolds_;
  PetscReal mach_;
  PetscReal prandtl_;
  PetscReal gamma_;
  PetscReal sutherlandRatio_;
};
