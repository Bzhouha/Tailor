/**
 * @file LNSCoefficients.hpp
 * @brief Pointwise physical-space linearized Navier--Stokes coefficients.
 */
#pragma once

#include <petscsys.h>

#include <array>
#include <cstddef>

#include "BaseFlowDerivatives.hpp"
#include "Recipe.hpp"

/** @brief One real five-component flow tuple ordered as rho, U, V, W, T. */
using FlowState =
    std::array<PetscReal, static_cast<std::size_t>(flowComponentCount)>;

/** @brief Base-flow state and all physical derivatives required by the LNS
 * model. */
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

/**
 * @brief Row-major dense 5-by-5 complex coefficient block.
 *
 * The contiguous row-major storage can be passed directly to PETSc blocked
 * matrix insertion routines.
 */
class Block5 {
public:
  /** @brief Mutable element access. */
  [[nodiscard]] PetscScalar &operator()(PetscInt row, PetscInt column);
  /** @brief Constant element access. */
  [[nodiscard]] const PetscScalar &operator()(PetscInt row,
                                              PetscInt column) const;
  /** @brief Return a read-only pointer to the row-major storage. */
  [[nodiscard]] const PetscScalar *data() const noexcept {
    return values_.data();
  }
  /** @brief Return a mutable pointer to the row-major storage. */
  [[nodiscard]] PetscScalar *data() noexcept { return values_.data(); }
  /** @brief Test that every matrix entry is finite. */
  [[nodiscard]] bool finite() const noexcept;

private:
  std::array<PetscScalar, 25> values_{};
};

/** @brief Eleven physical-space coefficient blocks of the compressible LNS
 * system. */
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

  /** @brief Test that all coefficient blocks are finite. */
  [[nodiscard]] bool finite() const noexcept;
};

inline constexpr PetscInt physicalCoefficientCount = 11;
inline constexpr std::array<const char *,
                            static_cast<std::size_t>(physicalCoefficientCount)>
    physicalCoefficientNames = {"Gamma", "A",   "B",   "C",   "D",  "Vxx",
                                "Vxy",   "Vxz", "Vyy", "Vyz", "Vzz"};

/** @brief Non-throwing validation result for pointwise coefficient operations.
 */
enum class CoefficientStatus {
  Success = 0,
  NonFiniteInput,
  NonPositiveDensity,
  NonPositiveTemperature,
  NonFiniteOutput,
};

/** @brief Return a stable diagnostic name for a coefficient status. */
[[nodiscard]] const char *
coefficientStatusName(CoefficientStatus status) noexcept;

/**
 * @brief Build a real BaseFlowPoint from distributed PETSc vector entries.
 * @param state Five base-flow values.
 * @param derivatives Twenty-five values in derivative-major layout.
 * @param point Destination point; x-containing derivatives are set to zero for
 *              the locally parallel BiGlobal model.
 * @return Validation status.
 */
[[nodiscard]] CoefficientStatus
makeBaseFlowPoint(const PetscScalar *state, const PetscScalar *derivatives,
                  BaseFlowPoint &point) noexcept;

/** @brief Evaluate the physical LNS coefficient blocks at one grid point. */
class LNSCoefficients {
public:
  /** @param recipe Nondimensional gas, transport, and flow parameters. */
  explicit LNSCoefficients(const Recipe &recipe);

  /**
   * @param flow Validated base-flow values and derivatives.
   * @param coefficients Destination physical coefficient blocks.
   * @return Validation status.
   */
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
