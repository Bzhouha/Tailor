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
  /** Basic-flow primitive state. */
  FlowState value{};
  /** First physical x derivative. */
  FlowState dx{};
  /** First physical y derivative. */
  FlowState dy{};
  /** First physical z derivative. */
  FlowState dz{};
  /** Second physical xx derivative. */
  FlowState dxx{};
  /** Second physical yy derivative. */
  FlowState dyy{};
  /** Second physical zz derivative. */
  FlowState dzz{};
  /** Mixed physical xy derivative. */
  FlowState dxy{};
  /** Mixed physical xz derivative. */
  FlowState dxz{};
  /** Mixed physical yz derivative. */
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
  /** Contiguous row-major matrix entries. */
  std::array<PetscScalar, 25> values_{};
};

/** @brief Eleven physical-space coefficient blocks of the compressible LNS
 * system. */
struct PhysicalLNSCoefficients {
  /** Generalized time/mass block. */
  Block5 Gamma;
  /** Inviscid/viscous physical x first-derivative block. */
  Block5 A;
  /** Inviscid/viscous physical y first-derivative block. */
  Block5 B;
  /** Inviscid/viscous physical z first-derivative block. */
  Block5 C;
  /** Zeroth-order physical block. */
  Block5 D;
  /** Physical xx viscous block. */
  Block5 Vxx;
  /** Physical xy viscous block. */
  Block5 Vxy;
  /** Physical xz viscous block. */
  Block5 Vxz;
  /** Physical yy viscous block. */
  Block5 Vyy;
  /** Physical yz viscous block. */
  Block5 Vyz;
  /** Physical zz viscous block. */
  Block5 Vzz;

  /** @brief Test that all coefficient blocks are finite. */
  [[nodiscard]] bool finite() const noexcept;
};

/**
 * @brief Inviscid physical-space blocks used by the far-field
 * characteristic boundary condition.
 */
struct InviscidLNSCoefficients {
  /** Generalized time/mass block. */
  Block5 Gamma;
  /** Inviscid physical-y flux Jacobian. */
  Block5 Bc;
  /** Inviscid physical-z flux Jacobian. */
  Block5 Cc;

  /** @brief Test that all three coefficient blocks are finite. */
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

  /**
   * @brief Evaluate the inviscid mass and y/z flux-Jacobian blocks.
   *
   * These are the canonical `G`, `B_c`, and `C_c` blocks from
   * `mod_cubes.f90/get_unadorned_cubes`.
   */
  [[nodiscard]] CoefficientStatus
  evaluateInviscid(const BaseFlowPoint &flow,
                   InviscidLNSCoefficients &coefficients) const noexcept;

private:
  /** Reynolds number. */
  PetscReal reynolds_;
  /** Mach number. */
  PetscReal mach_;
  /** Prandtl number. */
  PetscReal prandtl_;
  /** Ratio of specific heats. */
  PetscReal gamma_;
  /** Dimensionless Sutherland constant \f$S/T_{\rm ref}\f$. */
  PetscReal sutherlandRatio_;
};
