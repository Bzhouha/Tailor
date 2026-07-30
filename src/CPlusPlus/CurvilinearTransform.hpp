/**
 * @file CurvilinearTransform.hpp
 * @brief Pointwise transformation of Fourier-space LNS coefficients.
 */
#pragma once

#include <petscsys.h>

#include <array>
#include <cstddef>

#include "Metrics.hpp"
#include "StreamwiseFourier.hpp"

/** @brief First- and second-order inverse mapping metrics at one grid point. */
struct MetricPoint {
  /** First inverse derivative \f$\xi_y\f$. */
  PetscReal xiY = 0.0;
  /** First inverse derivative \f$\xi_z\f$. */
  PetscReal xiZ = 0.0;
  /** First inverse derivative \f$\eta_y\f$. */
  PetscReal etaY = 0.0;
  /** First inverse derivative \f$\eta_z\f$. */
  PetscReal etaZ = 0.0;
  /** Second inverse derivative \f$\xi_{yy}\f$. */
  PetscReal xiYY = 0.0;
  /** Second inverse derivative \f$\xi_{zz}\f$. */
  PetscReal xiZZ = 0.0;
  /** Mixed inverse derivative \f$\xi_{yz}\f$. */
  PetscReal xiYZ = 0.0;
  /** Second inverse derivative \f$\eta_{yy}\f$. */
  PetscReal etaYY = 0.0;
  /** Second inverse derivative \f$\eta_{zz}\f$. */
  PetscReal etaZZ = 0.0;
  /** Mixed inverse derivative \f$\eta_{yz}\f$. */
  PetscReal etaYZ = 0.0;

  /** @brief Test that every metric value is finite. */
  [[nodiscard]] bool finite() const noexcept;
};

/**
 * @brief Convert one raw metric-vector entry to a validated real MetricPoint.
 * @param metrics Pointer to the ten-component metric tuple.
 * @param point Destination point.
 * @return Validation status; complex or non-finite metrics are rejected.
 */
[[nodiscard]] CoefficientStatus makeMetricPoint(const PetscScalar *metrics,
                                                MetricPoint &point) noexcept;

/**
 * @brief LNS coefficient blocks expressed in computational coordinates.
 *
 * These blocks define
 * \f[
 * Lq=K_0q+K_\xi q_\xi+K_\eta q_\eta
 * -V_{\xi\xi}q_{\xi\xi}-V_{\eta\eta}q_{\eta\eta}
 * -V_{\xi\eta}q_{\xi\eta}.
 * \f]
 */
struct CurvilinearLNSCoefficients {
  /** Generalized time/mass block. */
  Block5 Gamma;
  /** Zeroth-order transformed block. */
  Block5 K0;
  /** Computational-xi first-derivative block. */
  Block5 Kxi;
  /** Computational-eta first-derivative block. */
  Block5 Keta;
  /** Computational-xi pure second-derivative block. */
  Block5 Vxixi;
  /** Computational mixed second-derivative block. */
  Block5 Vxieta;
  /** Computational-eta pure second-derivative block. */
  Block5 Vetaeta;

  /** @brief Test that all seven coefficient blocks are finite. */
  [[nodiscard]] bool finite() const noexcept;
};

inline constexpr PetscInt curvilinearCoefficientCount = 7;
inline constexpr std::array<const char *, static_cast<std::size_t>(
                                              curvilinearCoefficientCount)>
    curvilinearCoefficientNames = {"Gamma", "K0",     "Kxi",    "Keta",
                                   "Vxixi", "Vxieta", "Vetaeta"};

/** @brief Apply the physical-to-computational chain rule pointwise. */
class CurvilinearTransform {
public:
  /**
   * @param fourier Physical y/z coefficients after streamwise Fourier
   *                transformation.
   * @param metrics Inverse mapping metrics at the same grid point.
   * @param curvilinear Destination computational-coordinate coefficients.
   * @return Validation status.
   */
  [[nodiscard]] CoefficientStatus
  apply(const FourierLNSCoefficients &fourier, const MetricPoint &metrics,
        CurvilinearLNSCoefficients &curvilinear) const noexcept;
};
