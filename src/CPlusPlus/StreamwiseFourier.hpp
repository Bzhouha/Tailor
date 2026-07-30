/**
 * @file StreamwiseFourier.hpp
 * @brief Streamwise Fourier transformation of physical LNS coefficients.
 */
#pragma once

#include <petscsys.h>

#include <array>
#include <cstddef>

#include "LNSCoefficients.hpp"

/** @brief Physical y/z coefficient blocks after substituting d/dx = i alpha. */
struct FourierLNSCoefficients {
  /** Generalized time/mass block. */
  Block5 Gamma;
  /** Zeroth-order Fourier-space block. */
  Block5 K0;
  /** Physical-y first-derivative block. */
  Block5 Ky;
  /** Physical-z first-derivative block. */
  Block5 Kz;
  /** Physical-y second-derivative viscous block. */
  Block5 Vyy;
  /** Physical mixed-derivative viscous block. */
  Block5 Vyz;
  /** Physical-z second-derivative viscous block. */
  Block5 Vzz;

  /** @brief Test that all seven transformed blocks are finite. */
  [[nodiscard]] bool finite() const noexcept;
};

inline constexpr PetscInt fourierCoefficientCount = 7;
inline constexpr std::array<const char *,
                            static_cast<std::size_t>(fourierCoefficientCount)>
    fourierCoefficientNames = {"Gamma", "K0", "Ky", "Kz", "Vyy", "Vyz", "Vzz"};

/**
 * @brief Apply a prescribed complex streamwise wavenumber pointwise.
 *
 * The transformation uses
 * \f$K_0=D+i\alpha A+\alpha^2V_{xx}\f$,
 * \f$K_y=B-i\alpha V_{xy}\f$, and
 * \f$K_z=C-i\alpha V_{xz}\f$.
 */
class StreamwiseFourier {
public:
  /** @param alpha Complex streamwise wavenumber. */
  explicit StreamwiseFourier(PetscScalar alpha);

  /**
   * @param physical Physical-space coefficient blocks.
   * @param fourier Destination Fourier-space blocks.
   * @return Validation status.
   */
  [[nodiscard]] CoefficientStatus
  apply(const PhysicalLNSCoefficients &physical,
        FourierLNSCoefficients &fourier) const noexcept;

private:
  /** Complex streamwise Fourier wavenumber. */
  PetscScalar alpha_;
};
