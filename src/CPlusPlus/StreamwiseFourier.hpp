#pragma once

#include <petscsys.h>

#include <array>
#include <cstddef>

#include "LNSCoefficients.hpp"

struct FourierLNSCoefficients {
  Block5 Gamma;
  Block5 K0;
  Block5 Ky;
  Block5 Kz;
  Block5 Vyy;
  Block5 Vyz;
  Block5 Vzz;

  [[nodiscard]] bool finite() const noexcept;
};

inline constexpr PetscInt fourierCoefficientCount = 7;
inline constexpr std::array<const char *,
                            static_cast<std::size_t>(fourierCoefficientCount)>
    fourierCoefficientNames = {"Gamma", "K0", "Ky", "Kz", "Vyy", "Vyz", "Vzz"};

class StreamwiseFourier {
public:
  explicit StreamwiseFourier(PetscScalar alpha);

  [[nodiscard]] CoefficientStatus
  apply(const PhysicalLNSCoefficients &physical,
        FourierLNSCoefficients &fourier) const noexcept;

private:
  PetscScalar alpha_;
};
