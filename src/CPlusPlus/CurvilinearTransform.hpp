#pragma once

#include <petscsys.h>

#include <array>
#include <cstddef>

#include "Metrics.hpp"
#include "StreamwiseFourier.hpp"

struct MetricPoint {
  PetscReal xiY = 0.0;
  PetscReal xiZ = 0.0;
  PetscReal etaY = 0.0;
  PetscReal etaZ = 0.0;
  PetscReal xiYY = 0.0;
  PetscReal xiZZ = 0.0;
  PetscReal xiYZ = 0.0;
  PetscReal etaYY = 0.0;
  PetscReal etaZZ = 0.0;
  PetscReal etaYZ = 0.0;

  [[nodiscard]] bool finite() const noexcept;
};

[[nodiscard]] CoefficientStatus makeMetricPoint(const PetscScalar *metrics,
                                                MetricPoint &point) noexcept;

struct CurvilinearLNSCoefficients {
  Block5 Gamma;
  Block5 K0;
  Block5 Kxi;
  Block5 Keta;
  Block5 Vxixi;
  Block5 Vxieta;
  Block5 Vetaeta;

  [[nodiscard]] bool finite() const noexcept;
};

inline constexpr PetscInt curvilinearCoefficientCount = 7;
inline constexpr std::array<const char *, static_cast<std::size_t>(
                                              curvilinearCoefficientCount)>
    curvilinearCoefficientNames = {"Gamma", "K0",     "Kxi",    "Keta",
                                   "Vxixi", "Vxieta", "Vetaeta"};

class CurvilinearTransform {
public:
  [[nodiscard]] CoefficientStatus
  apply(const FourierLNSCoefficients &fourier, const MetricPoint &metrics,
        CurvilinearLNSCoefficients &curvilinear) const noexcept;
};
