#pragma once

#include <petscdmda.h>

#include <array>
#include <cstddef>

#include "ProblemData.hpp"

enum class MetricComponent : PetscInt {
  XiY = 0,
  XiZ,
  EtaY,
  EtaZ,
  XiYY,
  XiZZ,
  XiYZ,
  EtaYY,
  EtaZZ,
  EtaYZ,
};

inline constexpr PetscInt metricComponentCount = 10;

inline constexpr std::array<const char *,
                            static_cast<std::size_t>(metricComponentCount)>
    metricComponentNames = {"xi_y",   "xi_z",   "eta_y",  "eta_z",
                            "xi_yy",  "xi_zz",  "xi_yz",  "eta_yy",
                            "eta_zz", "eta_yz"};

[[nodiscard]] constexpr PetscInt metricIndex(MetricComponent component) {
  return static_cast<PetscInt>(component);
}

struct MetricDiagnostics {
  PetscReal minJacobian = 0.0;
  PetscReal maxJacobian = 0.0;
  PetscReal maxGridImaginary = 0.0;
  std::array<PetscReal, static_cast<std::size_t>(metricComponentCount)> norms{};
};

class Metrics {
public:
  explicit Metrics(MPI_Comm comm);

  PetscErrorCode compute(ProblemData &data,
                         MetricDiagnostics &diagnostics) const;

private:
  PetscErrorCode createStorage(ProblemData &data) const;
  PetscErrorCode computeFirstOrder(ProblemData &data,
                                   MetricDiagnostics &diagnostics) const;
  PetscErrorCode computeSecondOrder(ProblemData &data) const;
  PetscErrorCode computeNorms(const ProblemData &data,
                              MetricDiagnostics &diagnostics) const;

  MPI_Comm comm_;
};
