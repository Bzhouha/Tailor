/**
 * @file Metrics.hpp
 * @brief Curvilinear mapping metrics for the two-dimensional y/z grid.
 */
#pragma once

#include <petscdmda.h>

#include <array>
#include <cstddef>

#include "ProblemData.hpp"

/** @brief Fixed component ordering of the distributed metric vector. */
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
    metricComponentNames = {"xi_y",  "xi_z",  "eta_y",  "eta_z",  "xi_yy",
                            "xi_zz", "xi_yz", "eta_yy", "eta_zz", "eta_yz"};

/** @brief Return the zero-based metric-vector component index. */
[[nodiscard]] constexpr PetscInt metricIndex(MetricComponent component) {
  return static_cast<PetscInt>(component);
}

/** @brief Global validation data and two-norms for computed metrics. */
struct MetricDiagnostics {
  /** Global minimum direct mapping Jacobian. */
  PetscReal minJacobian = 0.0;
  /** Global maximum direct mapping Jacobian. */
  PetscReal maxJacobian = 0.0;
  /** Largest imaginary magnitude in physical y/z grid coordinates. */
  PetscReal maxGridImaginary = 0.0;
  /** Global two-norm of each metric component. */
  std::array<PetscReal, static_cast<std::size_t>(metricComponentCount)> norms{};
};

/**
 * @brief Compute inverse first- and second-order mapping metrics.
 *
 * First derivatives of the physical grid are evaluated with FD-q weights.
 * Second derivatives of the inverse mapping are then obtained by applying the
 * chain rule to synchronized first-order metrics.
 */
class Metrics {
public:
  /** @param comm Communicator shared by the grid DMDA and metric vector. */
  explicit Metrics(MPI_Comm comm);

  /**
   * @param data Prepared grid and FD-q rules; receives metric storage.
   * @param diagnostics Jacobian range, realness check, and component norms.
   * @return PETSc error code.
   */
  PetscErrorCode compute(ProblemData &data,
                         MetricDiagnostics &diagnostics) const;

private:
  /** @brief Create the compatible ten-DOF metric DMDA and vector. */
  PetscErrorCode createStorage(ProblemData &data) const;
  /** @brief Compute direct derivatives, Jacobian, and inverse metrics. */
  PetscErrorCode computeFirstOrder(ProblemData &data,
                                   MetricDiagnostics &diagnostics) const;
  /** @brief Differentiate first metrics and apply the second-order chain rule.
   */
  PetscErrorCode computeSecondOrder(ProblemData &data) const;
  /** @brief Compute global two-norms of all ten metric components. */
  PetscErrorCode computeNorms(const ProblemData &data,
                              MetricDiagnostics &diagnostics) const;

  /** Communicator used for ghost updates and global reductions. */
  MPI_Comm comm_;
};
