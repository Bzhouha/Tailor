/**
 * @file ProblemData.hpp
 * @brief Persistent distributed data and FD-q rules for one Tailor case.
 */
#pragma once

#include <petscdmda.h>
#include <petscmat.h>

#include <array>
#include <cstddef>
#include <vector>

/**
 * @brief Replicated one-dimensional local differentiation rule.
 *
 * Every MPI rank owns the complete rule. Arrays are row-major over
 * `(node, stencil slot)` and `weights[0..2]` stores d0, d1, and d2.
 */
struct FDQRuleData {
  /** Number of intervals; the rule contains N+1 nodes. */
  PetscInt N = 0;
  /** Local interpolation degree; every stencil contains q+1 nodes. */
  PetscInt q = 0;
  /** Strictly increasing computational coordinates. */
  std::vector<PetscReal> nodes;
  /** Global one-dimensional node index for every stencil entry. */
  std::vector<PetscInt> stencilIndices;
  /** Flattened interpolation and first/second derivative weights. */
  std::array<std::vector<PetscReal>, 3> weights;

  /** @brief Return the number of grid nodes represented by the rule. */
  [[nodiscard]] PetscInt nodeCount() const noexcept { return N + 1; }
  /** @brief Return the number of entries in each local stencil. */
  [[nodiscard]] PetscInt stencilSize() const noexcept { return q + 1; }

  /**
   * @brief Convert `(row, slot)` to the common flattened array offset.
   * @param row Differentiation node.
   * @param slot Position within its local stencil.
   */
  [[nodiscard]] std::size_t flatIndex(PetscInt row,
                                      PetscInt slot) const noexcept {
    return static_cast<std::size_t>(row) *
               static_cast<std::size_t>(stencilSize()) +
           static_cast<std::size_t>(slot);
  }

  /** @brief Return the global one-dimensional node used by a stencil slot. */
  [[nodiscard]] PetscInt stencilIndex(PetscInt row,
                                      PetscInt slot) const noexcept {
    return stencilIndices[flatIndex(row, slot)];
  }

  /**
   * @brief Return an interpolation or derivative weight.
   * @param derivative Derivative order: 0, 1, or 2.
   * @param row Differentiation node.
   * @param slot Position within its local stencil.
   */
  [[nodiscard]] PetscReal weight(PetscInt derivative, PetscInt row,
                                 PetscInt slot) const noexcept {
    return weights[static_cast<std::size_t>(derivative)][flatIndex(row, slot)];
  }

  /** @brief Release all replicated rule storage and reset its dimensions. */
  void clear() noexcept;
};

/**
 * @brief Owner of all long-lived PETSc resources for one configured problem.
 *
 * Resources are non-copyable and are released collectively by destroy() or the
 * destructor. Field-like vectors share compatible two-dimensional DMDAs.
 */
struct ProblemData {
  /** @brief Construct an empty resource owner. */
  ProblemData() = default;
  /** @brief Release all owned PETSc resources. */
  ~ProblemData();

  ProblemData(const ProblemData &) = delete;
  ProblemData &operator=(const ProblemData &) = delete;
  ProblemData(ProblemData &&) = delete;
  ProblemData &operator=(ProblemData &&) = delete;

  /**
   * @brief Destroy matrices, vectors, DMDAs, and replicated FD-q rules.
   * @return PETSc error code.
   */
  PetscErrorCode destroy();

  /** Number of xi/y nodes. */
  PetscInt ny = 0;
  /** Number of eta/z nodes. */
  PetscInt nz = 0;
  /** BOX ghost width used by compatible DMDAs. */
  PetscInt stencilWidth = 0;

  /** Five-DOF state DMDA ordered as rho, U, V, W, T. */
  DM fieldDM = nullptr;
  /** Three-DOF physical-grid DMDA ordered as x, y, z. */
  DM gridDM = nullptr;
  /** Ten-DOF inverse-metric DMDA. */
  DM metricDM = nullptr;
  /** Twenty-five-DOF physical base-flow derivative DMDA. */
  DM baseflowDerivativeDM = nullptr;
  /** Distributed five-component base-flow vector. */
  Vec baseflow = nullptr;
  /** Distributed physical-grid coordinate vector. */
  Vec grid = nullptr;
  /** Distributed ten-component metric vector. */
  Vec metrics = nullptr;
  /** Distributed twenty-five-component base-flow derivative vector. */
  Vec baseflowDerivatives = nullptr;
  /** Block-size-five generalized mass matrix M_Gamma. */
  Mat massMatrix = nullptr;
  /** Block-size-five unconstrained spatial matrix L. */
  Mat spatialMatrix = nullptr;

  /** HDF5 y rule, interpreted as differentiation in computational xi. */
  FDQRuleData xiRule;
  /** HDF5 z rule, interpreted as differentiation in computational eta. */
  FDQRuleData etaRule;
};
