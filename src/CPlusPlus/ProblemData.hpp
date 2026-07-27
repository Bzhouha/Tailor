#pragma once

#include <petscdmda.h>
#include <petscmat.h>

#include <array>
#include <cstddef>
#include <vector>

struct FDQRuleData {
  PetscInt N = 0;
  PetscInt q = 0;
  std::vector<PetscReal> nodes;
  std::vector<PetscInt> stencilIndices;
  std::array<std::vector<PetscReal>, 3> weights;

  [[nodiscard]] PetscInt nodeCount() const noexcept { return N + 1; }
  [[nodiscard]] PetscInt stencilSize() const noexcept { return q + 1; }

  [[nodiscard]] std::size_t flatIndex(PetscInt row,
                                      PetscInt slot) const noexcept {
    return static_cast<std::size_t>(row) *
               static_cast<std::size_t>(stencilSize()) +
           static_cast<std::size_t>(slot);
  }

  [[nodiscard]] PetscInt stencilIndex(PetscInt row,
                                      PetscInt slot) const noexcept {
    return stencilIndices[flatIndex(row, slot)];
  }

  [[nodiscard]] PetscReal weight(PetscInt derivative, PetscInt row,
                                 PetscInt slot) const noexcept {
    return weights[static_cast<std::size_t>(derivative)][flatIndex(row, slot)];
  }

  void clear() noexcept;
};

struct ProblemData {
  ProblemData() = default;
  ~ProblemData();

  ProblemData(const ProblemData &) = delete;
  ProblemData &operator=(const ProblemData &) = delete;
  ProblemData(ProblemData &&) = delete;
  ProblemData &operator=(ProblemData &&) = delete;

  PetscErrorCode destroy();

  PetscInt ny = 0;
  PetscInt nz = 0;
  PetscInt stencilWidth = 0;

  DM fieldDM = nullptr;
  DM gridDM = nullptr;
  DM metricDM = nullptr;
  DM baseflowDerivativeDM = nullptr;
  Vec baseflow = nullptr;
  Vec grid = nullptr;
  Vec metrics = nullptr;
  Vec baseflowDerivatives = nullptr;
  Mat massMatrix = nullptr;
  Mat spatialMatrix = nullptr;

  // The HDF5 y rule differentiates in computational xi; the z rule in eta.
  FDQRuleData xiRule;
  FDQRuleData etaRule;
};
