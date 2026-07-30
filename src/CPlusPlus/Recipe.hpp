/**
 * @file Recipe.hpp
 * @brief Plain configuration data shared by all Tailor stages.
 */
#pragma once

#include <complex>
#include <string>

/**
 * @brief Validated, MPI-replicated case configuration.
 *
 * Recipe owns no PETSc resources. Dimensional transport constants are retained
 * alongside nondimensional stability parameters for pointwise coefficient
 * construction.
 */
struct Recipe {
  /** User-facing case label. */
  std::string caseTitle;
  /** Original source-flow HDF5 path. */
  std::string sourceFile;
  /** Prepared FD-q HDF5 path consumed by PETSc. */
  std::string inputFile;

  /** Polynomial degree in the computational xi direction. */
  int qY = 0;
  /** Polynomial degree in the computational eta direction. */
  int qZ = 0;

  /** Freestream Mach number. */
  double mach = 0.0;
  /** Reynolds number. */
  double reynolds = 0.0;
  /** Prandtl number. */
  double prandtl = 0.0;
  /** Specific gas constant. */
  double gasConstant = 0.0;
  /** Ratio of specific heats. */
  double ratioOfSpecificHeats = 0.0;
  /** Dimensional reference dynamic viscosity. */
  double referenceViscosity = 0.0;
  /** Dimensional reference temperature. */
  double referenceTemperature = 0.0;
  /** Dimensional Sutherland constant. */
  double sutherlandConstant = 0.0;

  /** Complex streamwise wavenumber in the Fourier ansatz. */
  std::complex<double> alpha{};

  /** Complex frequency whose mapped lambda value is the local search target. */
  std::complex<double> searchCenterOmega{};
  /** Number of eigenpairs requested from the local spectrum. */
  int numberOfEigenvalues = 0;
  /** Relative convergence tolerance for SLEPc. */
  double eigenTolerance = 0.0;
  /** Maximum number of Krylov iterations. */
  int eigenMaximumIterations = 0;

  /** Final self-contained HDF5 eigenmode file. */
  std::string outputFile;
};
