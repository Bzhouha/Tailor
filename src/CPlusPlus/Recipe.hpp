#pragma once

#include <complex>
#include <string>

struct Recipe {
  std::string caseTitle;
  std::string sourceFile;
  std::string inputFile;

  int qY = 0;
  int qZ = 0;

  double mach = 0.0;
  double reynolds = 0.0;
  double prandtl = 0.0;
  double gasConstant = 0.0;
  double ratioOfSpecificHeats = 0.0;
  double referenceViscosity = 0.0;
  double referenceTemperature = 0.0;
  double sutherlandConstant = 0.0;

  std::complex<double> alpha{};
};
