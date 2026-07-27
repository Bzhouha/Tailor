#include "Parser.hpp"

#include <yaml-cpp/yaml.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

template <typename T> T required(const YAML::Node &node, const char *key) {
  const YAML::Node value = node[key];
  if (!value)
    throw std::runtime_error(std::string("Missing YAML key: ") + key);
  return value.as<T>();
}

void requireFinite(double value, const char *name) {
  if (!std::isfinite(value)) {
    throw std::runtime_error(std::string(name) + " must be finite");
  }
}

} // namespace

Parser::Parser(MPI_Comm comm) : comm_(comm) {}

Recipe Parser::parseOnRoot(const std::string &yamlConfig) {
  const YAML::Node root = YAML::LoadFile(yamlConfig);
  Recipe recipe;

  recipe.caseTitle = required<std::string>(root, "CaseTitle");
  const YAML::Node qValue = root["Q-Value"];
  if (!qValue || !qValue.IsMap()) {
    throw std::runtime_error("Missing or invalid YAML mapping: Q-Value");
  }
  recipe.qY = required<int>(qValue, "y");
  recipe.qZ = required<int>(qValue, "z");

  const std::filesystem::path configPath =
      std::filesystem::absolute(yamlConfig).lexically_normal();
  const std::filesystem::path folder = required<std::string>(root, "Folder");
  const std::filesystem::path file = required<std::string>(root, "File");
  const std::filesystem::path source =
      file.is_absolute() ? file : configPath.parent_path() / folder / file;
  const std::filesystem::path normalizedSource = source.lexically_normal();
  const std::string cacheName = "fdq_" + normalizedSource.stem().string() +
                                "_qy" + std::to_string(recipe.qY) + "_qz" +
                                std::to_string(recipe.qZ) + ".h5";
  const std::filesystem::path input =
      normalizedSource.parent_path() / "FD-q" / cacheName;
  recipe.sourceFile = normalizedSource.string();
  recipe.inputFile = input.lexically_normal().string();

  const YAML::Node alpha = root["Stability"]["Alpha"];
  if (!alpha)
    throw std::runtime_error("Missing YAML key: Stability.Alpha");
  recipe.alpha = {required<double>(alpha, "Real"),
                  required<double>(alpha, "Imag")};

  const YAML::Node physics = root["Physics"];
  const YAML::Node targets = physics["Targets"];
  const YAML::Node gas = physics["Gas"];
  const YAML::Node transport = physics["Transport"];
  recipe.reynolds = required<double>(targets, "ReynoldsNumber");
  recipe.mach = required<double>(targets, "MachNumber");
  recipe.prandtl = required<double>(gas, "PrandtlNumber");
  recipe.gasConstant = required<double>(gas, "GasConstant");
  recipe.ratioOfSpecificHeats = required<double>(gas, "RatioOfSpecificHeats");
  recipe.referenceViscosity = required<double>(transport, "ReferenceMu");
  recipe.referenceTemperature =
      required<double>(transport, "ReferenceTemperature");
  recipe.sutherlandConstant = required<double>(transport, "SutherlandConstant");

  if (recipe.qY < 2 || recipe.qZ < 2) {
    throw std::runtime_error("Q-Value.y and Q-Value.z must both be at least 2");
  }
  if (!std::filesystem::is_regular_file(recipe.inputFile)) {
    throw std::runtime_error(
        "FD-q input file does not exist: " + recipe.inputFile +
        ". Run Tailor.py -c " + configPath.string() + " first");
  }
  requireFinite(recipe.alpha.real(), "Stability.Alpha.Real");
  requireFinite(recipe.alpha.imag(), "Stability.Alpha.Imag");
  requireFinite(recipe.reynolds, "ReynoldsNumber");
  requireFinite(recipe.mach, "MachNumber");
  requireFinite(recipe.prandtl, "PrandtlNumber");
  requireFinite(recipe.gasConstant, "GasConstant");
  requireFinite(recipe.ratioOfSpecificHeats, "RatioOfSpecificHeats");
  requireFinite(recipe.referenceViscosity, "ReferenceMu");
  requireFinite(recipe.referenceTemperature, "ReferenceTemperature");
  requireFinite(recipe.sutherlandConstant, "SutherlandConstant");

  if (recipe.reynolds <= 0.0 || recipe.mach <= 0.0 || recipe.prandtl <= 0.0 ||
      recipe.gasConstant <= 0.0 || recipe.ratioOfSpecificHeats <= 1.0 ||
      recipe.referenceViscosity <= 0.0 || recipe.referenceTemperature <= 0.0 ||
      recipe.sutherlandConstant < 0.0) {
    throw std::runtime_error("The physical parameters in config.yaml are "
                             "outside their valid ranges");
  }

  return recipe;
}

PetscErrorCode Parser::broadcastString(std::string &value) const {
  PetscMPIInt rank = 0;
  PetscMPIInt length = 0;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm_, &rank));
  if (rank == 0) {
    PetscCheck(value.size() <= static_cast<std::size_t>(
                                   std::numeric_limits<PetscMPIInt>::max()),
               comm_, PETSC_ERR_ARG_SIZ, "String is too long to broadcast");
    length = static_cast<PetscMPIInt>(value.size());
  }
  PetscCallMPI(MPI_Bcast(&length, 1, MPI_INT, 0, comm_));
  if (rank != 0)
    value.resize(static_cast<std::size_t>(length));
  if (length > 0) {
    PetscCallMPI(MPI_Bcast(value.data(), length, MPI_CHAR, 0, comm_));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Parser::broadcast(Recipe &recipe) const {
  std::array<double, 10> values{};
  PetscMPIInt rank = 0;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm_, &rank));
  if (rank == 0) {
    values = {recipe.alpha.real(),
              recipe.alpha.imag(),
              recipe.mach,
              recipe.reynolds,
              recipe.prandtl,
              recipe.gasConstant,
              recipe.ratioOfSpecificHeats,
              recipe.referenceViscosity,
              recipe.referenceTemperature,
              recipe.sutherlandConstant};
  }
  PetscCallMPI(MPI_Bcast(values.data(), static_cast<PetscMPIInt>(values.size()),
                         MPI_DOUBLE, 0, comm_));
  std::array<int, 2> qValues{recipe.qY, recipe.qZ};
  PetscCallMPI(MPI_Bcast(qValues.data(),
                         static_cast<PetscMPIInt>(qValues.size()), MPI_INT, 0,
                         comm_));
  PetscCall(broadcastString(recipe.caseTitle));
  PetscCall(broadcastString(recipe.sourceFile));
  PetscCall(broadcastString(recipe.inputFile));

  if (rank != 0) {
    recipe.qY = qValues[0];
    recipe.qZ = qValues[1];
    recipe.alpha = {values[0], values[1]};
    recipe.mach = values[2];
    recipe.reynolds = values[3];
    recipe.prandtl = values[4];
    recipe.gasConstant = values[5];
    recipe.ratioOfSpecificHeats = values[6];
    recipe.referenceViscosity = values[7];
    recipe.referenceTemperature = values[8];
    recipe.sutherlandConstant = values[9];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Parser::parse(const std::string &yamlConfig,
                             Recipe &recipe) const {
  PetscMPIInt rank = 0;
  PetscMPIInt parsed = 1;
  std::string errorMessage;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm_, &rank));
  if (rank == 0) {
    try {
      recipe = parseOnRoot(yamlConfig);
    } catch (const std::exception &error) {
      parsed = 0;
      errorMessage = error.what();
    }
  }

  PetscCallMPI(MPI_Bcast(&parsed, 1, MPI_INT, 0, comm_));
  PetscCall(broadcastString(errorMessage));
  PetscCheck(parsed, comm_, PETSC_ERR_FILE_UNEXPECTED, "Failed to parse %s: %s",
             yamlConfig.c_str(), errorMessage.c_str());
  PetscCall(broadcast(recipe));
  PetscFunctionReturn(PETSC_SUCCESS);
}
