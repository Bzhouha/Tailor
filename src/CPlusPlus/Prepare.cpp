#include "Prepare.hpp"

#include <hdf5.h>
#include <petscviewerhdf5.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class HDF5Handle {
public:
  HDF5Handle(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {}
  ~HDF5Handle() {
    if (id_ >= 0)
      (void)close_(id_);
  }

  HDF5Handle(const HDF5Handle &) = delete;
  HDF5Handle &operator=(const HDF5Handle &) = delete;

  [[nodiscard]] hid_t get() const noexcept { return id_; }

private:
  hid_t id_;
  herr_t (*close_)(hid_t);
};

std::size_t checkedElementCount(const std::vector<hsize_t> &dimensions,
                                const std::string &path) {
  std::size_t count = 1;
  for (const hsize_t dimension : dimensions) {
    if (dimension >
        static_cast<hsize_t>(std::numeric_limits<std::size_t>::max() / count)) {
      throw std::runtime_error("Dataset is too large: " + path);
    }
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

void readDataset(hid_t file, const std::string &path,
                 const std::vector<hsize_t> &expectedDimensions,
                 hid_t memoryType, void *values) {
  const hid_t datasetId = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  if (datasetId < 0)
    throw std::runtime_error("Missing HDF5 dataset: " + path);
  HDF5Handle dataset(datasetId, H5Dclose);

  const hid_t dataspaceId = H5Dget_space(dataset.get());
  if (dataspaceId < 0) {
    throw std::runtime_error("Cannot read HDF5 dataspace: " + path);
  }
  HDF5Handle dataspace(dataspaceId, H5Sclose);

  const int rank = H5Sget_simple_extent_ndims(dataspace.get());
  if (rank < 0 || static_cast<std::size_t>(rank) != expectedDimensions.size()) {
    throw std::runtime_error("Unexpected rank for HDF5 dataset: " + path);
  }

  std::vector<hsize_t> actualDimensions(expectedDimensions.size());
  if (H5Sget_simple_extent_dims(dataspace.get(), actualDimensions.data(),
                                nullptr) < 0) {
    throw std::runtime_error("Cannot read HDF5 dimensions: " + path);
  }
  if (actualDimensions != expectedDimensions) {
    throw std::runtime_error("Unexpected shape for HDF5 dataset: " + path);
  }

  (void)checkedElementCount(expectedDimensions, path);
  if (H5Dread(dataset.get(), memoryType, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              values) < 0) {
    throw std::runtime_error("Cannot read HDF5 dataset: " + path);
  }
}

std::string readStringAttribute(hid_t file, const char *objectPath,
                                const char *attributeName) {
  const hid_t objectId = H5Oopen(file, objectPath, H5P_DEFAULT);
  if (objectId < 0) {
    throw std::runtime_error(std::string("Missing HDF5 object: ") + objectPath);
  }
  HDF5Handle object(objectId, H5Oclose);

  const hid_t attributeId = H5Aopen(object.get(), attributeName, H5P_DEFAULT);
  if (attributeId < 0) {
    throw std::runtime_error(std::string("Missing HDF5 attribute: ") +
                             objectPath + "/" + attributeName);
  }
  HDF5Handle attribute(attributeId, H5Aclose);

  const hid_t typeId = H5Aget_type(attribute.get());
  if (typeId < 0) {
    throw std::runtime_error(std::string("Cannot inspect HDF5 attribute: ") +
                             objectPath + "/" + attributeName);
  }
  HDF5Handle type(typeId, H5Tclose);

  if (H5Tis_variable_str(type.get()) > 0) {
    char *rawValue = nullptr;
    if (H5Aread(attribute.get(), type.get(), &rawValue) < 0 ||
        rawValue == nullptr) {
      throw std::runtime_error(std::string("Cannot read HDF5 attribute: ") +
                               objectPath + "/" + attributeName);
    }
    const std::string value(rawValue);
    (void)H5free_memory(rawValue);
    return value;
  }

  const std::size_t size = H5Tget_size(type.get());
  std::vector<char> rawValue(size + 1, '\0');
  if (H5Aread(attribute.get(), type.get(), rawValue.data()) < 0) {
    throw std::runtime_error(std::string("Cannot read HDF5 attribute: ") +
                             objectPath + "/" + attributeName);
  }
  return rawValue.data();
}

void resizeRule(FDQRuleData &rule, PetscInt N, PetscInt q) {
  const std::size_t nodeCount = static_cast<std::size_t>(N + 1);
  const std::size_t stencilSize = static_cast<std::size_t>(q + 1);
  const std::size_t entryCount = nodeCount * stencilSize;
  rule.N = N;
  rule.q = q;
  rule.nodes.resize(nodeCount);
  rule.stencilIndices.resize(entryCount);
  for (auto &derivativeWeights : rule.weights) {
    derivativeWeights.resize(entryCount);
  }
}

void readRuleOnRoot(hid_t file, const char *group, FDQRuleData &rule) {
  const hsize_t nodeCount = static_cast<hsize_t>(rule.nodeCount());
  const hsize_t stencilSize = static_cast<hsize_t>(rule.stencilSize());
  const std::vector<hsize_t> nodeDimensions{nodeCount};
  const std::vector<hsize_t> stencilDimensions{nodeCount, stencilSize};
  const std::string prefix = std::string("/discretization/") + group;
  const std::size_t entryCount = checkedElementCount(stencilDimensions, prefix);

  std::vector<double> rawNodes(static_cast<std::size_t>(nodeCount));
  std::vector<std::int64_t> rawIndices(entryCount);
  std::array<std::vector<double>, 3> rawWeights;
  for (auto &weights : rawWeights) {
    weights.resize(entryCount);
  }

  readDataset(file, prefix + "/nodes", nodeDimensions, H5T_NATIVE_DOUBLE,
              rawNodes.data());
  readDataset(file, prefix + "/stencil_indices", stencilDimensions,
              H5T_NATIVE_INT64, rawIndices.data());
  for (std::size_t derivative = 0; derivative < rawWeights.size();
       ++derivative) {
    readDataset(file, prefix + "/weights/d" + std::to_string(derivative),
                stencilDimensions, H5T_NATIVE_DOUBLE,
                rawWeights[derivative].data());
  }

  for (std::size_t index = 0; index < rawNodes.size(); ++index) {
    rule.nodes[index] = static_cast<PetscReal>(rawNodes[index]);
  }
  for (std::size_t index = 0; index < rawIndices.size(); ++index) {
    const std::int64_t value = rawIndices[index];
    if (value < static_cast<std::int64_t>(PETSC_INT_MIN) ||
        value > static_cast<std::int64_t>(PETSC_INT_MAX)) {
      throw std::runtime_error("FD-q stencil index does not fit PetscInt");
    }
    rule.stencilIndices[index] = static_cast<PetscInt>(value);
  }
  for (std::size_t derivative = 0; derivative < rawWeights.size();
       ++derivative) {
    for (std::size_t index = 0; index < rawWeights[derivative].size();
         ++index) {
      rule.weights[derivative][index] =
          static_cast<PetscReal>(rawWeights[derivative][index]);
    }
  }
}

PetscErrorCode broadcastString(MPI_Comm comm, PetscMPIInt root,
                               std::string &value) {
  PetscMPIInt rank = 0;
  PetscMPIInt length = 0;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == root) {
    PetscCall(PetscMPIIntCast(static_cast<MPIU_Count>(value.size()), &length));
  }
  PetscCallMPI(MPI_Bcast(&length, 1, MPI_INT, root, comm));
  if (rank != root)
    value.resize(static_cast<std::size_t>(length));
  if (length > 0) {
    PetscCallMPI(MPI_Bcast(value.data(), length, MPI_CHAR, root, comm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode broadcastRule(MPI_Comm comm, PetscMPIInt root,
                             FDQRuleData &rule) {
  PetscMPIInt nodeCount = 0;
  PetscMPIInt entryCount = 0;

  PetscFunctionBeginUser;
  PetscCall(
      PetscMPIIntCast(static_cast<MPIU_Count>(rule.nodes.size()), &nodeCount));
  PetscCall(PetscMPIIntCast(static_cast<MPIU_Count>(rule.stencilIndices.size()),
                            &entryCount));
  PetscCallMPI(MPI_Bcast(rule.nodes.data(), nodeCount, MPIU_REAL, root, comm));
  PetscCallMPI(
      MPI_Bcast(rule.stencilIndices.data(), entryCount, MPIU_INT, root, comm));
  for (auto &derivativeWeights : rule.weights) {
    PetscCallMPI(
        MPI_Bcast(derivativeWeights.data(), entryCount, MPIU_REAL, root, comm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal polynomialDerivative(PetscInt power, PetscInt derivative,
                               PetscReal location) {
  if (power < derivative)
    return 0.0;
  PetscReal coefficient = 1.0;
  for (PetscInt factor = 0; factor < derivative; ++factor) {
    coefficient *= static_cast<PetscReal>(power - factor);
  }
  return coefficient *
         PetscPowReal(location, static_cast<PetscReal>(power - derivative));
}

PetscErrorCode validateRule(MPI_Comm comm, const char *name,
                            const FDQRuleData &rule) {
  const PetscInt nodeCount = rule.nodeCount();
  const PetscInt stencilSize = rule.stencilSize();
  const std::size_t expectedEntries = static_cast<std::size_t>(nodeCount) *
                                      static_cast<std::size_t>(stencilSize);
  constexpr PetscReal momentTolerance = 5.0e-9;

  PetscFunctionBeginUser;
  PetscCheck(rule.N >= 2 && rule.q >= 2 && rule.q <= rule.N, comm,
             PETSC_ERR_FILE_UNEXPECTED,
             "%s FD-q metadata must satisfy N >= q >= 2", name);
  PetscCheck(rule.nodes.size() == static_cast<std::size_t>(nodeCount), comm,
             PETSC_ERR_FILE_UNEXPECTED, "%s FD-q nodes have the wrong size",
             name);
  PetscCheck(rule.stencilIndices.size() == expectedEntries, comm,
             PETSC_ERR_FILE_UNEXPECTED,
             "%s FD-q stencil indices have the wrong size", name);
  for (const auto &derivativeWeights : rule.weights) {
    PetscCheck(derivativeWeights.size() == expectedEntries, comm,
               PETSC_ERR_FILE_UNEXPECTED, "%s FD-q weights have the wrong size",
               name);
  }

  for (PetscInt row = 0; row < nodeCount; ++row) {
    PetscCheck(!PetscIsInfOrNanReal(rule.nodes[static_cast<std::size_t>(row)]),
               comm, PETSC_ERR_FILE_UNEXPECTED,
               "%s FD-q node %" PetscInt_FMT " is not finite", name, row);
    if (row > 0) {
      PetscCheck(rule.nodes[static_cast<std::size_t>(row)] >
                     rule.nodes[static_cast<std::size_t>(row - 1)],
                 comm, PETSC_ERR_FILE_UNEXPECTED,
                 "%s FD-q nodes must be strictly increasing", name);
    }

    PetscBool containsRow = PETSC_FALSE;
    for (PetscInt slot = 0; slot < stencilSize; ++slot) {
      const PetscInt column = rule.stencilIndex(row, slot);
      PetscCheck(column >= 0 && column < nodeCount, comm,
                 PETSC_ERR_FILE_UNEXPECTED,
                 "%s FD-q row %" PetscInt_FMT
                 " contains out-of-range index %" PetscInt_FMT,
                 name, row, column);
      if (column == row)
        containsRow = PETSC_TRUE;
      for (PetscInt derivative = 0; derivative <= 2; ++derivative) {
        PetscCheck(!PetscIsInfOrNanReal(rule.weight(derivative, row, slot)),
                   comm, PETSC_ERR_FILE_UNEXPECTED,
                   "%s FD-q weight is not finite at row %" PetscInt_FMT, name,
                   row);
      }
    }
    PetscCheck(containsRow, comm, PETSC_ERR_FILE_UNEXPECTED,
               "%s FD-q stencil row %" PetscInt_FMT
               " does not contain its evaluation node",
               name, row);

    for (PetscInt derivative = 0; derivative <= 2; ++derivative) {
      for (PetscInt power = 0; power <= 2; ++power) {
        PetscReal approximation = 0.0;
        for (PetscInt slot = 0; slot < stencilSize; ++slot) {
          const PetscInt column = rule.stencilIndex(row, slot);
          approximation +=
              rule.weight(derivative, row, slot) *
              PetscPowReal(rule.nodes[static_cast<std::size_t>(column)],
                           static_cast<PetscReal>(power));
        }
        const PetscReal expected = polynomialDerivative(
            power, derivative, rule.nodes[static_cast<std::size_t>(row)]);
        const PetscReal scale = PetscMax(1.0, PetscAbsReal(expected));
        PetscCheck(
            PetscAbsReal(approximation - expected) <= momentTolerance * scale,
            comm, PETSC_ERR_FILE_UNEXPECTED,
            "%s FD-q order-%" PetscInt_FMT " weights fail degree-%" PetscInt_FMT
            " polynomial check at row %" PetscInt_FMT,
            name, derivative, power, row);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

Prepare::Prepare(MPI_Comm comm) : comm_(comm) {}

PetscErrorCode Prepare::readMetadata(PetscViewer viewer, const Recipe &recipe,
                                     ProblemData &data) const {
  constexpr PetscMPIInt root = 0;
  PetscMPIInt rank = 0;
  PetscMPIInt loaded = 1;
  PetscInt schemaVersion = 0;
  PetscInt intervalY = 0;
  PetscInt intervalZ = 0;
  PetscInt qY = 0;
  PetscInt qZ = 0;
  std::string ordering;
  std::string errorMessage;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm_, &rank));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "Ny", PETSC_INT, nullptr,
                                         &data.ny));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "Nz", PETSC_INT, nullptr,
                                         &data.nz));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "schema_version",
                                         PETSC_INT, nullptr, &schemaVersion));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "N_y", PETSC_INT, nullptr,
                                         &intervalY));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "N_z", PETSC_INT, nullptr,
                                         &intervalZ));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "q_y", PETSC_INT, nullptr,
                                         &qY));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "q_z", PETSC_INT, nullptr,
                                         &qZ));

  if (rank == root) {
    try {
      hid_t file = -1;
      const PetscErrorCode ierr = PetscViewerHDF5GetFileId(viewer, &file);
      if (ierr != PETSC_SUCCESS || file < 0) {
        throw std::runtime_error("Cannot access the PETSc HDF5 file handle");
      }
      ordering = readStringAttribute(file, "/", "ordering");
    } catch (const std::exception &error) {
      loaded = 0;
      errorMessage = error.what();
    }
  }
  PetscCallMPI(MPI_Bcast(&loaded, 1, MPI_INT, root, comm_));
  PetscCall(broadcastString(comm_, root, errorMessage));
  PetscCheck(loaded, comm_, PETSC_ERR_FILE_UNEXPECTED,
             "Failed to read FD-q metadata: %s", errorMessage.c_str());
  PetscCall(broadcastString(comm_, root, ordering));

  PetscCheck(data.ny > 0 && data.nz > 0, comm_, PETSC_ERR_FILE_UNEXPECTED,
             "HDF5 attributes Ny and Nz must be positive");
  PetscCheck(schemaVersion == 1, comm_, PETSC_ERR_FILE_UNEXPECTED,
             "Unsupported FD-q schema version %" PetscInt_FMT, schemaVersion);
  PetscCheck(ordering == "k_j_dof", comm_, PETSC_ERR_FILE_UNEXPECTED,
             "FD-q ordering must be k_j_dof");
  PetscCheck(intervalY == data.ny - 1 && intervalZ == data.nz - 1, comm_,
             PETSC_ERR_FILE_UNEXPECTED,
             "FD-q interval counts must satisfy N_y=Ny-1 and N_z=Nz-1");
  PetscCheck(qY == static_cast<PetscInt>(recipe.qY) &&
                 qZ == static_cast<PetscInt>(recipe.qZ),
             comm_, PETSC_ERR_FILE_UNEXPECTED,
             "FD-q q_y/q_z attributes do not match the YAML configuration");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Prepare::createDMs(const Recipe &recipe,
                                  ProblemData &data) const {
  const PetscInt qY = static_cast<PetscInt>(recipe.qY);
  const PetscInt qZ = static_cast<PetscInt>(recipe.qZ);

  PetscFunctionBeginUser;
  PetscCheck(qY <= data.ny - 1, comm_, PETSC_ERR_ARG_OUTOFRANGE,
             "Q-Value.y (%" PetscInt_FMT
             ") must not exceed Ny-1 (%" PetscInt_FMT ")",
             qY, data.ny - 1);
  PetscCheck(qZ <= data.nz - 1, comm_, PETSC_ERR_ARG_OUTOFRANGE,
             "Q-Value.z (%" PetscInt_FMT
             ") must not exceed Nz-1 (%" PetscInt_FMT ")",
             qZ, data.nz - 1);

  // A degree-q one-sided Fornberg stencil contains q+1 points and reaches q
  // points from a boundary. DMDA uses one box width for both directions.
  data.stencilWidth = std::max(qY, qZ);
  PetscCall(DMDACreate2d(comm_, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX, data.ny, data.nz, PETSC_DECIDE,
                         PETSC_DECIDE, 5, data.stencilWidth, nullptr, nullptr,
                         &data.fieldDM));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(data.fieldDM),
                               "field_dm"));
  PetscCall(DMSetUp(data.fieldDM));

  PetscCall(DMDACreateCompatibleDMDA(data.fieldDM, 3, &data.gridDM));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(data.gridDM),
                               "grid_dm"));
  PetscCall(DMSetUp(data.gridDM));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Prepare::loadNaturalVector(PetscViewer viewer, DM dm,
                                          const char *name, Vec *global) const {
  Vec natural = nullptr;

  PetscFunctionBeginUser;
  PetscCheck(global, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "Output Vec pointer must not be null");
  PetscCall(DMDACreateNaturalVector(dm, &natural));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(natural), name));
  PetscCall(VecLoad(natural, viewer));

  PetscCall(DMCreateGlobalVector(dm, global));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(*global), name));
  PetscCall(DMDANaturalToGlobalBegin(dm, natural, INSERT_VALUES, *global));
  PetscCall(DMDANaturalToGlobalEnd(dm, natural, INSERT_VALUES, *global));
  PetscCall(VecDestroy(&natural));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Prepare::loadFields(PetscViewer viewer,
                                   ProblemData &data) const {
  PetscBool hasBaseflow = PETSC_FALSE;
  PetscBool hasGrid = PETSC_FALSE;
  PetscInt baseflowSize = 0;
  PetscInt gridSize = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerHDF5HasDataset(viewer, "baseflow", &hasBaseflow));
  PetscCall(PetscViewerHDF5HasDataset(viewer, "grid", &hasGrid));
  PetscCheck(hasBaseflow, comm_, PETSC_ERR_FILE_UNEXPECTED,
             "HDF5 dataset /baseflow is missing");
  PetscCheck(hasGrid, comm_, PETSC_ERR_FILE_UNEXPECTED,
             "HDF5 dataset /grid is missing");

  PetscCall(
      loadNaturalVector(viewer, data.fieldDM, "baseflow", &data.baseflow));
  PetscCall(loadNaturalVector(viewer, data.gridDM, "grid", &data.grid));
  PetscCall(VecGetSize(data.baseflow, &baseflowSize));
  PetscCall(VecGetSize(data.grid, &gridSize));
  PetscCheck(baseflowSize == data.ny * data.nz * 5, comm_,
             PETSC_ERR_FILE_UNEXPECTED,
             "Unexpected baseflow size: expected %" PetscInt_FMT
             ", got %" PetscInt_FMT,
             data.ny * data.nz * 5, baseflowSize);
  PetscCheck(
      gridSize == data.ny * data.nz * 3, comm_, PETSC_ERR_FILE_UNEXPECTED,
      "Unexpected grid size: expected %" PetscInt_FMT ", got %" PetscInt_FMT,
      data.ny * data.nz * 3, gridSize);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Prepare::loadDiscretization(PetscViewer viewer,
                                           const Recipe &recipe,
                                           ProblemData &data) const {
  constexpr PetscMPIInt root = 0;
  PetscMPIInt rank = 0;
  PetscMPIInt loaded = 1;
  PetscInt yN = 0;
  PetscInt yQ = 0;
  PetscInt zN = 0;
  PetscInt zQ = 0;
  std::string errorMessage;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm_, &rank));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/discretization/y", "N",
                                         PETSC_INT, nullptr, &yN));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/discretization/y", "q",
                                         PETSC_INT, nullptr, &yQ));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/discretization/z", "N",
                                         PETSC_INT, nullptr, &zN));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/discretization/z", "q",
                                         PETSC_INT, nullptr, &zQ));
  PetscCheck(yN == data.ny - 1 && yQ == static_cast<PetscInt>(recipe.qY), comm_,
             PETSC_ERR_FILE_UNEXPECTED,
             "The /discretization/y metadata is inconsistent");
  PetscCheck(zN == data.nz - 1 && zQ == static_cast<PetscInt>(recipe.qZ), comm_,
             PETSC_ERR_FILE_UNEXPECTED,
             "The /discretization/z metadata is inconsistent");

  resizeRule(data.xiRule, yN, yQ);
  resizeRule(data.etaRule, zN, zQ);

  if (rank == root) {
    try {
      hid_t file = -1;
      const PetscErrorCode ierr = PetscViewerHDF5GetFileId(viewer, &file);
      if (ierr != PETSC_SUCCESS || file < 0) {
        throw std::runtime_error("Cannot access the PETSc HDF5 file handle");
      }
      readRuleOnRoot(file, "y", data.xiRule);
      readRuleOnRoot(file, "z", data.etaRule);
    } catch (const std::exception &error) {
      loaded = 0;
      errorMessage = error.what();
    }
  }

  PetscCallMPI(MPI_Bcast(&loaded, 1, MPI_INT, root, comm_));
  PetscCall(broadcastString(comm_, root, errorMessage));
  PetscCheck(loaded, comm_, PETSC_ERR_FILE_UNEXPECTED,
             "Failed to load FD-q discretization: %s", errorMessage.c_str());

  PetscCall(broadcastRule(comm_, root, data.xiRule));
  PetscCall(broadcastRule(comm_, root, data.etaRule));
  PetscCall(validateRule(comm_, "xi", data.xiRule));
  PetscCall(validateRule(comm_, "eta", data.etaRule));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Prepare::initialize(const Recipe &recipe,
                                   ProblemData &data) const {
  PetscViewer viewer = nullptr;

  PetscFunctionBeginUser;
  PetscCheck(!data.fieldDM && !data.gridDM && !data.baseflow && !data.grid,
             comm_, PETSC_ERR_ARG_WRONGSTATE,
             "ProblemData must be empty before Prepare::initialize");
  PetscCall(PetscViewerHDF5Open(comm_, recipe.inputFile.c_str(), FILE_MODE_READ,
                                &viewer));
  PetscCall(PetscViewerHDF5SetCollective(viewer, PETSC_TRUE));
  PetscCall(readMetadata(viewer, recipe, data));
  PetscCall(createDMs(recipe, data));
  PetscCall(loadFields(viewer, data));
  PetscCall(loadDiscretization(viewer, recipe, data));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
