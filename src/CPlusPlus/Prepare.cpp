/**
 * @file Prepare.cpp
 * @brief Collective HDF5 loading and FD-q schema validation.
 *
 * PETSc-formatted complex fields are loaded collectively. Ordinary HDF5 rule
 * arrays are read through the viewer file identifier on rank zero and then
 * broadcast to every process.
 */
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

/**
 * @brief Scope-bound owner for an HDF5 identifier.
 *
 * The wrapper closes the identifier with the matching HDF5 close routine when
 * control leaves the current scope.
 */
class HDF5Handle {
public:
  /** @brief Adopt an HDF5 identifier and its close function. */
  HDF5Handle(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {}
  /** @brief Close the adopted identifier when it is valid. */
  ~HDF5Handle() {
    if (id_ >= 0)
      (void)close_(id_);
  }

  HDF5Handle(const HDF5Handle &) = delete;
  HDF5Handle &operator=(const HDF5Handle &) = delete;

  /** @brief Return the borrowed raw HDF5 identifier. */
  [[nodiscard]] hid_t get() const noexcept { return id_; }

private:
  hid_t id_;               ///< Adopted HDF5 identifier.
  herr_t (*close_)(hid_t); ///< Matching HDF5 close routine.
};

/**
 * @brief Compute a dataset element count while checking for size overflow.
 * @param dimensions Dataset dimensions.
 * @param path Dataset path used in diagnostics.
 * @return Product of all dimensions.
 */
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

/**
 * @brief Read an HDF5 dataset after verifying its exact shape.
 * @param file Open HDF5 file identifier.
 * @param path Absolute dataset path.
 * @param expectedDimensions Required dataset dimensions.
 * @param memoryType HDF5 type of the destination buffer.
 * @param values Destination buffer.
 */
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

/**
 * @brief Read a fixed- or variable-length string attribute.
 * @param file Open HDF5 file identifier.
 * @param objectPath HDF5 object containing the attribute.
 * @param attributeName Attribute name.
 * @return Attribute value.
 */
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

/**
 * @brief Resize and initialize one FD-q rule from its metadata.
 * @param rule Rule to initialize.
 * @param nodeCountValue Number of unique nodes.
 * @param q Polynomial degree; the stencil contains `q + 1` entries.
 * @param topology Bounded or periodic topology.
 * @param period Coordinate period or bounded coordinate span.
 */
void resizeRule(FDQRuleData &rule, PetscInt nodeCountValue, PetscInt q,
                FDQTopology topology, PetscReal period) {
  const std::size_t nodeCount = static_cast<std::size_t>(nodeCountValue);
  const std::size_t stencilSize = static_cast<std::size_t>(q + 1);
  const std::size_t entryCount = nodeCount * stencilSize;
  rule.nodeCountValue = nodeCountValue;
  rule.q = q;
  rule.topology = topology;
  rule.period = period;
  rule.nodes.resize(nodeCount);
  rule.stencilIndices.resize(entryCount);
  rule.stencilOffsets.resize(entryCount);
  for (auto &derivativeWeights : rule.weights) {
    derivativeWeights.resize(entryCount);
  }
}

/**
 * @brief Read one FD-q rule on the root MPI rank.
 * @param file Open HDF5 file identifier.
 * @param group Rule group name below `/discretization`.
 * @param rule Pre-sized destination rule.
 */
void readRuleOnRoot(hid_t file, const char *group, FDQRuleData &rule) {
  const hsize_t nodeCount = static_cast<hsize_t>(rule.nodeCount());
  const hsize_t stencilSize = static_cast<hsize_t>(rule.stencilSize());
  const std::vector<hsize_t> nodeDimensions{nodeCount};
  const std::vector<hsize_t> stencilDimensions{nodeCount, stencilSize};
  const std::string prefix = std::string("/discretization/") + group;
  const std::size_t entryCount = checkedElementCount(stencilDimensions, prefix);

  std::vector<double> rawNodes(static_cast<std::size_t>(nodeCount));
  std::vector<std::int64_t> rawIndices(entryCount);
  std::vector<std::int64_t> rawOffsets(entryCount);
  std::array<std::vector<double>, 3> rawWeights;
  for (auto &weights : rawWeights) {
    weights.resize(entryCount);
  }

  readDataset(file, prefix + "/nodes", nodeDimensions, H5T_NATIVE_DOUBLE,
              rawNodes.data());
  readDataset(file, prefix + "/stencil_indices", stencilDimensions,
              H5T_NATIVE_INT64, rawIndices.data());
  readDataset(file, prefix + "/stencil_offsets", stencilDimensions,
              H5T_NATIVE_INT64, rawOffsets.data());
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
    const std::int64_t offset = rawOffsets[index];
    if (offset < static_cast<std::int64_t>(PETSC_INT_MIN) ||
        offset > static_cast<std::int64_t>(PETSC_INT_MAX)) {
      throw std::runtime_error("FD-q stencil offset does not fit PetscInt");
    }
    rule.stencilOffsets[index] = static_cast<PetscInt>(offset);
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

/**
 * @brief Broadcast a variable-length string to all MPI ranks.
 * @param comm Collective communicator.
 * @param root Source rank.
 * @param value Source value on root and destination value elsewhere.
 */
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

/**
 * @brief Broadcast a pre-sized FD-q rule to all MPI ranks.
 * @param comm Collective communicator.
 * @param root Source rank.
 * @param rule Source rule on root and destination rule elsewhere.
 */
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
  PetscCallMPI(
      MPI_Bcast(rule.stencilOffsets.data(), entryCount, MPIU_INT, root, comm));
  for (auto &derivativeWeights : rule.weights) {
    PetscCallMPI(
        MPI_Bcast(derivativeWeights.data(), entryCount, MPIU_REAL, root, comm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Evaluate an analytic derivative of a coordinate monomial.
 * @param power Monomial power.
 * @param derivative Derivative order.
 * @param location Evaluation coordinate.
 * @return Derivative of `x^power` at `location`.
 */
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

/**
 * @brief Validate FD-q metadata, topology, stencils, and polynomial moments.
 * @param comm Collective communicator used for PETSc errors.
 * @param name Human-readable coordinate direction.
 * @param rule Rule to validate.
 */
PetscErrorCode validateRule(MPI_Comm comm, const char *name,
                            const FDQRuleData &rule) {
  const PetscInt nodeCount = rule.nodeCount();
  const PetscInt stencilSize = rule.stencilSize();
  const std::size_t expectedEntries = static_cast<std::size_t>(nodeCount) *
                                      static_cast<std::size_t>(stencilSize);
  constexpr PetscReal momentTolerance = 5.0e-9;

  PetscFunctionBeginUser;
  PetscCheck(nodeCount >= 3 && rule.q >= 2 && rule.q < nodeCount, comm,
             PETSC_ERR_FILE_UNEXPECTED,
             "%s FD-q metadata must satisfy node_count > q >= 2", name);
  PetscCheck(!PetscIsInfOrNanReal(rule.period) && rule.period > 0.0, comm,
             PETSC_ERR_FILE_UNEXPECTED,
             "%s FD-q period must be finite and positive", name);
  if (rule.topology == FDQTopology::Periodic)
    PetscCheck(rule.q % 2 == 0, comm, PETSC_ERR_FILE_UNEXPECTED,
               "%s periodic FD-q degree must be even", name);
  PetscCheck(rule.nodes.size() == static_cast<std::size_t>(nodeCount), comm,
             PETSC_ERR_FILE_UNEXPECTED, "%s FD-q nodes have the wrong size",
             name);
  PetscCheck(rule.stencilIndices.size() == expectedEntries, comm,
             PETSC_ERR_FILE_UNEXPECTED,
             "%s FD-q stencil indices have the wrong size", name);
  PetscCheck(rule.stencilOffsets.size() == expectedEntries, comm,
             PETSC_ERR_FILE_UNEXPECTED,
             "%s FD-q stencil offsets have the wrong size", name);
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
      const PetscInt offset = rule.stencilOffset(row, slot);
      PetscCheck(column >= 0 && column < nodeCount, comm,
                 PETSC_ERR_FILE_UNEXPECTED,
                 "%s FD-q row %" PetscInt_FMT
                 " contains out-of-range index %" PetscInt_FMT,
                 name, row, column);
      if (rule.topology == FDQTopology::Periodic) {
        PetscInt wrapped = (row + offset) % nodeCount;
        if (wrapped < 0)
          wrapped += nodeCount;
        PetscCheck(wrapped == column, comm, PETSC_ERR_FILE_UNEXPECTED,
                   "%s periodic row %" PetscInt_FMT
                   " has inconsistent stencil index/offset",
                   name, row);
      } else {
        PetscCheck(row + offset == column, comm, PETSC_ERR_FILE_UNEXPECTED,
                   "%s bounded row %" PetscInt_FMT
                   " has inconsistent stencil index/offset",
                   name, row);
      }
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
          const PetscReal sampleCoordinate =
              rule.topology == FDQTopology::Periodic
                  ? rule.nodes[static_cast<std::size_t>(row)] +
                        static_cast<PetscReal>(rule.stencilOffset(row, slot)) *
                            rule.period / static_cast<PetscReal>(nodeCount)
                  : rule.nodes[static_cast<std::size_t>(column)];
          approximation +=
              rule.weight(derivative, row, slot) *
              PetscPowReal(sampleCoordinate, static_cast<PetscReal>(power));
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
  PetscInt qY = 0;
  PetscInt qZ = 0;
  PetscInt spanwisePeriodic = 0;
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
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "q_y", PETSC_INT, nullptr,
                                         &qY));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "q_z", PETSC_INT, nullptr,
                                         &qZ));
  PetscCall(PetscViewerHDF5ReadAttribute(
      viewer, "/", "spanwise_periodic", PETSC_INT, nullptr, &spanwisePeriodic));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "spanwise_period",
                                         PETSC_REAL, nullptr,
                                         &data.spanwisePeriod));

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
  PetscCheck(schemaVersion == 2, comm_, PETSC_ERR_FILE_UNEXPECTED,
             "Unsupported FD-q schema version %" PetscInt_FMT, schemaVersion);
  PetscCheck(ordering == "k_j_dof", comm_, PETSC_ERR_FILE_UNEXPECTED,
             "FD-q ordering must be k_j_dof");
  PetscCheck(spanwisePeriodic == 1, comm_, PETSC_ERR_FILE_UNEXPECTED,
             "FD-q input must declare spanwise_periodic=1");
  PetscCheck(!PetscIsInfOrNanReal(data.spanwisePeriod) &&
                 data.spanwisePeriod > 0.0,
             comm_, PETSC_ERR_FILE_UNEXPECTED,
             "FD-q spanwise_period must be finite and positive");
  PetscCheck(qY == static_cast<PetscInt>(recipe.qY) &&
                 qZ == static_cast<PetscInt>(recipe.qZ),
             comm_, PETSC_ERR_FILE_UNEXPECTED,
             "FD-q q_y/q_z attributes do not match the YAML configuration");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Prepare::createDMs(const Recipe &recipe,
                                  ProblemData &data) const {
  PetscFunctionBeginUser;
  PetscCheck(data.xiRule.q == static_cast<PetscInt>(recipe.qY) &&
                 data.etaRule.q == static_cast<PetscInt>(recipe.qZ),
             comm_, PETSC_ERR_ARG_INCOMP,
             "Loaded FD-q degrees do not match the recipe");
  data.stencilWidth =
      std::max(data.xiRule.maxAbsOffset(), data.etaRule.maxAbsOffset());
  PetscCheck(data.stencilWidth > 0, comm_, PETSC_ERR_ARG_OUTOFRANGE,
             "FD-q stencil width must be positive");
  PetscCall(DMDACreate2d(comm_, DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC,
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
  PetscInt yNodeCount = 0;
  PetscInt yQ = 0;
  PetscInt zNodeCount = 0;
  PetscInt zQ = 0;
  PetscReal yPeriod = 0.0;
  PetscReal zPeriod = 0.0;
  std::string yTopology;
  std::string zTopology;
  std::string errorMessage;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm_, &rank));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/discretization/y",
                                         "node_count", PETSC_INT, nullptr,
                                         &yNodeCount));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/discretization/y", "q",
                                         PETSC_INT, nullptr, &yQ));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/discretization/z",
                                         "node_count", PETSC_INT, nullptr,
                                         &zNodeCount));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/discretization/z", "q",
                                         PETSC_INT, nullptr, &zQ));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/discretization/y", "period",
                                         PETSC_REAL, nullptr, &yPeriod));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/discretization/z", "period",
                                         PETSC_REAL, nullptr, &zPeriod));
  PetscCheck(yNodeCount == data.ny && yQ == static_cast<PetscInt>(recipe.qY),
             comm_, PETSC_ERR_FILE_UNEXPECTED,
             "The /discretization/y metadata is inconsistent");
  PetscCheck(zNodeCount == data.nz && zQ == static_cast<PetscInt>(recipe.qZ),
             comm_, PETSC_ERR_FILE_UNEXPECTED,
             "The /discretization/z metadata is inconsistent");

  if (rank == root) {
    try {
      hid_t file = -1;
      const PetscErrorCode ierr = PetscViewerHDF5GetFileId(viewer, &file);
      if (ierr != PETSC_SUCCESS || file < 0) {
        throw std::runtime_error("Cannot access the PETSc HDF5 file handle");
      }
      yTopology = readStringAttribute(file, "/discretization/y", "topology");
      zTopology = readStringAttribute(file, "/discretization/z", "topology");
      if (yTopology != "bounded" || zTopology != "periodic")
        throw std::runtime_error(
            "Expected bounded y and periodic z FD-q topologies");
      resizeRule(data.xiRule, yNodeCount, yQ, FDQTopology::Bounded, yPeriod);
      resizeRule(data.etaRule, zNodeCount, zQ, FDQTopology::Periodic, zPeriod);
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

  PetscCall(broadcastString(comm_, root, yTopology));
  PetscCall(broadcastString(comm_, root, zTopology));
  if (rank != root) {
    PetscCheck(yTopology == "bounded" && zTopology == "periodic", comm_,
               PETSC_ERR_FILE_UNEXPECTED,
               "Expected bounded y and periodic z FD-q topologies");
    resizeRule(data.xiRule, yNodeCount, yQ, FDQTopology::Bounded, yPeriod);
    resizeRule(data.etaRule, zNodeCount, zQ, FDQTopology::Periodic, zPeriod);
  }
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
  PetscCall(loadDiscretization(viewer, recipe, data));
  PetscCall(createDMs(recipe, data));
  PetscCall(loadFields(viewer, data));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
