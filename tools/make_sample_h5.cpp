#include <hdf5.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(herr_t status, const char *operation) {
  if (status < 0) throw std::runtime_error(operation);
}

hid_t require_id(hid_t id, const char *operation) {
  if (id < 0) throw std::runtime_error(operation);
  return id;
}

void write_attribute(hid_t file, const char *name, std::int64_t value) {
  const hid_t space = require_id(H5Screate(H5S_SCALAR), "create attribute dataspace");
  const hid_t attribute = require_id(
      H5Acreate2(file, name, H5T_STD_I64LE, space, H5P_DEFAULT, H5P_DEFAULT),
      "create attribute");
  check(H5Awrite(attribute, H5T_NATIVE_INT64, &value), "write attribute");
  check(H5Aclose(attribute), "close attribute");
  check(H5Sclose(space), "close attribute dataspace");
}

std::array<hsize_t, 5> dataset_shape(hid_t dataset, const char *name) {
  const hid_t space = require_id(H5Dget_space(dataset), "get dataset dataspace");
  if (H5Sget_simple_extent_ndims(space) != 5) {
    H5Sclose(space);
    throw std::runtime_error(std::string(name) + " must have rank 5");
  }

  std::array<hsize_t, 5> shape{};
  check(H5Sget_simple_extent_dims(space, shape.data(), nullptr), "read dataset shape");
  check(H5Sclose(space), "close dataset dataspace");
  return shape;
}

void copy_i0_slice(hid_t input, hid_t output, const char *source_name,
                   const char *destination_name, hsize_t expected_kn,
                   hsize_t expected_jn) {
  const hid_t source = require_id(H5Dopen2(input, source_name, H5P_DEFAULT),
                                  "open source dataset");
  const auto shape = dataset_shape(source, source_name);
  if (shape[0] != expected_kn || shape[1] != expected_jn || shape[2] == 0) {
    H5Dclose(source);
    throw std::runtime_error(std::string(source_name) + " has inconsistent dimensions");
  }

  // Input layout: [k, j, i, dof, complex]. Select i = 0.
  const std::array<hsize_t, 5> offset{0, 0, 0, 0, 0};
  const std::array<hsize_t, 5> count{shape[0], shape[1], 1, shape[3], shape[4]};
  const hid_t source_space = require_id(H5Dget_space(source), "get source dataspace");
  check(H5Sselect_hyperslab(source_space, H5S_SELECT_SET, offset.data(), nullptr,
                           count.data(), nullptr),
        "select i=0 hyperslab");

  // Structured sample layout: [k, j, dof, complex parts].
  const std::array<hsize_t, 4> output_shape{shape[0], shape[1], shape[3], shape[4]};
  const hid_t output_space = require_id(
      H5Screate_simple(static_cast<int>(output_shape.size()), output_shape.data(), nullptr),
      "create output dataspace");
  const hid_t destination = require_id(
      H5Dcreate2(output, destination_name, H5T_IEEE_F64LE, output_space, H5P_DEFAULT,
                 H5P_DEFAULT, H5P_DEFAULT),
      "create output dataset");
  const std::size_t value_count = static_cast<std::size_t>(shape[0] * shape[1] *
                                                           shape[3] * shape[4]);
  std::vector<double> values(value_count);
  check(H5Dread(source, H5T_NATIVE_DOUBLE, output_space, source_space, H5P_DEFAULT,
                values.data()),
        "read i=0 slice");
  check(H5Dwrite(destination, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 values.data()),
        "write output dataset");

  check(H5Dclose(destination), "close output dataset");
  check(H5Sclose(output_space), "close output dataspace");
  check(H5Sclose(source_space), "close source dataspace");
  check(H5Dclose(source), "close source dataset");
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " INPUT.h5 OUTPUT.h5\n";
    return 2;
  }

  hid_t input = -1;
  hid_t output = -1;
  const std::filesystem::path destination = std::filesystem::absolute(argv[2]);
  const std::filesystem::path temporary = destination.string() + ".tmp";
  try {
    input = require_id(H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT), "open input file");

    const hid_t flow = require_id(H5Dopen2(input, "flow", H5P_DEFAULT), "open flow");
    const auto flow_shape = dataset_shape(flow, "flow");
    check(H5Dclose(flow), "close flow");
    const hsize_t nz = flow_shape[0];
    const hsize_t ny = flow_shape[1];

    output = require_id(H5Fcreate(temporary.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                                  H5P_DEFAULT),
                        "create output file");
    copy_i0_slice(input, output, "grid", "grid", nz, ny);
    copy_i0_slice(input, output, "flow", "baseflow", nz, ny);
    write_attribute(output, "Ny", static_cast<std::int64_t>(ny));
    write_attribute(output, "Nz", static_cast<std::int64_t>(nz));

    check(H5Fclose(output), "close output file");
    output = -1;
    check(H5Fclose(input), "close input file");
    input = -1;
    std::filesystem::rename(temporary, destination);
  } catch (const std::exception &error) {
    if (output >= 0) H5Fclose(output);
    if (input >= 0) H5Fclose(input);
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
