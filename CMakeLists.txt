cmake_minimum_required(VERSION 3.24)

set(TAILOR_PETSC_PREFIX "/Users/becrazy/Wro/petsc/arch-complex" CACHE PATH
    "Prefix of the complex PETSc/SLEPc installation used by Tailor")
set(TAILOR_YAML_CPP_PREFIX "/opt/homebrew" CACHE PATH
    "Prefix containing yaml-cpp-config.cmake")
set(TAILOR_TEST_CONFIG
    "${CMAKE_CURRENT_LIST_DIR}/../cases/global/config.yaml" CACHE FILEPATH
    "Case configuration used by Tailor smoke tests")
set(TAILOR_TEST_PYTHON
    "${CMAKE_CURRENT_LIST_DIR}/../petsc-python/bin/python" CACHE FILEPATH
    "Python interpreter containing h5py and PyYAML for Tailor integration tests")
option(TAILOR_BUILD_TOOLS "Build input-data and FD-q verification tools" OFF)

set(_tailor_mpicxx "${TAILOR_PETSC_PREFIX}/bin/mpicxx")
if(NOT EXISTS "${_tailor_mpicxx}")
    message(FATAL_ERROR "PETSc MPI C++ wrapper not found: ${_tailor_mpicxx}")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${_tailor_mpicxx}" CACHE FILEPATH
        "C++ compiler from Tailor's PETSc arch")
endif()

project(Tailor VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER STREQUAL "${_tailor_mpicxx}")
    message(FATAL_ERROR
        "Tailor must be configured with ${_tailor_mpicxx}; got ${CMAKE_CXX_COMPILER}. "
        "Use a fresh build directory when changing PETSc arches.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(PREPEND CMAKE_BUILD_RPATH "${TAILOR_PETSC_PREFIX}/lib")

set(_tailor_pkgconfig_path "${TAILOR_PETSC_PREFIX}/lib/pkgconfig")
if(DEFINED ENV{PKG_CONFIG_PATH} AND NOT "$ENV{PKG_CONFIG_PATH}" STREQUAL "")
    set(ENV{PKG_CONFIG_PATH} "${_tailor_pkgconfig_path}:$ENV{PKG_CONFIG_PATH}")
else()
    set(ENV{PKG_CONFIG_PATH} "${_tailor_pkgconfig_path}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${TAILOR_YAML_CPP_PREFIX}")

find_package(PkgConfig REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
pkg_check_modules(SLEPC REQUIRED IMPORTED_TARGET "SLEPc>=3.25")
find_library(TAILOR_PETSC_HDF5_LIBRARY
    NAMES hdf5.310 hdf5
    PATHS "${TAILOR_PETSC_PREFIX}/lib"
    NO_DEFAULT_PATH
    REQUIRED)

if(NOT SLEPC_PREFIX STREQUAL TAILOR_PETSC_PREFIX)
    message(FATAL_ERROR
        "pkg-config selected SLEPc from ${SLEPC_PREFIX}, expected ${TAILOR_PETSC_PREFIX}")
endif()

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES PkgConfig::SLEPC)
check_cxx_source_compiles([=[
    #include <petscsys.h>
    #ifndef PETSC_USE_COMPLEX
    #error Tailor requires a complex-scalar PETSc build
    #endif
    int main() { return 0; }
]=] TAILOR_PETSC_IS_COMPLEX)
unset(CMAKE_REQUIRED_LIBRARIES)
if(NOT TAILOR_PETSC_IS_COMPLEX)
    message(FATAL_ERROR "TAILOR_PETSC_PREFIX must select a complex-scalar PETSc build")
endif()

function(tailor_enable_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>)
endfunction()

add_executable(tailor
    src/CPlusPlus/main.cpp
    src/CPlusPlus/Parser.cpp
    src/CPlusPlus/ProblemData.cpp
    src/CPlusPlus/Prepare.cpp
    src/CPlusPlus/Process.cpp
)
target_include_directories(tailor PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/CPlusPlus"
    "${TAILOR_PETSC_PREFIX}/include")
target_link_libraries(tailor PRIVATE
    PkgConfig::SLEPC
    yaml-cpp::yaml-cpp
    "${TAILOR_PETSC_HDF5_LIBRARY}")
tailor_enable_warnings(tailor)

if(TAILOR_BUILD_TOOLS)
    pkg_check_modules(HDF5 REQUIRED IMPORTED_TARGET "hdf5>=1.14")

    add_executable(make_sample_h5 tools/make_sample_h5.cpp)
    target_link_libraries(make_sample_h5 PRIVATE PkgConfig::HDF5)
    tailor_enable_warnings(make_sample_h5)

    add_executable(verify_fdq_load tools/verify_sample_load.cpp)
    target_link_libraries(verify_fdq_load PRIVATE PkgConfig::SLEPC)
    tailor_enable_warnings(verify_fdq_load)
endif()

include(CTest)
if(BUILD_TESTING)
    if(EXISTS "${TAILOR_TEST_CONFIG}")
        add_test(NAME tailor_smoke_serial
            COMMAND tailor -c "${TAILOR_TEST_CONFIG}")
        add_test(NAME tailor_smoke_mpi_2
            COMMAND "${TAILOR_PETSC_PREFIX}/bin/mpiexec" -n 2
                    $<TARGET_FILE:tailor> -c "${TAILOR_TEST_CONFIG}")
        if(EXISTS "${TAILOR_TEST_PYTHON}")
            add_test(NAME tailor_fdq_invalid_inputs
                COMMAND "${TAILOR_TEST_PYTHON}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/CPlusPlus/test_fdq_loader_failures.py"
                        --solver $<TARGET_FILE:tailor>
                        --config "${TAILOR_TEST_CONFIG}"
                        --mpiexec "${TAILOR_PETSC_PREFIX}/bin/mpiexec")
            set_tests_properties(tailor_fdq_invalid_inputs
                PROPERTIES TIMEOUT 180)
        else()
            message(STATUS
                "TAILOR_TEST_PYTHON does not exist; skipping malformed FD-q tests: "
                "${TAILOR_TEST_PYTHON}")
        endif()
    else()
        message(STATUS
            "TAILOR_TEST_CONFIG does not exist; skipping case smoke tests: "
            "${TAILOR_TEST_CONFIG}")
    endif()
endif()
