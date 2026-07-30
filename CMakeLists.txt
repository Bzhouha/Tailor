cmake_minimum_required(VERSION 3.24)

set(TAILOR_PETSC_PREFIX "" CACHE PATH
    "Prefix of the complex PETSc installation used by Tailor")
set(TAILOR_SLEPC_PREFIX "" CACHE PATH
    "Optional prefix of the compatible SLEPc installation")
set(TAILOR_YAML_CPP_PREFIX "" CACHE PATH
    "Optional prefix containing yaml-cpp-config.cmake")
set(TAILOR_TEST_CONFIG
    "${CMAKE_CURRENT_LIST_DIR}/case/config.yaml" CACHE FILEPATH
    "Case configuration used by Tailor smoke tests")
set(TAILOR_TEST_PYTHON "" CACHE FILEPATH
    "Python interpreter containing h5py and PyYAML for Tailor integration tests")
set(TAILOR_MPIEXEC "" CACHE FILEPATH
    "MPI launcher used by Tailor's parallel tests")

if(NOT TAILOR_PETSC_PREFIX AND DEFINED ENV{TAILOR_PETSC_PREFIX})
    set(TAILOR_PETSC_PREFIX "$ENV{TAILOR_PETSC_PREFIX}")
endif()
if(NOT TAILOR_PETSC_PREFIX AND DEFINED ENV{PETSC_DIR})
    if(DEFINED ENV{PETSC_ARCH} AND NOT "$ENV{PETSC_ARCH}" STREQUAL "")
        set(TAILOR_PETSC_PREFIX "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}")
    else()
        set(TAILOR_PETSC_PREFIX "$ENV{PETSC_DIR}")
    endif()
endif()
if(NOT TAILOR_PETSC_PREFIX)
    message(FATAL_ERROR
        "Set PETSC_DIR (and PETSC_ARCH for an in-place build), export "
        "TAILOR_PETSC_PREFIX, or pass -DTAILOR_PETSC_PREFIX=<prefix>.")
endif()
get_filename_component(TAILOR_PETSC_PREFIX
    "${TAILOR_PETSC_PREFIX}" ABSOLUTE)
set(TAILOR_PETSC_PREFIX "${TAILOR_PETSC_PREFIX}" CACHE PATH
    "Prefix of the complex PETSc installation used by Tailor" FORCE)

if(NOT TAILOR_SLEPC_PREFIX AND DEFINED ENV{TAILOR_SLEPC_PREFIX})
    set(TAILOR_SLEPC_PREFIX "$ENV{TAILOR_SLEPC_PREFIX}")
endif()
if(NOT TAILOR_SLEPC_PREFIX AND DEFINED ENV{SLEPC_DIR})
    if(DEFINED ENV{PETSC_ARCH}
       AND EXISTS "$ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig")
        set(TAILOR_SLEPC_PREFIX "$ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}")
    else()
        set(TAILOR_SLEPC_PREFIX "$ENV{SLEPC_DIR}")
    endif()
endif()
if(NOT TAILOR_SLEPC_PREFIX)
    set(TAILOR_SLEPC_PREFIX "${TAILOR_PETSC_PREFIX}")
endif()
get_filename_component(TAILOR_SLEPC_PREFIX
    "${TAILOR_SLEPC_PREFIX}" ABSOLUTE)
set(TAILOR_SLEPC_PREFIX "${TAILOR_SLEPC_PREFIX}" CACHE PATH
    "Optional prefix of the compatible SLEPc installation" FORCE)

find_program(_tailor_mpicxx NAMES mpicxx mpic++
    HINTS "${TAILOR_PETSC_PREFIX}/bin")
if(NOT _tailor_mpicxx)
    message(FATAL_ERROR
        "An MPI C++ compiler was not found. Add it to PATH or select it with "
        "-DCMAKE_CXX_COMPILER=<mpicxx>.")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${_tailor_mpicxx}" CACHE FILEPATH
        "MPI C++ compiler used by Tailor")
endif()

project(Tailor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(PREPEND CMAKE_BUILD_RPATH
    "${TAILOR_PETSC_PREFIX}/lib"
    "${TAILOR_SLEPC_PREFIX}/lib")

set(_tailor_pkgconfig_path "${TAILOR_PETSC_PREFIX}/lib/pkgconfig")
if(NOT TAILOR_SLEPC_PREFIX STREQUAL TAILOR_PETSC_PREFIX)
    set(_tailor_pkgconfig_path
        "${TAILOR_SLEPC_PREFIX}/lib/pkgconfig:${_tailor_pkgconfig_path}")
endif()
if(DEFINED ENV{PKG_CONFIG_PATH} AND NOT "$ENV{PKG_CONFIG_PATH}" STREQUAL "")
    set(ENV{PKG_CONFIG_PATH} "${_tailor_pkgconfig_path}:$ENV{PKG_CONFIG_PATH}")
else()
    set(ENV{PKG_CONFIG_PATH} "${_tailor_pkgconfig_path}")
endif()
if(TAILOR_YAML_CPP_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${TAILOR_YAML_CPP_PREFIX}")
endif()

find_package(PkgConfig REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
pkg_check_modules(SLEPC REQUIRED IMPORTED_TARGET "SLEPc>=3.25")
find_library(TAILOR_PETSC_HDF5_LIBRARY
    NAMES hdf5.310 hdf5
    HINTS "${TAILOR_PETSC_PREFIX}/lib"
    REQUIRED)

if(NOT TAILOR_MPIEXEC)
    find_program(_tailor_mpiexec NAMES mpiexec mpirun
        HINTS "${TAILOR_PETSC_PREFIX}/bin")
    if(_tailor_mpiexec)
        set(TAILOR_MPIEXEC "${_tailor_mpiexec}" CACHE FILEPATH
            "MPI launcher used by Tailor's parallel tests" FORCE)
    endif()
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

add_library(tailor_core STATIC
    src/CPlusPlus/BaseFlowDerivatives.cpp
    src/CPlusPlus/BoundaryConditions.cpp
    src/CPlusPlus/CurvilinearTransform.cpp
    src/CPlusPlus/EigenSolver.cpp
    src/CPlusPlus/EigenOutput.cpp
    src/CPlusPlus/GlobalOperator.cpp
    src/CPlusPlus/LNSCoefficients.cpp
    src/CPlusPlus/Metrics.cpp
    src/CPlusPlus/Parser.cpp
    src/CPlusPlus/ProblemData.cpp
    src/CPlusPlus/Prepare.cpp
    src/CPlusPlus/StreamwiseFourier.cpp
)
target_include_directories(tailor_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/CPlusPlus")
target_link_libraries(tailor_core PUBLIC
    PkgConfig::SLEPC
    yaml-cpp::yaml-cpp
    "${TAILOR_PETSC_HDF5_LIBRARY}"
    lapack
    blas)
tailor_enable_warnings(tailor_core)

add_executable(tailor
    src/CPlusPlus/main.cpp
    src/CPlusPlus/Process.cpp
)
target_link_libraries(tailor PRIVATE tailor_core)
tailor_enable_warnings(tailor)

include(CTest)
if(BUILD_TESTING)
    if(NOT TAILOR_MPIEXEC)
        message(FATAL_ERROR
            "An MPI launcher is required for tests. Add mpiexec to PATH or "
            "pass -DTAILOR_MPIEXEC=<path>.")
    endif()
    if(NOT TAILOR_TEST_PYTHON)
        set(Python3_FIND_VIRTUALENV FIRST)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            set(TAILOR_TEST_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH
                "Python interpreter containing Tailor test dependencies" FORCE)
        endif()
    endif()
    set(_tailor_python_tests_available FALSE)
    if(EXISTS "${TAILOR_TEST_PYTHON}")
        execute_process(
            COMMAND "${TAILOR_TEST_PYTHON}" -c
                    "import h5py, numpy, scipy, yaml"
            RESULT_VARIABLE _tailor_python_import_status
            OUTPUT_QUIET
            ERROR_QUIET)
        if(_tailor_python_import_status EQUAL 0)
            set(_tailor_python_tests_available TRUE)
        else()
            message(STATUS
                "TAILOR_TEST_PYTHON lacks h5py, NumPy, SciPy, or PyYAML; "
                "skipping Python/C++ integration tests: "
                "${TAILOR_TEST_PYTHON}")
        endif()
    endif()
    find_program(TAILOR_GFORTRAN NAMES gfortran REQUIRED)
    set(TAILOR_LNS_GOLD_GENERATOR
        "${CMAKE_CURRENT_BINARY_DIR}/generate_lns_coefficients_gold")
    set(TAILOR_LNS_GOLD_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/lns_coefficients_gold.txt")
    add_custom_command(
        OUTPUT "${TAILOR_LNS_GOLD_FILE}"
        BYPRODUCTS "${TAILOR_LNS_GOLD_GENERATOR}"
        COMMAND "${TAILOR_GFORTRAN}"
                -std=f2008 -Wall -Wextra -Werror
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/CPlusPlus/generate_lns_coefficients_gold.f90"
                -o "${TAILOR_LNS_GOLD_GENERATOR}"
        COMMAND "${TAILOR_LNS_GOLD_GENERATOR}" "${TAILOR_LNS_GOLD_FILE}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/CPlusPlus/generate_lns_coefficients_gold.f90"
        COMMENT "Generating canonical LNS coefficient fixture from mod_cubes formulas"
        VERBATIM)
    add_custom_target(tailor_lns_gold
        DEPENDS "${TAILOR_LNS_GOLD_FILE}")

    add_executable(test_lns_coefficients
        tests/CPlusPlus/test_lns_coefficients.cpp)
    target_link_libraries(test_lns_coefficients PRIVATE tailor_core)
    add_dependencies(test_lns_coefficients tailor_lns_gold)
    tailor_enable_warnings(test_lns_coefficients)

    set(TAILOR_LNS_TRANSFORM_GOLD_GENERATOR
        "${CMAKE_CURRENT_BINARY_DIR}/generate_lns_transform_gold")
    set(TAILOR_LNS_TRANSFORM_GOLD_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/lns_transform_gold.txt")
    add_custom_command(
        OUTPUT "${TAILOR_LNS_TRANSFORM_GOLD_FILE}"
        BYPRODUCTS "${TAILOR_LNS_TRANSFORM_GOLD_GENERATOR}"
        COMMAND "${TAILOR_GFORTRAN}"
                -std=f2008 -Wall -Wextra -Werror
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/CPlusPlus/generate_lns_transform_gold.f90"
                -o "${TAILOR_LNS_TRANSFORM_GOLD_GENERATOR}"
        COMMAND "${TAILOR_LNS_TRANSFORM_GOLD_GENERATOR}"
                "${TAILOR_LNS_TRANSFORM_GOLD_FILE}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/CPlusPlus/generate_lns_transform_gold.f90"
        COMMENT "Generating canonical LNS transform fixture from mod_cubes formulas"
        VERBATIM)
    add_custom_target(tailor_lns_transform_gold
        DEPENDS "${TAILOR_LNS_TRANSFORM_GOLD_FILE}")

    add_executable(test_lns_transforms
        tests/CPlusPlus/test_lns_transforms.cpp)
    target_link_libraries(test_lns_transforms PRIVATE tailor_core)
    add_dependencies(test_lns_transforms tailor_lns_transform_gold)
    tailor_enable_warnings(test_lns_transforms)

    add_executable(test_global_operator
        tests/CPlusPlus/test_global_operator.cpp)
    target_link_libraries(test_global_operator PRIVATE tailor_core)
    tailor_enable_warnings(test_global_operator)

    add_executable(test_boundary_conditions
        tests/CPlusPlus/test_boundary_conditions.cpp)
    target_link_libraries(test_boundary_conditions PRIVATE tailor_core)
    tailor_enable_warnings(test_boundary_conditions)

    add_executable(test_eigen_solver
        tests/CPlusPlus/test_eigen_solver.cpp)
    target_link_libraries(test_eigen_solver PRIVATE tailor_core)
    tailor_enable_warnings(test_eigen_solver)

    add_test(NAME tailor_eigen_solver_small
        COMMAND test_eigen_solver)

    if(EXISTS "${TAILOR_TEST_CONFIG}")
        add_test(NAME tailor_smoke_serial
            COMMAND tailor -c "${TAILOR_TEST_CONFIG}"
                    -tailor_assemble_only)
        add_test(NAME tailor_smoke_mpi_2
            COMMAND "${TAILOR_MPIEXEC}" -n 2
                    $<TARGET_FILE:tailor> -c "${TAILOR_TEST_CONFIG}"
                    -tailor_assemble_only)
        add_test(NAME tailor_lns_coefficients_serial
            COMMAND test_lns_coefficients
                    -c "${TAILOR_TEST_CONFIG}"
                    -gold "${TAILOR_LNS_GOLD_FILE}")
        add_test(NAME tailor_lns_coefficients_mpi_2
            COMMAND "${TAILOR_MPIEXEC}" -n 2
                    $<TARGET_FILE:test_lns_coefficients>
                    -c "${TAILOR_TEST_CONFIG}"
                    -gold "${TAILOR_LNS_GOLD_FILE}")
        add_test(NAME tailor_lns_transforms_serial
            COMMAND test_lns_transforms
                    -c "${TAILOR_TEST_CONFIG}"
                    -gold "${TAILOR_LNS_TRANSFORM_GOLD_FILE}")
        add_test(NAME tailor_lns_transforms_mpi_2
            COMMAND "${TAILOR_MPIEXEC}" -n 2
                    $<TARGET_FILE:test_lns_transforms>
                    -c "${TAILOR_TEST_CONFIG}"
                    -gold "${TAILOR_LNS_TRANSFORM_GOLD_FILE}")
        add_test(NAME tailor_global_operator_serial
            COMMAND test_global_operator
                    -c "${TAILOR_TEST_CONFIG}")
        add_test(NAME tailor_global_operator_mpi_2
            COMMAND "${TAILOR_MPIEXEC}" -n 2
                    $<TARGET_FILE:test_global_operator>
                    -c "${TAILOR_TEST_CONFIG}")
        add_test(NAME tailor_boundary_conditions_serial
            COMMAND test_boundary_conditions
                    -c "${TAILOR_TEST_CONFIG}")
        add_test(NAME tailor_boundary_conditions_mpi_2
            COMMAND "${TAILOR_MPIEXEC}" -n 2
                    $<TARGET_FILE:test_boundary_conditions>
                    -c "${TAILOR_TEST_CONFIG}")
        if(_tailor_python_tests_available)
            add_test(NAME tailor_fdq_invalid_inputs
                COMMAND "${TAILOR_TEST_PYTHON}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/CPlusPlus/test_fdq_loader_failures.py"
                        --solver $<TARGET_FILE:tailor>
                        --config "${TAILOR_TEST_CONFIG}"
                        --mpiexec "${TAILOR_MPIEXEC}")
            set_tests_properties(tailor_fdq_invalid_inputs
                PROPERTIES TIMEOUT 180)
            add_test(NAME tailor_metrics_regression
                COMMAND "${TAILOR_TEST_PYTHON}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/CPlusPlus/test_metrics.py"
                        --solver $<TARGET_FILE:tailor>
                        --config "${TAILOR_TEST_CONFIG}"
                        --mpiexec "${TAILOR_MPIEXEC}")
            set_tests_properties(tailor_metrics_regression
                PROPERTIES TIMEOUT 180)
            add_test(NAME tailor_baseflow_derivatives
                COMMAND "${TAILOR_TEST_PYTHON}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/CPlusPlus/test_baseflow_derivatives.py"
                        --solver $<TARGET_FILE:tailor>
                        --config "${TAILOR_TEST_CONFIG}"
                        --mpiexec "${TAILOR_MPIEXEC}")
            set_tests_properties(tailor_baseflow_derivatives
                PROPERTIES TIMEOUT 180)
            add_test(NAME tailor_physics_config
                COMMAND "${TAILOR_TEST_PYTHON}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/CPlusPlus/test_physics_config.py"
                        --solver $<TARGET_FILE:tailor>
                        --config "${TAILOR_TEST_CONFIG}"
                        --mpiexec "${TAILOR_MPIEXEC}")
            set_tests_properties(tailor_physics_config
                PROPERTIES TIMEOUT 180)
            add_test(NAME tailor_boundary_uniform
                COMMAND "${TAILOR_TEST_PYTHON}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/CPlusPlus/test_boundary_uniform.py"
                        --test-binary $<TARGET_FILE:test_boundary_conditions>
                        --config "${TAILOR_TEST_CONFIG}"
                        --mpiexec "${TAILOR_MPIEXEC}")
            set_tests_properties(tailor_boundary_uniform
                PROPERTIES TIMEOUT 180)
            add_test(NAME tailor_eigen_output
                COMMAND "${TAILOR_TEST_PYTHON}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/CPlusPlus/test_eigen_output.py"
                        --test-binary $<TARGET_FILE:test_eigen_solver>)
            set_tests_properties(tailor_eigen_output
                PROPERTIES TIMEOUT 120)
        else()
            message(STATUS
                "Python integration dependencies are unavailable; "
                "skipping Python/C++ integration tests: "
                "${TAILOR_TEST_PYTHON}")
        endif()
    else()
        message(STATUS
            "TAILOR_TEST_CONFIG does not exist; skipping case smoke tests: "
            "${TAILOR_TEST_CONFIG}")
    endif()
endif()
