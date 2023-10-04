cmake_minimum_required(VERSION 3.10)

project(PmscExercise6 LANGUAGES CXX)

include(${PROJECT_SOURCE_DIR}/cmake/check_mpi_vendor.cmake)

if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  endif()
  set(PMSC_PRIVATE_COMPILE_OPTIONS "/W4" "/WX" "/permissive-" "$<$<CONFIG:Release>:/arch:AVX2>")
elseif(CMAKE_COMPILER_IS_GNUCC)
  set(PMSC_PRIVATE_COMPILE_OPTIONS "-Wall" "-Wextra" "-pedantic" "-Werror" "$<$<CONFIG:Release>:-march=native")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

# =======
# third-party

find_package(MPI REQUIRED)
pmsc_check_mpi_vendor()

find_package(pybind11 REQUIRED)

# =======
# linear_algebra

set(linear_algebra_SOURCES
  linear_algebra/contiguousparallelpartition.cpp
  linear_algebra/exchangepattern.cpp
)

add_library(linear_algebra ${linear_algebra_SOURCES})
target_compile_features(linear_algebra PUBLIC cxx_std_17)
target_link_libraries(linear_algebra PUBLIC MPI::MPI_CXX)

if(NOT MSVC)
    target_link_libraries(linear_algebra PUBLIC m)
endif()

# =======
# solvers
add_library(solvers INTERFACE)
target_link_libraries(solvers INTERFACE linear_algebra)


# =======
# grid

set(grid_SOURCES
  grid/point.cpp
  grid/multiindex.cpp
  grid/grid.cpp
)

add_library(grid_1d ${grid_SOURCES})
target_compile_options(grid_1d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(grid_1d PUBLIC cxx_std_17)
target_compile_definitions(grid_1d PUBLIC "PMSC_SPACE_DIMENSION=1")
target_link_libraries(grid_1d PUBLIC linear_algebra)

add_library(grid_2d ${grid_SOURCES})
target_compile_options(grid_2d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(grid_2d PUBLIC cxx_std_17)
target_compile_definitions(grid_2d PUBLIC "PMSC_SPACE_DIMENSION=2")
target_link_libraries(grid_2d PUBLIC linear_algebra)

add_library(grid_3d ${grid_SOURCES})
target_compile_options(grid_3d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(grid_3d PUBLIC cxx_std_17)
target_compile_definitions(grid_3d PUBLIC "PMSC_SPACE_DIMENSION=3")
target_link_libraries(grid_3d PUBLIC linear_algebra)

# =======
# python

find_package(Python 3 COMPONENTS Development Interpreter)

set(python_module_SOURCES
  python/module.cpp
  python/mpi_comm_wrapper.cpp
)

pybind11_add_module(pmsc ${python_module_SOURCES})
target_compile_options(pmsc PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_link_libraries(pmsc PRIVATE linear_algebra solvers grid_2d)
#target_link_libraries(pmsc PRIVATE linear_algebra solvers grid_3d)

# =======
# TESTS

enable_testing()

add_subdirectory(third-party/gtest)

add_subdirectory(tests)