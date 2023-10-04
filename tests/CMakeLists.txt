cmake_minimum_required(VERSION 3.10)

set(PMSC_MPI_TEST_PROCESSES "1;2;3;4")

set(PMSC_MPIEXEC_EXTRAFLAGS)
if(PMSC_MPI_VENDOR STREQUAL "OpenMPI")
  option(PMSC_ALLOW_MPI_TESTS_AS_ROOT "Allows running MPI tests as root user" OFF)
  list(APPEND PMSC_MPIEXEC_EXTRAFLAGS "--oversubscribe")
  if(PMSC_ALLOW_MPI_TESTS_AS_ROOT)
    list(APPEND PMSC_MPIEXEC_EXTRAFLAGS "--allow-run-as-root")
  endif()
endif()

set(test_linear_algebra_tests)

# =======
# sequential linear algebra tests
add_executable(test_vector vector.cpp)
target_compile_options(test_vector PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_vector PRIVATE cxx_std_17)
target_link_libraries(test_vector linear_algebra)
target_link_libraries(test_vector third_party::gtest third_party::gtest_main)
add_test(NAME test_vector COMMAND test_vector)
list(APPEND test_linear_algebra_tests test_vector)

add_executable(test_matrix matrix.cpp)
target_compile_options(test_matrix PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_matrix PRIVATE cxx_std_17)
target_link_libraries(test_matrix linear_algebra)
target_link_libraries(test_matrix third_party::gtest third_party::gtest_main)
add_test(NAME test_matrix COMMAND test_matrix)
list(APPEND test_linear_algebra_tests test_matrix)

add_executable(test_operations operations.cpp)
target_compile_options(test_operations PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_operations PRIVATE cxx_std_17)
target_link_libraries(test_operations linear_algebra)
target_link_libraries(test_operations third_party::gtest third_party::gtest_main)
add_test(NAME test_operations COMMAND test_operations)
list(APPEND test_linear_algebra_tests test_operations)

# =======
# parallel linear algebra tests
add_executable(test_parallel_partition parallel_partition.cpp)
target_compile_options(test_parallel_partition PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_parallel_partition PRIVATE cxx_std_17)
target_link_libraries(test_parallel_partition linear_algebra)
target_link_libraries(test_parallel_partition third_party::gtest third_party::gtest_main)
list(APPEND test_linear_algebra_tests test_parallel_partition)

add_executable(test_parallel_vector parallel_vector.cpp)
target_compile_options(test_parallel_vector PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_parallel_vector PRIVATE cxx_std_17)
target_link_libraries(test_parallel_vector linear_algebra)
target_link_libraries(test_parallel_vector third_party::gtest third_party::gtest_main)
list(APPEND test_linear_algebra_tests test_parallel_vector)

add_executable(test_parallel_matrix parallel_matrix.cpp)
target_compile_options(test_parallel_matrix PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_parallel_matrix PRIVATE cxx_std_17)
target_link_libraries(test_parallel_matrix linear_algebra)
target_link_libraries(test_parallel_matrix third_party::gtest third_party::gtest_main)
list(APPEND test_linear_algebra_tests test_parallel_matrix)

add_executable(test_parallel_exchangepattern exchangepattern.cpp)
target_compile_options(test_parallel_exchangepattern PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_parallel_exchangepattern PRIVATE cxx_std_17)
target_link_libraries(test_parallel_exchangepattern linear_algebra)
target_link_libraries(test_parallel_exchangepattern third_party::gtest third_party::gtest_main)
list(APPEND test_linear_algebra_tests test_parallel_exchangepattern)

add_executable(test_parallel_exchangedata exchangedata.cpp)
target_compile_options(test_parallel_exchangedata PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_parallel_exchangedata PRIVATE cxx_std_17)
target_link_libraries(test_parallel_exchangedata linear_algebra)
target_link_libraries(test_parallel_exchangedata third_party::gtest third_party::gtest_main)
list(APPEND test_linear_algebra_tests test_parallel_exchangedata)

add_executable(test_parallel_operations parallel_operations.cpp)
target_compile_options(test_parallel_operations PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_parallel_operations PRIVATE cxx_std_17)
target_link_libraries(test_parallel_operations linear_algebra)
target_link_libraries(test_parallel_operations third_party::gtest third_party::gtest_main)
list(APPEND test_parallel_operations "test_parallel_operations_np${NP}")

foreach(NP IN LISTS PMSC_MPI_TEST_PROCESSES)
  add_test(NAME "test_parallel_partition_np${NP}" COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${NP}" ${PMSC_MPIEXEC_EXTRAFLAGS} ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:test_parallel_partition>" ${MPIEXEC_POSTFLAGS})
  set_tests_properties("test_parallel_partition_np${NP}" PROPERTIES PROCESSORS "${NP}")

  add_test(NAME "test_parallel_vector_np${NP}" COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${NP}" ${PMSC_MPIEXEC_EXTRAFLAGS} ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:test_parallel_vector>" ${MPIEXEC_POSTFLAGS})
  set_tests_properties("test_parallel_vector_np${NP}" PROPERTIES PROCESSORS "${NP}")

  add_test(NAME "test_parallel_matrix_np${NP}" COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${NP}" ${PMSC_MPIEXEC_EXTRAFLAGS} ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:test_parallel_matrix>" ${MPIEXEC_POSTFLAGS})
  set_tests_properties("test_parallel_matrix_np${NP}" PROPERTIES PROCESSORS "${NP}")

  add_test(NAME "test_parallel_exchangepattern_np${NP}" COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${NP}" ${PMSC_MPIEXEC_EXTRAFLAGS} ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:test_parallel_exchangepattern>" ${MPIEXEC_POSTFLAGS})
  set_tests_properties("test_parallel_exchangepattern_np${NP}" PROPERTIES PROCESSORS "${NP}")

  add_test(NAME "test_parallel_exchangedata_np${NP}" COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${NP}" ${PMSC_MPIEXEC_EXTRAFLAGS} ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:test_parallel_exchangedata>" ${MPIEXEC_POSTFLAGS})
  set_tests_properties("test_parallel_exchangedata_np${NP}" PROPERTIES PROCESSORS "${NP}")

  add_test(NAME "test_parallel_operations_np${NP}" COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${NP}" ${PMSC_MPIEXEC_EXTRAFLAGS} ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:test_parallel_operations>" ${MPIEXEC_POSTFLAGS})
  set_tests_properties("test_parallel_operations_np${NP}" PROPERTIES PROCESSORS "${NP}")
endforeach()

if(NOT TARGET test_linear_algebra)
  add_custom_target(test_linear_algebra)
  add_dependencies(test_linear_algebra ${test_linear_algebra_tests})
endif()

# =======
# solver tests
add_executable(test_solvers solvers.cpp)
target_compile_options(test_solvers PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_solvers PRIVATE cxx_std_17)
target_link_libraries(test_solvers solvers)
target_link_libraries(test_solvers third_party::gtest third_party::gtest_main)
add_test(NAME test_solvers COMMAND test_solvers)

set(test_grid_tests)
# =======
# sequential grid tests
add_executable(test_grid_1d grid.cpp)
target_compile_options(test_grid_1d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_grid_1d PRIVATE cxx_std_17)
target_link_libraries(test_grid_1d grid_1d)
target_link_libraries(test_grid_1d third_party::gtest third_party::gtest_main)
#add_test(NAME test_grid_1d COMMAND test_grid_1d)
list(APPEND test_grid_tests test_grid_1d)

add_executable(test_grid_2d grid.cpp)
target_compile_options(test_grid_2d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_grid_2d PRIVATE cxx_std_17)
target_link_libraries(test_grid_2d grid_2d)
target_link_libraries(test_grid_2d third_party::gtest third_party::gtest_main)
#add_test(NAME test_grid_2d COMMAND test_grid_2d)
list(APPEND test_grid_tests test_grid_2d)

add_executable(test_grid_3d grid.cpp)
target_compile_options(test_grid_3d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_grid_3d PRIVATE cxx_std_17)
target_link_libraries(test_grid_3d grid_3d)
target_link_libraries(test_grid_3d third_party::gtest third_party::gtest_main)
#add_test(NAME test_grid_3d COMMAND test_grid_3d)
list(APPEND test_grid_tests test_grid_3d)

# =======
# parallel grid tests
add_executable(test_parallel_grid_1d parallel_grid.cpp)
target_compile_options(test_parallel_grid_1d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_parallel_grid_1d PRIVATE cxx_std_17)
target_link_libraries(test_parallel_grid_1d grid_1d)
target_link_libraries(test_parallel_grid_1d third_party::gtest third_party::gtest_main)
list(APPEND test_grid_tests test_parallel_grid_1d)

add_executable(test_parallel_grid_2d parallel_grid.cpp)
target_compile_options(test_parallel_grid_2d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_parallel_grid_2d PRIVATE cxx_std_17)
target_link_libraries(test_parallel_grid_2d grid_2d)
target_link_libraries(test_parallel_grid_2d third_party::gtest third_party::gtest_main)
list(APPEND test_grid_tests test_parallel_grid_2d)

add_executable(test_parallel_grid_3d parallel_grid.cpp)
target_compile_options(test_parallel_grid_3d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_parallel_grid_3d PRIVATE cxx_std_17)
target_link_libraries(test_parallel_grid_3d grid_3d)
target_link_libraries(test_parallel_grid_3d third_party::gtest third_party::gtest_main)
list(APPEND test_grid_tests test_parallel_grid_3d)

foreach(NP IN LISTS PMSC_MPI_TEST_PROCESSES)
  add_test(NAME "test_parallel_grid_1d_np${NP}" COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${NP}" ${PMSC_MPIEXEC_EXTRAFLAGS} ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:test_parallel_grid_1d>" ${MPIEXEC_POSTFLAGS})
  set_tests_properties("test_parallel_grid_1d_np${NP}" PROPERTIES PROCESSORS "${NP}")

  add_test(NAME "test_parallel_grid_2d_np${NP}" COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${NP}" ${PMSC_MPIEXEC_EXTRAFLAGS} ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:test_parallel_grid_2d>" ${MPIEXEC_POSTFLAGS})
  set_tests_properties("test_parallel_grid_2d_np${NP}" PROPERTIES PROCESSORS "${NP}")

  add_test(NAME "test_parallel_grid_3d_np${NP}" COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${NP}" ${PMSC_MPIEXEC_EXTRAFLAGS} ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:test_parallel_grid_3d>" ${MPIEXEC_POSTFLAGS})
  set_tests_properties("test_parallel_grid_3d_np${NP}" PROPERTIES PROCESSORS "${NP}")
endforeach()

if(NOT TARGET test_grid)
  add_custom_target(test_grid)
  add_dependencies(test_grid ${test_grid_tests})
endif()

# =======
# poisson tests
add_executable(test_poisson_1d poisson.cpp)
target_compile_options(test_poisson_1d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_poisson_1d PRIVATE cxx_std_17)
target_link_libraries(test_poisson_1d grid_1d)
target_link_libraries(test_poisson_1d third_party::gtest third_party::gtest_main)
add_test(NAME test_poisson_1d COMMAND test_poisson_1d)

add_executable(test_poisson_2d poisson.cpp)
target_compile_options(test_poisson_2d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_poisson_2d PRIVATE cxx_std_17)
target_link_libraries(test_poisson_2d grid_2d)
target_link_libraries(test_poisson_2d third_party::gtest third_party::gtest_main)
add_test(NAME test_poisson_2d COMMAND test_poisson_2d)

add_executable(test_poisson_3d poisson.cpp)
target_compile_options(test_poisson_3d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_poisson_3d PRIVATE cxx_std_17)
target_link_libraries(test_poisson_3d grid_3d)
target_link_libraries(test_poisson_3d third_party::gtest third_party::gtest_main)
add_test(NAME test_poisson_3d COMMAND test_poisson_3d)

# =======
# heat equation tests
add_executable(test_heat_1d heat.cpp)
target_compile_options(test_heat_1d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_heat_1d PRIVATE cxx_std_17)
target_link_libraries(test_heat_1d grid_1d)
target_link_libraries(test_heat_1d third_party::gtest third_party::gtest_main)
add_test(NAME test_heat_1d COMMAND test_heat_1d)

add_executable(test_heat_2d heat.cpp)
target_compile_options(test_heat_2d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_heat_2d PRIVATE cxx_std_17)
target_link_libraries(test_heat_2d grid_2d)
target_link_libraries(test_heat_2d third_party::gtest third_party::gtest_main)
add_test(NAME test_heat_2d COMMAND test_heat_2d)

add_executable(test_heat_3d heat.cpp)
target_compile_options(test_heat_3d PRIVATE ${PMSC_PRIVATE_COMPILE_OPTIONS})
target_compile_features(test_heat_3d PRIVATE cxx_std_17)
target_link_libraries(test_heat_3d grid_3d)
target_link_libraries(test_heat_3d third_party::gtest third_party::gtest_main)
add_test(NAME test_heat_3d COMMAND test_heat_3d)
