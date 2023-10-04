cmake_minimum_required(VERSION 3.2.0)


function(pmsc_check_mpi_vendor)
    set(PMSC_MPI_VENDOR "Unknown")

    include(CheckSymbolExists)
    include(CMakePushCheckState)

    cmake_push_check_state()
    set(CMAKE_REQUIRED_LIBRARIES "${MPI_CXX_LIBRARIES}")
    set(CMAKE_REQUIRED_INCLUDES  "${MPI_CXX_INCLUDE_PATH}")

    if(PMSC_MPI_VENDOR STREQUAL "Unknown")
        check_symbol_exists(OPEN_MPI mpi.h mpi_is_openmpi)
        if(mpi_is_openmpi)
            set(PMSC_MPI_VENDOR "OpenMPI")
        endif()
    endif()

    if(PMSC_MPI_VENDOR STREQUAL "Unknown")
        check_symbol_exists(I_MPI_VERSION mpi.h mpi_is_intelmpi)
        if(mpi_is_intelmpi)
            set(PMSC_MPI_VENDOR "Intel MPI")
        endif()
    endif()

    if(PMSC_MPI_VENDOR STREQUAL "Unknown")
        check_symbol_exists(MSMPI_VER mpi.h mpi_is_msmpi)
        if(mpi_is_msmpi)
            set(PMSC_MPI_VENDOR "MSMPI")
        endif()
    endif()

    cmake_pop_check_state()

    set(PMSC_MPI_VENDOR "${PMSC_MPI_VENDOR}" PARENT_SCOPE)
endfunction()