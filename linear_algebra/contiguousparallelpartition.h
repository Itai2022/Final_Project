#pragma once

#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>

class ContiguousParallelPartition
{
private:
    MPI_Comm communicator_;
    std::vector<int> partition_;

public:
    // constructors and destructor
    ContiguousParallelPartition();
    ~ContiguousParallelPartition() = default;
    explicit ContiguousParallelPartition(MPI_Comm communicator, std::vector<int> partition);

    // getters
    MPI_Comm communicator() const;
    int local_size() const;
    int local_size(int process) const;
    int global_size() const;

    // match global indices
    int owner_process(int global_index) const;
    bool is_owned_by_local_process(int global_index) const;
    bool is_owned_by_process(int global_index, int process) const;

    // switch between local and global indices
    int to_global_index(int local_index) const;
    int to_local_index(int global_index) const;
    // for parallelized grid
    int to_global_index(int local_index, int owner_process) const;

    friend std::ostream& operator<<(std::ostream& os, const ContiguousParallelPartition& pat);
};

ContiguousParallelPartition create_partition(MPI_Comm communicator, int local_size);
ContiguousParallelPartition create_uniform_partition(MPI_Comm communicator, int global_size);