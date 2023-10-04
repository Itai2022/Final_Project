#pragma once

#include "contiguousparallelpartition.h"
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <vector>

template<typename T>
class SparseMatrix;

class ExchangePattern
{
private:
    std::vector<int> neighboring_processes_;
    std::vector<std::vector<int>> receive_indices_;
    std::vector<std::vector<int>> send_indices_;

public:
    ExchangePattern() = default;
    ExchangePattern(std::vector<int> neighboring_processes, std::vector<std::vector<int>> receive_indices, std::vector<std::vector<int>> send_indices);
    ~ExchangePattern() = default;

    const std::vector<int>& neighboring_processes() const;
    const std::vector<std::vector<int>>& receive_indices() const;
    const std::vector<std::vector<int>>& send_indices() const;

    friend std::ostream& operator<<(std::ostream& os, const ExchangePattern& ex);
};

template<typename T>
void unique_sorted(std::vector<T>& v)
{
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    // return std::move(v);
}

template<typename T>
inline ExchangePattern create_exchange_pattern(const SparseMatrix<T>& matrix, const ContiguousParallelPartition& column_partition)
{
    int number_of_processes;
    int myrank;
    MPI_Comm_size(column_partition.communicator(), &number_of_processes);
    MPI_Comm_rank(column_partition.communicator(), &myrank);
    std::vector<int> neighboring_processes;

    const auto& partition = matrix.row_partition();

    for(int row_index = 0; row_index < matrix.rows(); row_index++)
    {
        for(int nz_index = 0; nz_index < matrix.row_nz_size(row_index); nz_index++)
        {
            int global_index = matrix.row_nz_index(row_index, nz_index);
            int owner = column_partition.owner_process(global_index);
            if(owner != myrank)
            {
                neighboring_processes.push_back(owner);
            }
        }
    }

    unique_sorted(neighboring_processes);

    std::vector<std::vector<int>> receive_indices(neighboring_processes.size());
    std::vector<std::vector<int>> send_indices(neighboring_processes.size());

    for(int row_index = 0; row_index < matrix.rows(); row_index++)
    {
        for(int nz_index = 0; nz_index < matrix.row_nz_size(row_index); nz_index++)
        {
            int global_index = matrix.row_nz_index(row_index, nz_index);
            int owner = column_partition.owner_process(global_index);
            if(owner != myrank)
            {
                const auto it = std::find(neighboring_processes.begin(), neighboring_processes.end(), owner);
                const auto owner_index = it - neighboring_processes.begin();
                receive_indices[owner_index].push_back(global_index);
                send_indices[owner_index].push_back(partition.to_global_index(row_index));
            }
        }
    }
    for(auto& entry : receive_indices)
    {
        unique_sorted(entry);
    }

    for(auto& entry : send_indices)
    {
        unique_sorted(entry);
    }
    return ExchangePattern(neighboring_processes, receive_indices, send_indices);
}