#include <numeric>

#include <mpi.h>

#include <gtest/gtest.h>

#include "../linear_algebra/contiguousparallelpartition.h"

TEST(Partition, default_construct)
{
    const ContiguousParallelPartition partition;
    (void)partition;
}

TEST(Partition, create_with_partition)
{
    const auto communicator = MPI_COMM_WORLD;
    int number_of_processes;
    int local_process_rank;
    MPI_Comm_size(communicator, &number_of_processes);
    MPI_Comm_rank(communicator, &local_process_rank);

    std::vector<int> partition(number_of_processes + 1);
    std::iota(partition.begin(), partition.end(), 0);
    const int local_size = partition[local_process_rank + 1];
    const auto global_size = std::accumulate(partition.begin(), partition.end(), 0);
    std::partial_sum(partition.begin(), partition.end(), partition.begin());
    const auto local_begin = partition[local_process_rank];

    const ContiguousParallelPartition parallel_partition(communicator, std::move(partition));

    EXPECT_EQ(parallel_partition.global_size(), global_size);
    EXPECT_EQ(parallel_partition.local_size(), local_process_rank + 1);

    for(int p = 0; p < number_of_processes; ++p)
    {
        EXPECT_EQ(parallel_partition.local_size(p), p + 1);
    }

    for(int local_index = 0; local_index < local_size; ++local_index)
    {
        EXPECT_EQ(parallel_partition.to_global_index(local_index), local_index + local_begin);
    }

    int current_owner = 0;
    int last_process_begin = 0;
    for(int global_index = 0; global_index < global_size; ++global_index)
    {
        if(global_index >= last_process_begin + current_owner + 1)
        {
            last_process_begin += current_owner + 1;
            ++current_owner;
        }

        EXPECT_EQ(parallel_partition.owner_process(global_index), current_owner);
        EXPECT_TRUE(parallel_partition.is_owned_by_process(global_index, current_owner));

        if(current_owner == local_process_rank)
        {
            EXPECT_TRUE(parallel_partition.is_owned_by_local_process(global_index));
            EXPECT_EQ(parallel_partition.to_local_index(global_index), global_index - local_begin);
        }
    }
}

TEST(Partition, create_with_local_sizes)
{
    const auto communicator = MPI_COMM_WORLD;
    int number_of_processes;
    int local_process_rank;
    MPI_Comm_size(communicator, &number_of_processes);
    MPI_Comm_rank(communicator, &local_process_rank);

    const int local_size = local_process_rank + 1;
    const int global_size = (number_of_processes * (number_of_processes + 1)) / 2;

    const int local_begin = (local_process_rank * (local_process_rank + 1)) / 2;

    const auto parallel_partition = create_partition(communicator, local_size);

    EXPECT_EQ(parallel_partition.local_size(), local_size);
    EXPECT_EQ(parallel_partition.global_size(), global_size);

    for(int p = 0; p < number_of_processes; ++p)
    {
        EXPECT_EQ(parallel_partition.local_size(p), p + 1);
    }

    for(int local_index = 0; local_index < local_size; ++local_index)
    {
        EXPECT_EQ(parallel_partition.to_global_index(local_index), local_index + local_begin);
    }

    int current_owner = 0;
    int last_process_begin = 0;
    for(int global_index = 0; global_index < global_size; ++global_index)
    {
        if(global_index >= last_process_begin + current_owner + 1)
        {
            last_process_begin += current_owner + 1;
            ++current_owner;
        }

        EXPECT_EQ(parallel_partition.owner_process(global_index), current_owner);
        EXPECT_TRUE(parallel_partition.is_owned_by_process(global_index, current_owner));

        if(current_owner == local_process_rank)
        {
            EXPECT_TRUE(parallel_partition.is_owned_by_local_process(global_index));
            EXPECT_EQ(parallel_partition.to_local_index(global_index), global_index - local_begin);
        }
    }
}

TEST(Partition, create_uniform)
{
    const auto communicator = MPI_COMM_WORLD;
    int number_of_processes;
    int local_process_rank;
    MPI_Comm_size(communicator, &number_of_processes);
    MPI_Comm_rank(communicator, &local_process_rank);
    
    const auto global_size = number_of_processes*4 + 1;

    const auto parallel_partition = create_uniform_partition(communicator, global_size);

    EXPECT_EQ(parallel_partition.global_size(), global_size);
    EXPECT_EQ(parallel_partition.local_size(), local_process_rank == number_of_processes-1 ? 5 : 4);

    for(int p = 0; p < number_of_processes; ++p)
    {
        EXPECT_EQ(parallel_partition.local_size(p), p == number_of_processes-1 ? 5 : 4);
    }
    for(int global_index = 0; global_index < global_size; ++global_index)
    {
        const auto owner = global_index >= number_of_processes*4 ? (number_of_processes-1) : (global_index / 4);
        EXPECT_EQ(parallel_partition.owner_process(global_index), owner);
        EXPECT_TRUE(parallel_partition.is_owned_by_process(global_index, owner));
        EXPECT_EQ(parallel_partition.is_owned_by_local_process(global_index), owner == local_process_rank);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const auto test_result_status = RUN_ALL_TESTS();
    MPI_Finalize();
    return test_result_status;
}
