#include <mpi.h>

#include <gtest/gtest.h>

#include "../linear_algebra/exchangepattern.h"
#include "../linear_algebra/matrix.h"
#include "../linear_algebra/vector.h"

template<typename T>
std::vector<T> make_unique_sorted(std::vector<T> v)
{
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return std::move(v);
}

template<typename T>
std::vector<std::vector<T>> make_unique_sorted(std::vector<std::vector<T>> v)
{
    for(auto& entry : v)
    {
        std::sort(entry.begin(), entry.end());
        entry.erase(std::unique(entry.begin(), entry.end()), entry.end());
    }
    return std::move(v);
}

TEST(ExchangePattern, create_diagonal_matrix)
{
    using triplet = SparseMatrix<double>::triplet_type;

    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    int local_procces_rank;
    MPI_Comm_rank(communicator, &local_procces_rank);

    const auto partition = create_partition(communicator, 4);
    const SparseMatrix<double> m(partition, number_of_processes*4, { triplet{ 0, local_procces_rank*4 + 0, 1.0 },
                                                                     triplet{ 1, local_procces_rank*4 + 1, 3.0 },
                                                                     triplet{ 2, local_procces_rank*4 + 2, 4.0 },
                                                                     triplet{ 3, local_procces_rank*4 + 3, 7.0 } });

    const auto exchange_pattern = create_exchange_pattern(m, partition);

    EXPECT_EQ(exchange_pattern.neighboring_processes(), std::vector<int>{});
    EXPECT_EQ(exchange_pattern.send_indices(), std::vector<std::vector<int>>{});
    EXPECT_EQ(exchange_pattern.receive_indices(), std::vector<std::vector<int>>{});
}

TEST(ExchangePattern, create_tridiagonal_matrix)
{
    using triplet = SparseMatrix<double>::triplet_type;

    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    int local_procces_rank;
    MPI_Comm_rank(communicator, &local_procces_rank);

    const auto global_columns = number_of_processes*4;
    const auto partition = create_partition(communicator, 4);
    const SparseMatrix<double> m(partition, global_columns, { triplet{ 0, local_procces_rank*4 + 0, 1.0 }, triplet{ 0, local_procces_rank*4 + 1, 1.0 }, triplet{ 0, local_procces_rank > 0 ? (local_procces_rank*4 - 1) : (global_columns - 1), 1.0 },
                                                              triplet{ 1, local_procces_rank*4 + 1, 3.0 }, triplet{ 1, local_procces_rank*4 + 2, 1.0 }, triplet{ 1, local_procces_rank*4 + 0, 1.0 },
                                                              triplet{ 2, local_procces_rank*4 + 2, 4.0 }, triplet{ 2, local_procces_rank*4 + 3, 1.0 }, triplet{ 2, local_procces_rank*4 + 1, 1.0 },
                                                              triplet{ 3, local_procces_rank*4 + 3, 7.0 }, triplet{ 3, (local_procces_rank*4 + 4) % global_columns, 1.0 }, triplet{ 3, local_procces_rank*4 + 2, 1.0 } });

    const auto exchange_pattern = create_exchange_pattern(m, partition);

    EXPECT_TRUE(std::is_sorted(exchange_pattern.neighboring_processes().begin(), exchange_pattern.neighboring_processes().end()));

    if(number_of_processes > 1)
    {
        const auto prev_neighbor = local_procces_rank > 0 ? local_procces_rank - 1 : number_of_processes - 1;
        const auto next_neighbor = (local_procces_rank + 1) % number_of_processes;

        const auto prev_neighbor_index = prev_neighbor > next_neighbor ? 1 : 0;
        const auto next_neighbor_index = next_neighbor > prev_neighbor ? 1 : 0;

        const auto prev_index = local_procces_rank > 0 ? (local_procces_rank*4 - 1) : (global_columns - 1);
        const auto next_index = (local_procces_rank*4 + 4) % global_columns;

        const auto first_index = local_procces_rank*4 + 0;
        const auto last_index = local_procces_rank*4 + 3;

        const auto expected_neighboring_processes = make_unique_sorted(std::vector<int>{prev_neighbor, next_neighbor});

        EXPECT_EQ(exchange_pattern.neighboring_processes(), expected_neighboring_processes);

        std::vector<std::vector<int>> expected_send_indices(expected_neighboring_processes.size());
        expected_send_indices[prev_neighbor_index].push_back(first_index);
        expected_send_indices[next_neighbor_index].push_back(last_index);

        EXPECT_EQ(exchange_pattern.send_indices(), make_unique_sorted(expected_send_indices));

        std::vector<std::vector<int>> expected_receive_indices(expected_neighboring_processes.size());
        expected_receive_indices[prev_neighbor_index].push_back(prev_index);
        expected_receive_indices[next_neighbor_index].push_back(next_index);

        EXPECT_EQ(exchange_pattern.receive_indices(), make_unique_sorted(expected_receive_indices));
    }
    else
    {
        EXPECT_EQ(exchange_pattern.neighboring_processes(), std::vector<int>{});
        EXPECT_EQ(exchange_pattern.send_indices(), std::vector<std::vector<int>>{});
        EXPECT_EQ(exchange_pattern.receive_indices(), std::vector<std::vector<int>>{});
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
