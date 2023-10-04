#include <mpi.h>

#include <gtest/gtest.h>

#include "../linear_algebra/exchangedata.h"
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

TEST(ExchangeData, exchange_tri_diagonal)
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
    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    int local_procces_rank;
    MPI_Comm_rank(communicator, &local_procces_rank);

    const auto global_columns = number_of_processes*4;
    const auto partition = create_partition(communicator, 4);

    const auto prev_neighbor = local_procces_rank > 0 ? local_procces_rank - 1 : number_of_processes - 1;
    const auto next_neighbor = (local_procces_rank + 1) % number_of_processes;

    const auto prev_neighbor_index = prev_neighbor > next_neighbor ? 1 : 0;
    const auto next_neighbor_index = next_neighbor > prev_neighbor ? 1 : 0;

    const auto prev_index = local_procces_rank > 0 ? (local_procces_rank*4 - 1) : (global_columns - 1);
    const auto next_index = (local_procces_rank*4 + 4) % global_columns;

    const auto first_index = local_procces_rank*4 + 0;
    const auto last_index = local_procces_rank*4 + 3;

    auto neighboring_processes = make_unique_sorted(std::vector<int>{prev_neighbor, next_neighbor});

    std::vector<std::vector<int>> send_indices(neighboring_processes.size());
    send_indices[prev_neighbor_index].push_back(first_index);
    send_indices[next_neighbor_index].push_back(last_index);

    std::vector<std::vector<int>> receive_indices(neighboring_processes.size());
    receive_indices[prev_neighbor_index].push_back(prev_index);
    receive_indices[next_neighbor_index].push_back(next_index);

    const auto exchange_pattern = ExchangePattern(std::move(neighboring_processes),
                                                  make_unique_sorted(std::move(receive_indices)),
                                                  make_unique_sorted(std::move(send_indices)));

    const Vector<double> v(communicator, { local_procces_rank * 4 + 1.0,
                                           local_procces_rank * 4 + 2.0,
                                           local_procces_rank * 4 + 3.0,
                                           local_procces_rank * 4 + 4.0 });

    const auto remote_data = exchange_vector_data(exchange_pattern, v);

    if(number_of_processes > 1)
    {
        EXPECT_DOUBLE_EQ(remote_data.get(prev_neighbor, prev_index), prev_index + 1.0);
        EXPECT_DOUBLE_EQ(remote_data.get(next_neighbor, next_index), next_index + 1.0);
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
