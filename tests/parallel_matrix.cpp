#include <numeric>

#include <mpi.h>

#include <gtest/gtest.h>

#include "../linear_algebra/matrix.h"
#include "../linear_algebra/exchangepattern.h"

TEST(ParallelSparseMatrix, local_values_construct)
{
    using triplet = SparseMatrix<double>::triplet_type;

    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    int local_procces_rank;
    MPI_Comm_rank(communicator, &local_procces_rank);

    const SparseMatrix<double> m(communicator, 4, number_of_processes*3, { triplet{ 0, local_procces_rank*3 + 0, 1.0 }, triplet{ 0, local_procces_rank*3 + 2, 2.0 },
                                                                           triplet{ 1, local_procces_rank*3 + 0, 3.0 },
                                                                           triplet{ 2, local_procces_rank*3 + 0, 4.0 }, triplet{ 2, local_procces_rank*3 + 1, 5.0 }, triplet{ 2, local_procces_rank*3 + 2, 6.0 },
                                                                           triplet{ 3, local_procces_rank*3 + 2, 7.0 } });

    ASSERT_EQ(m.row_partition().global_size(), number_of_processes*4);
    ASSERT_EQ(m.row_partition().local_size(), 4);
    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.columns(), number_of_processes*3);
    EXPECT_EQ(m.non_zero_size(), 7);

    EXPECT_EQ(m.row_nz_size(0), 2);
    EXPECT_EQ(m.row_nz_size(1), 1);
    EXPECT_EQ(m.row_nz_size(2), 3);
    EXPECT_EQ(m.row_nz_size(3), 1);
    
    const double local_dense_values[4][3] = {{1.0, 0.0, 2.0},
                                             {3.0, 0.0, 0.0},
                                             {4.0, 5.0, 6.0},
                                             {0.0, 0.0, 7.0}};

    EXPECT_DOUBLE_EQ(m.row_nz_entry(0, 0), local_dense_values[0][m.row_nz_index(0, 0) - local_procces_rank*3]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(0, 1), local_dense_values[0][m.row_nz_index(0, 1) - local_procces_rank*3]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(1, 0), local_dense_values[1][m.row_nz_index(1, 0) - local_procces_rank*3]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 0), local_dense_values[2][m.row_nz_index(2, 0) - local_procces_rank*3]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 1), local_dense_values[2][m.row_nz_index(2, 1) - local_procces_rank*3]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 2), local_dense_values[2][m.row_nz_index(2, 2) - local_procces_rank*3]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(3, 0), local_dense_values[3][m.row_nz_index(3, 0) - local_procces_rank*3]);
}

TEST(ParallelSparseMatrix, partition_local_values_construct)
{
    using triplet = SparseMatrix<double>::triplet_type;

    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    int local_procces_rank;
    MPI_Comm_rank(communicator, &local_procces_rank);

    const auto partition = create_partition(communicator, 4);
    const SparseMatrix<double> m(partition, number_of_processes*3, { triplet{ 0, local_procces_rank*3 + 0, 1.0 }, triplet{ 0, local_procces_rank*3 + 2, 2.0 },
                                                                     triplet{ 1, local_procces_rank*3 + 0, 3.0 },
                                                                     triplet{ 2, local_procces_rank*3 + 0, 4.0 }, triplet{ 2, local_procces_rank*3 + 1, 5.0 }, triplet{ 2, local_procces_rank*3 + 2, 6.0 },
                                                                     triplet{ 3, local_procces_rank*3 + 2, 7.0 } });

    ASSERT_EQ(m.row_partition().global_size(), number_of_processes*4);
    ASSERT_EQ(m.row_partition().local_size(), 4);
    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.columns(), number_of_processes*3);
    EXPECT_EQ(m.non_zero_size(), 7);

    EXPECT_EQ(m.row_nz_size(0), 2);
    EXPECT_EQ(m.row_nz_size(1), 1);
    EXPECT_EQ(m.row_nz_size(2), 3);
    EXPECT_EQ(m.row_nz_size(3), 1);
    
    const double local_dense_values[4][3] = {{1.0, 0.0, 2.0},
                                             {3.0, 0.0, 0.0},
                                             {4.0, 5.0, 6.0},
                                             {0.0, 0.0, 7.0}};

    EXPECT_DOUBLE_EQ(m.row_nz_entry(0, 0), local_dense_values[0][m.row_nz_index(0, 0) - local_procces_rank*3]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(0, 1), local_dense_values[0][m.row_nz_index(0, 1) - local_procces_rank*3]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(1, 0), local_dense_values[1][m.row_nz_index(1, 0) - local_procces_rank*3]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 0), local_dense_values[2][m.row_nz_index(2, 0) - local_procces_rank*3]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 1), local_dense_values[2][m.row_nz_index(2, 1) - local_procces_rank*3]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 2), local_dense_values[2][m.row_nz_index(2, 2) - local_procces_rank*3]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(3, 0), local_dense_values[3][m.row_nz_index(3, 0) - local_procces_rank*3]);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const auto test_result_status = RUN_ALL_TESTS();
    MPI_Finalize();
    return test_result_status;
}
