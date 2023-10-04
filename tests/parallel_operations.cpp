#include <mpi.h>

#include <gtest/gtest.h>

#include "../linear_algebra/matrix.h"
#include "../linear_algebra/vector.h"
#include "../linear_algebra/operations.h"

TEST(Operations, equals_parallelvector_parallelvector)
{
    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    int local_procces_rank;
    MPI_Comm_rank(communicator, &local_procces_rank);

    const Vector<double> v1(communicator, {1.0, 2.0, 3.0});
    const Vector<double> v2(communicator, {1.0, 2.0, 3.0});
    const Vector<double> v3(communicator, {1.0, 3.0, 3.0});
    const Vector<double> v4(communicator, {1.0, local_procces_rank == 2 ? 3.0 : 2.0, 3.0});

    EXPECT_TRUE(equals(v1, v1));
    EXPECT_TRUE(equals(v2, v2));
    EXPECT_TRUE(equals(v3, v3));
    EXPECT_TRUE(equals(v4, v4));

    EXPECT_TRUE(equals(v1, v2));
    EXPECT_TRUE(equals(v2, v1));

    EXPECT_FALSE(equals(v1, v3));
    EXPECT_FALSE(equals(v3, v1));

    EXPECT_EQ(equals(v1, v4), number_of_processes <= 2);
    EXPECT_EQ(equals(v4, v1), number_of_processes <= 2);

    EXPECT_FALSE(equals(v2, v3));
    EXPECT_FALSE(equals(v3, v2));

    EXPECT_EQ(equals(v2, v4), number_of_processes <= 2);
    EXPECT_EQ(equals(v4, v2), number_of_processes <= 2);

    EXPECT_FALSE(equals(v3, v4));
    EXPECT_FALSE(equals(v4, v3));
}

TEST(Operations, assign_parallelvector_scalar)
{
    const auto communicator = MPI_COMM_WORLD;

    Vector<double> v(communicator, {1.0, 2.0, 3.0});
    assign(v, 5.0);
    const Vector<double> expected_result(communicator, {5.0, 5.0, 5.0});
    EXPECT_TRUE(equals(v, expected_result));
}

TEST(Operations, add_parallelvector_parallelvector)
{
    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    int local_procces_rank;
    MPI_Comm_rank(communicator, &local_procces_rank);

    const Vector<double> v1(communicator, {local_procces_rank * 3 + 1.0,
                                           local_procces_rank * 3 + 2.0,
                                           local_procces_rank * 3 + 3.0});
    const Vector<double> v2(communicator, {(number_of_processes + local_procces_rank) * 3 + 4.0,
                                           (number_of_processes + local_procces_rank) * 3 + 5.0,
                                           (number_of_processes + local_procces_rank) * 3 + 6.0});
    Vector<double> r(communicator, 3);
    add(r, v1, v2);
    const Vector<double> expected_result(communicator, { local_procces_rank * 3 + 1.0 + (number_of_processes + local_procces_rank) * 3 + 4.0,
                                                         local_procces_rank * 3 + 2.0 + (number_of_processes + local_procces_rank) * 3 + 5.0,
                                                         local_procces_rank * 3 + 3.0 + (number_of_processes + local_procces_rank) * 3 + 6.0 });
    EXPECT_TRUE(equals(r, expected_result));
}

TEST(Operations, subtract_parallelvector_parallelvector)
{
    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    int local_procces_rank;
    MPI_Comm_rank(communicator, &local_procces_rank);

    const Vector<double> v1(communicator, {local_procces_rank * 3 + 1.0,
                                           local_procces_rank * 3 + 2.0,
                                           local_procces_rank * 3 + 3.0});
    const Vector<double> v2(communicator, {(number_of_processes + local_procces_rank) * 3 + 4.0,
                                           (number_of_processes + local_procces_rank) * 3 + 5.0,
                                           (number_of_processes + local_procces_rank) * 3 + 6.0});
    Vector<double> r(communicator, 3);
    subtract(r, v1, v2);
    const Vector<double> expected_result(communicator, { local_procces_rank * 3 + 1.0 - (number_of_processes + local_procces_rank) * 3 - 4.0,
                                                         local_procces_rank * 3 + 2.0 - (number_of_processes + local_procces_rank) * 3 - 5.0,
                                                         local_procces_rank * 3 + 3.0 - (number_of_processes + local_procces_rank) * 3 - 6.0 });
    EXPECT_TRUE(equals(r, expected_result));
}

TEST(Operations, parallel_dot_product)
{
    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    int local_procces_rank;
    MPI_Comm_rank(communicator, &local_procces_rank);

    const Vector<double> v1(communicator, {local_procces_rank * 3 + 1.0,
                                           local_procces_rank * 3 + 2.0,
                                           local_procces_rank * 3 + 3.0});
    const Vector<double> v2(communicator, {(number_of_processes + local_procces_rank) * 3 + 4.0,
                                           (number_of_processes + local_procces_rank) * 3 + 5.0,
                                           (number_of_processes + local_procces_rank) * 3 + 6.0});

    double expected_result = 0;
    for(int p = 0; p < number_of_processes; ++p)
    {
        for(int k = 1; k <= 3; ++k)
        {
            expected_result += (3 * p + k) * (3 * (number_of_processes + p + 1) + k);
        }
    }

    EXPECT_DOUBLE_EQ(dot_product(v1, v2), expected_result);
}

TEST(Operations, multiply_parallelsparsematrix_parallelvector)
{
    using triplet = SparseMatrix<double>::triplet_type;

    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    int local_procces_rank;
    MPI_Comm_rank(communicator, &local_procces_rank);

    const auto global_columns = number_of_processes*4;
    const auto partition = create_partition(communicator, 4);
    SparseMatrix<double> m(partition, global_columns, { triplet{ 0, local_procces_rank*4 + 0, local_procces_rank * 4 + 10.0 }, triplet{ 0, local_procces_rank*4 + 1, local_procces_rank * 4 + 12.0 }, triplet{ 0, local_procces_rank > 0 ? (local_procces_rank*4 - 1) : (global_columns - 1), local_procces_rank * 4 + 8.0 },
                                                              triplet{ 1, local_procces_rank*4 + 1, local_procces_rank * 4 + 20.0 }, triplet{ 1, local_procces_rank*4 + 2, local_procces_rank * 4 + 22.0 }, triplet{ 1, local_procces_rank*4 + 0, local_procces_rank * 4 + 18.0 },
                                                              triplet{ 2, local_procces_rank*4 + 2, local_procces_rank * 4 + 30.0 }, triplet{ 2, local_procces_rank*4 + 3, local_procces_rank * 4 + 32.0 }, triplet{ 2, local_procces_rank*4 + 1, local_procces_rank * 4 + 28.0 },
                                                              triplet{ 3, local_procces_rank*4 + 3, local_procces_rank * 4 + 40.0 }, triplet{ 3, (local_procces_rank*4 + 4) % global_columns, local_procces_rank * 4 + 42.0 }, triplet{ 3, local_procces_rank*4 + 2, local_procces_rank * 4 + 38.0 } });

    m.initialize_exchange_pattern(partition);

    const Vector<double> v(communicator, { local_procces_rank * 4 + 1.0,
                                           local_procces_rank * 4 + 2.0,
                                           local_procces_rank * 4 + 3.0,
                                           local_procces_rank * 4 + 4.0 });

    Vector<double> r(partition);
    multiply(r, m, v);

    const Vector<double> expected_result(communicator, { (local_procces_rank * 4 + 1.0)*(local_procces_rank * 4 + 10.0) + (local_procces_rank == 0 ? ((number_of_processes-1)*4 + 4.0) : (local_procces_rank * 4 + 0.0))*(local_procces_rank * 4 + 8.0) + (local_procces_rank * 4 + 2.0)*(local_procces_rank * 4 + 12.0),
                                                         (local_procces_rank * 4 + 2.0)*(local_procces_rank * 4 + 20.0) + (local_procces_rank * 4 + 1.0)*(local_procces_rank * 4 + 18.0) + (local_procces_rank * 4 + 3.0)*(local_procces_rank * 4 + 22.0),
                                                         (local_procces_rank * 4 + 3.0)*(local_procces_rank * 4 + 30.0) + (local_procces_rank * 4 + 2.0)*(local_procces_rank * 4 + 28.0) + (local_procces_rank * 4 + 4.0)*(local_procces_rank * 4 + 32.0),
                                                         (local_procces_rank * 4 + 4.0)*(local_procces_rank * 4 + 40.0) + (local_procces_rank * 4 + 3.0)*(local_procces_rank * 4 + 38.0) + (local_procces_rank == number_of_processes - 1 ? 1.0 : (local_procces_rank * 4 + 5.0))*(local_procces_rank * 4 + 42.0) });

    EXPECT_TRUE(equals(r, expected_result));
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const auto test_result_status = RUN_ALL_TESTS();
    MPI_Finalize();
    return test_result_status;
}