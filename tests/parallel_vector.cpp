#include <numeric>

#include <mpi.h>

#include <gtest/gtest.h>

#include "../linear_algebra/vector.h"

TEST(ParallelVectorDouble, local_values_construct)
{
    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    int local_procces_rank;
    MPI_Comm_rank(communicator, &local_procces_rank);

    const Vector<double> v(communicator, { local_procces_rank * 3 + 1.0,
                                           local_procces_rank * 3 + 2.0,
                                           local_procces_rank * 3 + 3.0 });

    ASSERT_EQ(v.partition().global_size(), number_of_processes*3);
    ASSERT_EQ(v.partition().local_size(), 3);
    ASSERT_EQ(v.size(), 3);

    EXPECT_DOUBLE_EQ(v[0], local_procces_rank * 3 + 1);
    EXPECT_DOUBLE_EQ(v[1], local_procces_rank * 3 + 2);
    EXPECT_DOUBLE_EQ(v[2], local_procces_rank * 3 + 3);
}

TEST(ParallelVectorDouble, local_size_construct)
{
    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);

    const Vector<double> v(communicator, 3);

    ASSERT_EQ(v.partition().global_size(), number_of_processes*3);
    ASSERT_EQ(v.partition().local_size(), 3);
    ASSERT_EQ(v.size(), 3);
}

TEST(ParallelVectorDouble, partition_construct)
{
    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);

    const auto partition = create_partition(communicator, 3);
    const Vector<double> v(partition);

    ASSERT_EQ(v.partition().global_size(), number_of_processes*3);
    ASSERT_EQ(v.partition().local_size(), 3);
    ASSERT_EQ(v.size(), 3);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const auto test_result_status = RUN_ALL_TESTS();
    MPI_Finalize();
    return test_result_status;
}
