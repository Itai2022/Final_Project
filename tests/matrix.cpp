#include <mpi.h>

#include <gtest/gtest.h>

#include "../linear_algebra/matrix.h"
#include "../linear_algebra/vector.h"
#include "../linear_algebra/operations.h"

TEST(SparseMatrix, default_construct)
{
    const SparseMatrix<double> v;
    (void)v;
}

TEST(SparseMatrix, value_construct)
{
    using triplet = SparseMatrix<double>::triplet_type;

    const SparseMatrix<double> m(4, 3, { triplet{ 0, 0, 1.0 }, triplet{ 0, 2, 2.0 },
                                         triplet{ 1, 0, 3.0 },
                                         triplet{ 2, 0, 4.0 }, triplet{ 2, 1, 5.0 }, triplet{ 2, 2, 6.0 },
                                         triplet{ 3, 2, 7.0 } });

    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.columns(), 3);
    EXPECT_EQ(m.non_zero_size(), 7);

    EXPECT_EQ(m.row_nz_size(0), 2);
    EXPECT_EQ(m.row_nz_size(1), 1);
    EXPECT_EQ(m.row_nz_size(2), 3);
    EXPECT_EQ(m.row_nz_size(3), 1);
    
    const double dense_values[4][3] = {{1.0, 0.0, 2.0},
                                       {3.0, 0.0, 0.0},
                                       {4.0, 5.0, 6.0},
                                       {0.0, 0.0, 7.0}};

    EXPECT_DOUBLE_EQ(m.row_nz_entry(0, 0), dense_values[0][m.row_nz_index(0, 0)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(0, 1), dense_values[0][m.row_nz_index(0, 1)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(1, 0), dense_values[1][m.row_nz_index(1, 0)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 0), dense_values[2][m.row_nz_index(2, 0)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 1), dense_values[2][m.row_nz_index(2, 1)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 2), dense_values[2][m.row_nz_index(2, 2)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(3, 0), dense_values[3][m.row_nz_index(3, 0)]);
}

// TEST(SparseMatrix, size_construct)
// {
//     const SparseMatrix<double> m(4, 3, 7);

//     EXPECT_EQ(m.rows(), 4);
//     EXPECT_EQ(m.columns(), 3);
//     EXPECT_EQ(m.non_zero_size(), 7);
// }

TEST(SparseMatrix, copy_construct)
{
    using triplet = SparseMatrix<double>::triplet_type;

    const SparseMatrix<double> m1(4, 3, { triplet{ 0, 0, 1.0 }, triplet{ 0, 2, 2.0 },
                                          triplet{ 1, 0, 3.0 },
                                          triplet{ 2, 0, 4.0 }, triplet{ 2, 1, 5.0 }, triplet{ 2, 2, 6.0 },
                                          triplet{ 3, 2, 7.0 } });
    const SparseMatrix<double> m2 = m1;

    const double dense_values[4][3] = {{1.0, 0.0, 2.0},
                                         {3.0, 0.0, 0.0},
                                         {4.0, 5.0, 6.0},
                                         {0.0, 0.0, 7.0}};

    EXPECT_EQ(m1.rows(), 4);
    EXPECT_EQ(m1.columns(), 3);
    EXPECT_EQ(m1.non_zero_size(), 7);

    EXPECT_EQ(m1.row_nz_size(0), 2);
    EXPECT_EQ(m1.row_nz_size(1), 1);
    EXPECT_EQ(m1.row_nz_size(2), 3);
    EXPECT_EQ(m1.row_nz_size(3), 1);
    
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(0, 0), dense_values[0][m1.row_nz_index(0, 0)]);
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(0, 1), dense_values[0][m1.row_nz_index(0, 1)]);
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(1, 0), dense_values[1][m1.row_nz_index(1, 0)]);
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(2, 0), dense_values[2][m1.row_nz_index(2, 0)]);
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(2, 1), dense_values[2][m1.row_nz_index(2, 1)]);
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(2, 2), dense_values[2][m1.row_nz_index(2, 2)]);
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(3, 0), dense_values[3][m1.row_nz_index(3, 0)]);

    EXPECT_EQ(m2.rows(), 4);
    EXPECT_EQ(m2.columns(), 3);
    EXPECT_EQ(m2.non_zero_size(), 7);

    EXPECT_EQ(m2.row_nz_size(0), 2);
    EXPECT_EQ(m2.row_nz_size(1), 1);
    EXPECT_EQ(m2.row_nz_size(2), 3);
    EXPECT_EQ(m2.row_nz_size(3), 1);
    
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(0, 0), dense_values[0][m2.row_nz_index(0, 0)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(0, 1), dense_values[0][m2.row_nz_index(0, 1)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(1, 0), dense_values[1][m2.row_nz_index(1, 0)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 0), dense_values[2][m2.row_nz_index(2, 0)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 1), dense_values[2][m2.row_nz_index(2, 1)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 2), dense_values[2][m2.row_nz_index(2, 2)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(3, 0), dense_values[3][m2.row_nz_index(3, 0)]);
}

TEST(SparseMatrix, copy_assign)
{
    const double dense_values[4][3] = {{1.0, 0.0, 2.0},
                                       {3.0, 0.0, 0.0},
                                       {4.0, 5.0, 6.0},
                                       {0.0, 0.0, 7.0}};

    SparseMatrix<double> m2;

    {
        using triplet = SparseMatrix<double>::triplet_type;

        const SparseMatrix<double> m1(4, 3, { triplet{ 0, 0, 1.0 }, triplet{ 0, 2, 2.0 },
                                              triplet{ 1, 0, 3.0 },
                                              triplet{ 2, 0, 4.0 }, triplet{ 2, 1, 5.0 }, triplet{ 2, 2, 6.0 },
                                              triplet{ 3, 2, 7.0 } });

        m2 = m1;

        EXPECT_EQ(m1.rows(), 4);
        EXPECT_EQ(m1.columns(), 3);
        EXPECT_EQ(m1.non_zero_size(), 7);

        EXPECT_EQ(m1.row_nz_size(0), 2);
        EXPECT_EQ(m1.row_nz_size(1), 1);
        EXPECT_EQ(m1.row_nz_size(2), 3);
        EXPECT_EQ(m1.row_nz_size(3), 1);
        
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(0, 0), dense_values[0][m1.row_nz_index(0, 0)]);
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(0, 1), dense_values[0][m1.row_nz_index(0, 1)]);
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(1, 0), dense_values[1][m1.row_nz_index(1, 0)]);
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(2, 0), dense_values[2][m1.row_nz_index(2, 0)]);
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(2, 1), dense_values[2][m1.row_nz_index(2, 1)]);
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(2, 2), dense_values[2][m1.row_nz_index(2, 2)]);
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(3, 0), dense_values[3][m1.row_nz_index(3, 0)]);

        EXPECT_EQ(m2.rows(), 4);
        EXPECT_EQ(m2.columns(), 3);
        EXPECT_EQ(m2.non_zero_size(), 7);

        EXPECT_EQ(m2.row_nz_size(0), 2);
        EXPECT_EQ(m2.row_nz_size(1), 1);
        EXPECT_EQ(m2.row_nz_size(2), 3);
        EXPECT_EQ(m2.row_nz_size(3), 1);
        
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(0, 0), dense_values[0][m2.row_nz_index(0, 0)]);
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(0, 1), dense_values[0][m2.row_nz_index(0, 1)]);
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(1, 0), dense_values[1][m2.row_nz_index(1, 0)]);
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 0), dense_values[2][m2.row_nz_index(2, 0)]);
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 1), dense_values[2][m2.row_nz_index(2, 1)]);
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 2), dense_values[2][m2.row_nz_index(2, 2)]);
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(3, 0), dense_values[3][m2.row_nz_index(3, 0)]);
    }

    EXPECT_EQ(m2.rows(), 4);
    EXPECT_EQ(m2.columns(), 3);
    EXPECT_EQ(m2.non_zero_size(), 7);

    EXPECT_EQ(m2.row_nz_size(0), 2);
    EXPECT_EQ(m2.row_nz_size(1), 1);
    EXPECT_EQ(m2.row_nz_size(2), 3);
    EXPECT_EQ(m2.row_nz_size(3), 1);
    
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(0, 0), dense_values[0][m2.row_nz_index(0, 0)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(0, 1), dense_values[0][m2.row_nz_index(0, 1)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(1, 0), dense_values[1][m2.row_nz_index(1, 0)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 0), dense_values[2][m2.row_nz_index(2, 0)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 1), dense_values[2][m2.row_nz_index(2, 1)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 2), dense_values[2][m2.row_nz_index(2, 2)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(3, 0), dense_values[3][m2.row_nz_index(3, 0)]);
}

TEST(SparseMatrix, move_construct)
{
    const double dense_values[4][3] = {{1.0, 0.0, 2.0},
                                       {3.0, 0.0, 0.0},
                                       {4.0, 5.0, 6.0},
                                       {0.0, 0.0, 7.0}};

    using triplet = SparseMatrix<double>::triplet_type;

    const SparseMatrix<double> m1(4, 3, { triplet{ 0, 0, 1.0 }, triplet{ 0, 2, 2.0 },
                                          triplet{ 1, 0, 3.0 },
                                          triplet{ 2, 0, 4.0 }, triplet{ 2, 1, 5.0 }, triplet{ 2, 2, 6.0 },
                                          triplet{ 3, 2, 7.0 } });

    EXPECT_EQ(m1.rows(), 4);
    EXPECT_EQ(m1.columns(), 3);
    EXPECT_EQ(m1.non_zero_size(), 7);

    EXPECT_EQ(m1.row_nz_size(0), 2);
    EXPECT_EQ(m1.row_nz_size(1), 1);
    EXPECT_EQ(m1.row_nz_size(2), 3);
    EXPECT_EQ(m1.row_nz_size(3), 1);
    
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(0, 0), dense_values[0][m1.row_nz_index(0, 0)]);
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(0, 1), dense_values[0][m1.row_nz_index(0, 1)]);
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(1, 0), dense_values[1][m1.row_nz_index(1, 0)]);
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(2, 0), dense_values[2][m1.row_nz_index(2, 0)]);
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(2, 1), dense_values[2][m1.row_nz_index(2, 1)]);
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(2, 2), dense_values[2][m1.row_nz_index(2, 2)]);
    EXPECT_DOUBLE_EQ(m1.row_nz_entry(3, 0), dense_values[3][m1.row_nz_index(3, 0)]);

    const SparseMatrix<double> m2 = std::move(m1);

    EXPECT_EQ(m2.rows(), 4);
    EXPECT_EQ(m2.columns(), 3);
    EXPECT_EQ(m2.non_zero_size(), 7);

    EXPECT_EQ(m2.row_nz_size(0), 2);
    EXPECT_EQ(m2.row_nz_size(1), 1);
    EXPECT_EQ(m2.row_nz_size(2), 3);
    EXPECT_EQ(m2.row_nz_size(3), 1);
    
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(0, 0), dense_values[0][m2.row_nz_index(0, 0)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(0, 1), dense_values[0][m2.row_nz_index(0, 1)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(1, 0), dense_values[1][m2.row_nz_index(1, 0)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 0), dense_values[2][m2.row_nz_index(2, 0)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 1), dense_values[2][m2.row_nz_index(2, 1)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 2), dense_values[2][m2.row_nz_index(2, 2)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(3, 0), dense_values[3][m2.row_nz_index(3, 0)]);
}

TEST(SparseMatrix, move_assign)
{
    const double dense_values[4][3] = {{1.0, 0.0, 2.0},
                                       {3.0, 0.0, 0.0},
                                       {4.0, 5.0, 6.0},
                                       {0.0, 0.0, 7.0}};

    using triplet = SparseMatrix<double>::triplet_type;

    SparseMatrix<double> m2;

    {
        const SparseMatrix<double> m1(4, 3, { triplet{ 0, 0, 1.0 }, triplet{ 0, 2, 2.0 },
                                              triplet{ 1, 0, 3.0 },
                                              triplet{ 2, 0, 4.0 }, triplet{ 2, 1, 5.0 }, triplet{ 2, 2, 6.0 },
                                              triplet{ 3, 2, 7.0 } });

        EXPECT_EQ(m1.rows(), 4);
        EXPECT_EQ(m1.columns(), 3);
        EXPECT_EQ(m1.non_zero_size(), 7);

        EXPECT_EQ(m1.row_nz_size(0), 2);
        EXPECT_EQ(m1.row_nz_size(1), 1);
        EXPECT_EQ(m1.row_nz_size(2), 3);
        EXPECT_EQ(m1.row_nz_size(3), 1);
        
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(0, 0), dense_values[0][m1.row_nz_index(0, 0)]);
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(0, 1), dense_values[0][m1.row_nz_index(0, 1)]);
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(1, 0), dense_values[1][m1.row_nz_index(1, 0)]);
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(2, 0), dense_values[2][m1.row_nz_index(2, 0)]);
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(2, 1), dense_values[2][m1.row_nz_index(2, 1)]);
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(2, 2), dense_values[2][m1.row_nz_index(2, 2)]);
        EXPECT_DOUBLE_EQ(m1.row_nz_entry(3, 0), dense_values[3][m1.row_nz_index(3, 0)]);

        m2 = std::move(m1);

        EXPECT_EQ(m2.rows(), 4);
        EXPECT_EQ(m2.columns(), 3);
        EXPECT_EQ(m2.non_zero_size(), 7);

        EXPECT_EQ(m2.row_nz_size(0), 2);
        EXPECT_EQ(m2.row_nz_size(1), 1);
        EXPECT_EQ(m2.row_nz_size(2), 3);
        EXPECT_EQ(m2.row_nz_size(3), 1);

        EXPECT_DOUBLE_EQ(m2.row_nz_entry(0, 0), dense_values[0][m2.row_nz_index(0, 0)]);
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(0, 1), dense_values[0][m2.row_nz_index(0, 1)]);
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(1, 0), dense_values[1][m2.row_nz_index(1, 0)]);
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 0), dense_values[2][m2.row_nz_index(2, 0)]);
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 1), dense_values[2][m2.row_nz_index(2, 1)]);
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 2), dense_values[2][m2.row_nz_index(2, 2)]);
        EXPECT_DOUBLE_EQ(m2.row_nz_entry(3, 0), dense_values[3][m2.row_nz_index(3, 0)]);
    }

    EXPECT_EQ(m2.rows(), 4);
    EXPECT_EQ(m2.columns(), 3);
    EXPECT_EQ(m2.non_zero_size(), 7);

    EXPECT_EQ(m2.row_nz_size(0), 2);
    EXPECT_EQ(m2.row_nz_size(1), 1);
    EXPECT_EQ(m2.row_nz_size(2), 3);
    EXPECT_EQ(m2.row_nz_size(3), 1);
    
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(0, 0), dense_values[0][m2.row_nz_index(0, 0)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(0, 1), dense_values[0][m2.row_nz_index(0, 1)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(1, 0), dense_values[1][m2.row_nz_index(1, 0)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 0), dense_values[2][m2.row_nz_index(2, 0)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 1), dense_values[2][m2.row_nz_index(2, 1)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(2, 2), dense_values[2][m2.row_nz_index(2, 2)]);
    EXPECT_DOUBLE_EQ(m2.row_nz_entry(3, 0), dense_values[3][m2.row_nz_index(3, 0)]);
}

TEST(SparseMatrix, value_construct_modify)
{
    using triplet = SparseMatrix<double>::triplet_type;

    SparseMatrix<double> m(4, 3, { triplet{ 0, 0, 1.0 }, triplet{ 0, 2, 2.0 },
                                   triplet{ 1, 0, 3.0 },
                                   triplet{ 2, 0, 4.0 }, triplet{ 2, 1, 5.0 }, triplet{ 2, 2, 6.0 },
                                   triplet{ 3, 2, 7.0 } });

    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.columns(), 3);
    EXPECT_EQ(m.non_zero_size(), 7);

    EXPECT_EQ(m.row_nz_size(0), 2);
    EXPECT_EQ(m.row_nz_size(1), 1);
    EXPECT_EQ(m.row_nz_size(2), 3);
    EXPECT_EQ(m.row_nz_size(3), 1);

    const double dense_values[4][3] = {{1.0, 0.0, 2.0},
                                       {3.0, 0.0, 0.0},
                                       {4.0, 5.0, 6.0},
                                       {0.0, 0.0, 7.0}};

    EXPECT_DOUBLE_EQ(m.row_nz_entry(0, 0), dense_values[0][m.row_nz_index(0, 0)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(0, 1), dense_values[0][m.row_nz_index(0, 1)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(1, 0), dense_values[1][m.row_nz_index(1, 0)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 0), dense_values[2][m.row_nz_index(2, 0)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 1), dense_values[2][m.row_nz_index(2, 1)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 2), dense_values[2][m.row_nz_index(2, 2)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(3, 0), dense_values[3][m.row_nz_index(3, 0)]);

    m.row_nz_entry(1, 0) = 8.0;
    m.row_nz_entry(2, 2) = 9.0;

    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.columns(), 3);
    EXPECT_EQ(m.non_zero_size(), 7);

    EXPECT_EQ(m.row_nz_size(0), 2);
    EXPECT_EQ(m.row_nz_size(1), 1);
    EXPECT_EQ(m.row_nz_size(2), 3);
    EXPECT_EQ(m.row_nz_size(3), 1);

    EXPECT_DOUBLE_EQ(m.row_nz_entry(0, 0), dense_values[0][m.row_nz_index(0, 0)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(0, 1), dense_values[0][m.row_nz_index(0, 1)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(1, 0), 8.0);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 0), dense_values[2][m.row_nz_index(2, 0)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 1), dense_values[2][m.row_nz_index(2, 1)]);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(2, 2), 9.0);
    EXPECT_DOUBLE_EQ(m.row_nz_entry(3, 0), dense_values[3][m.row_nz_index(3, 0)]);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const auto test_result_status = RUN_ALL_TESTS();
    MPI_Finalize();
    return test_result_status;
}
