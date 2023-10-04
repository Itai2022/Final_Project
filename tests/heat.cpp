#include <mpi.h>

#include <gtest/gtest.h>

#include "../discretization/heat.h"

TEST(Poisson, assemble)
{
    if constexpr(space_dimension == 1)
    {
        const auto rhs_function = [](const Point&, const scalar_t) -> scalar_t { return 8.0; };
        const auto boundary_function = [](const Point&, const scalar_t) -> scalar_t { return 1.0; };
        const auto previous_temperature_function = [](const Point& x) -> scalar_t { return x[0]; };

        const scalar_t t = 0.34;
        const scalar_t delta_t = 1. / 100;

        const auto grid = std::make_shared<RegularGrid>(Point(1.0), Point(4.0), MultiIndex(5));
        GridFunction<scalar_t> previous_temperature(*grid, previous_temperature_function);

        const auto [A, b] = assemble_heat_matrix<scalar_t>(*grid, previous_temperature, t, delta_t, rhs_function, boundary_function);

        EXPECT_EQ(A.rows(), 5);
        EXPECT_EQ(A.columns(), 5);
        EXPECT_EQ(A.non_zero_size(), 11);

        EXPECT_EQ(A.row_nz_size(0), 1);
        EXPECT_EQ(A.row_nz_size(1), 3);
        EXPECT_EQ(A.row_nz_size(2), 3);
        EXPECT_EQ(A.row_nz_size(3), 3);
        EXPECT_EQ(A.row_nz_size(4), 1);

        const scalar_t h2 = 0.75 * 0.75;
        const scalar_t dense_values[5][5] = {{1.0, 0.0, 0.0, 0.0, 0.0},
                                             {0.0, 2.0 / h2 + 1.0 / delta_t, -1.0 / h2, 0.0, 0.0},
                                             {0.0, -1.0 / h2, 2.0 / h2 + 1.0 / delta_t, -1.0 / h2, 0.0},
                                             {0.0, 0.0, -1.0 / h2, 2.0 / h2 + 1.0 / delta_t, 0.0},
                                             {0.0, 0.0, 0.0, 0.0, 1.0}};

        EXPECT_DOUBLE_EQ(A.row_nz_entry(0, 0), dense_values[0][A.row_nz_index(0, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(1, 0), dense_values[1][A.row_nz_index(1, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(1, 1), dense_values[1][A.row_nz_index(1, 1)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(1, 2), dense_values[1][A.row_nz_index(1, 2)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(2, 0), dense_values[2][A.row_nz_index(2, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(2, 1), dense_values[2][A.row_nz_index(2, 1)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(2, 2), dense_values[2][A.row_nz_index(2, 2)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(3, 0), dense_values[3][A.row_nz_index(3, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(3, 1), dense_values[3][A.row_nz_index(3, 1)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(3, 2), dense_values[3][A.row_nz_index(3, 2)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(2, 0), dense_values[2][A.row_nz_index(2, 0)]);

        EXPECT_EQ(b.size(), 5);
        EXPECT_DOUBLE_EQ(b[0], 1.0);
        EXPECT_DOUBLE_EQ(b[1], 8.0 + 1.0 / h2 + grid->node_coordinates(1)[0] / delta_t);
        EXPECT_DOUBLE_EQ(b[2], 8.0 + grid->node_coordinates(2)[0] / delta_t);
        EXPECT_DOUBLE_EQ(b[3], 8.0 + 1.0 / h2 + grid->node_coordinates(3)[0] / delta_t);
        EXPECT_DOUBLE_EQ(b[4], 1.0);
    }
    else if constexpr(space_dimension == 2)
    {
        const auto rhs_function = [](const Point&, const scalar_t) -> scalar_t { return 8.0; };
        const auto boundary_function = [](const Point& x, const scalar_t) -> scalar_t { return x[0] + x[1]; };
        const auto previous_temperature_function = [](const Point& x) -> scalar_t { return x[0] + x[1]; };

        const scalar_t t = 0.34;
        const scalar_t delta_t = 1. / 100;

        const auto grid = std::make_shared<RegularGrid>(Point(1.0, 2.0), Point(2.0, 5.0), MultiIndex(3, 4));
        GridFunction<scalar_t> previous_temperature(*grid, previous_temperature_function);

        const auto [A, b] = assemble_heat_matrix<scalar_t>(*grid, previous_temperature, t, delta_t, rhs_function, boundary_function);

        EXPECT_EQ(A.rows(), 12);
        EXPECT_EQ(A.columns(), 12);
        EXPECT_EQ(A.non_zero_size(), 20);

        EXPECT_EQ(A.row_nz_size(0), 1);
        EXPECT_EQ(A.row_nz_size(1), 1);
        EXPECT_EQ(A.row_nz_size(2), 1);
        EXPECT_EQ(A.row_nz_size(3), 1);
        EXPECT_EQ(A.row_nz_size(4), 5);
        EXPECT_EQ(A.row_nz_size(5), 1);
        EXPECT_EQ(A.row_nz_size(6), 1);
        EXPECT_EQ(A.row_nz_size(7), 5);
        EXPECT_EQ(A.row_nz_size(8), 1);
        EXPECT_EQ(A.row_nz_size(9), 1);
        EXPECT_EQ(A.row_nz_size(10), 1);
        EXPECT_EQ(A.row_nz_size(11), 1);

        const scalar_t h2_x = 0.5 * 0.5;
        const scalar_t h2_y = 1.0 * 1.0;

        const scalar_t dense_values[12][12] = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 2.0 / h2_x + 2.0 / h2_y + 1.0 / delta_t, 0.0, 0.0, -1.0 / h2_y, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, -1.0 / h2_y, 0.0, 0.0, 2.0 / h2_x + 2.0 / h2_y + 1.0 / delta_t, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};

        EXPECT_DOUBLE_EQ(A.row_nz_entry(0, 0), dense_values[0][A.row_nz_index(0, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(1, 0), dense_values[1][A.row_nz_index(1, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(2, 0), dense_values[2][A.row_nz_index(2, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(3, 0), dense_values[3][A.row_nz_index(3, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(4, 0), dense_values[4][A.row_nz_index(4, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(4, 1), dense_values[4][A.row_nz_index(4, 1)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(4, 2), dense_values[4][A.row_nz_index(4, 2)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(4, 3), dense_values[4][A.row_nz_index(4, 3)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(4, 4), dense_values[4][A.row_nz_index(4, 4)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(5, 0), dense_values[5][A.row_nz_index(5, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(6, 0), dense_values[6][A.row_nz_index(6, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(7, 0), dense_values[7][A.row_nz_index(7, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(7, 1), dense_values[7][A.row_nz_index(7, 1)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(7, 2), dense_values[7][A.row_nz_index(7, 2)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(7, 3), dense_values[7][A.row_nz_index(7, 3)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(7, 4), dense_values[7][A.row_nz_index(7, 4)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(8, 0), dense_values[8][A.row_nz_index(8, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(9, 0), dense_values[9][A.row_nz_index(9, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(10, 0), dense_values[10][A.row_nz_index(10, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(11, 0), dense_values[11][A.row_nz_index(11, 0)]);

        EXPECT_EQ(b.size(), 12);
        EXPECT_DOUBLE_EQ(b[0], 1.0 + 2.0);
        EXPECT_DOUBLE_EQ(b[1], 1.5 + 2.0);
        EXPECT_DOUBLE_EQ(b[2], 2.0 + 2.0);
        EXPECT_DOUBLE_EQ(b[3], 1.0 + 3.0);
        EXPECT_DOUBLE_EQ(b[4], 8.0 + (1.0 + 3.0) / h2_x + (2.0 + 3.0) / h2_x + (1.5 + 2.0) / h2_y + (grid->node_coordinates(4)[0] + grid->node_coordinates(4)[1]) / delta_t);
        EXPECT_DOUBLE_EQ(b[5], 2.0 + 3.0);
        EXPECT_DOUBLE_EQ(b[6], 1.0 + 4.0);
        EXPECT_DOUBLE_EQ(b[7], 8.0 + (1.0 + 4.0) / h2_x + (2.0 + 4.0) / h2_x + (1.5 + 5.0) / h2_y + (grid->node_coordinates(7)[0] + grid->node_coordinates(7)[1]) / delta_t);
        EXPECT_DOUBLE_EQ(b[8], 2.0 + 4.0);
        EXPECT_DOUBLE_EQ(b[9], 1.0 + 5.0);
        EXPECT_DOUBLE_EQ(b[10], 1.5 + 5.0);
        EXPECT_DOUBLE_EQ(b[11], 2.0 + 5.0);
    }
    else if constexpr(space_dimension == 3)
    {
        const auto rhs_function = [](const Point&, const scalar_t) -> scalar_t { return 8.0; };
        const auto boundary_function = [](const Point& x, const scalar_t) -> scalar_t { return x[0] + x[1] + x[2]; };
        const auto previous_temperature_function = [](const Point& x) -> scalar_t { return x[0] + x[1] + x[2]; };

        const scalar_t t = 0.34;
        const scalar_t delta_t = 1. / 100;

        const auto grid = std::make_shared<RegularGrid>(Point(1.0, 2.0, 3.0), Point(4.0, 7.0, 9.0), MultiIndex(3, 3, 3));
        GridFunction<scalar_t> previous_temperature(*grid, previous_temperature_function);

        const auto [A, b] = assemble_heat_matrix<scalar_t>(*grid, previous_temperature, t, delta_t, rhs_function, boundary_function);

        EXPECT_EQ(A.rows(), 27);
        EXPECT_EQ(A.columns(), 27);
        EXPECT_EQ(A.non_zero_size(), 33);

        EXPECT_EQ(A.row_nz_size(0), 1);
        EXPECT_EQ(A.row_nz_size(1), 1);
        EXPECT_EQ(A.row_nz_size(2), 1);
        EXPECT_EQ(A.row_nz_size(3), 1);
        EXPECT_EQ(A.row_nz_size(4), 1);
        EXPECT_EQ(A.row_nz_size(5), 1);
        EXPECT_EQ(A.row_nz_size(6), 1);
        EXPECT_EQ(A.row_nz_size(7), 1);
        EXPECT_EQ(A.row_nz_size(8), 1);
        EXPECT_EQ(A.row_nz_size(9), 1);
        EXPECT_EQ(A.row_nz_size(10), 1);
        EXPECT_EQ(A.row_nz_size(11), 1);
        EXPECT_EQ(A.row_nz_size(12), 1);
        EXPECT_EQ(A.row_nz_size(13), 7);
        EXPECT_EQ(A.row_nz_size(14), 1);
        EXPECT_EQ(A.row_nz_size(15), 1);
        EXPECT_EQ(A.row_nz_size(16), 1);
        EXPECT_EQ(A.row_nz_size(17), 1);
        EXPECT_EQ(A.row_nz_size(18), 1);
        EXPECT_EQ(A.row_nz_size(19), 1);
        EXPECT_EQ(A.row_nz_size(20), 1);
        EXPECT_EQ(A.row_nz_size(21), 1);
        EXPECT_EQ(A.row_nz_size(22), 1);
        EXPECT_EQ(A.row_nz_size(23), 1);

        const scalar_t h2_x = 1.5 * 1.5;
        const scalar_t h2_y = 2.5 * 2.5;
        const scalar_t h2_z = 3.0 * 3.0;

        const scalar_t dense_values[27][27] = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0 / h2_x + 2.0 / h2_y + 2.0 / h2_z + 1.0 / delta_t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};

        EXPECT_DOUBLE_EQ(A.row_nz_entry(0, 0), dense_values[0][A.row_nz_index(0, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(1, 0), dense_values[1][A.row_nz_index(1, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(2, 0), dense_values[2][A.row_nz_index(2, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(3, 0), dense_values[3][A.row_nz_index(3, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(4, 0), dense_values[4][A.row_nz_index(4, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(5, 0), dense_values[5][A.row_nz_index(5, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(6, 0), dense_values[6][A.row_nz_index(6, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(7, 0), dense_values[7][A.row_nz_index(7, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(8, 0), dense_values[8][A.row_nz_index(8, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(9, 0), dense_values[9][A.row_nz_index(9, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(10, 0), dense_values[10][A.row_nz_index(10, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(11, 0), dense_values[11][A.row_nz_index(11, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(12, 0), dense_values[12][A.row_nz_index(12, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(13, 0), dense_values[13][A.row_nz_index(13, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(13, 1), dense_values[13][A.row_nz_index(13, 1)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(13, 2), dense_values[13][A.row_nz_index(13, 2)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(13, 3), dense_values[13][A.row_nz_index(13, 3)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(13, 4), dense_values[13][A.row_nz_index(13, 4)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(13, 5), dense_values[13][A.row_nz_index(13, 5)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(13, 6), dense_values[13][A.row_nz_index(13, 6)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(14, 0), dense_values[14][A.row_nz_index(14, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(15, 0), dense_values[15][A.row_nz_index(15, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(16, 0), dense_values[16][A.row_nz_index(16, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(17, 0), dense_values[17][A.row_nz_index(17, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(18, 0), dense_values[18][A.row_nz_index(18, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(19, 0), dense_values[19][A.row_nz_index(19, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(20, 0), dense_values[20][A.row_nz_index(20, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(21, 0), dense_values[21][A.row_nz_index(21, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(22, 0), dense_values[22][A.row_nz_index(22, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(23, 0), dense_values[23][A.row_nz_index(23, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(24, 0), dense_values[24][A.row_nz_index(24, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(25, 0), dense_values[25][A.row_nz_index(25, 0)]);
        EXPECT_DOUBLE_EQ(A.row_nz_entry(26, 0), dense_values[26][A.row_nz_index(26, 0)]);

        EXPECT_EQ(b.size(), 27);
        EXPECT_DOUBLE_EQ(b[0], 1.0 + 2.0 + 3.0);
        EXPECT_DOUBLE_EQ(b[1], 2.5 + 2.0 + 3.0);
        EXPECT_DOUBLE_EQ(b[2], 4.0 + 2.0 + 3.0);
        EXPECT_DOUBLE_EQ(b[3], 1.0 + 4.5 + 3.0);
        EXPECT_DOUBLE_EQ(b[4], 2.5 + 4.5 + 3.0);
        EXPECT_DOUBLE_EQ(b[5], 4.0 + 4.5 + 3.0);
        EXPECT_DOUBLE_EQ(b[6], 1.0 + 7.0 + 3.0);
        EXPECT_DOUBLE_EQ(b[7], 2.5 + 7.0 + 3.0);
        EXPECT_DOUBLE_EQ(b[8], 4.0 + 7.0 + 3.0);
        EXPECT_DOUBLE_EQ(b[9], 1.0 + 2.0 + 6.0);
        EXPECT_DOUBLE_EQ(b[10], 2.5 + 2.0 + 6.0);
        EXPECT_DOUBLE_EQ(b[11], 4.0 + 2.0 + 6.0);
        EXPECT_DOUBLE_EQ(b[12], 1.0 + 4.5 + 6.0);
        EXPECT_DOUBLE_EQ(b[13], 8.0 + (1.0 + 4.5 + 6.0) / h2_x + (4.0 + 4.5 + 6.0) / h2_x + (2.5 + 2.0 + 6.0) / h2_y + (2.5 + 7.0 + 6.0) / h2_y + (2.5 + 4.5 + 3.0) / h2_z + (2.5 + 4.5 + 9.0) / h2_z + (grid->node_coordinates(13)[0] + grid->node_coordinates(13)[1] + grid->node_coordinates(13)[2]) / delta_t);
        EXPECT_DOUBLE_EQ(b[14], 4.0 + 4.5 + 6.0);
        EXPECT_DOUBLE_EQ(b[15], 1.0 + 7.0 + 6.0);
        EXPECT_DOUBLE_EQ(b[16], 2.5 + 7.0 + 6.0);
        EXPECT_DOUBLE_EQ(b[17], 4.0 + 7.0 + 6.0);
        EXPECT_DOUBLE_EQ(b[18], 1.0 + 2.0 + 9.0);
        EXPECT_DOUBLE_EQ(b[19], 2.5 + 2.0 + 9.0);
        EXPECT_DOUBLE_EQ(b[20], 4.0 + 2.0 + 9.0);
        EXPECT_DOUBLE_EQ(b[21], 1.0 + 4.5 + 9.0);
        EXPECT_DOUBLE_EQ(b[22], 2.5 + 4.5 + 9.0);
        EXPECT_DOUBLE_EQ(b[23], 4.0 + 4.5 + 9.0);
        EXPECT_DOUBLE_EQ(b[24], 1.0 + 7.0 + 9.0);
        EXPECT_DOUBLE_EQ(b[25], 2.5 + 7.0 + 9.0);
        EXPECT_DOUBLE_EQ(b[26], 4.0 + 7.0 + 9.0);
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
