#include <mpi.h>

#include <gtest/gtest.h>

#include "../grid/grid.h"

std::array<std::pair<int, int>, space_dimension> make_1d_neighborhood(std::pair<int, int> x)
{
    if constexpr(space_dimension == 1)
    {
        std::array<std::pair<int, int>, space_dimension> result;
        result[0] = x;
        return result;
    }
    else
    {
        return std::array<std::pair<int, int>, space_dimension>();
    }
}

std::array<std::pair<int, int>, space_dimension> make_2d_neighborhood(std::pair<int, int> x, std::pair<int, int> y)
{
    if constexpr(space_dimension == 2)
    {
        std::array<std::pair<int, int>, space_dimension> result;
        result[0] = x;
        result[1] = y;
        return result;
    }
    else
    {
        return std::array<std::pair<int, int>, space_dimension>();
    }
}

std::array<std::pair<int, int>, space_dimension> make_3d_neighborhood(std::pair<int, int> x, std::pair<int, int> y, std::pair<int, int> z)
{
    if constexpr(space_dimension == 3)
    {
        std::array<std::pair<int, int>, space_dimension> result;
        result[0] = x;
        result[1] = y;
        result[2] = z;
        return result;
    }
    else
    {
        return std::array<std::pair<int, int>, space_dimension>();
    }
}

TEST(RegularGrid, node_counts)
{
    if constexpr(space_dimension == 1)
    {
        RegularGrid grid(Point(1.0), Point(2.0), MultiIndex(3));

        EXPECT_EQ(grid.number_of_nodes(), 3);
        EXPECT_EQ(grid.number_of_inner_nodes(), 1);
        EXPECT_EQ(grid.number_of_boundary_nodes(), 2);
    }
    else if constexpr(space_dimension == 2)
    {
        RegularGrid grid(Point(1.0, 2.0), Point(3.0, 5.0), MultiIndex(3, 4));

        EXPECT_EQ(grid.number_of_nodes(), 12);
        EXPECT_EQ(grid.number_of_inner_nodes(), 2);
        EXPECT_EQ(grid.number_of_boundary_nodes(), 10);
    }
    else if constexpr(space_dimension == 3)
    {
        RegularGrid grid(Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0), MultiIndex(3, 3, 3));

        EXPECT_EQ(grid.number_of_nodes(), 27);
        EXPECT_EQ(grid.number_of_inner_nodes(), 1);
        EXPECT_EQ(grid.number_of_boundary_nodes(), 26);
    }
}

TEST(RegularGrid, neighborhoods)
{
    if constexpr(space_dimension == 1)
    {
        RegularGrid grid(Point(1.0), Point(2.0), MultiIndex(3));

        EXPECT_EQ(grid.number_of_nodes(), 3);
        EXPECT_EQ(grid.number_of_inner_nodes(), 1);
        EXPECT_EQ(grid.number_of_boundary_nodes(), 2);

        EXPECT_EQ(grid.number_of_neighbors(0), 1);
        EXPECT_EQ(grid.number_of_neighbors(1), 2);
        EXPECT_EQ(grid.number_of_neighbors(2), 1);

        std::array<std::pair<int, int>, space_dimension> neighbors;
        EXPECT_EQ(grid.neighbors_of(0, neighbors), 1);
        EXPECT_EQ(neighbors, make_1d_neighborhood({-1, 1}));
        EXPECT_EQ(grid.neighbors_of(1, neighbors), 2);
        EXPECT_EQ(neighbors, make_1d_neighborhood({0, 2}));
        EXPECT_EQ(grid.neighbors_of(2, neighbors), 1);
        EXPECT_EQ(neighbors, make_1d_neighborhood({1, -1}));
    }
    else if constexpr(space_dimension == 2)
    {
        RegularGrid grid(Point(1.0, 2.0), Point(3.0, 5.0), MultiIndex(3, 4));

        EXPECT_EQ(grid.number_of_nodes(), 12);
        EXPECT_EQ(grid.number_of_inner_nodes(), 2);
        EXPECT_EQ(grid.number_of_boundary_nodes(), 10);

        EXPECT_EQ(grid.number_of_neighbors(0), 2);
        EXPECT_EQ(grid.number_of_neighbors(1), 3);
        EXPECT_EQ(grid.number_of_neighbors(2), 2);
        EXPECT_EQ(grid.number_of_neighbors(3), 3);
        EXPECT_EQ(grid.number_of_neighbors(4), 4);
        EXPECT_EQ(grid.number_of_neighbors(5), 3);
        EXPECT_EQ(grid.number_of_neighbors(6), 3);
        EXPECT_EQ(grid.number_of_neighbors(7), 4);
        EXPECT_EQ(grid.number_of_neighbors(8), 3);
        EXPECT_EQ(grid.number_of_neighbors(9), 2);
        EXPECT_EQ(grid.number_of_neighbors(10), 3);
        EXPECT_EQ(grid.number_of_neighbors(11), 2);

        std::array<std::pair<int, int>, space_dimension> neighbors;
        EXPECT_EQ(grid.neighbors_of(0, neighbors), 2);
        EXPECT_EQ(neighbors, make_2d_neighborhood({-1, 1}, {-1, 3}));
        EXPECT_EQ(grid.neighbors_of(1, neighbors), 3);
        EXPECT_EQ(neighbors, make_2d_neighborhood({0, 2}, {-1, 4}));
        EXPECT_EQ(grid.neighbors_of(2, neighbors), 2);
        EXPECT_EQ(neighbors, make_2d_neighborhood({1, -1}, {-1, 5}));
        EXPECT_EQ(grid.neighbors_of(3, neighbors), 3);
        EXPECT_EQ(neighbors, make_2d_neighborhood({-1, 4}, {0, 6}));
        EXPECT_EQ(grid.neighbors_of(4, neighbors), 4);
        EXPECT_EQ(neighbors, make_2d_neighborhood({3, 5}, {1, 7}));
        EXPECT_EQ(grid.neighbors_of(5, neighbors), 3);
        EXPECT_EQ(neighbors, make_2d_neighborhood({4, -1}, {2, 8}));
        EXPECT_EQ(grid.neighbors_of(6, neighbors), 3);
        EXPECT_EQ(neighbors, make_2d_neighborhood({-1, 7}, {3, 9}));
        EXPECT_EQ(grid.neighbors_of(7, neighbors), 4);
        EXPECT_EQ(neighbors, make_2d_neighborhood({6, 8}, {4, 10}));
        EXPECT_EQ(grid.neighbors_of(8, neighbors), 3);
        EXPECT_EQ(neighbors, make_2d_neighborhood({7, -1}, {5, 11}));
        EXPECT_EQ(grid.neighbors_of(9, neighbors), 2);
        EXPECT_EQ(neighbors, make_2d_neighborhood({-1, 10}, {6, -1}));
        EXPECT_EQ(grid.neighbors_of(10, neighbors), 3);
        EXPECT_EQ(neighbors, make_2d_neighborhood({9, 11}, {7, -1}));
        EXPECT_EQ(grid.neighbors_of(11, neighbors), 2);
        EXPECT_EQ(neighbors, make_2d_neighborhood({10, -1}, {8, -1}));
    }
    else if constexpr(space_dimension == 3)
    {
        RegularGrid grid(Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0), MultiIndex(3, 3, 3));

        EXPECT_EQ(grid.number_of_nodes(), 27);
        EXPECT_EQ(grid.number_of_inner_nodes(), 1);
        EXPECT_EQ(grid.number_of_boundary_nodes(), 26);

        EXPECT_EQ(grid.number_of_neighbors(0), 3);
        EXPECT_EQ(grid.number_of_neighbors(1), 4);
        EXPECT_EQ(grid.number_of_neighbors(2), 3);
        EXPECT_EQ(grid.number_of_neighbors(3), 4);
        EXPECT_EQ(grid.number_of_neighbors(4), 5);
        EXPECT_EQ(grid.number_of_neighbors(5), 4);
        EXPECT_EQ(grid.number_of_neighbors(6), 3);
        EXPECT_EQ(grid.number_of_neighbors(7), 4);
        EXPECT_EQ(grid.number_of_neighbors(8), 3);
        EXPECT_EQ(grid.number_of_neighbors(9), 4);
        EXPECT_EQ(grid.number_of_neighbors(10), 5);
        EXPECT_EQ(grid.number_of_neighbors(11), 4);
        EXPECT_EQ(grid.number_of_neighbors(12), 5);
        EXPECT_EQ(grid.number_of_neighbors(13), 6);
        EXPECT_EQ(grid.number_of_neighbors(14), 5);
        EXPECT_EQ(grid.number_of_neighbors(15), 4);
        EXPECT_EQ(grid.number_of_neighbors(16), 5);
        EXPECT_EQ(grid.number_of_neighbors(17), 4);
        EXPECT_EQ(grid.number_of_neighbors(18), 3);
        EXPECT_EQ(grid.number_of_neighbors(19), 4);
        EXPECT_EQ(grid.number_of_neighbors(20), 3);
        EXPECT_EQ(grid.number_of_neighbors(21), 4);
        EXPECT_EQ(grid.number_of_neighbors(22), 5);
        EXPECT_EQ(grid.number_of_neighbors(23), 4);
        EXPECT_EQ(grid.number_of_neighbors(24), 3);
        EXPECT_EQ(grid.number_of_neighbors(25), 4);
        EXPECT_EQ(grid.number_of_neighbors(26), 3);

        std::array<std::pair<int, int>, space_dimension> neighbors;
        EXPECT_EQ(grid.neighbors_of(0, neighbors), 3);
        EXPECT_EQ(neighbors, make_3d_neighborhood({-1, 1}, {-1, 3}, {-1, 9}));
        EXPECT_EQ(grid.neighbors_of(1, neighbors), 4);
        EXPECT_EQ(neighbors, make_3d_neighborhood({0, 2}, {-1, 4}, {-1, 10}));
        EXPECT_EQ(grid.neighbors_of(2, neighbors), 3);
        EXPECT_EQ(neighbors, make_3d_neighborhood({1, -1}, {-1, 5}, {-1, 11}));
        EXPECT_EQ(grid.neighbors_of(3, neighbors), 4);
        EXPECT_EQ(neighbors, make_3d_neighborhood({-1, 4}, {0, 6}, {-1, 12}));
        EXPECT_EQ(grid.neighbors_of(4, neighbors), 5);
        EXPECT_EQ(neighbors, make_3d_neighborhood({3, 5}, {1, 7}, {-1, 13}));
        EXPECT_EQ(grid.neighbors_of(5, neighbors), 4);
        EXPECT_EQ(neighbors, make_3d_neighborhood({4, -1}, {2, 8}, {-1, 14}));
        EXPECT_EQ(grid.neighbors_of(6, neighbors), 3);
        EXPECT_EQ(neighbors, make_3d_neighborhood({-1, 7}, {3, -1}, {-1, 15}));
        EXPECT_EQ(grid.neighbors_of(7, neighbors), 4);
        EXPECT_EQ(neighbors, make_3d_neighborhood({6, 8}, {4, -1}, {-1, 16}));
        EXPECT_EQ(grid.neighbors_of(8, neighbors), 3);
        EXPECT_EQ(neighbors, make_3d_neighborhood({7, -1}, {5, -1}, {-1, 17}));
        EXPECT_EQ(grid.neighbors_of(9, neighbors), 4);
        EXPECT_EQ(neighbors, make_3d_neighborhood({-1, 10}, {-1, 12}, {0, 18}));
        EXPECT_EQ(grid.neighbors_of(10, neighbors), 5);
        EXPECT_EQ(neighbors, make_3d_neighborhood({9, 11}, {-1, 13}, {1, 19}));
        EXPECT_EQ(grid.neighbors_of(11, neighbors), 4);
        EXPECT_EQ(neighbors, make_3d_neighborhood({10, -1}, {-1, 14}, {2, 20}));
        EXPECT_EQ(grid.neighbors_of(12, neighbors), 5);
        EXPECT_EQ(neighbors, make_3d_neighborhood({-1, 13}, {9, 15}, {3, 21}));
        EXPECT_EQ(grid.neighbors_of(13, neighbors), 6);
        EXPECT_EQ(neighbors, make_3d_neighborhood({12, 14}, {10, 16}, {4, 22}));
        EXPECT_EQ(grid.neighbors_of(14, neighbors), 5);
        EXPECT_EQ(neighbors, make_3d_neighborhood({13, -1}, {11, 17}, {5, 23}));
        EXPECT_EQ(grid.neighbors_of(15, neighbors), 4);
        EXPECT_EQ(neighbors, make_3d_neighborhood({-1, 16}, {12, -1}, {6, 24}));
        EXPECT_EQ(grid.neighbors_of(16, neighbors), 5);
        EXPECT_EQ(neighbors, make_3d_neighborhood({15, 17}, {13, -1}, {7, 25}));
        EXPECT_EQ(grid.neighbors_of(17, neighbors), 4);
        EXPECT_EQ(neighbors, make_3d_neighborhood({16, -1}, {14, -1}, {8, 26}));
        EXPECT_EQ(grid.neighbors_of(18, neighbors), 3);
        EXPECT_EQ(neighbors, make_3d_neighborhood({-1, 19}, {-1, 21}, {9, -1}));
        EXPECT_EQ(grid.neighbors_of(19, neighbors), 4);
        EXPECT_EQ(neighbors, make_3d_neighborhood({18, 20}, {-1, 22}, {10, -1}));
        EXPECT_EQ(grid.neighbors_of(20, neighbors), 3);
        EXPECT_EQ(neighbors, make_3d_neighborhood({19, -1}, {-1, 23}, {11, -1}));
        EXPECT_EQ(grid.neighbors_of(21, neighbors), 4);
        EXPECT_EQ(neighbors, make_3d_neighborhood({-1, 22}, {18, 24}, {12, -1}));
        EXPECT_EQ(grid.neighbors_of(22, neighbors), 5);
        EXPECT_EQ(neighbors, make_3d_neighborhood({21, 23}, {19, 25}, {13, -1}));
        EXPECT_EQ(grid.neighbors_of(23, neighbors), 4);
        EXPECT_EQ(neighbors, make_3d_neighborhood({22, -1}, {20, 26}, {14, -1}));
        EXPECT_EQ(grid.neighbors_of(24, neighbors), 3);
        EXPECT_EQ(neighbors, make_3d_neighborhood({-1, 25}, {21, -1}, {15, -1}));
        EXPECT_EQ(grid.neighbors_of(25, neighbors), 4);
        EXPECT_EQ(neighbors, make_3d_neighborhood({24, 26}, {22, -1}, {16, -1}));
        EXPECT_EQ(grid.neighbors_of(26, neighbors), 3);
        EXPECT_EQ(neighbors, make_3d_neighborhood({25, -1}, {23, -1}, {17, -1}));
    }
}

TEST(RegularGrid, boundary_flags)
{
    if constexpr(space_dimension == 1)
    {
        RegularGrid grid(Point(1.0), Point(2.0), MultiIndex(3));

        EXPECT_TRUE(grid.is_boundary_node(0));
        EXPECT_FALSE(grid.is_boundary_node(1));
        EXPECT_TRUE(grid.is_boundary_node(2));
    }
    else if constexpr(space_dimension == 2)
    {
        RegularGrid grid(Point(1.0, 2.0), Point(3.0, 5.0), MultiIndex(3, 4));

        EXPECT_TRUE(grid.is_boundary_node(0));
        EXPECT_TRUE(grid.is_boundary_node(1));
        EXPECT_TRUE(grid.is_boundary_node(2));
        EXPECT_TRUE(grid.is_boundary_node(3));
        EXPECT_FALSE(grid.is_boundary_node(4));
        EXPECT_TRUE(grid.is_boundary_node(5));
        EXPECT_TRUE(grid.is_boundary_node(6));
        EXPECT_FALSE(grid.is_boundary_node(7));
        EXPECT_TRUE(grid.is_boundary_node(8));
        EXPECT_TRUE(grid.is_boundary_node(9));
        EXPECT_TRUE(grid.is_boundary_node(10));
        EXPECT_TRUE(grid.is_boundary_node(11));
    }
    else if constexpr(space_dimension == 3)
    {
        RegularGrid grid(Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0), MultiIndex(3, 3, 3));

        EXPECT_TRUE(grid.is_boundary_node(0));
        EXPECT_TRUE(grid.is_boundary_node(1));
        EXPECT_TRUE(grid.is_boundary_node(2));
        EXPECT_TRUE(grid.is_boundary_node(3));
        EXPECT_TRUE(grid.is_boundary_node(4));
        EXPECT_TRUE(grid.is_boundary_node(5));
        EXPECT_TRUE(grid.is_boundary_node(6));
        EXPECT_TRUE(grid.is_boundary_node(7));
        EXPECT_TRUE(grid.is_boundary_node(8));
        EXPECT_TRUE(grid.is_boundary_node(9));
        EXPECT_TRUE(grid.is_boundary_node(10));
        EXPECT_TRUE(grid.is_boundary_node(11));
        EXPECT_TRUE(grid.is_boundary_node(12));
        EXPECT_FALSE(grid.is_boundary_node(13));
        EXPECT_TRUE(grid.is_boundary_node(14));
        EXPECT_TRUE(grid.is_boundary_node(15));
        EXPECT_TRUE(grid.is_boundary_node(16));
        EXPECT_TRUE(grid.is_boundary_node(17));
        EXPECT_TRUE(grid.is_boundary_node(18));
        EXPECT_TRUE(grid.is_boundary_node(19));
        EXPECT_TRUE(grid.is_boundary_node(20));
        EXPECT_TRUE(grid.is_boundary_node(21));
        EXPECT_TRUE(grid.is_boundary_node(22));
        EXPECT_TRUE(grid.is_boundary_node(23));
        EXPECT_TRUE(grid.is_boundary_node(24));
        EXPECT_TRUE(grid.is_boundary_node(25));
        EXPECT_TRUE(grid.is_boundary_node(26));
    }
}

TEST(RegularGrid, node_coordinates)
{
    if constexpr(space_dimension == 1)
    {
        RegularGrid grid(Point(1.0), Point(2.0), MultiIndex(3));

        EXPECT_TRUE(equals(grid.node_coordinates(0), Point(1.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(1), Point(1.5)));
        EXPECT_TRUE(equals(grid.node_coordinates(2), Point(2.0)));
    }
    else if constexpr(space_dimension == 2)
    {
        RegularGrid grid(Point(1.0, 3.0), Point(2.0, 6.0), MultiIndex(3, 4));

        EXPECT_TRUE(equals(grid.node_coordinates(0), Point(1.0, 3.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(1), Point(1.5, 3.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(2), Point(2.0, 3.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(3), Point(1.0, 4.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(4), Point(1.5, 4.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(5), Point(2.0, 4.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(6), Point(1.0, 5.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(7), Point(1.5, 5.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(8), Point(2.0, 5.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(9), Point(1.0, 6.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(10), Point(1.5, 6.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(11), Point(2.0, 6.0)));
    }
    else if constexpr(space_dimension == 3)
    {
        RegularGrid grid(Point(1.0, 2.0, 3.0), Point(4.0, 6.0, 9.0), MultiIndex(3, 3, 3));

        EXPECT_TRUE(equals(grid.node_coordinates(0), Point(1.0, 2.0, 3.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(1), Point(2.5, 2.0, 3.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(2), Point(4.0, 2.0, 3.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(3), Point(1.0, 4.0, 3.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(4), Point(2.5, 4.0, 3.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(5), Point(4.0, 4.0, 3.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(6), Point(1.0, 6.0, 3.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(7), Point(2.5, 6.0, 3.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(8), Point(4.0, 6.0, 3.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(9), Point(1.0, 2.0, 6.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(10), Point(2.5, 2.0, 6.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(11), Point(4.0, 2.0, 6.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(12), Point(1.0, 4.0, 6.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(13), Point(2.5, 4.0, 6.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(14), Point(4.0, 4.0, 6.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(15), Point(1.0, 6.0, 6.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(16), Point(2.5, 6.0, 6.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(17), Point(4.0, 6.0, 6.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(18), Point(1.0, 2.0, 9.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(19), Point(2.5, 2.0, 9.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(20), Point(4.0, 2.0, 9.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(21), Point(1.0, 4.0, 9.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(22), Point(2.5, 4.0, 9.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(23), Point(4.0, 4.0, 9.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(24), Point(1.0, 6.0, 9.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(25), Point(2.5, 6.0, 9.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(26), Point(4.0, 6.0, 9.0)));
    }
}

TEST(RegularGrid, node_neighbor_distance)
{
    if constexpr(space_dimension == 1)
    {
        RegularGrid grid(Point(1.0), Point(2.0), MultiIndex(3));

        EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(1, 0, NeighborSuccession::predecessor), 0.5);
        EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(1, 0, NeighborSuccession::successor), 0.5);
    }
    else if constexpr(space_dimension == 2)
    {
        RegularGrid grid(Point(1.0, 3.0), Point(2.0, 6.0), MultiIndex(3, 4));

        EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(4, 0, NeighborSuccession::predecessor), 0.5);
        EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(4, 0, NeighborSuccession::successor), 0.5);
        EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(4, 1, NeighborSuccession::predecessor), 1.0);
        EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(4, 1, NeighborSuccession::successor), 1.0);
    }
    else if constexpr(space_dimension == 3)
    {
        RegularGrid grid(Point(1.0, 2.0, 3.0), Point(4.0, 6.0, 9.0), MultiIndex(3, 3, 3));

        EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(13, 0, NeighborSuccession::predecessor), 1.5);
        EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(13, 0, NeighborSuccession::successor), 1.5);
        EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(13, 1, NeighborSuccession::predecessor), 2.0);
        EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(13, 1, NeighborSuccession::successor), 2.0);
        EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(13, 2, NeighborSuccession::predecessor), 3.0);
        EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(13, 2, NeighborSuccession::successor), 3.0);
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
