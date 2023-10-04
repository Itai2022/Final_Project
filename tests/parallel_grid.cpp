#include <mpi.h>

#include <gtest/gtest.h>

#include <numeric>

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

std::array<std::pair<int, int>, space_dimension> map_neighbors(const std::array<std::pair<int, int>, space_dimension>& neighbors, const std::vector<int>& map)
{
    std::array<std::pair<int, int>, space_dimension> result;

    for(int d = 0; d < space_dimension; ++d)
    {
        result[d].first = neighbors[d].first == -1 ? neighbors[d].first : map[neighbors[d].first];
        result[d].second = neighbors[d].second == -1 ? neighbors[d].second : map[neighbors[d].second];
    }

    return result;
}

std::vector<int> compute_inverse_order(const std::vector<int>& order)
{
    const auto max_index = order.size();

    std::vector<int> result;
    result.reserve(max_index);

    for(int index = 0; index < max_index; ++index)
    {
        const auto new_index = std::distance(order.begin(), std::find(order.begin(), order.end(), index));

        result.push_back((int)new_index);
    }

    return result;
}

TEST(RegularGrid, node_counts)
{
    const auto communicator = MPI_COMM_WORLD;

    if constexpr(space_dimension == 1)
    {
        RegularGrid grid(communicator, Point(1.0), Point(2.0), MultiIndex(5));

        EXPECT_EQ(grid.number_of_nodes(), 5);
        EXPECT_EQ(grid.number_of_inner_nodes(), 3);
        EXPECT_EQ(grid.number_of_boundary_nodes(), 2);
    }
    else if constexpr(space_dimension == 2)
    {
        RegularGrid grid(communicator, Point(1.0, 2.0), Point(3.0, 5.0), MultiIndex(3, 4));

        EXPECT_EQ(grid.number_of_nodes(), 12);
        EXPECT_EQ(grid.number_of_inner_nodes(), 2);
        EXPECT_EQ(grid.number_of_boundary_nodes(), 10);
    }
    else if constexpr(space_dimension == 3)
    {
        RegularGrid grid(communicator, Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0), MultiIndex(3, 3, 3));

        EXPECT_EQ(grid.number_of_nodes(), 27);
        EXPECT_EQ(grid.number_of_inner_nodes(), 1);
        EXPECT_EQ(grid.number_of_boundary_nodes(), 26);
    }
}

TEST(RegularGrid, neighborhoods)
{
    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);

    if constexpr(space_dimension == 1)
    {
        RegularGrid grid(communicator, Point(1.0), Point(2.0), MultiIndex(5));
        const auto& partition = grid.partition();

        const std::vector<int> expected_number_of_neighbors = {1, 2, 2, 2, 1};
        const std::vector<std::array<std::pair<int, int>, space_dimension>> expected_neighbors = {make_1d_neighborhood({-1, 1}), make_1d_neighborhood({0, 2}), make_1d_neighborhood({1, 3}), make_1d_neighborhood({2, 4}), make_1d_neighborhood({3, -1})};

        EXPECT_EQ(grid.number_of_nodes(), 5);
        EXPECT_EQ(grid.number_of_inner_nodes(), 3);
        EXPECT_EQ(grid.number_of_boundary_nodes(), 2);

        std::array<std::pair<int, int>, space_dimension> neighbors;

        const auto local_size = partition.local_size();
        for(int local_node_index = 0; local_node_index < local_size; ++local_node_index)
        {
            const auto global_node_index = partition.to_global_index(local_node_index);
            EXPECT_EQ(grid.number_of_neighbors(local_node_index), expected_number_of_neighbors[global_node_index]);

            EXPECT_EQ(grid.neighbors_of(local_node_index, neighbors), expected_number_of_neighbors[global_node_index]);
            EXPECT_EQ(neighbors, expected_neighbors[global_node_index]);
        }
    }
    else if constexpr(space_dimension == 2)
    {
        RegularGrid grid(communicator, Point(1.0, 2.0), Point(3.0, 5.0), MultiIndex(3, 4));
        const auto& partition = grid.partition();

        std::vector<int> sequential_node_index;

        if(number_of_processes == 1)
        {
            sequential_node_index.resize(partition.global_size());
            std::iota(sequential_node_index.begin(), sequential_node_index.end(), 0);
        }
        else if(number_of_processes == 2)
        {
            sequential_node_index = {0, 3, 6, 9, 1, 2, 4, 5, 7, 8, 10, 11};
        }
        else if(number_of_processes == 3)
        {
            sequential_node_index = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
        }
        else if(number_of_processes == 4)
        {
            sequential_node_index = {0, 3, 6, 9, 1, 2, 4, 5, 7, 8, 10, 11};
        }

        const auto parallel_node_index = compute_inverse_order(sequential_node_index);

        const std::vector<int> expected_number_of_neighbors = {2, 3, 2, 3, 4, 3, 3, 4, 3, 2, 3, 2};
        const std::vector<std::array<std::pair<int, int>, space_dimension>> expected_neighbors = {make_2d_neighborhood({-1, 1}, {-1, 3}), make_2d_neighborhood({0, 2}, {-1, 4}), make_2d_neighborhood({1, -1}, {-1, 5}), make_2d_neighborhood({-1, 4}, {0, 6}), make_2d_neighborhood({3, 5}, {1, 7}), make_2d_neighborhood({4, -1}, {2, 8}), make_2d_neighborhood({-1, 7}, {3, 9}), make_2d_neighborhood({6, 8}, {4, 10}), make_2d_neighborhood({7, -1}, {5, 11}), make_2d_neighborhood({-1, 10}, {6, -1}), make_2d_neighborhood({9, 11}, {7, -1}), make_2d_neighborhood({10, -1}, {8, -1})};

        EXPECT_EQ(grid.number_of_nodes(), 12);
        EXPECT_EQ(grid.number_of_inner_nodes(), 2);
        EXPECT_EQ(grid.number_of_boundary_nodes(), 10);

        std::array<std::pair<int, int>, space_dimension> neighbors;

        const auto local_size = partition.local_size();
        for(int local_node_index = 0; local_node_index < local_size; ++local_node_index)
        {
            const auto global_node_index = partition.to_global_index(local_node_index);
            const auto sequential_global_node_index = sequential_node_index[global_node_index];

            EXPECT_EQ(grid.number_of_neighbors(local_node_index), expected_number_of_neighbors[sequential_global_node_index]);

            EXPECT_EQ(grid.neighbors_of(local_node_index, neighbors), expected_number_of_neighbors[sequential_global_node_index]);
            EXPECT_EQ(neighbors, map_neighbors(expected_neighbors[sequential_global_node_index], parallel_node_index));
        }
    }
    else if constexpr(space_dimension == 3)
    {
        RegularGrid grid(communicator, Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0), MultiIndex(3, 3, 3));
        const auto& partition = grid.partition();

        std::vector<int> sequential_node_index;

        if(number_of_processes == 1)
        {
            sequential_node_index.resize(partition.global_size());
            std::iota(sequential_node_index.begin(), sequential_node_index.end(), 0);
        }
        else if(number_of_processes == 2)
        {
            sequential_node_index = {0, 3, 6, 9, 12, 15, 18, 21, 24, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26};
        }
        else if(number_of_processes == 3)
        {
            sequential_node_index = {0, 3, 6, 9, 12, 15, 18, 21, 24, 1, 4, 7, 10, 13, 16, 19, 22, 25, 2, 5, 8, 11, 14, 17, 20, 23, 26};
        }
        else if(number_of_processes == 4)
        {
            sequential_node_index = {0, 9, 18, 3, 6, 12, 15, 21, 24, 1, 2, 10, 11, 19, 20, 4, 5, 7, 8, 13, 14, 16, 17, 22, 23, 25, 26};
        }

        const auto parallel_node_index = compute_inverse_order(sequential_node_index);

        const std::vector<int> expected_number_of_neighbors = {3, 4, 3, 4, 5, 4, 3, 4, 3, 4, 5, 4, 5, 6, 5, 4, 5, 4, 3, 4, 3, 4, 5, 4, 3, 4, 3};
        const std::vector<std::array<std::pair<int, int>, space_dimension>> expected_neighbors = {make_3d_neighborhood({-1, 1}, {-1, 3}, {-1, 9}), make_3d_neighborhood({0, 2}, {-1, 4}, {-1, 10}), make_3d_neighborhood({1, -1}, {-1, 5}, {-1, 11}), make_3d_neighborhood({-1, 4}, {0, 6}, {-1, 12}), make_3d_neighborhood({3, 5}, {1, 7}, {-1, 13}), make_3d_neighborhood({4, -1}, {2, 8}, {-1, 14}), make_3d_neighborhood({-1, 7}, {3, -1}, {-1, 15}), make_3d_neighborhood({6, 8}, {4, -1}, {-1, 16}), make_3d_neighborhood({7, -1}, {5, -1}, {-1, 17}), make_3d_neighborhood({-1, 10}, {-1, 12}, {0, 18}), make_3d_neighborhood({9, 11}, {-1, 13}, {1, 19}), make_3d_neighborhood({10, -1}, {-1, 14}, {2, 20}), make_3d_neighborhood({-1, 13}, {9, 15}, {3, 21}), make_3d_neighborhood({12, 14}, {10, 16}, {4, 22}), make_3d_neighborhood({13, -1}, {11, 17}, {5, 23}), make_3d_neighborhood({-1, 16}, {12, -1}, {6, 24}), make_3d_neighborhood({15, 17}, {13, -1}, {7, 25}), make_3d_neighborhood({16, -1}, {14, -1}, {8, 26}), make_3d_neighborhood({-1, 19}, {-1, 21}, {9, -1}), make_3d_neighborhood({18, 20}, {-1, 22}, {10, -1}), make_3d_neighborhood({19, -1}, {-1, 23}, {11, -1}), make_3d_neighborhood({-1, 22}, {18, 24}, {12, -1}), make_3d_neighborhood({21, 23}, {19, 25}, {13, -1}), make_3d_neighborhood({22, -1}, {20, 26}, {14, -1}), make_3d_neighborhood({-1, 25}, {21, -1}, {15, -1}), make_3d_neighborhood({24, 26}, {22, -1}, {16, -1}), make_3d_neighborhood({25, -1}, {23, -1}, {17, -1})};

        EXPECT_EQ(grid.number_of_nodes(), 27);
        EXPECT_EQ(grid.number_of_inner_nodes(), 1);
        EXPECT_EQ(grid.number_of_boundary_nodes(), 26);

        std::array<std::pair<int, int>, space_dimension> neighbors;

        const auto local_size = partition.local_size();
        for(int local_node_index = 0; local_node_index < local_size; ++local_node_index)
        {
            const auto global_node_index = partition.to_global_index(local_node_index);
            const auto sequential_global_node_index = sequential_node_index[global_node_index];

            EXPECT_EQ(grid.number_of_neighbors(local_node_index), expected_number_of_neighbors[sequential_global_node_index]);

            EXPECT_EQ(grid.neighbors_of(local_node_index, neighbors), expected_number_of_neighbors[sequential_global_node_index]);
            EXPECT_EQ(neighbors, map_neighbors(expected_neighbors[sequential_global_node_index], parallel_node_index));
        }
    }
}

TEST(RegularGrid, boundary_flags)
{
    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);

    if constexpr(space_dimension == 1)
    {
        RegularGrid grid(communicator, Point(1.0), Point(2.0), MultiIndex(5));

        EXPECT_TRUE(grid.is_boundary_node(0));
        EXPECT_FALSE(grid.is_boundary_node(1));
        EXPECT_FALSE(grid.is_boundary_node(2));
        EXPECT_FALSE(grid.is_boundary_node(3));
        EXPECT_TRUE(grid.is_boundary_node(4));
    }
    else if constexpr(space_dimension == 2)
    {
        RegularGrid grid(communicator, Point(1.0, 2.0), Point(3.0, 5.0), MultiIndex(3, 4));

        const auto& partition = grid.partition();

        std::vector<int> sequential_node_index;

        if(number_of_processes == 1)
        {
            sequential_node_index.resize(partition.global_size());
            std::iota(sequential_node_index.begin(), sequential_node_index.end(), 0);
        }
        else if(number_of_processes == 2)
        {
            sequential_node_index = {0, 3, 6, 9, 1, 2, 4, 5, 7, 8, 10, 11};
        }
        else if(number_of_processes == 3)
        {
            sequential_node_index = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
        }
        else if(number_of_processes == 4)
        {
            sequential_node_index = {0, 3, 6, 9, 1, 2, 4, 5, 7, 8, 10, 11};
        }

        std::vector<bool> expected_is_boundary_node = {true, true, true, true, false, true, true, false, true, true, true, true};

        const auto global_size = partition.global_size();

        for(int global_index = 0; global_index < global_size; ++global_index)
        {
            EXPECT_EQ(grid.is_boundary_node(global_index), expected_is_boundary_node[sequential_node_index[global_index]]);
        }
    }
    else if constexpr(space_dimension == 3)
    {
        RegularGrid grid(communicator, Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0), MultiIndex(3, 3, 3));
        const auto& partition = grid.partition();

        std::vector<int> sequential_node_index;

        if(number_of_processes == 1)
        {
            sequential_node_index.resize(partition.global_size());
            std::iota(sequential_node_index.begin(), sequential_node_index.end(), 0);
        }
        else if(number_of_processes == 2)
        {
            sequential_node_index = {0, 3, 6, 9, 12, 15, 18, 21, 24, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26};
        }
        else if(number_of_processes == 3)
        {
            sequential_node_index = {0, 3, 6, 9, 12, 15, 18, 21, 24, 1, 4, 7, 10, 13, 16, 19, 22, 25, 2, 5, 8, 11, 14, 17, 20, 23, 26};
        }
        else if(number_of_processes == 4)
        {
            sequential_node_index = {0, 9, 18, 3, 6, 12, 15, 21, 24, 1, 2, 10, 11, 19, 20, 4, 5, 7, 8, 13, 14, 16, 17, 22, 23, 25, 26};
        }

        std::vector<bool> expected_is_boundary_node = {true, true, true, true, true, true, true, true, true, true, true, true, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true};

        const auto global_size = partition.global_size();

        for(int global_index = 0; global_index < global_size; ++global_index)
        {
            EXPECT_EQ(grid.is_boundary_node(global_index), expected_is_boundary_node[sequential_node_index[global_index]]);
        }
    }
}

TEST(RegularGrid, node_coordinates)
{
    const auto communicator = MPI_COMM_WORLD;

    int number_of_processes;

    MPI_Comm_size(communicator, &number_of_processes);

    if constexpr(space_dimension == 1)
    {
        RegularGrid grid(communicator, Point(1.0), Point(2.0), MultiIndex(5));

        EXPECT_TRUE(equals(grid.node_coordinates(0), Point(1.0)));
        EXPECT_TRUE(equals(grid.node_coordinates(1), Point(1.25)));
        EXPECT_TRUE(equals(grid.node_coordinates(2), Point(1.5)));
        EXPECT_TRUE(equals(grid.node_coordinates(3), Point(1.75)));
        EXPECT_TRUE(equals(grid.node_coordinates(4), Point(2.0)));
    }
    else if constexpr(space_dimension == 2)
    {
        RegularGrid grid(communicator, Point(1.0, 3.0), Point(2.0, 6.0), MultiIndex(3, 4));
        const auto& partition = grid.partition();

        std::vector<int> sequential_node_index;

        if(number_of_processes == 1)
        {
            sequential_node_index.resize(partition.global_size());
            std::iota(sequential_node_index.begin(), sequential_node_index.end(), 0);
        }
        else if(number_of_processes == 2)
        {
            sequential_node_index = {0, 3, 6, 9, 1, 2, 4, 5, 7, 8, 10, 11};
        }
        else if(number_of_processes == 3)
        {
            sequential_node_index = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
        }
        else if(number_of_processes == 4)
        {
            sequential_node_index = {0, 3, 6, 9, 1, 2, 4, 5, 7, 8, 10, 11};
        }

        const std::vector<Point> expected_node_coordinates = {Point(1.0, 3.0), Point(1.5, 3.0), Point(2.0, 3.0), Point(1.0, 4.0), Point(1.5, 4.0), Point(2.0, 4.0), Point(1.0, 5.0), Point(1.5, 5.0), Point(2.0, 5.0), Point(1.0, 6.0), Point(1.5, 6.0), Point(2.0, 6.0)};

        const auto global_size = partition.global_size();

        for(int global_index = 0; global_index < global_size; ++global_index)
        {
            EXPECT_TRUE(equals(grid.node_coordinates(global_index), expected_node_coordinates[sequential_node_index[global_index]]));
        }
    }
    else if constexpr(space_dimension == 3)
    {
        RegularGrid grid(communicator, Point(1.0, 2.0, 3.0), Point(4.0, 6.0, 9.0), MultiIndex(3, 3, 3));
        const auto& partition = grid.partition();

        std::vector<int> sequential_node_index;

        if(number_of_processes == 1)
        {
            sequential_node_index.resize(partition.global_size());
            std::iota(sequential_node_index.begin(), sequential_node_index.end(), 0);
        }
        else if(number_of_processes == 2)
        {
            sequential_node_index = {0, 3, 6, 9, 12, 15, 18, 21, 24, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26};
        }
        else if(number_of_processes == 3)
        {
            sequential_node_index = {0, 3, 6, 9, 12, 15, 18, 21, 24, 1, 4, 7, 10, 13, 16, 19, 22, 25, 2, 5, 8, 11, 14, 17, 20, 23, 26};
        }
        else if(number_of_processes == 4)
        {
            sequential_node_index = {0, 9, 18, 3, 6, 12, 15, 21, 24, 1, 2, 10, 11, 19, 20, 4, 5, 7, 8, 13, 14, 16, 17, 22, 23, 25, 26};
        }

        const std::vector<Point> expected_node_coordinates = {Point(1.0, 2.0, 3.0), Point(2.5, 2.0, 3.0), Point(4.0, 2.0, 3.0), Point(1.0, 4.0, 3.0), Point(2.5, 4.0, 3.0), Point(4.0, 4.0, 3.0), Point(1.0, 6.0, 3.0), Point(2.5, 6.0, 3.0), Point(4.0, 6.0, 3.0), Point(1.0, 2.0, 6.0), Point(2.5, 2.0, 6.0), Point(4.0, 2.0, 6.0), Point(1.0, 4.0, 6.0), Point(2.5, 4.0, 6.0), Point(4.0, 4.0, 6.0), Point(1.0, 6.0, 6.0), Point(2.5, 6.0, 6.0), Point(4.0, 6.0, 6.0), Point(1.0, 2.0, 9.0), Point(2.5, 2.0, 9.0), Point(4.0, 2.0, 9.0), Point(1.0, 4.0, 9.0), Point(2.5, 4.0, 9.0), Point(4.0, 4.0, 9.0), Point(1.0, 6.0, 9.0), Point(2.5, 6.0, 9.0), Point(4.0, 6.0, 9.0)};

        const auto global_size = partition.global_size();

        for(int global_index = 0; global_index < global_size; ++global_index)
        {
            EXPECT_TRUE(equals(grid.node_coordinates(global_index), expected_node_coordinates[sequential_node_index[global_index]]));
        }
    }
}

TEST(RegularGrid, node_neighbor_distance)
{
    const auto communicator = MPI_COMM_WORLD;

    if constexpr(space_dimension == 1)
    {
        RegularGrid grid(communicator, Point(1.0), Point(2.0), MultiIndex(5));

        const auto& partition = grid.partition();

        if(partition.is_owned_by_local_process(1))
        {
            const auto local_node_index = partition.to_local_index(1);

            EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(local_node_index, 0, NeighborSuccession::predecessor), 0.25);
            EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(local_node_index, 0, NeighborSuccession::successor), 0.25);
        }
    }
    else if constexpr(space_dimension == 2)
    {
        RegularGrid grid(communicator, Point(1.0, 3.0), Point(2.0, 6.0), MultiIndex(3, 4));

        const auto& partition = grid.partition();

        int number_of_processes;
        MPI_Comm_size(partition.communicator(), &number_of_processes);

        const auto global_index = number_of_processes == 1 ? 4 : 6;

        if(partition.is_owned_by_local_process(global_index))
        {
            const auto local_node_index = partition.to_local_index(global_index);

            EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(local_node_index, 0, NeighborSuccession::predecessor), 0.5);
            EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(local_node_index, 0, NeighborSuccession::successor), 0.5);

            EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(local_node_index, 1, NeighborSuccession::predecessor), 1.0);
            EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(local_node_index, 1, NeighborSuccession::successor), 1.0);
        }
    }
    else if constexpr(space_dimension == 3)
    {
        RegularGrid grid(communicator, Point(1.0, 2.0, 3.0), Point(4.0, 6.0, 9.0), MultiIndex(3, 3, 3));

        const auto& partition = grid.partition();

        int number_of_processes;
        MPI_Comm_size(partition.communicator(), &number_of_processes);

        const auto global_index = number_of_processes == 1 || number_of_processes == 3 ? 13 : number_of_processes == 2 ? 17
                                                                                                                       : 19;

        if(partition.is_owned_by_local_process(global_index))
        {
            const auto local_node_index = partition.to_local_index(global_index);

            EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(local_node_index, 0, NeighborSuccession::predecessor), 1.5);
            EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(local_node_index, 0, NeighborSuccession::successor), 1.5);
            EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(local_node_index, 1, NeighborSuccession::predecessor), 2.0);
            EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(local_node_index, 1, NeighborSuccession::successor), 2.0);
            EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(local_node_index, 2, NeighborSuccession::predecessor), 3.0);
            EXPECT_DOUBLE_EQ(grid.node_neighbor_distance(local_node_index, 2, NeighborSuccession::successor), 3.0);
        }
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