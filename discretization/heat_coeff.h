#pragma once
#include "../grid/gridfunction.h"
#include "../linear_algebra/matrix.h"
#include "../linear_algebra/operations.h"
#include "../linear_algebra/vector.h"

template<typename T>
std::pair<SparseMatrix<T>, Vector<T>> assemble_heat_matrix(const RegularGrid& grid,
                                                           const GridFunction<T>& previous_temperature,
                                                           const scalar_t t,
                                                           const scalar_t delta_t,
                                                           const std::function<T(const Point&, const scalar_t)>& rhs_function,
                                                           const std::function<T(const Point&, const scalar_t)>& boundary_function,
                                                           const std::function<T(const Point&)>& alpha,
                                                           const std::function<T(const Point&)>& rho,
                                                           const std::function<T(const Point&)>& c)
{
    const auto& partition = grid.partition();
    const auto local_size = partition.local_size();
    std::vector<std::tuple<int, int, T>> Init_A;
    Vector<T> rhs(partition);
    std::array<std::pair<int, int>, space_dimension> neighbors;
    const auto& node_distance = grid.node_distance();

    // Haupt Schleife
    for(auto local_node_index = 0; local_node_index < local_size; ++local_node_index)
    {
        const auto global_node_index = partition.to_global_index(local_node_index);

        Point point_x = grid.node_coordinates(global_node_index);
        Point forward_point;
        Point backward_point;

        T diag_entry = 0.0; // compute the diag entry;
        for(int count = 0; count < space_dimension; ++count)
        {
            forward_point = point_x;
            forward_point[count] += node_distance[count] / 2;
            diag_entry += alpha(forward_point) / std::pow(node_distance[count], 2);

            backward_point = point_x;
            backward_point[count] -= node_distance[count] / 2;
            diag_entry += alpha(backward_point) / std::pow(node_distance[count], 2);
        }

        if(grid.is_boundary_node(global_node_index))
        {
            Init_A.push_back({local_node_index, global_node_index, 1.0});
            rhs[local_node_index] = boundary_function(point_x, t);
        }
        else
        {
            grid.neighbors_of(local_node_index, neighbors);
            Init_A.push_back({local_node_index, global_node_index, diag_entry + rho(point_x) * c(point_x) / delta_t});
            rhs[local_node_index] = rhs_function(point_x, t) + rho(point_x) * c(point_x) * previous_temperature.value(local_node_index) / delta_t;
            for(int i = 0; i < space_dimension; i++)
            {
                if(grid.is_boundary_node(neighbors[i].first))
                {
                    rhs[local_node_index] += boundary_function(grid.node_coordinates(neighbors[i].first), t) / std::pow(node_distance[i], 2);
                    Init_A.push_back({local_node_index, neighbors[i].first, 0});
                }
                else
                {
                    backward_point = point_x;
                    backward_point[i] -= node_distance[i] / 2;
                    Init_A.push_back({local_node_index, neighbors[i].first, -alpha(backward_point) / std::pow(node_distance[i], 2)});
                }
                if(grid.is_boundary_node(neighbors[i].second))
                {
                    rhs[local_node_index] += boundary_function(grid.node_coordinates(neighbors[i].second), t) / std::pow(node_distance[i], 2);
                    Init_A.push_back({local_node_index, neighbors[i].second, 0});
                }
                else
                {
                    forward_point = point_x;
                    forward_point[i] += node_distance[i] / 2;
                    Init_A.push_back({local_node_index, neighbors[i].second, -alpha(forward_point) / std::pow(node_distance[i], 2)});
                }
            }
        }
    }

    SparseMatrix<T> A(partition, partition.global_size(), Init_A);
    A.initialize_exchange_pattern(partition);
    return std::make_pair(A, rhs);
}