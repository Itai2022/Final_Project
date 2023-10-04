#pragma once
#include "../grid/gridfunction.h"
#include "../linear_algebra/matrix.h"
#include "../linear_algebra/operations.h"
#include "../linear_algebra/vector.h"

template<typename T>
std::pair<SparseMatrix<T>, Vector<T>> assemble_poisson_matrix(const RegularGrid& grid,
                                                              const std::function<T(const Point&)>& rhs_function,
                                                              const std::function<T(const Point&)>& boundary_function)
{
    const auto& partition = grid.partition();
    const auto local_size = partition.local_size();
    std::vector<std::tuple<int, int, T>> Init_A;
    Vector<T> rhs(partition);
    std::array<std::pair<int, int>, space_dimension> neighbors;
    T diag_entry = 0.0;
    /* Da wir reguläres Grid haben, ist der Abstand in einer bestimmten Richtung gleich.
     Müsse wir also nur 1-mal das 2.0/h^2 rechnen wobei h=(hx,hy,hz);*/
    for(int count = 0; count < space_dimension; ++count)
    {
        diag_entry += 2.0 / std::pow(grid.node_distance()[count], 2);
    }
    // Haupt Schleife
    for(auto local_node_index = 0; local_node_index < local_size; ++local_node_index)
    {
        const auto global_node_index = partition.to_global_index(local_node_index);
        if(grid.is_boundary_node(global_node_index))
        {
            Init_A.push_back({local_node_index, global_node_index, 1.0});
            rhs[local_node_index] = boundary_function(grid.node_coordinates(global_node_index));
        }
        else
        {
            grid.neighbors_of(local_node_index, neighbors);
            Init_A.push_back({local_node_index, global_node_index, diag_entry});
            rhs[local_node_index] = rhs_function(grid.node_coordinates(global_node_index));
            for(int i = 0; i < space_dimension; i++)
            {
                if(grid.is_boundary_node(neighbors[i].first))
                {
                    rhs[local_node_index] += boundary_function(grid.node_coordinates(neighbors[i].first)) / std::pow(grid.node_distance()[i], 2);
                    Init_A.push_back({local_node_index, neighbors[i].first, 0}); // Comment out if there are to little number of nodes
                }
                else
                {
                    Init_A.push_back({local_node_index, neighbors[i].first, -1.0 / std::pow(grid.node_distance()[i], 2)});
                }
                if(grid.is_boundary_node(neighbors[i].second))
                {
                    rhs[local_node_index] += boundary_function(grid.node_coordinates(neighbors[i].second)) / std::pow(grid.node_distance()[i], 2);
                    Init_A.push_back({local_node_index, neighbors[i].second, 0}); // Comment out if there are to little number of nodes
                }
                else
                {
                    Init_A.push_back({local_node_index, neighbors[i].second, -1.0 / std::pow(grid.node_distance()[i], 2)});
                }
            }
        }
    }
    SparseMatrix<T> A(partition, partition.global_size(), Init_A);
    A.initialize_exchange_pattern(partition);
    return std::make_pair(A, rhs);
}
