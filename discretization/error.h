#pragma once
#include "../grid/grid.h"
#include "../grid/gridfunction.h"
#include <cmath>

template<typename T>
T compute_l_infinity_error(const RegularGrid& grid,
                           const GridFunction<T>& computed_solution,
                           const std::function<T(const Point&)>& analytical_solution)
{
    // find the maximum of the computed solution
    const auto& partition = grid.partition();
    const auto size = partition.local_size();
    const auto global_id = partition.to_global_index(0);
    T result = std::fabs(analytical_solution(grid.node_coordinates(global_id)) - computed_solution.value(0));
    for(int local_node_index = 1; local_node_index < size; ++local_node_index)
    {
        const auto global = partition.to_global_index(local_node_index);
        T temp = std::fabs(analytical_solution(grid.node_coordinates(global)) - computed_solution.value(local_node_index));
        if(temp > result)
        {
            result = temp;
        }
    }
    T max = 0;
    MPI_Allreduce(&result, &max, 1, MPI_DOUBLE, MPI_MAX, partition.communicator());
    return max;
}