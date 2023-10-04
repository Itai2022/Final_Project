
#ifndef PMSC_GRID_IO_H
#define PMSC_GRID_IO_H

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "grid.h"
#include "gridfunction.h"

namespace detail
{

inline std::string make_extend_string(const MultiIndex& start, const MultiIndex& count, const MultiIndex& global_count, int ghost_level)
{
    std::stringstream extend;
    for(int d = 0; d < start.size(); ++d)
    {
        assert(count[d] > 0);
        extend << std::max(start[d] - ghost_level, 0) << " "
               << std::min(start[d] + count[d] - 1 + ghost_level, global_count[d] - 1) << " ";
    }
    for(int d = start.size(); d < 3; ++d)
    {
        extend << "0 0 ";
    }
    return extend.str();
}

inline void write_point(std::ostream& file, Point point)
{
    for(int d = 0; d < point.size(); ++d)
    {
        file << point[d] << " ";
    }
    for(int d = point.size(); d < 3; ++d)
    {
        file << 0 << " ";
    }
    file << "\n";
}

inline MultiIndex node_offset_per_dimension(const MPI_Comm& communicator, int process_rank, const MultiIndex& global_node_count_per_dimension, const MultiIndex& processes_per_dimension)
{
    std::array<int, space_dimension> processes_coordinates;
    MPI_Cart_coords(communicator, process_rank, space_dimension, processes_coordinates.data());

    MultiIndex result;
    for(int i = 0; i < result.size(); ++i)
    {
        const auto local_process_count_in_dimension = global_node_count_per_dimension[i] / processes_per_dimension[i];
        result[i] = local_process_count_in_dimension * processes_coordinates[i];
    }
    return result;
}

} // namespace detail

template<typename T>
void write_to_vtk(const std::filesystem::path& file_path, const GridFunction<T>& grid_function, const std::string& name)
{
    const auto& partition = grid_function.grid().partition();
    const auto& processes_per_dimension = grid_function.grid().processes_per_dimension();
    const auto& local_process_coordinates = grid_function.grid().local_process_coordinates();

    int number_of_processes;
    MPI_Comm_size(partition.communicator(), &number_of_processes);
    int local_process_rank;
    MPI_Comm_rank(partition.communicator(), &local_process_rank);
    const auto is_master = local_process_rank == 0;

    const auto process_filename = file_path.stem().string() + "_processor" + std::to_string(local_process_rank) + ".vts";
    const auto master_filename = file_path.stem().string() + "_master.pvts";

    std::ofstream master_file;
    if(is_master)
    {
        master_file.open(file_path.parent_path() / master_filename);
        if(!master_file.is_open()) throw std::runtime_error("Failed to open file '" + (file_path.parent_path() / master_filename).string() + "'");
    }

    std::ofstream process_file(file_path.parent_path() / process_filename);
    if(!process_file.is_open()) throw std::runtime_error("Failed to open file '" + (file_path.parent_path() / process_filename).string() + "'");

    process_file << std::scientific << std::setprecision(std::numeric_limits<scalar_t>::max_digits10) << std::boolalpha;

    const auto global_node_count_per_dimension = grid_function.grid().global_node_count_per_dimension();

    MultiIndex extened_start_per_dimension;
    MultiIndex extened_end_per_dimension;
    MultiIndex extened_count_per_dimension;
    int extend_count = 1;
    const int ghost_level = 1;
    for(int i = 0; i < space_dimension; ++i)
    {
        const auto local_process_count_in_dimension = global_node_count_per_dimension[i] / processes_per_dimension[i];
        extened_start_per_dimension[i] = std::max(local_process_count_in_dimension * local_process_coordinates[i] - ghost_level, 0);
        if(local_process_coordinates[i] == processes_per_dimension[i] - 1)
        {
            extened_end_per_dimension[i] = global_node_count_per_dimension[i];
        }
        else
        {
            extened_end_per_dimension[i] = local_process_count_in_dimension * (local_process_coordinates[i] + 1) + 1;
        }
        extened_count_per_dimension[i] = extened_end_per_dimension[i] - extened_start_per_dimension[i];
        extend_count *= extened_count_per_dimension[i];
    }

    const auto global_extend = detail::make_extend_string(MultiIndex(0), global_node_count_per_dimension, global_node_count_per_dimension, 0);
    const auto local_node_offset_per_dimension = detail::node_offset_per_dimension(partition.communicator(), local_process_rank, global_node_count_per_dimension, processes_per_dimension);
    const auto local_extend = detail::make_extend_string(local_node_offset_per_dimension, grid_function.grid().node_count_per_dimension(), global_node_count_per_dimension, 1);

    if(is_master)
    {
        master_file << "<?xml version=\"1.0\"?>"
                    << "\n"
                    << "<VTKFile type=\"PStructuredGrid\">"
                    << "\n"
                    << "<PStructuredGrid WholeExtent=\"" << global_extend << "\" GhostLevel=\"1\">"
                    << "\n"
                    << "<PPoints>"
                    << "\n"
                    << "<PDataArray type=\"Float64\" Name=\"coordinate\" NumberOfComponents=\"3\" format=\"ascii\">"
                    << "\n";
    }

    process_file << "<?xml version=\"1.0\"?>"
                 << "\n"
                 << "<VTKFile type=\"StructuredGrid\">"
                 << "\n"
                 << "<StructuredGrid WholeExtent=\"" << local_extend << "\">"
                 << "\n"
                 << "<Piece Extent=\"" << local_extend << "\">"
                 << "\n"
                 << "<Points>"
                 << "\n"
                 << "<DataArray type=\"Float64\" Name=\"coordinate\" NumberOfComponents=\"3\" format=\"ascii\">"
                 << "\n";

    // const auto number_of_nodes = partition.local_size();
    // std::array<std::pair<int, int>, space_dimension> neighbors;

    for(int node_index = 0; node_index < extend_count; ++node_index)
    {
        const auto node_multi_index = to_multi_index(node_index, extened_count_per_dimension);
        bool is_ghost = false;
        for(int d = 0; d < space_dimension; ++d)
        {
            if((local_process_coordinates[d] != 0 && node_multi_index[d] == 0) ||
               (local_process_coordinates[d] != processes_per_dimension[d] - 1 && (node_multi_index[d] == extened_count_per_dimension[d] - 1)))
            {
                is_ghost = true;
                break;
            }
        }

        if(is_ghost)
        {
            detail::write_point(process_file, Point(0));
        }
        else
        {
            auto local_node_multi_index = node_multi_index;
            for(int d = 0; d < space_dimension; ++d)
            {
                if(local_process_coordinates[d] != 0)
                {
                    local_node_multi_index[d] -= 1;
                }
            }
            const auto local_node_index = from_multi_index(local_node_multi_index, grid_function.grid().node_count_per_dimension());
            detail::write_point(process_file, grid_function.grid().node_coordinates(grid_function.grid().partition().to_global_index(local_node_index)));
        }
    }

    if(is_master)
    {
        master_file << "</PDataArray>"
                    << "\n"
                    << "</PPoints>"
                    << "\n"
                    << "<PPointData>"
                    << "\n"
                    << "<PDataArray type=\"Float64\" Name=\"" << name << "\" NumberOfComponents=\"1\" format=\"ascii\">"
                    << "\n";
    }

    process_file << "</DataArray>"
                 << "\n"
                 << "</Points>"
                 << "\n"
                 << "<PointData>"
                 << "\n"
                 << "<DataArray type=\"Float64\" Name=\"" << name << "\" NumberOfComponents=\"1\" format=\"ascii\">"
                 << "\n";

    for(int node_index = 0; node_index < extend_count; ++node_index)
    {
        const auto node_multi_index = to_multi_index(node_index, extened_count_per_dimension);
        bool is_ghost = false;
        for(int d = 0; d < space_dimension; ++d)
        {
            if((local_process_coordinates[d] != 0 && node_multi_index[d] == 0) ||
               (local_process_coordinates[d] != processes_per_dimension[d] - 1 && (node_multi_index[d] == extened_count_per_dimension[d] - 1)))
            {
                is_ghost = true;
                break;
            }
        }

        if(is_ghost)
        {
            process_file << 0.0 << "\n";
        }
        else
        {
            auto local_node_multi_index = node_multi_index;
            for(int d = 0; d < space_dimension; ++d)
            {
                if(local_process_coordinates[d] != 0)
                {
                    local_node_multi_index[d] -= 1;
                }
            }
            const auto local_node_index = from_multi_index(local_node_multi_index, grid_function.grid().node_count_per_dimension());
            process_file << grid_function.value(local_node_index) << "\n";
        }
    }

    if(is_master)
    {
        master_file << "</PDataArray>"
                    << "\n"
                    << "</PPointData>"
                    << "\n";
        for(int p = 0; p < number_of_processes; ++p)
        {
            const auto process_node_offset_per_dimension = detail::node_offset_per_dimension(partition.communicator(), p, global_node_count_per_dimension, processes_per_dimension);
            const auto process_extend = detail::make_extend_string(process_node_offset_per_dimension, grid_function.grid().node_count_per_dimension(p), global_node_count_per_dimension, 1);
            master_file << "<Piece Extent=\"" << process_extend << "\" Source=\"" << file_path.stem().string() << "_processor" << p << ".vts"
                        << "\">"
                        << "\n"
                        << "</Piece>"
                        << "\n";
        }
        master_file << "</PStructuredGrid>"
                    << "\n"
                    << "</VTKFile>"
                    << "\n";
    }
    process_file << "</DataArray>"
                 << "\n"
                 << "</PointData>"
                 << "\n"
                 << "</Piece>"
                 << "\n"
                 << "</StructuredGrid>"
                 << "\n"
                 << "</VTKFile>"
                 << "\n";
}

#endif // PMSC_GRID_IO_H
