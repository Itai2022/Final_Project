
#include "grid.h"
#include <cassert>

RegularGrid::RegularGrid(Point min_corner, Point max_corner, MultiIndex node_count_per_dimension)
{
    min_corner_ = min_corner;
    node_count_per_dimension_ = node_count_per_dimension;
    global_node_count_per_dimension_ = node_count_per_dimension;
    int size = 1;
    for(int i = 0; i < space_dimension; i++)
    {
        node_distance_[i] = (max_corner[i] - min_corner[i]) / (node_count_per_dimension[i] - 1);
        size *= node_count_per_dimension[i];
        dims[i] = 1;
    }
    int periods[space_dimension] = {0};
    MPI_Comm new_comm;
    MPI_Cart_create(MPI_COMM_SELF, space_dimension, &dims[0], periods, 1, &new_comm);
    partition_ = create_partition(new_comm, size);
}

// for MPI
RegularGrid::RegularGrid(MPI_Comm communicator, Point min_corner, Point max_corner, MultiIndex global_node_count_per_dimension)
{
    // initialize MPI
    int number_of_processes;
    int myrank;
    MPI_Comm_size(communicator, &number_of_processes);

    MPI_Dims_create(number_of_processes, space_dimension, &dims[0]);

    MPI_Comm cart_comm;
    int periods[space_dimension] = {0};
    MPI_Cart_create(communicator, space_dimension, &dims[0], periods, 0, &cart_comm);

    MPI_Comm_rank(cart_comm, &myrank);
    MPI_Cart_coords(cart_comm, myrank, space_dimension, &current_coords[0]);

    // initialize
    int node_count = 1;       // help creating partition
    min_corner_ = min_corner; // half of initializing min_corner_
    for(int i = 0; i < space_dimension; i++)
    {
        // initialize node_count_per_dimension_ without rest_
        node_count_per_dimension_[i] = global_node_count_per_dimension[i] / dims[i];
        // initialize the rest_
        rest_[i] = global_node_count_per_dimension[i] % dims[i];

        // initialize node_distance_
        node_distance_[i] = (max_corner[i] - min_corner[i]) / (global_node_count_per_dimension[i] - 1);

        // initialize node_count_per_dimension of last process
        if(current_coords[i] == dims[i] - 1)
        {
            node_count_per_dimension_[i] += rest_[i];
        }

        // get node_count for initializing partition_
        node_count *= node_count_per_dimension_[i];
    }
    // initialize global_node_count_per_dimension_
    global_node_count_per_dimension_ = global_node_count_per_dimension;
    // initialize partition_
    partition_ = create_partition(cart_comm, node_count);
}

MultiIndex RegularGrid::node_count_per_dimension() const
{
    return node_count_per_dimension_;
}

int RegularGrid::number_of_nodes() const
{
    return partition_.global_size();
}

int RegularGrid::number_of_inner_nodes() const
{
    int node_count = 1;
    for(int i = 0; i < space_dimension; i++)
    {
        node_count *= global_node_count_per_dimension_[i] - 2;
    }
    return node_count;
}

int RegularGrid::number_of_boundary_nodes() const
{
    return number_of_nodes() - number_of_inner_nodes();
}

int RegularGrid::number_of_neighbors(int local_node_index) const
{
    int neigh = space_dimension * 2;
    MultiIndex multi_index = to_multi_index(local_node_index, node_count_per_dimension_);
    auto coords = local_process_coordinates();
    for(int i = 0; i < space_dimension; i++)
    {
        multi_index[i] += coords[i] * (global_node_count_per_dimension_[i] / dims[i]);
        if(multi_index[i] == 0 || multi_index[i] == global_node_count_per_dimension_[i] - 1)
        {
            neigh -= 1;
        }
    }
    return neigh;
}

int RegularGrid::neighbors_of(int local_node_index, std::array<std::pair<int, int>, space_dimension>& neighbors) const
{

    MultiIndex local_multi_index = to_multi_index(local_node_index, node_count_per_dimension_);
    MultiIndex local_process_coords = local_process_coordinates();

    int offset_point = 1;
    int direction_point;
    for(int i = 0; i < space_dimension; i++)
    {
        // predecessor
        direction_point = local_node_index - offset_point;
        if(local_multi_index[i] == 0)
        {
            if(local_process_coords[i] == 0)
            {
                neighbors[i].first = -1;
            }
            else
            {
                MultiIndex index = to_multi_index(local_node_index, node_count_per_dimension_);
                MultiIndex coords = local_process_coordinates();
                coords[i] -= 1;
                int rank;
                MPI_Cart_rank(partition_.communicator(), &coords[0], &rank);
                MultiIndex node_count_per_dim = node_count_per_dimension(rank);
                index[i] = node_count_per_dim[i] - 1;
                int flat_index = 0;
                int count = 1;
                for(int j = 0; j < space_dimension; j++)
                {
                    flat_index += index[j] * count;
                    count *= node_count_per_dim[j];
                }
                neighbors[i].first = partition_.to_global_index(flat_index, rank);
            }
        }
        else
        {
            neighbors[i].first = partition_.to_global_index(direction_point);
        }
        // successor
        direction_point = local_node_index + offset_point;
        if(local_multi_index[i] == node_count_per_dimension_[i] - 1)
        {
            if(local_process_coords[i] == processes_per_dimension()[i] - 1)
            {
                neighbors[i].second = -1;
            }
            else
            {
                MultiIndex index = to_multi_index(local_node_index, node_count_per_dimension_);
                MultiIndex coords = local_process_coordinates();
                coords[i] += 1;
                int rank;
                MPI_Cart_rank(partition_.communicator(), &coords[0], &rank);
                MultiIndex node_count_per_dim = node_count_per_dimension(rank);
                index[i] = 0;
                int flat_index = 0;
                int count = 1;
                for(int j = 0; j < space_dimension; j++)
                {
                    flat_index += index[j] * count;
                    count *= node_count_per_dim[j];
                }
                neighbors[i].second = partition_.to_global_index(flat_index, rank);
            }
        }
        else
        {
            neighbors[i].second = partition_.to_global_index(direction_point);
        }
        offset_point *= node_count_per_dimension_[i];
    }
    return number_of_neighbors(local_node_index);
}

bool RegularGrid::is_boundary_node(int global_node_index) const
{
    assert(global_node_index < number_of_nodes());
    int local_index = partition_.to_local_index(global_node_index);
    int owner_process = partition_.owner_process(global_node_index);
    MultiIndex multi_index = to_multi_index(local_index, node_count_per_dimension(owner_process));
    MultiIndex coords;
    MPI_Cart_coords(partition_.communicator(), owner_process, space_dimension, &coords[0]);
    for(int i = 0; i < space_dimension; i++)
    {
        multi_index[i] += coords[i] * (global_node_count_per_dimension_[i] / dims[i]);
        if(multi_index[i] == 0 || multi_index[i] == global_node_count_per_dimension_[i] - 1)
        {
            return true;
        }
    }
    return false;
}

Point RegularGrid::node_coordinates(int global_node_index) const
{

    int local_index = partition_.to_local_index(global_node_index);
    int owner_process = partition_.owner_process(global_node_index);
    MultiIndex multi_index = to_multi_index(local_index, node_count_per_dimension(owner_process));
    MultiIndex coords;
    MPI_Cart_coords(partition_.communicator(), owner_process, space_dimension, &coords[0]);
    Point data;
    for(int i = 0; i < space_dimension; i++)
    {
        data[i] = min_corner_[i] + node_distance_[i] * ((global_node_count_per_dimension_[i] / dims[i]) * coords[i] + multi_index[i]);
    }
    return data;
}

scalar_t RegularGrid::node_neighbor_distance(int local_node_index, int neighbor_direction, NeighborSuccession neighbor_succession) const
{
    assert(0 <= local_node_index && local_node_index < partition_.local_size());
    assert(neighbor_direction >= 0 && neighbor_direction < space_dimension);
    std::array<std::pair<int, int>, space_dimension> neighbors;
    neighbors_of(local_node_index, neighbors);
    int global_node_index = partition_.to_global_index(local_node_index);
    if(neighbor_succession == NeighborSuccession::predecessor)
    {
        return node_coordinates(global_node_index)[neighbor_direction] - node_coordinates(neighbors[neighbor_direction].first)[neighbor_direction];
    }
    if(neighbor_succession == NeighborSuccession::successor)
    {
        return node_coordinates(neighbors[neighbor_direction].second)[neighbor_direction] - node_coordinates(global_node_index)[neighbor_direction];
    }
    else
    {
        return -1;
    }
}

// additional function
MultiIndex RegularGrid::to_multi_index(int local_node_index, const MultiIndex& node_count_per_dimension) const
{
    MultiIndex multi_index;
    int node_count = 1;
    for(int i = 0; i < space_dimension; i++)
    {
        multi_index[i] = (local_node_index / node_count) % node_count_per_dimension[i];
        node_count *= node_count_per_dimension[i];
    }
    return multi_index;
}

// additional function
Point RegularGrid::node_distance() const
{
    return node_distance_;
}

// for MPI
const ContiguousParallelPartition& RegularGrid::partition() const
{
    return partition_;
}

MultiIndex RegularGrid::processes_per_dimension() const
{
    return dims;
}
MultiIndex RegularGrid::local_process_coordinates() const
{
    return current_coords;
}

MultiIndex RegularGrid::global_node_count_per_dimension() const
{
    return global_node_count_per_dimension_;
}

MultiIndex RegularGrid::node_count_per_dimension(int process_rank) const
{
    MultiIndex coords;
    MPI_Cart_coords(partition_.communicator(), process_rank, space_dimension, &coords[0]);
    MultiIndex mycoords = local_process_coordinates();
    MultiIndex count = node_count_per_dimension();
    for(int i = 0; i < space_dimension; i++)
    {
        if(mycoords[i] == processes_per_dimension()[i] - 1 && coords[i] != processes_per_dimension()[i] - 1)
        {
            count[i] -= rest_[i];
        }
        else if(mycoords[i] != processes_per_dimension()[i] - 1 && coords[i] == processes_per_dimension()[i] - 1)
        {
            count[i] += rest_[i];
        }
    }
    return count;
}

// additional function to compute multi index with one dim. index for io.h
MultiIndex to_multi_index(int local_node_index, const MultiIndex& node_count_per_dimension)
{
    MultiIndex multi_index;
    int node_count = 1;
    for(int i = 0; i < space_dimension; i++)
    {
        multi_index[i] = (local_node_index / node_count) % node_count_per_dimension[i];
        node_count *= node_count_per_dimension[i];
    }
    return multi_index;
}

// to 1 dim index for io.h
int from_multi_index(const MultiIndex& local_node_multi_index, const MultiIndex& node_count_per_dimension)
{
    int flat_index = 0;
    int count = 1;
    for(int i = 0; i < space_dimension; i++)
    {
        flat_index += local_node_multi_index[i] * count;
        count *= node_count_per_dimension[i];
    }
    return flat_index;
}
