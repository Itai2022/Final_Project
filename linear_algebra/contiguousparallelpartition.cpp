#include "contiguousparallelpartition.h"
#include <cassert>
#include <numeric>
ContiguousParallelPartition::ContiguousParallelPartition()
{
    communicator_ = MPI_COMM_SELF;
}

ContiguousParallelPartition::ContiguousParallelPartition(MPI_Comm communicator, std::vector<int> partition)
{
    communicator_ = communicator;
    partition_ = partition;
}

MPI_Comm ContiguousParallelPartition::communicator() const
{
    return communicator_;
}

int ContiguousParallelPartition::local_size() const
{
    int myrank;
    MPI_Comm_rank(communicator_, &myrank);
    return partition_[myrank + 1] - partition_[myrank];
}

int ContiguousParallelPartition::local_size(int process) const
{
    assert(process >= 0 && process < partition_.size() - 1);
    return partition_[process + 1] - partition_[process];
}

int ContiguousParallelPartition::global_size() const
{
    return partition_.back();
}

int ContiguousParallelPartition::owner_process(int global_index) const
{
    assert(0 <= global_index && global_index < global_size());
    for(int i = 0; i < partition_.size(); i++)
    {
        if(partition_[i] <= global_index && global_index < partition_[i + 1])
        {
            return i;
        }
    }
    return -1;
}

bool ContiguousParallelPartition::is_owned_by_local_process(int global_index) const
{
    assert(0 <= global_index && global_index < global_size());
    int myrank;
    MPI_Comm_rank(communicator_, &myrank);
    return myrank == owner_process(global_index) ? true : false;
}

bool ContiguousParallelPartition::is_owned_by_process(int global_index, int process) const
{
    assert(0 <= process && process < partition_.size() - 1);
    assert(0 <= global_index && global_index < global_size());
    return process == owner_process(global_index) ? true : false;
}

int ContiguousParallelPartition::to_global_index(int local_index) const
{
    assert(0 <= local_index && local_index < local_size());
    int myrank;
    MPI_Comm_rank(communicator_, &myrank);
    return partition_[myrank] + local_index;
}

int ContiguousParallelPartition::to_local_index(int global_index) const
{
    assert(0 <= global_index && global_index < global_size());
    int process;
    process = owner_process(global_index);
    return global_index - partition_[process];
}

// for parallelized grid
int ContiguousParallelPartition::to_global_index(int local_index, int owner_process) const
{
    assert(0 <= local_index && local_index < local_size(owner_process));
    return partition_[owner_process] + local_index;
}

ContiguousParallelPartition create_partition(MPI_Comm communicator, int local_size)
{
    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    std::vector<int> partition(number_of_processes);
    MPI_Allgather(&local_size, 1, MPI_INT, partition.data(), 1, MPI_INT, communicator);

    partition.insert(partition.begin(), 0);
    std::partial_sum(partition.begin(), partition.end(), partition.begin());
    return ContiguousParallelPartition(communicator, std::move(partition));
}

ContiguousParallelPartition create_uniform_partition(MPI_Comm communicator, int global_size)
{
    int number_of_processes;
    MPI_Comm_size(communicator, &number_of_processes);
    int local_size = global_size / number_of_processes;
    std::vector<int> partition;
    int sum = 0;
    for(int i = 0; i < number_of_processes; i++)
    {
        partition.push_back(sum);
        sum += local_size;
    }
    partition.push_back(global_size);
    return ContiguousParallelPartition(communicator, partition);
}

std::ostream& operator<<(std::ostream& os, const ContiguousParallelPartition& pat)
{
    os << "{";
    for(int i = 0; i < pat.partition_.size() - 1; ++i)
    {
        os << pat.partition_[i] << ", ";
    }
    os << pat.partition_.back() << "}" << std::endl;
    return os;
}