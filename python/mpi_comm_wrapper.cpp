
#include "mpi_comm_wrapper.h"

MpiCommWrapper::MpiCommWrapper(MPI_Comm communicator) :
    communicator_(communicator)
{}

MpiCommWrapper::operator MPI_Comm() const
{
    return communicator_;
}

int MpiCommWrapper::rank() const
{
    int rank;
    MPI_Comm_rank(communicator_, &rank);
    return rank;
}

int MpiCommWrapper::size() const
{
    int size;
    MPI_Comm_size(communicator_, &size);
    return size;
}