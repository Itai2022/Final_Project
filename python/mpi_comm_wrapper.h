#ifndef PMSC_MPI_COMM_WRAPPER_H
#define PMSC_MPI_COMM_WRAPPER_H

#include <mpi.h>

class MpiCommWrapper
{
public:
    MpiCommWrapper(MPI_Comm communicator);

    operator MPI_Comm() const;

    int rank() const;
    int size() const;
private:
    MPI_Comm communicator_;
};

#endif // PMSC_MPI_COMM_WRAPPER_H
