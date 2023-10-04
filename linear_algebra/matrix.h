#pragma once

#include "../common/scalar.h"
#include "contiguousparallelpartition.h"
#include "exchangepattern.h"
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

class ExchangePattern;

template<typename T>
class SparseMatrix
{
public:
    using triplet_type = std::tuple<int, int, T>;

    // constructors with 1 process
    SparseMatrix();
    SparseMatrix(const SparseMatrix& other);
    SparseMatrix(SparseMatrix&& other) noexcept;
    SparseMatrix& operator=(const SparseMatrix& other);
    SparseMatrix& operator=(SparseMatrix&& other) noexcept;
    explicit SparseMatrix(int rows, int columns,
                          const std::vector<triplet_type>& entries);

    // constructor MPI
    explicit SparseMatrix(ContiguousParallelPartition row_partition, int global_columns, std::function<int(int)> nz_per_row);
    explicit SparseMatrix(ContiguousParallelPartition row_partition, int global_columns, const std::vector<triplet_type>& entries);
    explicit SparseMatrix(MPI_Comm communicator, int local_rows, int global_columns, std::function<int(int)> nz_per_row);
    explicit SparseMatrix(MPI_Comm communicator, int local_rows, int global_columns, const std::vector<triplet_type>& entries);

    // getter for the row partition
    const ContiguousParallelPartition& row_partition() const;

    // create exchange_pattern for multiprocess multiply
    void initialize_exchange_pattern(const ContiguousParallelPartition& column_partition);
    const ExchangePattern& exchange_pattern() const;

    // r = row_index nz_i = non zero index
    int rows() const;
    int columns() const;
    int non_zero_size() const;
    int row_nz_size(int r) const;
    const int& row_nz_index(int r, int nz_i) const;
    int& row_nz_index(int r, int nz_i);
    const T& row_nz_entry(int r, int nz_i) const;
    T& row_nz_entry(int r, int nz_i);
    ~SparseMatrix() = default;

    // helper function that prints the matrix
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const SparseMatrix<U>& M);

private:
    std::unique_ptr<T[]> A_;    // the data of non zero elements
    std::unique_ptr<int[]> JA_; // the column data
    std::unique_ptr<int[]> IA_; // the number of non zero elements that prior rows have
    int columns_;
    int nnz_;
    int rows_;
    ContiguousParallelPartition row_partition_;
    ExchangePattern pattern_;
};

template<typename T>
SparseMatrix<T>::SparseMatrix() :
    A_(nullptr), JA_(nullptr), IA_(nullptr), columns_(0), nnz_(0), rows_(0) {}

// copy constructor
template<typename T>
SparseMatrix<T>::SparseMatrix(const SparseMatrix<T>& other)
{
    columns_ = other.columns_;
    nnz_ = other.nnz_;
    rows_ = other.rows_;
    A_ = std::make_unique<T[]>(other.nnz_);
    JA_ = std::make_unique<int[]>(other.nnz_);
    IA_ = std::make_unique<int[]>(other.rows_ + 1);
    std::copy(other.A_.get(), other.A_.get() + other.nnz_, A_.get());
    std::copy(other.JA_.get(), other.JA_.get() + other.nnz_, JA_.get());
    std::copy(other.IA_.get(), other.IA_.get() + (other.rows_ + 1), IA_.get());
    row_partition_ = other.row_partition_;
    pattern_ = other.pattern_;
}

// Move constructor
template<typename T>
SparseMatrix<T>::SparseMatrix(SparseMatrix<T>&& other) noexcept
{
    columns_ = other.columns_;
    nnz_ = other.nnz_;
    rows_ = other.rows_;
    A_ = std::move(other.A_);
    JA_ = std::move(other.JA_);
    IA_ = std::move(other.IA_);
    row_partition_ = std::move(other.row_partition_);
    pattern_ = std::move(other.pattern_);
}

// copy assignment
template<typename T>
SparseMatrix<T>& SparseMatrix<T>::operator=(const SparseMatrix<T>& other)
{
    columns_ = other.columns_;
    nnz_ = other.nnz_;
    rows_ = other.rows_;
    A_ = std::make_unique<T[]>(nnz_);
    JA_ = std::make_unique<int[]>(nnz_);
    IA_ = std::make_unique<int[]>(rows_ + 1);
    std::copy(other.A_.get(), other.A_.get() + nnz_, A_.get());
    std::copy(other.JA_.get(), other.JA_.get() + nnz_, JA_.get());
    std::copy(other.IA_.get(), other.IA_.get() + (rows_ + 1), IA_.get());
    row_partition_ = other.row_partition_;
    pattern_ = other.pattern_;
    return *this;
}

// Move assignment
template<typename T>
SparseMatrix<T>& SparseMatrix<T>::operator=(SparseMatrix<T>&& other) noexcept
{
    columns_ = other.columns_;
    nnz_ = other.nnz_;
    rows_ = other.rows_;
    A_ = std::move(other.A_);
    JA_ = std::move(other.JA_);
    IA_ = std::move(other.IA_);
    row_partition_ = std::move(other.row_partition_);
    pattern_ = std::move(other.pattern_);
    return *this;
}

template<typename T>
SparseMatrix<T>::SparseMatrix(int rows, int columns,
                              const std::vector<triplet_type>& entries)
{
    rows_ = rows;
    columns_ = columns;
    nnz_ = (int)entries.size();
    A_ = std::make_unique<T[]>(nnz_);
    JA_ = std::make_unique<int[]>(nnz_);
    IA_ = std::make_unique<int[]>(rows_ + 1);
    for(int i = 0; i < nnz_; ++i)
    {
        JA_[i] = std::get<1>(entries[i]);
        A_[i] = std::get<2>(entries[i]);
    }
    IA_[0] = 0;
    int nz_i = 0;
    for(int r = 0; r < rows; ++r)
    {
        while(nz_i < nnz_ && std::get<0>(entries[nz_i]) == r)
        {
            ++nz_i;
        }
        IA_[r + 1] = nz_i;
    }
    row_partition_ = create_partition(MPI_COMM_SELF, rows);
    initialize_exchange_pattern(row_partition_);
}

// constructor MPI with row partition and function of non zero numbers per row
template<typename T>
SparseMatrix<T>::SparseMatrix(ContiguousParallelPartition row_partition, int global_columns, std::function<int(int)> nz_per_row)
{
    row_partition_ = row_partition;
    columns_ = global_columns;
    rows_ = row_partition_.local_size();
    nnz_ = 0;
    for(int row_index = 0; row_index < rows_; row_index++)
    {
        nnz_ += nz_per_row(row_index);
    }
    A_ = std::make_unique<T[]>(nnz_);
    JA_ = std::make_unique<int[]>(nnz_);
    IA_ = std::make_unique<int[]>(rows_ + 1);
}

// constructor MPI with row partition and entries
template<typename T>
SparseMatrix<T>::SparseMatrix(ContiguousParallelPartition row_partition, int global_columns, const std::vector<triplet_type>& entries)
{
    row_partition_ = row_partition;
    columns_ = global_columns;
    rows_ = row_partition_.local_size();
    nnz_ = (int)entries.size();
    A_ = std::make_unique<T[]>(nnz_);
    JA_ = std::make_unique<int[]>(nnz_);
    IA_ = std::make_unique<int[]>(rows_ + 1);
    for(int i = 0; i < nnz_; ++i)
    {
        JA_[i] = std::get<1>(entries[i]);
        A_[i] = std::get<2>(entries[i]);
    }
    IA_[0] = 0;
    int nz_i = 0;
    for(int r = 0; r < rows_; ++r)
    {
        while(nz_i < nnz_ && std::get<0>(entries[nz_i]) == r)
        {
            ++nz_i;
        }
        IA_[r + 1] = nz_i;
    }
}

// constructor MPI with local size and function of non zero numbers per row
template<typename T>
SparseMatrix<T>::SparseMatrix(MPI_Comm communicator, int local_rows, int global_columns, std::function<int(int)> nz_per_row)
{

    row_partition_ = create_partition(communicator, local_rows);
    columns_ = global_columns;
    rows_ = local_rows;
    nnz_ = 0;
    for(int row_index = 0; row_index < rows_; row_index++)
    {
        nnz_ += nz_per_row(row_index);
    }
    A_ = std::make_unique<T[]>(nnz_);
    JA_ = std::make_unique<int[]>(nnz_);
    IA_ = std::make_unique<int[]>(rows_ + 1);
}

// constructor MPI with local size and entries
template<typename T>
SparseMatrix<T>::SparseMatrix(MPI_Comm communicator, int local_rows, int global_columns, const std::vector<triplet_type>& entries)
{
    row_partition_ = create_partition(communicator, local_rows);
    columns_ = global_columns;
    rows_ = local_rows;
    nnz_ = (int)entries.size();
    A_ = std::make_unique<T[]>(nnz_);
    JA_ = std::make_unique<int[]>(nnz_);
    IA_ = std::make_unique<int[]>(rows_ + 1);
    for(int i = 0; i < nnz_; ++i)
    {
        JA_[i] = std::get<1>(entries[i]);
        A_[i] = std::get<2>(entries[i]);
    }
    IA_[0] = 0;
    int nz_i = 0;
    for(int r = 0; r < rows_; ++r)
    {
        while(nz_i < nnz_ && std::get<0>(entries[nz_i]) == r)
        {
            ++nz_i;
        }
        IA_[r + 1] = nz_i;
    }
}

// getter for the row partition
template<typename T>
const ContiguousParallelPartition& SparseMatrix<T>::row_partition() const
{
    return row_partition_;
}

// create exchange_pattern for multiprocess multiply
template<typename T>
void SparseMatrix<T>::initialize_exchange_pattern(const ContiguousParallelPartition& column_partition)
{
    pattern_ = create_exchange_pattern(*this, column_partition);
    // pattern_ = std::make_shared<ExchangePattern>(create_exchange_pattern(*this, column_partition));
}
template<typename T>
const ExchangePattern& SparseMatrix<T>::exchange_pattern() const
{
    // assert(pattern_ != nullptr);
    return pattern_;
}

template<typename T>
int SparseMatrix<T>::rows() const
{
    return rows_;
}
template<typename T>
int SparseMatrix<T>::columns() const
{
    return columns_;
}
template<typename T>
int SparseMatrix<T>::non_zero_size() const
{
    return nnz_;
}
template<typename T>
int SparseMatrix<T>::row_nz_size(int r) const
{
    assert(r >= 0 && r < rows_);
    return IA_[r + 1] - IA_[r];
}
template<typename T>
const int& SparseMatrix<T>::row_nz_index(int r, int nz_i) const
{
    assert(r >= 0 && r < rows_);
    assert(nz_i >= 0 && nz_i < row_nz_size(r));
    return JA_[IA_[r] + nz_i];
}

template<typename T>
int& SparseMatrix<T>::row_nz_index(int r, int nz_i)
{
    assert(r >= 0 && r < rows_);
    assert(nz_i >= 0 && nz_i < row_nz_size(r));
    return JA_[IA_[r] + nz_i];
}

template<typename T>
const T& SparseMatrix<T>::row_nz_entry(int r, int nz_i) const
{
    assert(r >= 0 && r < rows_);
    assert(nz_i >= 0 && nz_i < row_nz_size(r));
    return A_[IA_[r] + nz_i];
}
template<typename T>
T& SparseMatrix<T>::row_nz_entry(int r, int nz_i)
{
    assert(r >= 0 && r < rows_);
    assert(nz_i >= 0 && nz_i < row_nz_size(r));
    return A_[IA_[r] + nz_i];
}

// helper function that prints the matrix
template<typename T>
std::ostream& operator<<(std::ostream& os, const SparseMatrix<T>& M)
{
    const auto rows = M.rows();
    const auto cols = M.columns();
    for(int row_index = 0; row_index < rows; ++row_index)
    {
        const int _nnzi = M.row_nz_size(row_index);
        for(int col_index = 0; col_index < cols; ++col_index)
        {
            bool is_set = false;
            for(int sparse_index = 0; sparse_index < _nnzi; ++sparse_index)
            {
                const int temp = M.row_nz_index(row_index, sparse_index);
                if(temp == col_index)
                {
                    os << M.row_nz_entry(row_index, sparse_index) << " ";
                    is_set = true;
                }
            }
            if(!is_set)
            {
                os << 0 << " ";
            }
        }
        if(row_index != row_index - 1)
        {
            os << std::endl;
        }
    }
    return os;
}