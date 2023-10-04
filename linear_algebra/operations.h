#pragma once

#include "../common/equals.h"
#include "exchangedata.h"
#include "matrix.h"
#include "vector.h"
#include <cmath>

template<typename T>
bool equals(const Vector<T>& lhs, const Vector<T>& rhs)
{
    int is_equal = 1;
    if(lhs.size() != rhs.size())
    {
        is_equal = 0;
    }
    else
    {
        int vec_size = lhs.size();
        for(int row_index = 0; row_index < vec_size; ++row_index)
        {
            if(!equals(lhs[row_index], rhs[row_index]))
            {
                is_equal = 0;
                break;
            }
        }
    }
    int result = 0;
    MPI_Allreduce(&is_equal, &result, 1, MPI_INT, MPI_MIN, lhs.partition().communicator());
    return (bool)result;
}

template<typename T>
bool equals(const SparseMatrix<T>& lhs, const SparseMatrix<T>& rhs)
{
    int is_equal = 1;
    if(lhs.rows() != rhs.rows() || lhs.columns() != rhs.columns() || lhs.non_zero_size() != rhs.non_zero_size())
    {
        is_equal = 0;
    }
    else
    {
        int rows = lhs.rows();
        for(int row_index = 0; row_index < rows; ++row_index)
        {
            if(lhs.row_nz_size(row_index) != rhs.row_nz_size(row_index))
            {
                is_equal = 0;
                break;
            }
            int nz_i = lhs.row_nz_size(row_index);
            for(int j = 0; j < nz_i; ++j)
            {
                if((!equals(lhs.row_nz_entry(row_index, j), rhs.row_nz_entry(row_index, j))) || lhs.row_nz_index(row_index, j) != rhs.row_nz_index(row_index, j))
                {
                    is_equal = 0;
                    break;
                }
            }
        }
    }
    int result = 0;
    MPI_Allreduce(&is_equal, &result, 1, MPI_INT, MPI_MIN, lhs.partition().communicator());
    return (bool)result;
}

template<typename T>
void assign(Vector<T>& lhs, const T& rhs)
{
    int vec_size = lhs.size();
    for(int row_index = 0; row_index < vec_size; ++row_index)
    {
        lhs[row_index] = rhs;
    }
}

template<typename T>
void add(Vector<T>& result, const Vector<T>& lhs,
         const Vector<T>& rhs)
{
    assert(lhs.size() == rhs.size());
    if(result.size() == 0)
    {
        result = lhs;
    }
    int vec_size = lhs.size();
    for(int row_index = 0; row_index < vec_size; ++row_index)
    {
        result[row_index] = lhs[row_index] + rhs[row_index];
    }
}

template<typename T>
void subtract(Vector<T>& result, const Vector<T>& lhs,
              const Vector<T>& rhs)
{
    assert(lhs.size() == rhs.size());
    if(result.size() == 0)
    {
        result = lhs;
    }
    int vec_size = lhs.size();
    for(int row_index = 0; row_index < vec_size; ++row_index)
    {
        result[row_index] = lhs[row_index] - rhs[row_index];
    }
}

template<typename T>
T dot_product(const Vector<T>& lhs, const Vector<T>& rhs)
{
    assert(lhs.size() == rhs.size());
    T scalar_prod = 0;
    int vec_size = lhs.size();
    for(int row_index = 0; row_index < vec_size; ++row_index)
    {
        scalar_prod += lhs[row_index] * rhs[row_index];
    }
    T sum = 0;
    MPI_Allreduce(&scalar_prod, &sum, 1, mpi_get_type<T>(), MPI_SUM, lhs.partition().communicator());
    return sum;
}

template<typename T>
T norm(const Vector<T>& r)
{
    return std::sqrt(dot_product(r, r));
}

template<typename T>
void multiply(Vector<T>& result, const Vector<T>& lhs,
              const T& rhs)
{
    int vec_size = lhs.size();
    for(int row_index = 0; row_index < vec_size; ++row_index)
    {
        result[row_index] = lhs[row_index] * rhs;
    }
}

template<typename T>
void multiply(Vector<T>& result, const SparseMatrix<T>& lhs,
              const Vector<T>& rhs)
{
    if(rhs.partition().communicator() == MPI_COMM_SELF)
    {
        assert(lhs.columns() == rhs.size());

        int lhs_row_size = lhs.rows();
        for(int row_index = 0; row_index < lhs_row_size; ++row_index)
        {
            int nz_i = lhs.row_nz_size(row_index);
            T res = 0;
            for(int j = 0; j < nz_i; ++j)
            {
                res += lhs.row_nz_entry(row_index, j) * rhs[lhs.row_nz_index(row_index, j)];
            }
            result[row_index] = res;
        }
    }
    else
    {
        assert(lhs.rows() == rhs.size());
        assert(lhs.columns() == rhs.partition().global_size());

        ExchangeData<T> data = exchange_vector_data(lhs.exchange_pattern(), rhs);

        int lhs_row_size = lhs.rows();
        for(int row_index = 0; row_index < lhs_row_size; ++row_index)
        {
            int nz_i = lhs.row_nz_size(row_index);
            T res = 0;
            for(int j = 0; j < nz_i; ++j)
            {
                auto global_index = lhs.row_nz_index(row_index, j);
                auto owner_rank = rhs.partition().owner_process(global_index);
                if(rhs.partition().is_owned_by_local_process(global_index))
                {
                    auto local_index = rhs.partition().to_local_index(global_index);
                    res += lhs.row_nz_entry(row_index, j) * rhs[local_index];
                }
                else
                {
                    res += lhs.row_nz_entry(row_index, j) * data.get(owner_rank, global_index);
                }
            }
            result[row_index] = res;
        }
    }
}
