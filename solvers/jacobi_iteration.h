#pragma once
#include "preconditioner.h"
#include <iostream>
#include <memory>

template<typename T>
class JacobiIteration : public Preconditioner<T>
{
public:
    virtual void apply(Vector<T>& x, const Vector<T>& b);
    JacobiIteration() = default;
    ~JacobiIteration() = default;
};

template<typename T>
void JacobiIteration<T>::apply(Vector<T>& x, const Vector<T>& b)
{
    int rows = this->_pA->rows();
    const auto& partition = this->_pA->row_partition();
    for(int row_index = 0; row_index < rows; ++row_index)
    {
        int nz_row = this->_pA->row_nz_size(row_index);
        for(int nz_index = 0; nz_index < nz_row; ++nz_index)
        {
            int global_row_index = partition.to_global_index(row_index);
            if(this->_pA->row_nz_index(row_index, nz_index) == global_row_index)
            {
                x[row_index] = b[row_index] / this->_pA->row_nz_entry(row_index, nz_index);
            }
        }
    }
}
