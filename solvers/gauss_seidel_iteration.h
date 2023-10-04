#pragma once
#include "preconditioner.h"

template<typename T>
class GaussSeidelIteration : public Preconditioner<T>
{
public:
    virtual void apply(Vector<T>& x, const Vector<T>& b);
    GaussSeidelIteration() = default;
};

template<typename T>
void GaussSeidelIteration<T>::apply(Vector<T>& x, const Vector<T>& b)
{
    int rows = this->_pA->rows();
    for(int row_index = 0; row_index < rows; ++row_index)
    {
        T sum = 0.0;
        int nz_row = this->_pA->row_nz_size(row_index);
        int diag_index = 0;
        for(int nz_index = 0; nz_index < nz_row; ++nz_index)
        {
            int index = this->_pA->row_nz_index(row_index, nz_index);
            if(index == row_index)
            {
                diag_index = nz_index;
                continue;
            }
            else
            {
                sum += this->_pA->row_nz_entry(row_index, nz_index) * x[index];
            }
        }
        x[row_index] = (b[row_index] - sum) / this->_pA->row_nz_entry(row_index, diag_index);
    }
}
