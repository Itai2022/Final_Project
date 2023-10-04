#pragma once
#include "../linear_algebra/matrix.h"
#include "../linear_algebra/operations.h"
#include "../linear_algebra/vector.h"
#include <memory>
template<typename T>
class Preconditioner
{
public:
    virtual void set_operator(const SparseMatrix<T>& A);
    virtual void setup(){};
    virtual void apply(Vector<T>& x, const Vector<T>& b) = 0;
    Preconditioner() = default;
    virtual ~Preconditioner() = default;

protected:
    std::shared_ptr<SparseMatrix<T>> _pA;
};

template<typename T>
void Preconditioner<T>::set_operator(const SparseMatrix<T>& A)
{
    _pA = std::make_shared<SparseMatrix<T>>(A);
}