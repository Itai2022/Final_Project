#pragma once

#include "solver.h"

template<typename T>
class RichardsonSolver : public IterativeSolver<T>
{
public:
    void solve(Vector<T>& x, const Vector<T>& b);
    void set_operator(const SparseMatrix<T>& A);
    void setup() { IterativeSolver<T>::setup(); }
    RichardsonSolver() = default;
    ~RichardsonSolver() = default;
};

template<typename T>
void RichardsonSolver<T>::solve(Vector<T>& x, const Vector<T>& b)
{
    const auto& partition = b.partition();
    this->_last_iterations = 1;
    Vector<T> residue(partition);
    Vector<T> x_1(partition);
    T start_residue_norm = 0.0;
    while(this->_last_iterations <= this->max_iterations().value())
    {
        multiply(residue, *(Solver<T>::_pA), x); // res1 = Ax_k
        subtract(residue, b, residue);           // res1 = b - Ax_k
        if(this->_last_iterations == 1)
        {
            start_residue_norm = norm(residue);
        }
        // compute P(b-Ax_k)
        assign(x_1, 0.0);
        if(this->_preconditioner != nullptr)
        {
            this->_preconditioner->apply(x_1, residue);
        }
        add(x, x, x_1);
        multiply(residue, *(Solver<T>::_pA), x);
        subtract(residue, b, residue);
        if((this->_relative_tolerance != std::nullopt && norm(residue) / start_residue_norm < this->relative_tolerance()) || norm(residue) < this->absolute_tolerance())
        {
            Solver<T>::last_stop_reason_ = StopReason::converged;
            this->_last_residual_norm = norm(residue);
            return;
        }
        ++this->_last_iterations;
        this->_last_residual_norm = norm(residue);
    }
}
template<typename T>
void RichardsonSolver<T>::set_operator(const SparseMatrix<T>& A)
{
    IterativeSolver<T>::set_operator(A);
}