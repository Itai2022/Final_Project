#pragma once

#include "solver.h"

template<typename T>
class CgSolver : public IterativeSolver<T>
{
public:
    CgSolver() = default;
    ~CgSolver() = default;
    virtual void solve(Vector<T>& x, const Vector<T>& b);
    void set_operator(const SparseMatrix<T>& A);
    void setup() { IterativeSolver<T>::setup(); }
};

template<typename T>
void CgSolver<T>::set_operator(const SparseMatrix<T>& A)
{
    IterativeSolver<T>::set_operator(A);
}

template<typename T>
void CgSolver<T>::solve(Vector<T>& x, const Vector<T>& b)
{
    const auto& partition = x.partition();
    Vector<T> start_residue(partition);
    multiply(start_residue, *(Solver<T>::_pA), x);
    subtract(start_residue, b, start_residue); // start_residue = b - Ax_0
    if(norm(start_residue) == 0)
    {
        return; // Initial guess is the correct answer.
    }
    Vector<T> z_0(partition);
    assign(z_0, 0.0);
    if(this->_preconditioner != nullptr)
    {
        this->_preconditioner->apply(z_0, start_residue);
    } // z_0 = M^-1 start_residue
    this->_last_iterations = 1;
    Vector<T> conjugated_residue(z_0); // conjugated_residue = z_0
    Vector<T> next_residue(partition);
    Vector<T> temp(conjugated_residue);
    T start_residue_norm = norm(start_residue);
    while(this->_last_iterations <= this->max_iterations().value())
    {
        multiply(temp, *(Solver<T>::_pA), conjugated_residue);                             //  temp = Aconjugated_residue
        T alpha = dot_product(start_residue, z_0) / dot_product(temp, conjugated_residue); // alpha = (r_j,z_j)/(Ap_j,p_j)
        multiply(conjugated_residue, conjugated_residue, alpha);                           // p_j = alpha p_j;
        add(x, x, conjugated_residue);                                                     // x_j+1 = x_j + aplha p_j
        multiply(temp, temp, alpha);                                                       // Ap_j = alpha Ap_j
        subtract(next_residue, start_residue, temp);                                       // r_j+1 = r_j - alpha Ap_j
        assign(temp, 0.0);
        if(this->_preconditioner != nullptr)
        {
            this->_preconditioner->apply(temp, next_residue);
        }                                                                           // z_j+1 = M^-1 r_j+1
        T beta = dot_product(next_residue, temp) / dot_product(start_residue, z_0); // beta = (r_j+1, z_j+1)/(r_j, z_j)
        multiply(conjugated_residue, conjugated_residue, beta / alpha);             // p_j+1 = beta * p_j
        add(conjugated_residue, temp, conjugated_residue);                          // p_j+1 = z_j+1 + beta p_j;
        if((this->_relative_tolerance != std::nullopt && norm(next_residue) / start_residue_norm < this->relative_tolerance()) || norm(next_residue) < this->absolute_tolerance())
        {
            Solver<T>::last_stop_reason_ = StopReason::converged;
            this->_last_residual_norm = norm(next_residue);
            return;
        }
        start_residue = next_residue;
        z_0 = temp;
        ++this->_last_iterations;
        this->_last_residual_norm = norm(start_residue);
    }
}
