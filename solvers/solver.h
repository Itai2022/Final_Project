#ifndef SOLVER_H
#define SOLVER_H
#include "../linear_algebra/matrix.h"
#include "../linear_algebra/operations.h"
#include "../linear_algebra/vector.h"
#include "preconditioner.h"
#include <cassert>
#include <limits.h>
#include <optional>

enum class StopReason
{
    unknown,
    converged,
    undefined
};

template<typename T>
class Solver
{
public:
    Solver() = default;
    virtual void set_operator(const SparseMatrix<T>& A);
    virtual void setup(){};
    virtual void solve(Vector<T>& x, const Vector<T>& b) = 0;
    StopReason last_stop_reason() const;
    virtual ~Solver() = default;

protected:
    // Ax=b
    std::shared_ptr<SparseMatrix<T>> _pA{};
    StopReason last_stop_reason_{StopReason::unknown};
};

template<typename T>
void Solver<T>::set_operator(const SparseMatrix<T>& A)
{
    _pA = std::make_shared<SparseMatrix<T>>(A);
}
template<typename T>
StopReason Solver<T>::last_stop_reason() const
{
    return last_stop_reason_;
} // Base Class

template<typename T>
class IterativeSolver : public Solver<T>
{
public:
    IterativeSolver() = default;
    virtual void set_operator(const SparseMatrix<T>& A);
    virtual void setup();
    virtual void solve(Vector<T>& x, const Vector<T>& b) = 0;
    void set_preconditioner(std::shared_ptr<Preconditioner<T>> preconditioner);
    void max_iterations(std::optional<int> value);
    std::optional<int> max_iterations() const;
    void absolute_tolerance(T value);
    T absolute_tolerance() const;
    void relative_tolerance(std::optional<T> value);
    std::optional<T> relative_tolerance() const;
    int last_iterations() const;
    T last_residual_norm() const;
    virtual ~IterativeSolver() = default;

protected:
    std::optional<int> _max_iteration{std::nullopt};
    T _absolute_tolerance{0.0};
    std::optional<T> _relative_tolerance{std::nullopt};
    int _last_iterations{};
    T _last_residual_norm{};
    std::shared_ptr<Preconditioner<T>> _preconditioner{nullptr};
}; // Derived Class

template<typename T>
void IterativeSolver<T>::set_operator(const SparseMatrix<T>& A)
{
    Solver<T>::set_operator(A);
    if(_preconditioner != nullptr)
    {
        _preconditioner->set_operator(A);
    }
}

template<typename T>
void IterativeSolver<T>::setup()
{
    Solver<T>::setup();
}

template<typename T>
void IterativeSolver<T>::set_preconditioner(std::shared_ptr<Preconditioner<T>> preconditioner)
{
    //_preconditioner = std::move(preconditioner);
    _preconditioner = preconditioner;
}

template<typename T>
void IterativeSolver<T>::max_iterations(std::optional<int> value)
{
    if(value == std::nullopt)
    {
        value = {INT_MAX};
    }
    else
    {
        _max_iteration = value;
    }
}

template<typename T>
std::optional<int> IterativeSolver<T>::max_iterations() const
{
    if(_max_iteration)
    {
        return _max_iteration;
    }
    else
    {
        return {INT_MAX};
    }
}

template<typename T>
void IterativeSolver<T>::absolute_tolerance(T value)
{
    _absolute_tolerance = value;
}

template<typename T>
T IterativeSolver<T>::absolute_tolerance() const
{
    return _absolute_tolerance;
}

template<typename T>
void IterativeSolver<T>::relative_tolerance(std::optional<T> value)
{
    if(value == std::nullopt)
    {
        _relative_tolerance = {0.0};
    }
    else
    {
        _relative_tolerance = value;
    }
}

template<typename T>
std::optional<T> IterativeSolver<T>::relative_tolerance() const
{
    if(_relative_tolerance)
    {
        return _relative_tolerance;
    }
    else
    {
        return {0.0};
    }
}

template<typename T>
int IterativeSolver<T>::last_iterations() const
{
    return _last_iterations;
}
template<typename T>
T IterativeSolver<T>::last_residual_norm() const
{
    return _last_residual_norm;
}
#endif // solver.h