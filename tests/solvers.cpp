#include <iostream>
#include <iomanip>

#include <mpi.h>

#include <gtest/gtest.h>

#include "../linear_algebra/matrix.h"
#include "../linear_algebra/vector.h"
#include "../linear_algebra/operations.h"

#include "../solvers/richardson.h"
#include "../solvers/cg.h"
#include "../solvers/gauss_seidel_iteration.h"
#include "../solvers/jacobi_iteration.h"

template<typename T>
std::tuple<SparseMatrix<T>, Vector<T>, Vector<T>> build_equation_system()
{
    using triplet = typename SparseMatrix<T>::triplet_type;

    const int N = 9;
    const T h = 1.0 / (N - 1);
    const T h2 = h*h;

    const SparseMatrix<T> A(N - 2, N - 2, {
        triplet{ 0, 0, 2.0/h2 }, triplet{ 0, 1,-1.0/h2 },
        triplet{ 1, 1, 2.0/h2 }, triplet{ 1, 0,-1.0/h2 }, triplet{ 1, 2,-1.0/h2 },
        triplet{ 2, 2, 2.0/h2 }, triplet{ 2, 1,-1.0/h2 }, triplet{ 2, 3,-1.0/h2 },
        triplet{ 3, 3, 2.0/h2 }, triplet{ 3, 2,-1.0/h2 }, triplet{ 3, 4,-1.0/h2 },
        triplet{ 4, 4, 2.0/h2 }, triplet{ 4, 3,-1.0/h2 }, triplet{ 4, 5,-1.0/h2 },
        triplet{ 5, 5, 2.0/h2 }, triplet{ 5, 4,-1.0/h2 }, triplet{ 5, 6,-1.0/h2 },
        triplet{ 6, 6, 2.0/h2 }, triplet{ 6, 5,-1.0/h2 } });
    const Vector<T> b = {8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0};
    const Vector<T> expected_result = {28./64, 48./64, 60./64, 1.0, 60./64, 48./64, 28./64};

    return std::make_tuple(std::move(A), std::move(b), std::move(expected_result));
}

TEST(Solvers, richardson_jacobi)
{
    const auto [A, b, expected_result] = build_equation_system<double>();
    Vector<double> x(A.columns());
    assign(x, 0.0);
    RichardsonSolver<double> solver;
    solver.set_preconditioner(std::make_unique<JacobiIteration<double>>());
    solver.relative_tolerance(1e-15);
    solver.max_iterations(1000);
    solver.absolute_tolerance(0.0);
    solver.set_operator(A);
    solver.setup();
    solver.solve(x, b);
    EXPECT_EQ(solver.last_stop_reason(), StopReason::converged);
    EXPECT_LE(solver.last_residual_norm(), 1e-10);
    EXPECT_EQ(solver.last_iterations(), 437);
    EXPECT_TRUE(equals(x, expected_result));
}

TEST(Solvers, richardson_gauss_seidel)
{
    const auto [A, b, expected_result] = build_equation_system<double>();
    Vector<double> x(A.columns());
    assign(x, 0.0);
    RichardsonSolver<double> solver;
    solver.set_preconditioner(std::make_unique<GaussSeidelIteration<double>>());
    solver.relative_tolerance(1e-15);
    solver.max_iterations(1000);
    solver.absolute_tolerance(0.0);
    solver.set_operator(A);
    solver.setup();
    solver.solve(x, b);
    EXPECT_EQ(solver.last_stop_reason(), StopReason::converged);
    EXPECT_LE(solver.last_residual_norm(), 1e-10);
    EXPECT_EQ(solver.last_iterations(), 220);
    EXPECT_TRUE(equals(x, expected_result));
}

TEST(Solvers, cg_jacobi)
{
    const auto [A, b, expected_result] = build_equation_system<double>();
    Vector<double> x(A.columns());
    assign(x, 0.0);
    CgSolver<double> solver;
    solver.set_preconditioner(std::make_unique<JacobiIteration<double>>());
    solver.relative_tolerance(1e-15);
    solver.max_iterations(1000);
    solver.absolute_tolerance(0.0);
    solver.set_operator(A);
    solver.setup();
    solver.solve(x, b);
    EXPECT_EQ(solver.last_stop_reason(), StopReason::converged);
    EXPECT_LE(solver.last_residual_norm(), 1e-10);
    EXPECT_EQ(solver.last_iterations(), 4);
    EXPECT_TRUE(equals(x, expected_result));
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const auto test_result_status = RUN_ALL_TESTS();
    MPI_Finalize();
    return test_result_status;
}
