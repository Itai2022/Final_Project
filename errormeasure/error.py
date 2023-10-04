import sys
sys.path.append("/Users/winter/Desktop/pmsc-exercise-6-whiletrue/build")
from pmsc import *
from math import exp, sin, pi
import matplotlib.pyplot as plt

number_of_nodes_per_dimension = []
error = []

def calculate_error(boundary_function_poisson, rhs_function_poisson):
    for m in range(1, 7):
        gridpoisson = grid = RegularGrid(mpi_comm_world(),Point(0.0), Point(1.0), MultiIndex(2 ** m, 2**m))
        output = assemble_poisson_matrix(gridpoisson, rhs_function_poisson, boundary_function_poisson)
        A = output[0]
        b = output[1]
        u = Vector(b.partition())
        assign(u, 0.0)
        solver = CgSolver()
        solver.set_preconditioner(JacobiIteration())
        solver.relative_tolerance(1e-15)
        solver.max_iterations(10000)
        solver.absolute_tolerance(1e-15)
        solver.set_operator(A)
        solver.setup()
        solver.solve(u, b)
        GF_poisson = GridFunction(gridpoisson, u)
        error.append(compute_l_infinity_error(grid, GF_poisson, boundary_function_poisson))
        number_of_nodes_per_dimension.append(2**m)
    

if __name__ == "__main__":
    MPI_Init()

    boundary_function_poisson = lambda point: sin(pi * point[0]) * sin(pi * point[1]) #* sin(pi * point[2])
    rhs_function_poisson = lambda point : 2 * pi ** 2 * sin(pi * point[0]) * sin(pi * point[1]) #m* sin(pi * point[2])
    calculate_error(boundary_function_poisson, rhs_function_poisson)
    
    if mpi_comm_world().rank() == 0:
        plt.title('Error against number of nodes in 2D', fontsize=24)
        plt.xlabel('Number_of_nodes_per_dimension', fontsize=14)
        plt.ylabel('Error', fontsize=14)
        plt.semilogy(number_of_nodes_per_dimension, error)
        plt.savefig('error2d.png')
        plt.show()
        plt.close()

    MPI_Finalize()
