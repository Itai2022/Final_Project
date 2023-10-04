import sys
sys.path.append("/Users/winter/Desktop/praktikum/pmsc-exercise-6-whiletrue/build")
from pmsc import *
from math import exp, sin, pi, cos



def BTCS(grid, previous_temperature, t_start, t_end, rhs_function, boundary_function, M):
    delta_t = t_end / M
    t_start += delta_t
    for i in range(1, M + 1):
        output = assemble_heat_matrix(grid, previous_temperature,t_start, delta_t, rhs_function, boundary_function)
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
        previous_temperature = GridFunction(grid, u)
        t_start += delta_t
        write_to_vtk("vtk/heat_timestep_{}".format(i), previous_temperature, "heat")    

def BTCS_coeff(grid, previous_temperature, t_start, t_end, rhs_function, boundary_function, M, alpha, rho, c):
    delta_t = t_end / M
    t_start += delta_t
    for i in range(1, M + 1):
        output = assemble_heat_matrix(grid, previous_temperature,t_start, delta_t, rhs_function, boundary_function, alpha, rho, c)
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
        previous_temperature = GridFunction(grid, u)
        t_start += delta_t
        write_to_vtk("vtk/heat_timestep_{}".format(i), previous_temperature, "heat")   


if __name__ == "__main__":
    #exercise = int(input("Enter the number of exercise:"))
    MPI_Init()

    exercise = 5
    if exercise == 5:
        grid = RegularGrid(mpi_comm_world(),Point(0.0, 0.0), Point(1.0, 1.0), MultiIndex(30, 30)) #domain [0,1]^2
        boundary_function = lambda point, t: 1 + point[0] ** 2 + 2 * point[1] ** 2 + 2 * t
        #boundary_function_poisson = lambda point, t: point[0] +  point[1] 
        rhs_function = lambda point,t : -4.0 
        init_previous_temperature_function = lambda point: 1 + point[0] ** 2 + 2 * point[1] ** 2
        t_start = 0.0
        t_end = 2.0
        init_previous_temperature = GridFunction(grid, init_previous_temperature_function)
        BTCS(grid, init_previous_temperature, t_start, t_end, rhs_function, boundary_function,10)

    elif exercise == 6:
        grid = RegularGrid(mpi_comm_world(),Point(-2.0, -2.0), Point(2.0, 2.0), MultiIndex(30, 30)) #domain [-2,2]^2
        boundary_function = lambda point, t: 0.0
        rhs_function = lambda point,t : 0.0
        a = 1
        b = 2
        c = 2
        init_previous_temperature_function = lambda point: c * exp(-a*point[0]**2-b*point[1]**2)
        t_start = 0.0
        t_end = 1.0
        init_previous_temperature = GridFunction(grid, init_previous_temperature_function)
        BTCS(grid, init_previous_temperature, t_start, t_end, rhs_function, boundary_function,10)
        #consider domains for example [-2,0]^2 or [-2,2]*[-2,0]

    elif exercise == 9:
        grid = RegularGrid(mpi_comm_world(),Point(-1.0, -1.0), Point(1.0, 1.0), MultiIndex(100, 100)) #domain [-1,1]^2
        boundary_function = lambda point, t: 0.0
        rhs_function = lambda point,t : 0.0
        
        def alpha(x):
            radius_sq = x[0]**2 + x[1]**2
            return 0.000024 if 0.3**2 <= radius_sq and radius_sq <= 0.4**2 else 1
        
        '''
        def alpha(x):
            return 1/ (x[0]**2 + x[1]**2)
        '''
        rho = lambda point: 1
        c = lambda point: 1
        init_previous_temperature_function = lambda point: 10
        t_start = 0.0
        t_end = 2.0
        init_previous_temperature = GridFunction(grid, init_previous_temperature_function)
        BTCS_coeff(grid, init_previous_temperature, t_start, t_end, rhs_function, boundary_function,10,alpha, rho, c)
    MPI_Finalize()