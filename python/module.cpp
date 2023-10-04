#include "../discretization/error.h"
#include "../discretization/heat.h"
#include "../discretization/heat_coeff.h"
#include "../discretization/poisson.h"
#include "../grid/grid.h"
#include "../grid/io.h"
#include "../linear_algebra/operations.h"
#include "../solvers/cg.h"
#include "../solvers/gauss_seidel_iteration.h"
#include "../solvers/jacobi_iteration.h"
#include "../solvers/richardson.h"
#include <filesystem>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <string>

#include "mpi_comm_wrapper.h"

void my_mpi_init()
{
    MPI_Init(nullptr, nullptr);
}

void my_mpi_finalize()
{
    MPI_Finalize();
}
/*
void write_to_vtk_1(const std::string& string, const RegularGrid& grid)
{
    std::filesystem::path path(string);
    write_to_vtk(path, grid);
}*/

template<typename T>
void write_to_vtk_2(const std::string& string, const GridFunction<T>& grid_function, const std::string& name)
{
    std::filesystem::path path(string);
    write_to_vtk(path, grid_function, name);
}

template<typename T>
void set_vector(Vector<T>& self, int i, T value)
{
    self.operator[](i) = value;
}

void set_point(Point& self, int i, scalar_t value)
{
    self.operator[](i) = value;
}

void set_multiindex(MultiIndex& self, int i, int value)
{
    self.operator[](i) = value;
}

std::string printpat(const ContiguousParallelPartition& pat)
{
    std::cout << pat << std::endl;
    return "";
}

PYBIND11_MODULE(pmsc, mod)
{
    mod.doc() = "PMSC python module";

    namespace py = pybind11;

    py::class_<MpiCommWrapper>(mod, "MpiComm")
        .def("rank", &MpiCommWrapper::rank)
        .def("size", &MpiCommWrapper::size);

    mod.def("mpi_comm_world", []() { return MpiCommWrapper(MPI_COMM_WORLD); });
    mod.def("mpi_comm_self", []() { return MpiCommWrapper(MPI_COMM_SELF); });

    // MPI
    mod.def("MPI_Init", &my_mpi_init);
    mod.def("MPI_Finalize", &my_mpi_finalize);

    // parallelcontiguouspartition
    py::class_<ContiguousParallelPartition>(mod, "ContiguousParallelPartition")
        .def(py::init<>())
        .def(py::init<MpiCommWrapper, std::vector<int>>())
        .def("communicator", [](const ContiguousParallelPartition& self) { return MpiCommWrapper(self.communicator()); })
        .def("local_size", py::overload_cast<>(&ContiguousParallelPartition::local_size, py::const_))
        .def("local_size", py::overload_cast<int>(&ContiguousParallelPartition::local_size, py::const_))
        .def("global_size", &ContiguousParallelPartition::global_size)
        .def("owner_process", &ContiguousParallelPartition::owner_process)
        .def("is_owned_by_local_process", &ContiguousParallelPartition::is_owned_by_local_process)
        .def("is_owned_by_process", &ContiguousParallelPartition::is_owned_by_process)
        .def("to_local_index", &ContiguousParallelPartition::to_local_index)
        .def("to_global_index", py::overload_cast<int>(&ContiguousParallelPartition::to_global_index, py::const_))
        .def("to_global_index", py::overload_cast<int, int>(&ContiguousParallelPartition::to_global_index, py::const_))
        .def("__repr__", &printpat);

    // create_partition
    mod.def(
        "create_partition", [](MpiCommWrapper MpiComm, int local_size) { return create_partition(MpiComm, local_size); }, py::arg("communicator"), py::arg("local_size"));
    mod.def(
        "create_uniform_partition", [](MpiCommWrapper MpiComm, int global_size) { return create_uniform_partition(MpiComm, global_size); }, py::arg("communicator"), py::arg("global_size"));

    // vector
    py::class_<Vector<scalar_t>>(mod, "Vector")
        .def(py::init<>())
        .def(py::init<const Vector<scalar_t>&>(), py::arg("other"))
        .def(py::init<int>(), py::arg("size"))
        .def(py::init<ContiguousParallelPartition>(), py::arg("partition"))
        .def(py::init<MpiCommWrapper, int>(), py::arg("communicator"), py::arg("local_size"))
        .def(py::init<MpiCommWrapper, std::initializer_list<scalar_t>>(), py::arg("communicator"), py::arg("init"))
        .def("partition", &Vector<scalar_t>::partition)
        .def("assign", py::overload_cast<const Vector<scalar_t>&>(&Vector<scalar_t>::operator=), py::arg("other"), py::return_value_policy::reference_internal)
        .def("__setitem__", &set_vector<scalar_t>, py::arg("i"), py::arg("value"))
        .def("__getitem__", py::overload_cast<int>(&Vector<scalar_t>::operator[], py::const_), py::arg("i"))
        .def("size", &Vector<scalar_t>::size);

    // SparseMatrix
    py::class_<SparseMatrix<scalar_t>>(mod, "SparseMatrix")
        .def(py::init<>())
        .def(py::init<const SparseMatrix<scalar_t>&>(), py::arg("other"))
        .def(py::init<int, int, const std::vector<SparseMatrix<scalar_t>::triplet_type>&>(), py::arg("rows"), py::arg("columns"), py::arg("entries"))
        .def(py::init<ContiguousParallelPartition, int, std::function<int(int)>>(), py::arg("row_partition"), py::arg("global_columns"), py::arg("nz_per_row"))
        .def(py::init<ContiguousParallelPartition, int, const std::vector<SparseMatrix<scalar_t>::triplet_type>&>(), py::arg("row_partition"), py::arg("global_columns"), py::arg("entries"))
        .def(py::init<MpiCommWrapper, int, int, std::function<int(int)>>(), py::arg("communicator"), py::arg("local_rows"), py::arg("global_columns"), py::arg("nz_per_row"))
        .def(py::init<MpiCommWrapper, int, int, const std::vector<SparseMatrix<scalar_t>::triplet_type>&>(), py::arg("communicator"), py::arg("local_rows"), py::arg("global_columns"), py::arg("entries"))
        .def("row_partition", &SparseMatrix<scalar_t>::row_partition)
        .def("initialize_exchange_pattern", &SparseMatrix<scalar_t>::initialize_exchange_pattern, py::arg("column_partition"))
        .def("exchange_pattern", &SparseMatrix<scalar_t>::exchange_pattern)
        .def("rows", &SparseMatrix<scalar_t>::rows)
        .def("columns", &SparseMatrix<scalar_t>::columns)
        .def("non_zero_size", &SparseMatrix<scalar_t>::non_zero_size)
        .def("row_nz_size", &SparseMatrix<scalar_t>::row_nz_size, py::arg("r"))
        .def("row_nz_index", py::overload_cast<int, int>(&SparseMatrix<scalar_t>::row_nz_index), py::arg("r"), py::arg("nz_i"))
        .def("row_nz_index", py::overload_cast<int, int>(&SparseMatrix<scalar_t>::row_nz_index, py::const_), py::arg("r"), py::arg("nz_i"))
        .def("row_nz_entry", py::overload_cast<int, int>(&SparseMatrix<scalar_t>::row_nz_entry), py::arg("r"), py::arg("nz_i"))
        .def("row_nz_entry", py::overload_cast<int, int>(&SparseMatrix<scalar_t>::row_nz_entry, py::const_), py::arg("r"), py::arg("nz_i"));

    // assemble_poisson_matrix
    mod.def("assemble_poisson_matrix", &assemble_poisson_matrix<scalar_t>, py::arg("grid"), py::arg("rhs_function"), py::arg("boundary_function"));
    mod.def("assemble_heat_matrix", py::overload_cast<const RegularGrid&, const GridFunction<scalar_t>&, const scalar_t, const scalar_t, const std::function<scalar_t(const Point&, const scalar_t)>&, const std::function<scalar_t(const Point&, const scalar_t)>&>(&assemble_heat_matrix<scalar_t>), py::arg("grid"), py::arg("previous_temperature"), py::arg("t"), py::arg("delta_t"), py::arg("rhs_function"), py::arg("boundary_function"));
    mod.def("assemble_heat_matrix", py::overload_cast<const RegularGrid&, const GridFunction<scalar_t>&, const scalar_t, const scalar_t, const std::function<scalar_t(const Point&, const scalar_t)>&, const std::function<scalar_t(const Point&, const scalar_t)>&, const std::function<scalar_t(const Point&)>&, const std::function<scalar_t(const Point&)>&, const std::function<scalar_t(const Point&)>&>(&assemble_heat_matrix<scalar_t>), py::arg("grid"), py::arg("previous_temperature"), py::arg("t"), py::arg("delta_t"), py::arg("rhs_function"), py::arg("boundary_function"), py::arg("alpha"), py::arg("rho"), py::arg("c"));
    mod.def("compute_l_infinity_error", &compute_l_infinity_error<scalar_t>, py::arg("grid"), py::arg("computed_solution"), py::arg("analytical_solution"));

    // operations
    mod.def("equals", py::overload_cast<const Vector<scalar_t>&, const Vector<scalar_t>&>(&equals<scalar_t>), py::arg("lhs"), py::arg("rhs"));
    // mod.def("equals", py::overload_cast<const SparseMatrix<scalar_t>&, const SparseMatrix<scalar_t>&>(&equals<scalar_t>), py::arg("lhs"), py::arg("rhs"));
    //    equals in Point.cpp
    //  mod.def("equals", py::overload_cast<const Point&, const Point&>(&equals), py::arg("lhs"), py::arg("rhs"));
    mod.def("assign", &assign<scalar_t>, py::arg("lhs"), py::arg("rhs"));
    mod.def("add", &add<scalar_t>, py::arg("result"), py::arg("lhs"), py::arg("rhs"));
    mod.def("subtract", &subtract<scalar_t>, py::arg("result"), py::arg("lhs"), py::arg("rhs"));
    mod.def("dot_product", &dot_product<scalar_t>, py::arg("lhs"), py::arg("rhs"));
    mod.def("norm", &norm<scalar_t>, py::arg("r"));
    mod.def("multiply", py::overload_cast<Vector<scalar_t>&, const Vector<scalar_t>&, const scalar_t&>(&multiply<scalar_t>), py::arg("result"), py::arg("lhs"), py::arg("rhs"));
    mod.def("multiply", py::overload_cast<Vector<scalar_t>&, const SparseMatrix<scalar_t>&, const Vector<scalar_t>&>(&multiply<scalar_t>), py::arg("result"), py::arg("lhs"), py::arg("rhs"));

    // solvers
    py::enum_<StopReason>(mod, "StopReason")
        .value("unknown", StopReason::unknown)
        .value("converged", StopReason::converged)
        .value("undefined", StopReason::undefined);

    py::class_<Solver<scalar_t>>(mod, "Solver")
        .def("set_operator", &Solver<scalar_t>::set_operator, py::arg("const SparseMatrix<T>& A"), "sets the operator of a linear equation system.")
        .def("setup", &Solver<scalar_t>::setup, "do some preprations.")
        .def("solve", &Solver<scalar_t>::solve, py::arg("Vector<T>& x"), py::arg("const Vector<T>& b"), "Solve the eqation system.")
        .def("last_stop_reason", &Solver<scalar_t>::last_stop_reason);

    py::class_<IterativeSolver<scalar_t>, Solver<scalar_t>>(mod, "IterativeSolver")
        .def("set_preconditioner", &IterativeSolver<scalar_t>::set_preconditioner, py::arg("std::shared_ptr<Preconditioner<scalar_t>> preconditioner"))
        .def("max_iterations", py::overload_cast<std::optional<int>>(&IterativeSolver<scalar_t>::max_iterations), py::arg("std::optional<int> value"))
        .def("max_iterations", py::overload_cast<>(&IterativeSolver<scalar_t>::max_iterations, py::const_))
        .def("absolute_tolerance", py::overload_cast<scalar_t>(&IterativeSolver<scalar_t>::absolute_tolerance), py::arg("scalar_t value"))
        .def("absolute_tolerance", py::overload_cast<>(&IterativeSolver<scalar_t>::absolute_tolerance, py::const_))
        .def("last_residual_norm", &IterativeSolver<scalar_t>::last_residual_norm) // Ohne py::overload_cast<> gib's Fehler
        .def("relative_tolerance", py::overload_cast<std::optional<scalar_t>>(&IterativeSolver<scalar_t>::relative_tolerance), py::arg("std::optional<scalar_t> value"))
        .def("relative_tolerance", py::overload_cast<>(&IterativeSolver<scalar_t>::relative_tolerance, py::const_))
        .def("last_iterations", py::overload_cast<>(&IterativeSolver<scalar_t>::last_iterations, py::const_));

    py::class_<Preconditioner<scalar_t>, std::shared_ptr<Preconditioner<scalar_t>>>(mod, "Preconditioner")
        .def("set_operator", &Preconditioner<scalar_t>::set_operator)
        .def("setup", &Preconditioner<scalar_t>::setup)
        .def("apply", &Preconditioner<scalar_t>::apply);

    py::class_<RichardsonSolver<scalar_t>, IterativeSolver<scalar_t>>(mod, "RichardsonSolver")
        .def(py::init<>())
        .def("solve", &RichardsonSolver<scalar_t>::solve)
        .def("set_operator", &RichardsonSolver<scalar_t>::set_operator)
        .def("setup", &RichardsonSolver<scalar_t>::setup);

    py::class_<CgSolver<scalar_t>, IterativeSolver<scalar_t>>(mod, "CgSolver")
        .def(py::init<>())
        .def("solve", &CgSolver<scalar_t>::solve)
        .def("set_operator", &CgSolver<scalar_t>::set_operator)
        .def("setup", &CgSolver<scalar_t>::setup);

    py::class_<JacobiIteration<scalar_t>, Preconditioner<scalar_t>, std::shared_ptr<JacobiIteration<scalar_t>>>(mod, "JacobiIteration")
        .def(py::init<>())
        .def("apply", &JacobiIteration<scalar_t>::apply);

    py::class_<GaussSeidelIteration<scalar_t>, Preconditioner<scalar_t>, std::shared_ptr<GaussSeidelIteration<scalar_t>>>(mod, "GaussSeidelIteration")
        .def(py::init<>())
        .def("apply", &GaussSeidelIteration<scalar_t>::apply);

    // point
    py::class_<Point>(mod, "Point")
        .def(py::init<>())
        .def(py::init<scalar_t>(), py::arg("value"))
        .def(py::init<scalar_t, scalar_t>(), py::arg("x"), py::arg("y"))
        .def(py::init<scalar_t, scalar_t, scalar_t>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def("size", &Point::size)
        .def("__setitem__", &set_point, py::arg("i"), py::arg("value"))
        .def("__getitem__", py::overload_cast<int>(&Point::operator[], py::const_), py::arg("i"));

    // multiindex

    py::class_<MultiIndex>(mod, "MultiIndex")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("value"))
        .def(py::init<int, int>(), py::arg("x"), py::arg("y"))
        .def(py::init<int, int, int>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def("size", &MultiIndex::size)
        .def("__eq__", &MultiIndex::operator==, py::arg("other"))
        .def("__ne__", &MultiIndex::operator!=, py::arg("other"))
        .def("__setitem__", &set_multiindex, py::arg("i"), py::arg("value"))
        .def("__getitem__", py::overload_cast<int>(&MultiIndex::operator[], py::const_));

    // neighborsuccession
    py::enum_<NeighborSuccession>(mod, "NeighborSuccession")
        .value("predecessor", NeighborSuccession::predecessor)
        .value("successor", NeighborSuccession::successor);

    // regulargrid

    py::class_<RegularGrid>(mod, "RegularGrid")
        .def(py::init<>())
        .def(py::init<const RegularGrid&>(), py::arg("other"))
        .def(py::init<Point, Point, MultiIndex>(), py::arg("min_corner"), py::arg("max_corner"), py::arg("node_count_per_dimension"))
        .def(py::init<MpiCommWrapper, Point, Point, MultiIndex>(), py::arg("communicator"), py::arg("min_corner"), py::arg("max_corner"), py::arg("global_node_count_per_dimension"))
        .def("node_count_per_dimension", py::overload_cast<>(&RegularGrid::node_count_per_dimension, py::const_))
        .def("node_count_per_dimension", py::overload_cast<int>(&RegularGrid::node_count_per_dimension, py::const_))
        .def("number_of_nodes", &RegularGrid::number_of_nodes)
        .def("number_of_inner_nodes", &RegularGrid::number_of_inner_nodes)
        .def("number_of_boundary_nodes", &RegularGrid::number_of_boundary_nodes)
        .def("number_of_neighbors", &RegularGrid::number_of_neighbors, py::arg("node_index"))
        .def("neighbors_of", &RegularGrid::neighbors_of, py::arg("node_index"), py::arg("neighbors"))
        .def("is_boundary_node", &RegularGrid::is_boundary_node, py::arg("node_index"))
        .def("node_coordinates", &RegularGrid::node_coordinates, py::arg("node_index"))
        .def("node_neighbor_distance", &RegularGrid::node_neighbor_distance, py::arg("node_index"), py::arg("neighbor_direction"), py::arg("neighbor_succession"))
        .def("node_distance", &RegularGrid::node_distance);

    // gridfunction
    py::class_<GridFunction<scalar_t>>(mod, "GridFunction")
        .def(py::init<>())
        .def(py::init<const RegularGrid&, scalar_t>(), py::arg("grid"), py::arg("value"))
        .def(py::init<const RegularGrid&, const std::function<scalar_t(const Point&)>&>(), py::arg("grid"), py::arg("function"))
        //.def(py::init<const RegularGrid&, const std::function<scalar_t(const Point&, const scalar_t)>&>(), py::arg("grid"), py::arg("heat_function"))
        .def(py::init<const RegularGrid&, const Vector<scalar_t>&>(), py::arg("grid"), py::arg("values"))
        .def("grid", &GridFunction<scalar_t>::grid)
        .def("value", py::overload_cast<int>(&GridFunction<scalar_t>::value, py::const_), py::arg("node_index"));
    //.def("value", py::overload_cast<int, const scalar_t>(&GridFunction<scalar_t>::value, py::const_), py::arg("node_index"), py::arg("t"));

    // io
    // mod.def("write_to_vtk", &write_to_vtk_1, py::arg("file_path"), py::arg("grid"));
    mod.def("write_to_vtk", &write_to_vtk_2<scalar_t>, py::arg("file_path"), py::arg("grid_function"), py::arg("name"));

    // TODO: add other exports here
}
