/*
    Python bindings for EigenSolver
*/

#include "eigensolver.cpp"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

std::tuple<VectorXd, MatrixXd> eigendecomposition(MatrixXd M, int niter, double eps) {
    EigenSolver E(M, niter, eps);
    auto res = E.eigen();
    return std::tuple<VectorXd, MatrixXd>(
        get<0>(res),
        get<1>(res)
    );
}

std::tuple<VectorXd, MatrixXd, vector<int>> decompositionIterations(MatrixXd M, int niter, double
eps) {
    EigenSolver E(M, niter, eps);
    return E.eigen();
}

PYBIND11_MODULE(eigenvalues, m) {
    m.def("decomposition", &eigendecomposition);
    m.def("decompositionIterations", &decompositionIterations);
}