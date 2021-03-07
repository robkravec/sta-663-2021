

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace Eigen;

VectorXd logistic_cpp(VectorXd x){
    return 1 / (1 + exp((-x).array()));
}

PYBIND11_MODULE(hw_cpp, m) {
    m.def("logistic_cpp", &logistic_cpp);
}