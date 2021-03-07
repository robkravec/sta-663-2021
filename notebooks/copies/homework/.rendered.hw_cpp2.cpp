

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace Eigen;

// Define logistic function
VectorXd logistic_cpp(VectorXd x){
    return 1 / (1 + exp((-x).array()));
}

// Define gradient descent function
VectorXd gd_cpp(MatrixXd X, VectorXd y, VectorXd beta, double alpha, int niter) {
    
    // Define variables
    int n = X.rows();
    int p = X.cols();
    MatrixXd Xt = X.transpose();
    VectorXd y_pred = VectorXd::Zero(n);
    VectorXd epsilon = VectorXd::Zero(n);
    VectorXd grad = VectorXd::Zero(p);
    
    // Loop through specified number of iterations
    for (int i = 0; i < niter; i++) {
        y_pred = logistic_cpp(X * beta);
        epsilon = y - y_pred;
        grad = Xt * epsilon / n;
        beta += alpha * grad;
    }
    return beta;
}

PYBIND11_MODULE(hw_cpp2, m) {
    m.def("logistic_cpp", &logistic_cpp);
    m.def("gd_cpp", &gd_cpp);
}