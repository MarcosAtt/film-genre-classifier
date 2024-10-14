#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>

using std::cout;
using std::endl;
using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::VectorXd;
using std::tuple;
using std::get;
using std::vector;

using eigenvalueHistory = vector<double>;
using errorHistory      = vector<double>;

struct eigenPair {
    double eigenvalue;
    VectorXd eigenvector;
    int iterations;
};

class EigenSolver {
    public:
    explicit EigenSolver(MatrixXd aMatrix, int niter = 10000, double eps = 1e-12)
    : A(aMatrix), niter(niter), eps(eps) {
        cols = A.cols();
    }

    void deflation(double eigenvalue, const Ref<const VectorXd>& eigenvector);

    eigenPair power_iteration();

    tuple<eigenvalueHistory, errorHistory, VectorXd> power_iteration_evolution();

    tuple<VectorXd, MatrixXd, vector<int>>   eigen();

    private:
    double infinityNorm(VectorXd v);
    double rayleigh_quotient(VectorXd eigenvector);
    bool convergenceCriteria(VectorXd previousVector, VectorXd newVector);
    double methodError(VectorXd v_i, double lambda_i);

    Eigen::MatrixXd A;
    int niter;
    int cols;
    double eps;
};

VectorXd positiveRandomVector(int cols) {
    VectorXd res = VectorXd::Random(cols);
    for(int i = 0; i < cols; i++) {
        res[i] = abs(res[i]);
    }
    return res;
}

double EigenSolver::rayleigh_quotient(VectorXd eigenvector) {
    auto dividend     = (eigenvector.transpose() * A * eigenvector);
    auto divisor      = (eigenvector.transpose() * eigenvector);
    double eigenvalue = dividend(0) / divisor(0);
    return eigenvalue;
}

double EigenSolver::infinityNorm(VectorXd v) {
    return v.lpNorm<Eigen::Infinity>();
}

bool EigenSolver::convergenceCriteria(VectorXd a, VectorXd b) {
    return infinityNorm(a - b) < eps;
}

eigenPair EigenSolver::power_iteration() {
    // Quiero que quede un vector positivo para testear facil.
    auto previousEigenvector = positiveRandomVector(cols);
    VectorXd newEigenvector;

    int iterations = 0;

    while(iterations < niter) {
        newEigenvector = (A * previousEigenvector).normalized();
        if(convergenceCriteria(previousEigenvector, newEigenvector))
            break;
        previousEigenvector = newEigenvector;
        iterations++;
    }

    struct eigenPair result {
        rayleigh_quotient(newEigenvector), newEigenvector, iterations
    };
    return result;
}

void EigenSolver::deflation(double eigenvalue, const Ref<const VectorXd>& eigenvector) {
    A = A - (eigenvalue * (eigenvector * (eigenvector.transpose())));
}

tuple<VectorXd, MatrixXd, vector<int>> EigenSolver::eigen() {
    auto eigenvalues  = VectorXd(cols);
    auto eigenvectors = MatrixXd(cols, cols);
    vector<int> iterations(cols);

    for(int i = 0; i < cols; i++) {
        auto [eigenvalue, eigenvector, it_k] = power_iteration();
        eigenvalues[i]                 = eigenvalue;
        eigenvectors.col(i)            = eigenvector;
        iterations[i]                  = it_k;
        deflation(eigenvalue, eigenvector);
    }

    return tuple<VectorXd, MatrixXd, vector<int>>(eigenvalues, eigenvectors, iterations);
}

double EigenSolver::methodError(VectorXd v_i, double lambda_i) {
    return (A * v_i - lambda_i * v_i).lpNorm<2>();
}

tuple<eigenvalueHistory, errorHistory, VectorXd> EigenSolver::power_iteration_evolution() {
    //
    eigenvalueHistory eigenH;
    errorHistory errorH;
    //
    auto previousEigenvector = positiveRandomVector(cols);
    VectorXd newEigenvector;

    int iteraciones = 0;

    while(iteraciones < niter) {
        newEigenvector = (A * previousEigenvector).normalized();
        //
        double eigenValue = rayleigh_quotient(newEigenvector);
        double error      = methodError(newEigenvector, eigenValue);

        eigenH.push_back(eigenValue);
        errorH.push_back(error);
        //
        if(convergenceCriteria(previousEigenvector, newEigenvector))
            break;
        previousEigenvector = newEigenvector;
        iteraciones++;
    }

    return tuple<eigenvalueHistory, errorHistory, VectorXd>(eigenH, errorH, newEigenvector);
}
