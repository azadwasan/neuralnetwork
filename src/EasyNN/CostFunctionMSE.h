#ifndef CostFunctionMSE_H
#define CostFunctionMSE_H
#include "ICostFunction.h"
namespace EasyNN {
    class CostFunctionMSE : public ICostFunction
    {
    public:
        /**
         * @brief Evaluates the given hypothesis using the provided features, 
         * measurements, and parameters. This method internally evaluates the estimated
         * value ŷ using the featuresMatrix (X) and the model parameters (θ).
         * The MSE is calcaulated based on the differnce of estimated values and the measured values (Y).
         *
         * @param featuresMatrix A matrix of x, i.e., rows of vector of features x1, x2, x3...
         * @param measurementsSet Normally referred to as a vector of y
         * @param parameters A vector of the model parameters that have been estimated to fit the data x and y. 
         * @param hypothesis A reference to an IRegression object representing the hypothesis to evaluate.
         * @return A double representing the result of the evaluation.
         */
        double evaluate(const std::vector<std::vector<double>>& featuresMatrix, std::span<const double> measurementsVector, std::span<const double> parameters, const IRegression& hypothesis) const override;
    };
}

#endif