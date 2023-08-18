#ifndef CostFunctionMSE_H
#define CostFunctionMSE_H
#include "ICostFunction.h"
namespace EasyNN {
    class CostFunctionMSE : public ICostFunction
    {
    public:
        /**
        * @param hypothesis A reference to an IRegression object representing the hypothesis function of the regression model.
        * @param lambda Regulartion rate.
        */
        CostFunctionMSE(std::unique_ptr<IRegression> hypothesis, double lambda = 0.0) : ICostFunction(std::move(hypothesis), lambda) {}
        /**
         * @brief Evaluates the given hypothesis using the provided features, 
         * measurements, and parameters. This method internally evaluates the estimated
         * value ŷ using the featuresMatrix (X) and the model parameters (θ).
         * The MSE is calcaulated based on the differnce of estimated values and the measured values (Y).
         *
         * @param featuresMatrix A matrix of x, i.e., rows of vector of features x1, x2, x3...
         * @param measurementsSet Normally referred to as a vector of y
         * @param parameters A vector of the model parameters that have been estimated to fit the data x and y. 
         * @return A double representing the result of the evaluation.
         */
        double evaluate(const std::vector<std::vector<double>>& featuresMatrix, std::span<const double> measurementsVector, std::span<const double> parameters) const override;
    };
}

#endif