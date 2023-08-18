#ifndef COSTFUNCTIONLOGISTIC_H
#define COSTFUNCTIONLOGISTIC_H

#include "ICostFunction.h"

namespace EasyNN {
    /**
     * @class CostFunctionLogistic
     * @brief A cost function implementation for logistic regression.
     *
     * The CostFunctionLogistic class inherits from the ICostFunction interface and provides
     * methods to evaluate the cost associated with the predictions made by a logistic regression model.
     * This cost function is used to optimize the model parameters based on the discrepancies between
     * predicted probabilities and actual measurements.
     * 
     * @param hypothesis A reference to an IRegression object representing the hypothesis function of the regression model.
     * @param lambda Regulartion rate.
     */
    class CostFuntionLogistic : public ICostFunction
    {
    public:
        CostFuntionLogistic(std::unique_ptr<IRegression> hypothesis, double lambda = 0.0) : ICostFunction(std::move(hypothesis), lambda) {}
     /**
     * @brief Evaluate the cost associated with logistic regression predictions.
     *
     * Given a features matrix, measurements vector, model parameters, and a logistic regression hypothesis,
     * this method calculates and returns the cost associated with the predictions made by the hypothesis.
     * The cost is computed based on the discrepancies between predicted probabilities and actual measurements.
     *
     * @param featuresMatrix A matrix containing the feature vectors of the training examples.
     * @param measurementsVector A span containing the actual measurements (ground truth).
     * @param parameters A span containing the model parameters.
     * @return The computed cost associated with the logistic regression predictions.
     */
        double evaluate(const std::vector<std::vector<double>>& featuresMatrix, std::span<const double> measurementsVector, std::span<const double> parameters) const override;

    };
}

#endif