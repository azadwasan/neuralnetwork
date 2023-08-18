#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include "IRegression.h"
#include "ICostFunction.h"

#include<vector>
#include<functional>

namespace EasyNN {
	/**
	 * @class GradientDescent
	 * @brief An optimization algorithm for finding the parameters of a regression model.
	 *
	 * This class implements the gradient descent algorithm for finding the optimal values of the
	 * parameters of a regression model. The algorithm iteratively updates the values of the parameters
	 * to minimize the cost function of the regression model.
	 */
class GradientDescent
{
public:
	/**
	 * @brief Finds the optimal values for the parameters of a regression model.
	 *
	 * This function uses the gradient descent algorithm to find the optimal values for the
	 * parameters of a regression model. The algorithm iteratively updates the values of the
	 * parameters to minimize the cost function of the regression model.
	 *
	 * @param featuresMatrix A matrix of features, where each row represents a sample and each column represents a feature.
	 * @param measurementsVector A vector of measurements, where each element represents the measured value for the corresponding sample in the featuresMatrix.
	 * @param costFunction A reference to an ICostFunction object representing the cost function of the regression model.
	 * @param alpha The learning rate of the gradient descent algorithm.
	 * @param stopThreshold The stopping threshold for the gradient descent algorithm. The algorithm stops when the change in cost between two consecutive iterations is below this threshold.
	 * @param[out] parameters A vector of parameters representing the initial values for the parameters of the regression model. On output, this vector contains the optimal values for the parameters found by the gradient descent algorithm.
	 */
	void evaluate(const std::vector<std::vector<double>>& featuresMatrix,
		const std::vector<double>& measurementsVector,
		std::vector<double>& parameters,
		const ICostFunction& costFunction,
		double alpha, double stopThreshold);
private:
	/**
	 * @brief Computes the cost of a regression model given its current parameters.
	 *
	 * This function computes the cost of a regression model given its current parameters. The cost
	 * is computed using a user-defined cost derivative function that specifies how to compute
	 * the derivative of the cost function with respect to each parameter.
	 *
	 * @param featuresMatrix A matrix of features, where each row represents a sample and each column represents a feature.
	 * @param measurementsVector A vector of measurements, where each element represents the measured value for the corresponding sample in the featuresMatrix.
	 * @param parameters A vector of parameters representing the current values for the parameters of the regression model.
	 * @param hypothesis A reference to an IRegression object representing the hypothesis function of the regression model.
	 * @param costDerivative A user-defined function that specifies how to compute
	 *                       the derivative of the cost function with respect to each parameter.
	 * @param index The index of the parameter with respect to which to compute the derivative of the cost function. Defaults to 0.
	 *
	 * @return The computed cost of the regression model given its current parameters.
	 */
	double computeCost(const std::vector<std::vector<double>>& featuresMatrix,
		const std::vector<double>& measurementsVector,
		const std::vector<double>& parameters,
		const IRegression& hypothesis, 
		std::function<double(const std::vector<double>&, const std::vector<double>&, double, size_t)> costDeivative, size_t index = 0);
};
}

#endif