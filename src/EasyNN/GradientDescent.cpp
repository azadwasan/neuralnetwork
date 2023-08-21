#include "GradientDescent.h"
#include "Commons.h"
#include "CostFunctionMSE.h"
#include "CostFuntionLogistic.h"
#include <numeric>
#include <stdexcept>

using namespace EasyNN;

void GradientDescent::evaluate(const std::vector<std::vector<double>>& featuresMatrix,
	const std::vector<double>& measurementsVector,
	std::vector<double>& parameters,
	const ICostFunction& costFunction,
	double alpha, double stopThreshold, const size_t maxIterations /*= 3000*/) {

	if (featuresMatrix.empty() || measurementsVector.empty() || parameters.empty()) {
		throw std::invalid_argument("Feature matrix, measurements vectors and parameters vector size must be greater than zero.");
	}
	else if (featuresMatrix.size() != measurementsVector.size() || featuresMatrix[0].size() != parameters.size() - 1) {
		throw std::invalid_argument("Sample size of features matrix and measured values must be equal and parameters must be one element more than the total number of features.");
	}

	std::vector<double> parametersNew(parameters.size());

	auto oldCost = costFunction.evaluate(featuresMatrix, measurementsVector, parameters);
	auto i = 0;
	auto newCost = 0.0;
	//std::vector<double> costHistory;
	//std::vector<std::vector<double>> parameterHistory;
	double m = measurementsVector.size();
	double regFactor = (1 - alpha * costFunction.getLambda() / m);
	auto costDerivativeZero = [&](const auto& featuresVector, const auto& parameters, auto measurement, size_t index) {return costFunction.getHypothesis().evaluate(featuresVector, parameters) - measurement; };
	auto costDerivative = [&](const auto& featuresVector, const auto& parameters, auto measurement, size_t index) {return (costFunction.getHypothesis().evaluate(featuresVector, parameters) - measurement) * featuresVector[index - 1]; };

	for (; i < maxIterations; i++) {

		parametersNew[0] = parameters[0] - alpha * 1 / m *
			computeCost(featuresMatrix, measurementsVector, parameters, costDerivativeZero);
		for (size_t index = 1; index < parameters.size(); index++) {
			parametersNew[index] = parameters[index] * regFactor - alpha * 1 / m
				* computeCost(featuresMatrix, measurementsVector, parameters, costDerivative, index);
		}

		parameters = parametersNew;

		newCost = costFunction.evaluate(featuresMatrix, measurementsVector, parametersNew);
		//costHistory.push_back(newCost);
		//parameterHistory.push_back(parameters);
		if (abs(oldCost - newCost) < stopThreshold) {
			break;
		}
		oldCost = newCost;
	}
}

double GradientDescent::computeCost(const std::vector<std::vector<double>>& featuresMatrix,
	const std::vector<double>& measurementsVector,
	const std::vector<double>& parameters,
	std::function<double(const std::vector<double>&, const std::vector<double>&, double, size_t)> costDerivative, size_t index /*=0*/) {

	double differenceSum = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
		std::plus<>(),
		[&](const auto& featuresVector, auto measurement) {
			return costDerivative(featuresVector, parameters, measurement, index);
		}
	);
	return differenceSum;
}
