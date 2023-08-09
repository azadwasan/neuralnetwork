#include "GradientDescent.h"
#include "Commons.h"
#include "CostFunctionMSE.h"

#include <numeric>
#include <stdexcept>

using namespace EasyNN;

void GradientDescent::evaluate(const std::vector<std::vector<double>>& featuresMatrix,
	const std::vector<double>& measurementsVector,
	const IRegression& hypothesis,
	double alpha, double stopThreshold,
	std::vector<double>& parameters) {

	if (featuresMatrix.empty() || measurementsVector.empty() || parameters.empty()) {
		throw std::invalid_argument("Feature matrix, measurements vectors and parameters vector size must be greater than zero.");
	}
	else if (featuresMatrix.size() != measurementsVector.size() || featuresMatrix[0].size() != parameters.size() - 1) {
		throw std::invalid_argument("Sample size of features matrix and measured values must be equal and parameters must be one element more than the total number of features.");
	}

	//auto parametersOld = parameters;
	std::vector<double> parametersNew(parameters.size());
	constexpr auto MAX_ITERATIONS = 1000;
	CostFunctionMSE costMSE{};
	
	auto oldCost = costMSE.evaluate(featuresMatrix, measurementsVector, parameters, hypothesis);
	auto i = 0;
	auto newCost = 0.0;
	std::vector<double> costHistory;
	std::vector<std::vector<double>> parameterHistory;
	for (; i < MAX_ITERATIONS; i++) {
		//parametersNew[0] = parametersOld[0] - alpha * 1 / measurementsVector.size() * computeCost(featuresMatrix, measurementsVector, parametersOld, hypothesis);
		//for (size_t index = 1; index < parametersOld.size(); index++) {
		//	parametersNew[index] = parametersOld[index] - alpha * 1 / measurementsVector.size() * computeCost(featuresMatrix, measurementsVector, parametersOld, hypothesis, index);
		//}

		auto costDerivativeZero = [&](const auto& featuresVector, const auto& parameters, auto measurement, size_t index) {return hypothesis.evaluate(featuresVector, parameters) - measurement; };
		auto costDerivative = [&](const auto& featuresVector, const auto& parameters, auto measurement, size_t index) {return (hypothesis.evaluate(featuresVector, parameters) - measurement) * featuresVector[index - 1]; };

		parametersNew[0] = parameters[0] - alpha * 1 / measurementsVector.size() *
			computeCost(featuresMatrix, measurementsVector, parameters, hypothesis, costDerivativeZero);
		for (size_t index = 1; index < parameters.size(); index++) {
			parametersNew[index] = parameters[index] - alpha * 1 / measurementsVector.size()
				* computeCost(featuresMatrix, measurementsVector, parameters, hypothesis, costDerivative, index);
		}

		parameters = parametersNew;

		newCost = costMSE.evaluate(featuresMatrix, measurementsVector, parametersNew, hypothesis);
		costHistory.push_back(newCost);
		parameterHistory.push_back(parameters);
		if (abs(oldCost - newCost) < stopThreshold) {
			break;
		}
		oldCost = newCost;
	}
	auto temp = costMSE.evaluate(featuresMatrix, measurementsVector, parametersNew, hypothesis);
}

double GradientDescent::computeCost(const std::vector<std::vector<double>>& featuresMatrix,
	const std::vector<double>& measurementsVector,
	const std::vector<double>& parameters,
	const IRegression& hypothesis, 
	std::function<double(const std::vector<double>&, const std::vector<double>&, double, size_t)> costDerivative, size_t index /*=0*/) {

	double differenceSum = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
		std::plus<>(),
		[&](const auto& featuresVector, auto measurement) {
			return costDerivative(featuresVector, parameters, measurement, index);
		}
	);
	return differenceSum;
}












double GradientDescent::computeCost(const std::vector<std::vector<double>>& featuresMatrix,
	const std::vector<double>& measurementsVector, const std::vector<double>& parameters, const IRegression& hypothesis) {
	double differenceSum = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
		std::plus<>(),
		[&](const std::vector<double>& featuresVector, double measurement) {return hypothesis.evaluate(featuresVector, parameters) - measurement; });
	return differenceSum;
}

double GradientDescent::computeCost(const std::vector<std::vector<double>>& featuresMatrix,
	const std::vector<double>& measurementsVector, const std::vector<double>& parameters, const IRegression& hypothesis, size_t index) {
	double differenceSum = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
		std::plus<>(),
		[&](const std::vector<double>& featuresVector, double measurement) {
			return (hypothesis.evaluate(featuresVector, parameters) - measurement) * featuresVector[index - 1]; }
	);
	return differenceSum;
}
