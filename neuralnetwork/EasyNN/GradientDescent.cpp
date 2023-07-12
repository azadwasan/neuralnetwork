#include "GradientDescent.h"
#include "Commons.h"
#include "CostFunctionMSE.h"

#include <numeric>
#include <stdexcept>

using namespace EasyNN;

std::vector<double> GradientDescent::evaluate(const std::vector<std::vector<double>>& featuresMatrix,
	const std::vector<double>& measurementsVector,
	const std::vector<double>& parameters,
	const IHypothesis& hypothesis,
	double alpha, double stopThreshold) {

	if (featuresMatrix.size() == 0 || measurementsVector.size() == 0 || parameters.size()==0) {
		throw std::invalid_argument("Feature matrix, measurements vectors and parameters vector size must be greater than zero.");
	}
	else if (featuresMatrix.size() != measurementsVector.size() || featuresMatrix[0].size() != parameters.size() - 1) {
		throw std::invalid_argument("Sample size of features matrix and measured values must be equal and parameters must be one element more than the total number of features.");
	}


	auto parametersOld = parameters;
	std::vector<double> parametersNew(parameters.size());
	size_t MAX_ITERATIONS = 400;
	CostFunctionMSE costMSE{};
	
	double oldCost = costMSE.evaluate(featuresMatrix, measurementsVector, parametersOld, hypothesis);
	
	for (size_t i = 0; i < MAX_ITERATIONS; i++) {
		parametersNew[0] = parametersOld[0] - alpha * 1 / measurementsVector.size() * computeCost(featuresMatrix, measurementsVector, parametersOld, hypothesis);
		for (size_t index = 1; index < parametersOld.size(); index++) {
			parametersNew[index] = parametersOld[index] - alpha * 1 / measurementsVector.size() * computeCost(featuresMatrix, measurementsVector, parametersOld, hypothesis, index);
		}

		parametersOld = parametersNew;

		double newCost = costMSE.evaluate(featuresMatrix, measurementsVector, parametersNew, hypothesis);
		if (abs(oldCost - newCost) < stopThreshold) {
			break;
		}
	}
	return parametersOld;
}

double GradientDescent::computeCost(const std::vector<std::vector<double>>& featuresMatrix,
	const std::vector<double>& measurementsVector, 	const std::vector<double>& parameters, 	const IHypothesis& hypothesis) {
	double differenceSum = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
		std::plus<>(),
		[&](const std::vector<double>& featuresVector, double measurement) {return hypothesis.evaluate(featuresVector, parameters) - measurement; });
	return differenceSum;
}

double GradientDescent::computeCost(const std::vector<std::vector<double>>& featuresMatrix,
	const std::vector<double>& measurementsVector,	const std::vector<double>& parameters,	const IHypothesis& hypothesis, size_t index) {
	double differenceSum = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
		std::plus<>(),
		[&](const std::vector<double>& featuresVector, double measurement) {
			return (hypothesis.evaluate(featuresVector, parameters) - measurement) * featuresVector[index-1]; }
		);
	return differenceSum;
}
