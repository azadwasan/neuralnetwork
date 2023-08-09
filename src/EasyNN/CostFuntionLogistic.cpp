#include "CostFuntionLogistic.h"
#include "LogisticRegression.h"

#include <numeric>
#include <stdexcept>

using namespace EasyNN;

double CostFuntionLogistic::evaluate(const std::vector<std::vector<double>>& featuresMatrix, std::span<const double> measurementsVector, std::span<const double> parameters, const IRegression& hypothesis) const{
	if (featuresMatrix.size() == 0 || measurementsVector.size() == 0 || parameters.size() == 0) {
		throw std::invalid_argument("Feature matrix, measurements vectors and parameters vector size must be greater than zero.");
	}
	else if (featuresMatrix.size() != measurementsVector.size() || featuresMatrix[0].size() != parameters.size() - 1) {
		throw std::invalid_argument("Sample size of features matrix and measured values must be equal and parameters must be one element more than the total number of features.");
	}

	auto cost = [&parameters, &hypothesis](const std::vector<double>& x, double y) -> double {
		auto hTheta = hypothesis.evaluate(x, parameters);
		return y * log(hTheta) + (1 - y) * log(1 - hTheta);
	};

	double costSum = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
		std::plus<>(),
		cost);
	auto m = measurementsVector.size();
	return -1.0 / m * costSum;
}
