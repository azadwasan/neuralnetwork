#include "CostFunctionMSE.h"
#include "Commons.h"

#include <numeric>
#include <cmath>
#include <stdexcept>

using namespace EasyNN;


double CostFunctionMSE::evaluate(const std::vector<std::vector<double>>& featuresMatrix, std::span<const double> measurementsVector, std::span<const double> parameters) const
{
	if (featuresMatrix.size() == 0  || measurementsVector.size() == 0 || parameters.size() == 0) {
		throw std::invalid_argument("Feature matrix, measurements vectors and parameters vector size must be greater than zero.");
	}
	else if (featuresMatrix.size() != measurementsVector.size() || featuresMatrix[0].size() != parameters.size() - 1) {
		throw std::invalid_argument("Sample size of features matrix and measured values must be equal and parameters must be one element more than the total number of features.");
	}
	// NOTE: Read all the following comments in one go, to determine what is being computed.

						//For each value of		feature matrix (x)							and the measurement vector (y)
	double mse = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
		// compute a sum of
		std::plus<>(),
		//										square of the difference between the hypothesis estimate for a single sample (x1, x2) and the measured value (y)
		[&](const std::vector<double>& featuresVector, double measurement) {return std::pow(hypothesis->evaluate(featuresVector, parameters) - measurement, 2); });
	
	auto m = measurementsVector.size();
	double regFactor = lambda / (2 * m) * std::accumulate(parameters.begin() + 1, parameters.end(), 0, [](auto acc, auto x) { return acc + x * x; });

	return mse / (2 * m) + regFactor;
}