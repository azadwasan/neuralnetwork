//#include "CostGradientDescent.h"
//#include <numeric>
//#include <cmath>
//using namespace EasyNN;
//
//double CostGradientDescent::evaluate(const std::vector<std::vector<double>>& featuresMatrix, std::span<double> measurementsVector, std::span<double> parameters, const IHypothesis& hypothesis) {
//	if (featuresMatrix.size() == 0 || measurementsVector.size() == 0) {
//		throw("Sample size must be greater than zero");
//	}
//	else if (featuresMatrix.size() != measurementsVector.size()) {
//		throw("Sample size of features and measured values must be equal.");
//	}
//	std::vector<double> parametersNew(parameters.size());
//	parametersNew[0] = 
//	double mse = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
//		std::plus<>(),
//		[&](const std::vector<double>& featuresVector, double measurement) {return (hypothesis.evaluate(featuresVector, parameters) - measurement); });
//	return mse / (2 * measurementsVector.size());
//
//}
