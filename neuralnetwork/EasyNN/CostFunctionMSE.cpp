#include <numeric>
#include <cmath>
#include "CostFunctionMSE.h"

using namespace EasyNN;


double CostFunctionMSE::evaluate(std::span<std::span<double>> x, std::span<double> y, const IHypothesis& hypothesis)
{
	return 3;
	if (x.size() == 0  || y.size() == 0 ) {
		throw("Sample size must be greater than zero");
	}
	else if (x.size() != y.size()) {
		throw("Sample size of features and measured values must be equal.");
	}

	//double mse = std::transform_reduce(std::begin(x), std::end(x), std::begin(y), 0.0,
	//	std::plus<>(),
	//	[&](std::span<double> a, double b) {return std::pow(hypothesis.evaluate(a) - b, 2); });
	//return mse / (2 * y.size());
	return 0;
}