#include "LinearRegression.h"
#include "Commons.h"

#include <numeric>
#include <stdexcept>

using namespace EasyNN;

double LinearRegression::evaluate(std::span<const double> featureVector, const std::span<const double> parameters, std::unique_ptr<IRegression> underlyingHypothesis /*= nullptr*/) const{
	if (featureVector.size() == 0 || parameters.size() == 0) {
		throw std::invalid_argument("Feature vector and parameters must have at least one element");
	}
	else if (featureVector.size() != parameters.size() - 1) {
		throw std::invalid_argument("Feature vector size must be parameters vector size - 1.");
	}

	auto sum = parameters[0];
	sum += std::inner_product(std::begin(featureVector), std::end(featureVector), std::begin(parameters) + 1, 0.0);
	return sum;
}