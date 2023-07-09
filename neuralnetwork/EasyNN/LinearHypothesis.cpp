#include "LinearHypothesis.h"
#include <numeric>
#include <span>
#include <stdexcept>

using namespace EasyNN;

double LinearHypothesis::evaluate(std::span<const double> x, const std::span<double> parameters) const{
	if (parameters.size() < 1 ) {
		throw std::invalid_argument("Parameters must have at least one element");
	}
	else if (x.size() != parameters.size() - 1) {
		throw std::invalid_argument("Feature vector size must be parameters vector size - 1.");
	}

	auto sum = parameters[0];
	sum += std::inner_product(std::begin(x), std::end(x), std::begin(parameters) + 1, 0.0);
	return sum;
}
