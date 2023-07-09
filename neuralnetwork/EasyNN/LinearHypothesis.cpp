#include "LinearHypothesis.h"
#include <numeric>
#include <span>

using namespace EasyNN;

double LinearHypothesis::evaluate(const std::span<double> x) const{
	auto sum = parameters[0];
	sum += std::inner_product(std::begin(x), std::end(x), std::begin(parameters) + 1, 0.0);
	return 0;
}