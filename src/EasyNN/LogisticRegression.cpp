#include "LogisticRegression.h"
#include "LinearRegression.h"

#include <stdexcept>
using namespace EasyNN;

double LogisticRegression::evaluate(std::span<const double> featureVector, const std::span<const double> parameters) const {
	if (featureVector.size() == 0 || parameters.size() == 0) {
		throw std::invalid_argument("Feature vector and parameters must have at least one element");
	}
	else if (featureVector.size() != parameters.size() - 1) {
		throw std::invalid_argument("Feature vector size must be parameters vector size - 1.");
	}

	auto z = LinearRegression{}.evaluate(featureVector, parameters);
	auto gz = 1 / (1 + exp(-1.0 * z));
	return gz;
}
