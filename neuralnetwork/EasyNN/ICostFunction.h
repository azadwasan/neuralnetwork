#ifndef ICostFunction_H
#define ICostFunction_H

#include <span>
#include "IHypothesis.h"

namespace EasyNN {
	class ICostFunction {
	public:
		virtual double evaluate(const std::vector<std::vector<double>>& featuresMatrix, std::span<double> measurementsVector, std::span<double> parameters, const IHypothesis& hypothesis) = 0;
	};
}

#endif