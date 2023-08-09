#ifndef ICOSTFUNCTION_H
#define ICOSTFUNCTION_H

#include <span>
#include "IRegression.h"

namespace EasyNN {
	class ICostFunction {
	public:
		virtual double evaluate(const std::vector<std::vector<double>>& featuresMatrix, std::span<const double> measurementsVector, std::span<const double> parameters, const IRegression& hypothesis) = 0;
	};
}

#endif