#ifndef ICostFunction_H
#define ICostFunction_H

#include <span>
#include "IHypothesis.h"

namespace EasyNN {
	class ICostFunction {
	public:
		virtual double evaluate(std::span<std::span<double>> x, std::span<double> y, const IHypothesis& hypothesis) = 0;
	};
}

#endif