#ifndef IREGRESSION_H
#define IREGRESSION_H

#include <vector>
#include <ranges>

namespace EasyNN {
	class IRegression {
	public:
		virtual double evaluate(std::span<const double> featureVector, const std::span<const double> parameters) const = 0;
	};
}


#endif