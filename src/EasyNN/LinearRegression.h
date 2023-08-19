#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include "IRegression.h"

namespace EasyNN {
	class LinearRegression : public IRegression {
	public:
		double evaluate(std::span<const double> featureVector, const std::span<const double> parameters) const override;
	};
}
#endif

