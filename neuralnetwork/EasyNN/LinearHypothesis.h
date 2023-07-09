#ifndef LinearHypothesis_H
#define LinearHypothesis_H

#include "IHypothesis.h"

namespace EasyNN {
	class LinearHypothesis : public IHypothesis {
	public:
		double evaluate(std::span<const double> x, const std::span<double> parameters) const override;
	};

}
#endif

