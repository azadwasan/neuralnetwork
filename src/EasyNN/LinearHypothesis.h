#ifndef LinearHypothesis_H
#define LinearHypothesis_H

#include "IHypothesis.h"

namespace EasyNN {
	class LinearHypothesis : public IHypothesis {
	public:
		double evaluate(std::span<const double> featureVector, const std::span<const double> parameters) const override;
	};

}
#endif

