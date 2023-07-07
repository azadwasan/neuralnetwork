#ifndef LinearHypothesis_H
#define LinearHypothesis_H

#include "IHypothesis.h"

namespace EasyNN {
	class LinearHypothesis : public IHypothesis {
	public:
		double evaluate(const std::span<double> x) const override;
	private:
		std::vector<double> parameters;
	};

}
#endif

