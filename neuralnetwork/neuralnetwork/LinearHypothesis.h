#ifndef LinearHypothesis_H
#define LinearHypothesis_H

#include "IHypothesis.h"

namespace EasyNN {
	class LinearHypothesis : public IHypothesis {
	public:
		double evaluate(std::span<double> x) override;
	private:
		std::vector<double> parameters;
	};

}
#endif

