#ifndef IHypothesis_H
#define IHypothesis_H

#include <vector>
#include <ranges>

namespace EasyNN {
	class IHypothesis {
	public:
		virtual double evaluate(std::span<double> x) = 0;
	};
}


#endif