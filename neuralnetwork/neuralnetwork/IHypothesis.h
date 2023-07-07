#ifndef IHypothesis_H
#define IHypothesis_H

#include <vector>
#include <ranges>

namespace EasyNN {
	class IHypothesis {
	public:
		virtual double evaluate(const std::span<double> x) const = 0;
	};
}


#endif