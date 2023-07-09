#ifndef IHypothesis_H
#define IHypothesis_H

#include <vector>
#include <ranges>

namespace EasyNN {
	class IHypothesis {
	public:
		virtual double evaluate(std::span<const double> x, const std::span<double> parameters) const = 0;
	};
}


#endif