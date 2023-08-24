#ifndef NEURON_H
#define NEURON_H

#include "ICostFunction.h"

#include <memory>

// I initially thought to have a neuron class to hold the value of a single neuron. 
// But, as it turned out it would very negatively affect the data structures as data points would be
// distributed to single elements. Hence, I am not following this approach at least not for now.
// Instead, I am going to have layer and network class and layer class will take care of 
// neuron abstraction.

namespace EasyNN {
	class Neuron final
	{
	public:
		explicit Neuron(std::shared_ptr<ICostFunction> actFunc, double val = 0.0)  : value{ val }, costFunction{ actFunc }  {
		}
		void setValue(double val) noexcept { value = val; }
		inline const double getValue() const noexcept { (*this)(); }
		inline const double operator()() const noexcept  { return value; }
	private:
		std::shared_ptr<ICostFunction>	costFunction;
		double							value;
	public:
	};
}

#endif