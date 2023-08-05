#ifndef ALGORITHMS_H
#define ALGORITHMS_H
#include <vector>

namespace EasyNNPyPlugin {
	class Algorithms
	{
	public:
		static std::vector<double> RunGD(const std::vector<std::vector<double>>& X, const std::vector<double> y, size_t paramCount);
	};
}

#endif