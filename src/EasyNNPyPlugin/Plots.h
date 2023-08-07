#ifndef PLOTS_H
#define PLOTS_H
#include <vector>

namespace EasyNNPyPlugin {
	class Plots
	{
	public:
		static void CompareHypothesis(const std::vector<std::vector<double>>& X, const std::vector<double> y, const std::vector<double>& theta1, const std::vector<double>& theta2);
	};
}

#endif