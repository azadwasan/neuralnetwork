#ifndef PLOTS_H
#define PLOTS_H
#include <vector>
#include <optional>

namespace EasyNNPyPlugin {
	class Plots
	{
	public:
		static void CompareHypothesis(const std::vector<std::vector<double>>& X, const std::vector<double> y, const std::vector<double>& theta1, const std::vector<double>& theta2);
		static void PlotClassificationData(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<double>& theta, const std::optional<std::vector<double>> theta2 = std::nullopt);
	};
}

#endif