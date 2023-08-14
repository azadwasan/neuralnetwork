#ifndef ALGORITHMS_H
#define ALGORITHMS_H
#include <vector>

namespace EasyNNPyPlugin {
	class Algorithms
	{
	public:
		static std::vector<double> RunGD(const std::vector<std::vector<double>>& X, const std::vector<double>& y, size_t paramCount);
		static std::vector<double> FitLogisticRegression(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
		static std::vector<double> FitLogisticRegressionTF(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
	};
}

#endif