#ifndef PLOTS_H
#define PLOTS_H
#include <vector>
#include <optional>

#include "PyInterpreter.h"
#include "Common.h"

namespace EasyNNPyPlugin {
	class Plots
	{
	public:
		static void CompareHypothesis(const std::vector<std::vector<double>>& X, const std::vector<double> y, const std::vector<double>& theta1, const std::vector<double>& theta2);
		template <typename... Args>
		static void PlotClassificationData(const std::vector<std::vector<double>>& X, const std::vector<double>& y, Args... args) {
			auto scriptName{ "PlotData" };
			auto methodName{ "PlotClassificationData" };
			auto& interpreter = PyInterpreter::getInstance();
			easyNN_unique_ptr argsAggregate{ interpreter.convertArgumentsToPyTuple(X, y, args...) };
			easyNN_unique_ptr pResult{ interpreter.executeMethod(scriptName, methodName, argsAggregate.get()) };
			if (pResult == nullptr) {
				PyErr_Print();
				throw std::runtime_error("Python method call failed.");
			}
		}
	};
}

#endif