#include "pch.h"
#include "Plots.h"
#include "PyInterpreter.h"
#include "Common.h"

using namespace EasyNNPyPlugin;

void Plots::CompareHypothesis(const std::vector<std::vector<double>>& X, const std::vector<double> y, const std::vector<double>& theta1, const std::vector<double>& theta2) {
	auto scriptName{ "PlotData" };
	auto methodName{ "CompareHypothesis" };
	auto& interpreter = PyInterpreter::getInstance();

	easyNN_unique_ptr args{ interpreter.convertArgumentsToPyTuple(X, y, theta1, theta2) };

	easyNN_unique_ptr pResult{ interpreter.executeMethod(scriptName, methodName, args.get()) };
}

void Plots::PlotClassificationData(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<double>& theta, const std::optional<std::vector<double>> theta2){
	auto scriptName{ "PlotData" };
	auto methodName{ "PlotClassificationData" };
	auto& interpreter = PyInterpreter::getInstance();
	easyNN_unique_ptr args{ interpreter.convertArgumentsToPyTuple(X, y, theta, theta2) };
	easyNN_unique_ptr pResult{ interpreter.executeMethod(scriptName, methodName, args.get()) };
	if (pResult == nullptr) {
		PyErr_Print();
		throw std::runtime_error("Python method call failed.");
	}
}