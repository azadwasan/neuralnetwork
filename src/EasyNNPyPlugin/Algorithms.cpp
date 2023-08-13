#include "pch.h"
#include "Algorithms.h"
#include "Common.h"
#include "PyInterpreter.h"


using namespace EasyNNPyPlugin;


std::vector<double> Algorithms::RunGD(const std::vector<std::vector<double>>& X, const std::vector<double>& y, size_t paramCount) {
	auto scriptName{ "GradientDescent" };
	auto methodName{ "OptimizeGD" };
	auto& interpreter = PyInterpreter::getInstance();
	// Convert the C++ method arguments first to Python tuple so that we can pass them while calling the python method.
	easyNN_unique_ptr args{ interpreter.convertArgumentsToPyTuple(X, y, paramCount)};

	easyNN_unique_ptr pResult{ interpreter.executeMethod(scriptName, methodName, args.get()) };
	std::vector<double> result;
	
	if (pResult != nullptr && PyList_Check(pResult.get()) /* && PyList_Size(pResult.get()) == 1*/) {
		interpreter.extractVector(pResult.get(), result);
	}
	else {
		PyErr_Print(); // Print any Python exceptions
		throw std::runtime_error("Python method call failed.");
	}
	return result;
}

std::vector<double> Algorithms::FitLogisticRegression(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
	auto scriptName{ "LogisticRegression" };
	auto methodName{ "FitLogisticRegression" };
	auto& interpreter = PyInterpreter::getInstance();
	// Convert the C++ method arguments first to Python tuple so that we can pass them while calling the python method.
	easyNN_unique_ptr args{ interpreter.convertArgumentsToPyTuple(X, y) };

	easyNN_unique_ptr pResult{ interpreter.executeMethod(scriptName, methodName, args.get()) };
	std::vector<double> result;

	if (pResult == nullptr) {
		throw std::runtime_error("Null result was returned!");
	}

	if (pResult != nullptr && PyList_Check(pResult.get())) {
		interpreter.extractVector(pResult.get(), result);
	}
	else {
		PyErr_Print(); // Print any Python exceptions
		throw std::runtime_error("Python method call failed.");
	}
	return result;

}
