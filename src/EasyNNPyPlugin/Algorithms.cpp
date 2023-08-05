#include "pch.h"
#include "Algorithms.h"
#include "PyInterpreter.h"
#include "Common.h"

using namespace EasyNNPyPlugin;

std::vector<double> Algorithms::RunGD(const std::vector<std::vector<double>>& X, const std::vector<double> y, size_t paramCount) {
	auto scriptName{ "GradientDescent" };
	auto methodName{ "OptimizeGD" };
	PyInterpreter interpreter{};

	// Convert the C++ method arguments first to Python tuple so that we can pass them while calling the python method.
	easyNN_unique_ptr args{ interpreter.convertArgumentsToPyTuple(X, y, paramCount) };

	easyNN_unique_ptr pResult{ interpreter.executeMethod(scriptName, methodName, args.get()) };
	std::vector<double> result;
	if (pResult != nullptr && PyTuple_Check(pResult.get()) && PyTuple_Size(pResult.get()) == 1) {
		interpreter.extractVector(pResult.get(), result);
	}
	else {
		PyErr_Print(); // Print any Python exceptions
		throw std::runtime_error("Python method call failed.");
	}
	return result;
}
