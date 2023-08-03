#include "pch.h"
#include "DataChannel.h"
#include "PyInterpreter.h"
#include "Common.h"

using namespace EasyNNPyPlugin;

void DataChannel::getRegressionData(std::vector<std::vector<double>>& X, std::vector<double>& y, size_t nSamples, size_t nFeatures, double noise)
{
	auto scriptName{ "DataGenerator"};
	auto methodName {"getRegressionData"};
	PyInterpreter interpreter{};

	// Convert the C++ method arguments first to Python tuple so that we can pass them while calling the python method.
	PyObject* args = interpreter.convertArgumentsToPyTuple(nSamples, nFeatures, noise);

	PyObject* pResult = interpreter.executeMethod(scriptName, methodName, args);
	if (pResult != nullptr && PyTuple_Check(pResult) && PyTuple_Size(pResult) == 2) {
		interpreter.retrieveMatrixAndVector(X, y, pResult);
		Py_DECREF(pResult);
	}
	else {
		PyErr_Print(); // Print any Python exceptions
		throw std::runtime_error("Python method call failed.");
	}
	Py_DECREF(args);
}
