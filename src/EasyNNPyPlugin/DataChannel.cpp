#include "pch.h"
#include "DataChannel.h"
#include "PyInterpreter.h"
#include "Common.h"

using namespace EasyNNPyPlugin;

void DataChannel::getRegressionData(std::vector<std::vector<double>>& X, std::vector<double>& y, size_t nSamples, size_t nFeatures, double noise){
	auto scriptName{ "DataGenerator"};
	auto methodName {"getRegressionData"};
	auto& interpreter = PyInterpreter::getInstance();

	// Convert the C++ method arguments first to Python tuple so that we can pass them while calling the python method.
	easyNN_unique_ptr args{ interpreter.convertArgumentsToPyTuple(nSamples, nFeatures, noise) };

	easyNN_unique_ptr pResult{ interpreter.executeMethod(scriptName, methodName, args.get()) };
	if (pResult != nullptr && PyTuple_Check(pResult.get()) && PyTuple_Size(pResult.get()) == 2) {
		interpreter.retrieveMatrixAndVector(X, y, pResult.get());
	}
	else {
		PyErr_Print(); // Print any Python exceptions
		throw std::runtime_error("Python method call failed.");
	}
}

void DataChannel::getClassificationData(std::vector<std::vector<double>>& X, std::vector<double>& y, size_t nSamples /*= 100*/, size_t nFeatures /*= 2*/, size_t redundantFeatures /*= 0*/, size_t clustersPerClass /*= 1*/, size_t randomState /*= 42*/){
	auto scriptName{ "DataGenerator" };
	auto methodName{ "getClassificationData" };
	auto& interpreter = PyInterpreter::getInstance();

	// Convert the C++ method arguments first to Python tuple so that we can pass them while calling the python method.
	easyNN_unique_ptr args{ interpreter.convertArgumentsToPyTuple(nSamples, nFeatures, redundantFeatures, clustersPerClass, randomState) };

	easyNN_unique_ptr pResult{ interpreter.executeMethod(scriptName, methodName, args.get()) };
	if (pResult != nullptr && PyTuple_Check(pResult.get()) && PyTuple_Size(pResult.get()) == 2) {
		interpreter.retrieveMatrixAndVector(X, y, pResult.get());
	}
	else {
		PyErr_Print(); // Print any Python exceptions
		throw std::runtime_error("Python method call failed.");
	}
}

