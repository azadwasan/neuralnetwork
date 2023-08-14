#include "pch.h"
#include "CppUnitTest.h"
#include "LogisticRegression.h"
#include "DataChannel.h"
#include "Algorithms.h"
#include "Plots.h"

#include <vector>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace EasyNNTest
{
	TEST_CLASS(LogisticRegressionTest) {
public:
		TEST_METHOD(TestLogisticRegressionEvaluation)
		{
			//EasyNN::LogisticRegression regression{};
			std::vector<std::vector<double>> X;
			std::vector<double> y;
			EasyNNPyPlugin::DataChannel::getClassificationData(X, y);
			auto logisticRegressionFit = EasyNNPyPlugin::Algorithms::FitLogisticRegression(X, y);
			EasyNNPyPlugin::Plots::PlotClassificationData(X, y, logisticRegressionFit);
			EasyNN::LogisticRegression lg{};

			size_t index = 0;
			for (const auto& vec : X) {
				Assert::AreEqual(lg.evaluate(vec, logisticRegressionFit), y[index++]);
			}
		}
	};
}
