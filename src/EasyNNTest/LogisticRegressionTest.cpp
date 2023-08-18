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
		/**
		 * @brief Test method for evaluating the performance of a logistic regression model.
		 *
		 * This test method retrieves classification data using the EasyNNPyPlugin::DataChannel::getClassificationData function,
		 * fits a logistic regression model to this data using the EasyNNPyPlugin::Algorithms::FitLogisticRegression function,
		 * which internally uses sklearn LogsiticRegression model to find the best parameters,
		 * and plots the classification data and fitted model using the EasyNNPyPlugin::Plots::PlotClassificationData function.
		 * The method then evaluates the performance of the fitted model on the classification data. The percentage of correct 
		 predictions is calculated, logged, and asserted to be greater than 80%.
		 */
		TEST_METHOD(TestLogisticRegressionEvaluation)
		{
			std::vector<std::vector<double>> X;
			std::vector<double> y;
			EasyNNPyPlugin::DataChannel::getClassificationData(X, y);
			auto logisticRegressionFit = EasyNNPyPlugin::Algorithms::FitLogisticRegression(X, y);
			EasyNNPyPlugin::Plots::PlotClassificationData(X, y, logisticRegressionFit);
			EasyNN::LogisticRegression lg{};

			size_t index = 0;
			double correctPercentage = 0;
			for (const auto& vec : X) {
				correctPercentage += (lg.evaluate(vec, logisticRegressionFit) > 0.5 ? 1 : 0 )== y[index++];
			}
			correctPercentage = correctPercentage / y.size() * 100;
			Logger::WriteMessage(("EasyNN % = " + std::to_string(correctPercentage)).c_str());
			Assert::IsTrue(correctPercentage > 80);
		}
		/**
		 * @brief Test method for evaluating the performance of a logistic regression model implemented using TensorFlow via the EasyNN library.
		 *
		 * It does exactly the same as above, but this time it uses TensorFlow instead.
		 */
		TEST_METHOD(TestLogisticRegressionEvaluationTF)
		{
			std::vector<std::vector<double>> X;
			std::vector<double> y;
			EasyNNPyPlugin::DataChannel::getClassificationData(X, y);
			auto logisticRegressionFit = EasyNNPyPlugin::Algorithms::FitLogisticRegressionTF(X, y);
			EasyNNPyPlugin::Plots::PlotClassificationData(X, y, logisticRegressionFit);
			EasyNN::LogisticRegression lg{};

			size_t index = 0;
			double correctPercentage = 0;
			for (const auto& vec : X) {
				correctPercentage += (lg.evaluate(vec, logisticRegressionFit) > 0.5 ? 1 : 0) == y[index++];
			}
			correctPercentage = correctPercentage / y.size() * 100;
			Logger::WriteMessage(("EasyNN % = " + std::to_string(correctPercentage)).c_str());

			Assert::IsTrue(correctPercentage > 80);
		}
	};
}
