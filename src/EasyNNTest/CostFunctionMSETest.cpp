#include "pch.h"
#include "CppunitTest.h"
#include "CostFunctionMSE.h"
#include "LinearRegression.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace EasyNNTest {
	// NOTE: Please run LinearRegression.py in EasyNNPyScripts project to see how does this function actually look like.
	// and how the points are scattered around the plane that is represented by the model parameters.
	TEST_CLASS(CostFunctioNMSETest) {
	public:
		//Estimate the cost function MSE, when estimating using linear hypothesis.
		// Please refer to the documentaiton of LinearRegressionTest for details about various parts of the test.
		TEST_METHOD(TestCostFunctionMSE) {
			std::vector<double> parameters{ -6.867, 3.148, -1.656 };
			std::vector<std::vector<double>> x = {
													{60, 22},
													{62, 25},
													{67, 24},
													{70, 20},
													{71, 15},
													{72, 14},
													{75, 14},
													{78, 11}
			};
			std::vector<double> y = { 140, 155, 159, 179, 192, 200, 212, 215 };
			// Calcuate the MSE for the given feature vector, the measurement vector, model parameters and linear hypothesis.
			auto MSE = EasyNN::CostFunctionMSE{ std::make_unique<EasyNN::LinearRegression>() }.evaluate(x, y, parameters);
			Assert::AreEqual(MSE, 12.715159, 1.0E-5);
		}
	};
}