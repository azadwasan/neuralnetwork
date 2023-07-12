#include "pch.h"
#include "CppunitTest.h"
#include "CostFunctionMSE.h"
#include "LinearHypothesis.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace EasyNNTest {
	TEST_CLASS(CostFunctioNMSETest) {
	public:
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
			std::vector<double> estimates = { 145.581, 146.909, 164.305, 180.373, 191.801, 196.605, 206.049, 220.461 };
			auto hypothesis = EasyNN::LinearHypothesis{};
			
			auto MSE = EasyNN::CostFunctionMSE{}.evaluate(x, y, parameters, hypothesis);
			Assert::AreEqual(MSE, 12.715159, 0.00001);
		}
	};
}