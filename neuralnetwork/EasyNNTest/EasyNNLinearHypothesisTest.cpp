#include "pch.h"
#include "CppUnitTest.h"
#include "IHypothesis.h"
#include "CostFunctionMSE.h"
#include "LinearHypothesis.h"
#include <span>
#include <vector>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace EasyNNTest
{
	TEST_CLASS(EasyNNTest)
	{
	public:
		
		TEST_METHOD(TestLinearHypothesis)
		{
			EasyNN::LinearHypothesis hypothesis{};
			// This example has been taken from the following link
			// https://www.statology.org/multiple-linear-regression-by-hand/
			// If solved using this, https://www.statskingdom.com/410multi_linear_regression.html, the model parameters are different.
			// Note that the model doesn't really fit exactly. Hence, we will not take the values of Y, rather I used the model
			// and the input values to find the Y estimated using excel. That is what we will use to check the values against.
			std::vector<double> parameters{ -6.867, 3.148, -1.656};
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
			std::vector<double> y = { 145.581, 146.909, 164.305, 180.373, 191.801, 196.605, 206.049, 220.461};
			auto index = 0;
			for (const auto& val : x) {
				Assert::AreEqual(y[index++], hypothesis.evaluate(val, parameters), 0.00001);
			}
		}
	};
}
