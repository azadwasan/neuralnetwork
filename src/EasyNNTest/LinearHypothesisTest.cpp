#include "pch.h"
#include "CppUnitTest.h"
#include "LinearHypothesis.h"
#include <span>
#include <vector>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace EasyNNTest
{
	TEST_CLASS(LinearHypothesisTest){
	public:
		// This is a very basic test that will verify that the LinearHypythesis computed estimate is same as our pre-computed measured value.
		// NOTE: Please run LinearHypothesisFunctionDepictions.py in EasyNNPythonDepictions project to see how does this function actually look like.
		// and how the points are scattered around the plane that is represented by the model parameters.
		TEST_METHOD(TestLinearHypothesisEvaluation)
		{
			EasyNN::LinearHypothesis hypothesis{};
			// This example has been taken from the following link
			// https://www.statology.org/multiple-linear-regression-by-hand/
			// If solved using this, https://www.statskingdom.com/410multi_linear_regression.html, the model parameters are different.
			// Note that the model doesn't really fit exactly. Hence, we will not take the values of Y, rather I used the model
			// and the input values to find the Y estimated using excel. That is what we will use to check the values against.

			// A simple model with three paramaters theta1, theta2, theta3 to estimate feature vector, each containing two features.
			// These values have been taken from the link provided above. 
			// We can see in test "TestGradientDescentEvaluation2Features", 
			// our estimated parameters are very different from this. But, both are pretty close.
			std::vector<double> parameters{ -6.867, 3.148, -1.656};
			// Feature vector containing two features, x1 and x2.
			// Each row corresponds to a sample of the two features.
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
			// Estimated values for each feature vector sample. E.g., for sample {60, 22}, the esimate is -6.867 + 3.148 * 60 - 1.656 * 22 = 145.581
			std::vector<double> estimates = { 145.581, 146.909, 164.305, 180.373, 191.801, 196.605, 206.049, 220.461};
			auto estimate = begin(estimates);
			// Finally, calculate the estimate using the linear hypothesis and compare is to the measured value (y). 
			// There should be no difference (slight difference is due to computation error).
			for (const auto& val : x) {
				Assert::AreEqual(*estimate++, hypothesis.evaluate(val, parameters), 1.0E-5);
			}
		}
	};
}
