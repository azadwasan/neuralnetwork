#include "pch.h"
#include "CppUnitTest.h"
#include "IHypothesis.h"
#include "CostFunctionMSE.h"
#include "LinearHypothesis.h"
#include <span>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace EasyNNTest
{
	TEST_CLASS(EasyNNTest)
	{
	public:
		
		TEST_METHOD(TestMethod1)
		{
			//// Arrange
			EasyNN::CostFunctionMSE costFunction;
			std::span<std::span<double>> x;
			std::span<double> y;

			EasyNN::LinearHypothesis hypothesis;
			auto expectedResult = 3;
			// Act
			double result = costFunction.evaluate(x, y, hypothesis);

			// Assert
			Assert::AreEqual(expectedResult,result,0.00001);
		}
	};
}
