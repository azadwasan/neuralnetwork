#include "pch.h"
#include "CppUnitTest.h"
#include "GradientDescent.h"
#include "LinearHypothesis.h"

#include <span>
#include <vector>
#include <chrono>
#include <iostream>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace EasyNNTest
{
	TEST_CLASS(GradientDescentTest){
	public:
		
		TEST_METHOD(TestGradientDescentEvaluation)
		{
			EasyNN::LinearHypothesis hypothesis{};
			std::vector<double> parameters{ 0, 0};
			std::vector<std::vector<double>> x = { {0}, {1}, {2}, {3}, {4} };
			std::vector<double> y = { 1, 3, 5, 7, 9};
			auto start = std::chrono::high_resolution_clock::now();
			parameters = EasyNN::GradientDescent{}.evaluate(x, y, parameters, hypothesis, 0.1, 0.000000001);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			std::string message = "Execution time: " + std::to_string(duration) + " microseconds\n";
			Logger::WriteMessage(message.c_str());
			std::vector<double> expectedParameters{1, 2};
			Assert::IsTrue(std::equal(std::begin(parameters), std::end(parameters), std::begin(expectedParameters), 
				[](double a, double b) {
					return std::abs(a - b) < 0.001; 
				}));
		}
	};
}
