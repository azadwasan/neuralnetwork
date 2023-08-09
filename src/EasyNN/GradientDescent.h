#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include "IRegression.h"
#include "ICostFunction.h"

#include<vector>
#include<functional>

namespace EasyNN {
class GradientDescent
{
public:
	void evaluate(const std::vector<std::vector<double>>& featuresMatrix,
		const std::vector<double>& measurementsVector,
		const IRegression& hypothesis,
		double alpha, double stopThreshold,
		std::vector<double>& parameters);
private:
	double computeCost(const std::vector<std::vector<double>>& featuresMatrix,
		const std::vector<double>& measurementsVector,
		const std::vector<double>& parameters,
		const IRegression& hypothesis, 
		std::function<double(const std::vector<double>&, const std::vector<double>&, double, size_t)> costDeivative, size_t index = 0);

	// The following methods are redundant with seperate implementation for Theta(0) and Thota(i), where i>0. They can be safely removed now.
	double computeCost(const std::vector<std::vector<double>>& featuresMatrix,
		const std::vector<double>& measurementsVector,
		const std::vector<double>& parameters,
		const IRegression& hypothesis);
	double computeCost(const std::vector<std::vector<double>>& featuresMatrix,
		const std::vector<double>& measurementsVector,
		const std::vector<double>& parameters,
		const IRegression& hypothesis, size_t index);
};
}

#endif