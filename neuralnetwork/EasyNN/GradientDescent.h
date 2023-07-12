#ifndef GradientDescent_H
#define GradientDescent_H

#include "IHypothesis.h"
#include "ICostFunction.h"

#include<vector>

namespace EasyNN {
class GradientDescent
{
public:
	std::vector<double> evaluate(const std::vector<std::vector<double>>& featuresMatrix,
		const std::vector<double>& measurementsVector,
		const std::vector<double>& parameters,
		const IHypothesis& hypothesis,
		double alpha, double stopThreshold);
private:
	double computeCost(const std::vector<std::vector<double>>& featuresMatrix,
		const std::vector<double>& measurementsVector,
		const std::vector<double>& parameters,
		const IHypothesis& hypothesis);
	double computeCost(const std::vector<std::vector<double>>& featuresMatrix,
		const std::vector<double>& measurementsVector,
		const std::vector<double>& parameters,
		const IHypothesis& hypothesis, size_t index);

};
}

#endif