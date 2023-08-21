#ifndef ICOSTFUNCTION_H
#define ICOSTFUNCTION_H

#include <span>
#include <memory>
#include <stdexcept>
#include <numeric>
#include <cmath>

#include "IRegression.h"

namespace EasyNN {
	class ICostFunction {
	public:
		ICostFunction(std::unique_ptr<IRegression> hypo, double lmda = 0.0) : hypothesis{ std::move(hypo) }, lambda{ lmda } {
			if (hypothesis == nullptr) {
				throw std::runtime_error("Hypothesis is not allowed to be Null!");
			}
		}
		virtual double evaluate(const std::vector<std::vector<double>>& featuresMatrix, std::span<const double> measurementsVector, std::span<const double> parameters) const = 0;
		const IRegression& getHypothesis() const noexcept{
			return *hypothesis.get();
		}
		double getLambda() const noexcept { return lambda; }
	protected:
		double getRegFactor(std::span<const double> parameters, double m) const noexcept{
			return lambda / (2 * m) * std::accumulate(parameters.begin() + 1, parameters.end(), 0, [](auto acc, auto x) { return acc + x * x; });
		}

		std::unique_ptr<IRegression> hypothesis;
		double lambda;
	};
}

#endif