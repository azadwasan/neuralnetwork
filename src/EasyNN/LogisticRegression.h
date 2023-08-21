#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H
#include "IRegression.h"

namespace EasyNN {
	/**
	 * @class LogisticRegression
	 * @brief An implementation of logistic regression for binary classification.
	 *
	 * The LogisticRegression class provides methods for binary classification based on logistic regression.
	 * It calculates the probability of a positive outcome (class 1) given input features
	 * and model parameters.
	 */
	class LogisticRegression : public IRegression
	{
	public:
		/**
		 * @brief Calculate the probability of a positive outcome (class 1).
		 *
		 * Given a feature vector and model parameters, this method evaluates the logistic
		 * regression model and returns the probability of the positive outcome (class 1).
		 *
		 * @param featureVector A span containing the input feature vector.
		 * @param parameters A span containing the model parameters.
		 * @return The probability of a positive outcome (class 1).
		 */
		double evaluate(std::span<const double> featureVector, const std::span<const double> parameters, std::unique_ptr<IRegression> underlyingHypothesis = nullptr) const override;
	private:
		/**
		 * @brief The decision boundary for classification.
		 *
		 * This private constant defines the probability threshold that separates the two classes.
		 * If the predicted probability is greater than or equal to this threshold, the instance
		 * is classified as class 1; otherwise, it's classified as class 0.
		 */
		constexpr static double DECISION_BOUNDARY = 0.5;
	};
}

#endif
