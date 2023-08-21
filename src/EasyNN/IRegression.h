#ifndef IREGRESSION_H
#define IREGRESSION_H

#include <vector>
#include <span>
#include <memory>

namespace EasyNN {
    /**
     * @class IRegression
     * @brief An interface representing a regression model.
     *
     * This class defines a common interface for regression models, which can be used to make predictions based on a set of input features and model parameters.
     */
    class IRegression {
    public:
        /**
         * @brief Evaluates the regression model for the given feature vector and parameters.
         *
         * This method computes the predicted output value for the given feature vector, based on the current values of the model parameters.
         *
         * @param featureVector The feature vector for which to make a prediction.
         * @param parameters The current values of the model parameters.
         * @param underlyingHypothesis An optional underlying hypothesis, it is useful specially in the case of logistic regression,
         * where the underlying hypothesis is used to model the data. 
         * @return The predicted output value for the given feature vector and parameters.
         */
        virtual double evaluate(std::span<const double> featureVector, const std::span<const double> parameters, std::unique_ptr<IRegression> underlyingHypothesis = nullptr) const = 0;
    };
}


#endif