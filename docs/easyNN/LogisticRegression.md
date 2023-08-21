# Implementing Logistic Regression

## Fundamentals
I will summarize the various forms of logistic regression hypothesis as taught in Ng's course here

$\large{h_{\theta}(x) = g(z) = g(\theta^Tx) = \frac{1}{1 + e^{-z}}}$

$\large{h_{\theta}(x) = \frac{1}{1 + e^{-\theta^Tx}}}$

We will be using the last equation above for the implementation.

## Key Observations
From the above equations we have $z = \theta^Tx$, if we look closely this is the same as linear regression. However, it is not limited to just linear regression it could be any underlying linear or non-linear operations on the features and parameters. Luckily, we already modeled such flexibility in our design earlier by defining the hypothesis through IRegression. Hence, we can call $z$ as the underlying hypothesis.

## Implementation

Once, we have established that $z$ is an underlying hypothesis, all that is need to evaluate the logistic regression hypothesis to calculate the underlying hypothesis $z$ value and compute the sigmoid function using the equation given above.

Hence, the implementation would look as follows:

```cpp
double LogisticRegression::evaluate(const std::vector<double>& featureVector, const std::vector<double>& parameters, const IRegression& underlyingHypothesis) const {
   auto z = underlyingHypothesis.evaluate(featureVector, parameters);
   auto gz = 1 / (1 + exp(-1.0 * z));
   return gz;
}
```

That is literally all that is to evaluate the logistic regression and due to our flexible design, we do not care how does the underlying hypothesis look like and whether it is linear or linear etc.

## Software Design - Extending the interface to accept underlying hypothesis

The code I provided above is technically not correct, because the interface we defined for IRegression::evaluate(..) doesn't allow the additional parameter. Though, it conveys the message the but we still need to improve the design.

The underlying hypothesis is not always needed, like in the case of linear regression. Hence, when we extend the interface to accept the additional parameter, we can make it optional and use it only when we need it. The underlying hypothesis parameter can be added to the base interface as follows

```cpp
class IRegression {
public:
    virtual double evaluate(std::span<const double> featureVector, const std::span<const double> parameters, std::unique_ptr<IRegression> underlyingHypothesis = nullptr) const = 0;
};
```

It is added as a pointer with default value set to nullptr. Hence, the existing code and the tests stay unaffected by the change. However, we will have to adapt all the implementations of the interface, i.e., linear regression and logistic regression. However, we will only show the change for logistic regression here.

```cpp
double LogisticRegression::evaluate(std::span<const double> featureVector, const std::span<const double> parameters, std::unique_ptr<IRegression> underlyingHypothesis /*= nullptr*/) const {
	std::unique_ptr<IRegression> hypothesis = (underlyingHypothesis == nullptr ? std::make_unique<LinearRegression>() : std::move(underlyingHypothesis));

	auto z = hypothesis->evaluate(featureVector, parameters);
	auto gz = 1 / (1 + exp(-1.0 * z));
	return gz;
}
```

Logistic regression can now accept additional parameter for underlying hypothesis. Based on the value of the pointer it can either use the passed hypothesis if it is not nullptr, otherwise it creates an instance of linear regression and use it as underlying hypothesis. I am sorry to non-C++ programmers here for some weird syntax but all it does that it checks if the user has pass a valid object of underlying hypothesis then use it, otherwise create a new object of LinearRegression type.

## Testing

Logistic regression tests look similar to linear regression tests. For specific example, one can look at EasyNN tests. However, these tests are implemented on dynamic data that is generated from Python libraries. The details about this kind of live data testing are discussed separately.


## Important Links
* [Next: Logistic regression cost function](./CostFunctionLogisticRegression.md).
* [Back: Linear regression cost function](./CostFunctionLinearRegression.md).
* [Go back to Implementing Neural Networks in C++](./index.md)
* EasyNN logistic regression implementation [header](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/LogisticRegression.h).
* EasyNN logistic regression implementation [code](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/LogisticRegression.cpp).
* EasyNN logistic regression [test](https://github.com/azadwasan/neuralnetwork/blob/main/src/EasyNNTest/LogisticRegressionTest.cpp).
