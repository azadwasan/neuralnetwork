# Implementing Logistic Regression

## Fundamentals
I will summarize the various forms of logistic regression hypothesis as taught in Ng's course here

$\large{h_{\theta}(x) = g(z) = g(\theta^Tx) = \frac{1}{1 + e^{-z}}}$

$\large{h_{\theta}(x) = \frac{1}{1 + e^{-\theta^Tx}}}$

We will be using the last equation above for the implementation.

## Key Observations
From the above equations we have $z = \theta^Tx$, if we look closely this is the same as linear regression. However, it is not limited to just linear regression it could be any underlying linear or non-linear operations on the features and parameters. Luckily, we already modeled such flexiblity in our design earlier by defining the hypothesis through IRegression. Hence, we can call $z$ as the underlying hypothesis.

## Implementation

Once, we have established that $z$ is an underlying hypothesis, all that is need to evaluate the logistic regression hypotsis to calculate the underlying hypothesis $z$ value and compute the sigmoid function using the equation given above.

Hence, the implementation would look as follows:

```cpp
double LogisticRegression::evaluate(const std::vector<double>& featureVector, const std::vector<double>& parameters, const IRegression& underlyingHypothesis) const {
	auto z = underlyingHypothesis.evaluate(featureVector, parameters);
	auto gz = 1 / (1 + exp(-1.0 * z));
	return gz;
}

```

That is literally all that is to evaluate the logistic regression and due to our flexible design, we do not care how does the underlying hypothesis look like and whether it is linear or linear etc.

## Testing

Logistic regression tests look similar to linear regression tests. For specific example, one can look at EasyNN tests. However, these tests are implemented on dynamic data that is generated from Python libraries. The details about this kind of live data testing are discussed separately.


## Important Links
* [Back: Linear regression cost function](./CostFunctionLinearRegression.md).
* [Next: Logistic regression cost function](./CostFunctionLogisticRegression.md).
* [Go back to Implementing Neural Networks in C++](./index.md)
* EasyNN logistic regression implementation [header](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/LogisticRegression.h).
* EasyNN logistic regression implementation [code](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/LogisticRegression.cpp).
* EasyNN logistic regression [test](https://github.com/azadwasan/neuralnetwork/blob/main/src/EasyNNTest/LogisticRegressionTest.cpp).
