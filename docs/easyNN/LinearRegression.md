# Implementing Linear Regression

## Fundamentals

Let's delve into the world of linear regression and get a grip on its key components:

Parameters: $\large{\theta = \theta_0, \theta_1, ... , \theta_n }$

It is an $n+1$ dimensional vector, where $n$ is the number of features.

Hypothesis: $\large{h_{\theta}(x) = \theta^Tx = \theta_0 x_0 + \theta_1 x_1 + ... + \theta_n x_n}$

## Key Observations

Here are key observations:

1. $x_0=1$
2. The hypothesis $h_{\theta}(x)$ involves $n$ features.
3. Subscript $n$ in $x_n$ denotes the $n$ features. This will be very important when we will implement the training, where both subscripts and superscripts will be involved.
4. The hypothesis is essentially a dot product of the parameter vector $\theta$ and the feature vector $x$.

## Implementation

With this groundwork laid, implementing the linear hypothesis is a breeze. Linear regression boils down to adding up the products of corresponding feature values and parameter values. The implementation would look something like this:

```cpp
// Linear regression implementation using raw loops
double LinearRegression::evaluate(const std::vector<double>& featureVector, const std::vector<double>& parameters){
    auto sum = 0;
    for (size_t i = 0; i < featureVector.size(); ++i) {
        sum += featureVector[i] * parameters[i + 1];
    }
	return sum;
}
```
LinearRegression::evaluate only signifies that evaluate is a method belonding to LinearRegression class. The implementation consits of a simple loop that sums the product of the feature values and the parameter. However, because $x_0 = 1$, we can save some effort. This adjusted version highlights the point:

```cpp
// Linear regression implementation using raw loops with shorter feature vector
double LinearRegression::evaluate(const std::vector<double>& featureVector, const std::vector<double>& parameters){
	auto sum = parameters[0];
    for (size_t i = 0; i < featureVector.size(); ++i) {
        sum += featureVector[i] * parameters[i + 1];
    }
	return sum;
}
```
Notice that the feature vector is one element shorter than the parameter vector. We begin the sum with parameters[0], representing $\theta_0$. In the loop, we remember to add one to the index for the parameters vector to align with the matching parameter value. 

This implementation can easily be simplified further by using standard library method std::inner_product() as follows

```cpp
// Linear regression implementation using std::inner_product
double LinearRegression::evaluate(const std::vector<double>& featureVector, const std::vector<double>& parameters){
	auto sum = parameters[0];
	sum += std::inner_product(std::begin(featureVector), std::end(featureVector), std::begin(parameters) + 1, 0.0);
	return sum;
}
```

## Testing

Let's peek at a typical test scenario to assess our implementation:
```cpp
// Linear regression implementation using std::inner_product
TEST_METHOD(TestLinearRegressionEvaluation)
{
    // Parameter vector
    std::vector<double> parameters{ -6.867, 3.148, -1.656};
    // Feature vector
    std::vector<std::vector<double>> x = { {60, 22}, {62, 25},{67, 24} };
    // Estimated values for each feature vector sample. E.g., for sample {60, 22}, the esimate is -6.867 + 3.148 * 60 - 1.656 * 22 = 145.581
    std::vector<double> estimates = { 145.581, 146.909, 164.305};
    auto estimate = begin(estimates);
    // Finally, calculate the estimate using the linear regression and compare is to the measured value 'y' (estimates).
    LinearRegression hypothesis{};
    for (const auto& val : x) {
        Assert::AreEqual(*estimate++, hypothesis.evaluate(val, parameters), 1.0E-5);
    }
}
```

## Software Design - Generalizing the Hypothesis

The implementation above solves the core problem, however, in order to make software scalable we need to generalize the design. It can be achieved by defining an interface that can represent various classes of regressions, i.e., linear regression, logistic regression etc.  EasyNN defines the interace for regressions as follow:

```cpp
namespace EasyNN {
	class IRegression {
	public:
		virtual double evaluate(const std::vector<double>& featureVector, const std::vector<double>& parameters) const = 0;
	};
}
```

Linear regression is an implementation of IRegression interface, as follows:

```cpp
namespace EasyNN {
	class LinearRegression : public IRegression {
	public:
		double evaluate(const std::vector<double>& featureVector, const std::vector<double>& parameters) const override;
	};
}
```

This design element is the key to generalized implementation of hypothesis and it will allow us to pass various hypothesis to different algorithms like gradient descent.

## Important Links

* [Next: Cost function for linear regression](./CostFunctionLinearRegression.md).
* [Go back to Implementing Neural Networks in C++](./index.md)
* EasyNN linear regression implementation [header](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/LinearRegression.h).
* EasyNN linear regression implementation [code](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/LinearRegression.cpp).
* EasyNN linear regression [test](https://github.com/azadwasan/neuralnetwork/blob/main/src/EasyNNTest/LinearRegressionTest.cpp).
