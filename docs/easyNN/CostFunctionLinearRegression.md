# Linear Regression Cost Function

## Fundamentals
The cost function for linear regression  can be defined as follows:

$\large{J(\theta) = \frac{1}{2n} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2}$ **(1)**

This is the form of equation that we will use for the implementation. However, for the sake of clarify we can expand $h_\theta(x^{(i)})$ 

$\large{J(\theta) = \frac{1}{2n} \sum_{i=1}^{m} (\theta^Tx - y^{(i)})^2}$ **(2)**

Which can be expanded even further into individual feature compoents for $h_\theta(x^{(i)})$

$\large{J(\theta) = \frac{1}{2n} \sum_{i=1}^{m} (\theta_0 x_0 + \theta_1 x_1 + ... + \theta_n x_n - y^{(i)})^2}$ **(3)**

This equation is useful for implementing without seperately defining the hypothesis.

## Notations and Key Observations

$J(\theta)$ is the cost funciton

$m$ is the number of samples

$x$ is a feature **vector** (and *not a single feature*)

$x^{(i)}$ is the feature vector of $i^{th}$ sample

$h_\theta(x^{(i)})$ is the linear regression hypothesis, [implemented here](./LinearRegression.md)

$y^{(i)}$ are the measured values

The superscript for $x$ and $y$ are used to denote the sample number and the subscripts (as it was pointed out [earlier](./LinearRegression.md)) denote a particular feature.

### Visualizing the parameters
Before we delve any further, here is a simple example to visualize these parameters (taken from Ng's lecture notes)

| Size ($feet^2$) | # of bedrooms     | # of floors | Price ($1000) |
|:----------------|:------------------|:------------|:--------------|
| 2104            | 5                 | 1           | 460           |
| 1416            | 3                 | 2           | 232           |
| 1534            | 3                 | 2           | 315           |

Here, 

$m=3$ as there are 3 samples in total.

$x^{(2)}$ denotes the second feature vector with values [1416 3 2].

$y^{(2)}$ denotes the second measured value, i.e., 232.

## Implementing

Referring back to equation (1), implementing cost functional is trivial. It involves the following simple steps
* Compute hypothesis value for a feature vector
* Compute the squared difference of hypotehesis and the measured value
* Sum the squared differences for all the samples
* Normalize the sum by dividing it with $2m$

In order to implement the cost function would accept the following inputs
* A feature matrix, where each row of the matrix will represent a feature as shown above
* A measuredment vector
* A parameters vector containing the model parameters to compute the cost
* Linear regression as hypothesis, as [implemented here](./LinearRegression.md)


```cpp
// Cost function implementing using raw loops
double CostFunctionMSE::evaluate(const std::vector<std::vector<double>>& featuresMatrix, const std::vector<double>& measurementsVector, const std::vector<double>& parameters, const LinearRegression& hypothesis) const
{
    double mse = 0.0;
    for (size_t i = 0; i < featuresMatrix.size(); ++i) {
        const auto& featuresVector = featuresMatrix[i];
        double measurement = measurementsVector[i];
        mse += std::pow(hypothesis.evaluate(featuresVector, parameters) - measurement, 2);
    }
    auto m = measurementsVector.size();
    return mse / (2 * m);
}

```
The code implements exactly as we discussed above. It runs a loop over feature matrix, extracts a feature vector and computes a sum of squared difference and finally normalizes it. CostFunctionMSE::evaluate only signifies that evaluate is a method of CostFunctionMSE class. This code can easily be improved and written succinctly as follows

```cpp
double CostFunctionMSE::evaluate(const std::vector<std::vector<double>>& featuresMatrix, const std::vector<double>& measurementsVector, const std::vector<double>& parameters, const LinearRegression& hypothesis) const
{
	double mse = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
		std::plus<>(),
		[&](const std::vector<double>& featuresVector, double measurement) {return std::pow(hypothesis.evaluate(featuresVector, parameters) - measurement, 2); });
	
	auto m = measurementsVector.size();

	return mse / (2 * m) + regFactor;
}
```
We hav replaced the loops with std::transform_reduce and specificed the summation operation through std::plus<>() to be performed for each feature vector. The operation of squared difference is specified through a lambda.

## Testing
```cpp
TEST_CLASS(CostFunctioNMSETest) {
public:
    TEST_METHOD(TestCostFunctionMSE) {
        std::vector<double> parameters{ -6.867, 3.148, -1.656 };
        std::vector<std::vector<double>> x = {
                                                {60, 22},
                                                {62, 25},
                                                {67, 24}
        };
        std::vector<double> y = { 140, 155, 159, 179, 192, 200, 212, 215 };
        auto hypothesis = LinearRegression{};
        auto MSE = CostFunctionMSE{}.evaluate(x, y, parameters, hypothesis);
        Assert::AreEqual(MSE, 12.715159, 1.0E-5);
    }
};
```
## EasyNN Implementation
EasyNN core implementaiton for cost function is exactly the same but it differs in terms of the design, because it is designed to be more flexible and scalable. Hence, it implements interfaces for regression classes and cost function and the regression instances are passed through dependency injection to the cost functions. The design decisions and their tradeoffs are discussed in detail [here](./EasyNNDesign.md).

## Important Links

* [Next: Logistic regression](./LogisticRegression.md).
* [Go back to Implementing Neural Networks in C++](./index.md)
* EasyNN linear regression cost function implementation [header](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/CostFunctionMSE.h).
* EasyNN linear regression cost function  implementation [code](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/CostFunctionMSE.cpp).
* EasyNN linear regression cost function [test](https://github.com/azadwasan/neuralnetwork/blob/main/src/EasyNNTest/CostFunctionMSETest.cpp).
