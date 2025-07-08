# Implementing Logistic Regression Cost Function

## Fundamentals

Logistic regression cost function is given as follows

$\large{J(\theta) = \frac{1}{m} \sum_{i=1}^{m} Cost(h_{\theta}(x^{(i)}), y^{(i)})}$

where

$\large{Cost(h_{\theta}(x), y) = \begin{cases} -\log(h_{\theta}(x)), & \text{if } y =1.\\\\ -\log (1 - h_{\theta}(x)), & \text{if } y = 0. \end{cases}}$

Hence,

$\large{J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log (1 - h_{\theta}(x^{(i)})) \right]}$

where 

$m$ is the number of samples.

$h_{\theta}(x^{(i)})$ is the logistic regression hypothesis for feature vector of $i^{th}$ sample.

$y^{(i)}$ is the $i^{th}$ measured value.

This final equation is the one that we will use for the implementation.

## Key observations

The hypothesis used in the equation above is the logistic regression hypothesis as already discussed and implemented [here](./LogisticRegression.md). It is being repeated it here for conveneince

$\large{h_{\theta}(x) = \frac{1}{1 + e^{-\theta^Tx}}}$


## Implementation

The implementation of logistic regression consists of the following steps
1. Compute the hypothesis (logistic regression)
2. Compute the log of the hypothesis results from step 1
3. Compute two two parts in the summation as shown in the equation above using the results from step 2
4. Compute sum of all the resultant values of step 3.
5. Normalize the final summation value with $-\frac{1}{m}$

In order to implement the cost function would accept the following inputs
* A feature matrix, where each row of the matrix will represent a feature vector
* A measurement vector
* A parameters vector containing the model parameters to compute the cost

```cpp
double CostFuntionLogistic::evaluate(const std::vector<std::vector<double>>& featuresMatrix, const std::vector<double>& measurementsVector, const std::vector<const double>& parameters) const {
    double costSum = 0.0;
    for (size_t i = 0; i < featuresMatrix.size(); ++i) {
        const auto& x = featuresMatrix[i];
        double y = measurementsVector[i];
        auto hTheta = hypothesis->evaluate(x, parameters);
        auto cost = y * log(hTheta) + (1 - y) * log(1 - hTheta);
        costSum += cost;
    }
    auto m = measurementsVector.size();
    return -1.0 / m * costSum;
}
```
To compute the logistic regression we run a simple loop, extracting the feature vector from the featureMatrix and the corresponding measured value. The hypothesis value is computed based on the feature vector and the measured value. Next, we compute the cost, which is summed over the duration of the loop, which is finally normalized.

Note from the details software design discussion for [linear regression cost function](./CostFunctionLinearRegression.md), we already know that the hypothesis is part of the ICostFunction interface, hence it is readily available to CostFuntionLogistic::evaluate(...) method.

This implementation can be further improved as follows by using the standard library algorithms:

```cpp
double CostFuntionLogistic::evaluate(const std::vector<std::vector<double>>& featuresMatrix, const std::vector<double>& measurementsVector, const std::vector<const double>& parameters) const{
   auto cost = [&parameters, this](const std::vector<double>& x, double y) -> double {
      auto hTheta = hypothesis->evaluate(x, parameters);
      auto cost = y * log(hTheta) + (1 - y) * log(1 - hTheta);
      return cost;
   };

   double costSum = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
      std::plus<>(),
      cost);
   auto m = measurementsVector.size();
   return -1.0 / m * costSum;
}
```

The cost computation has been moved to a lambda and cost summation calculation is exactly the same as the liner regression cost.

## Testing

EasyNN currently doesn't have any explicit tests for logistic regression. However, it is extensively tested while testing the gradient descent for data classification.

## Important Links
* [Next: Gradient Descent](./GradientDescent.md).
* [Back: Logistic regression](./LogisticRegression.md).
* [Go back to Implementing Neural Networks in C++](./index.md)
* EasyNN logistic regression cost function implementation [header](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/CostFunctionLogistic.h).
* EasyNN logistic regression cost function  implementation [code](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/CostFunctionLogistic.cpp).

## Index

## Index

**Linear Regression**

[Linear Regression](./LinearRegression.md)

[Linear Regression Cost Function](./CostFunctionLinearRegression.md)

**Logistic Regression**

[Logistic Regression](./LogisticRegression.md)

[Logistic Regression Cost Function](./CostFunctionLogisticRegression.md)

**Regularization**

[Regularization](./Regularization.md)

**Gradient Descent**

[Gradient Descent](./GradientDescent.md)

[Gradient Descent Evaluation](./GradientDescentTest.md)

**Neural Networks**

[Back Propagation](./BackPropagation.md)
