# Linear Regression Implementaion 

For linear regression, parameters and the hypothesis are defined as follows:

### Parameters: $\theta = \theta_0, \theta_1, ... , \theta_n $

It is an $n+1$ dimensional vector.

### Hypothesis: $h_{\theta}(x) = \theta^Tx = \theta_0 x_0 + \theta_1 x_1 + ... + \theta_n x_n$

Here are key observations:

1. $x_0=1$
2. We have $n$ features being represented by the hypothesis $h_{\theta}(x)$.
3. Subscript $n$ in $x_n$ denotes the $n$ features. This will be very important when we will implement the training, where both subscripts and superscripts will be involved.
4. Hypothesis is simply an dot product of parameter vector $\theta$ and feature vector $x$.

Based on the above discussion, the implementation of linear hypothesis is extremely trivial. Linear regerssion is sum of the products of the corresponding feature value and the parameter value. This is how the implementation would look like:

```cpp
// Linear regression implementation using raw loops
double LinearRegressionEvaluate(const std::vector<double>& featureVector, const std::vector<double>& parameters){
    auto sum = 0;
    for (size_t i = 0; i < featureVector.size(); ++i) {
        sum += featureVector[i] * parameters[i + 1];
    }
	return sum;
}
```
The implementation consits of a simple loop that sums the product of the feature values and the parameter. However, because $x_0 = 1$, we don't have to have both features vector and parameters vector be of the same length. Hence, the updated code would look like the following 
```cpp
// Linear regression implementation using raw loops
double LinearRegressionEvaluate(const std::vector<double>& featureVector, const std::vector<double>& parameters){
	auto sum = parameters[0];
    for (size_t i = 0; i < featureVector.size(); ++i) {
        sum += featureVector[i] * parameters[i + 1];
    }
	return sum;
}
```
Here, that feature vector is one length shorter than the parameter vector and we start with sum equating to parameters[0], i.e., $\theta_0$. However, we have to be careful in the loop to add one to the parameters vector to fetch the matching parameter value.

This implementation can easily be simplified further by using standard library method std::inner_product() as follows

```cpp
// Linear regression implementation using std::inner_product
double LinearRegressionEvaluate(const std::vector<double>& featureVector, const std::vector<double>& parameters){
	auto sum = parameters[0];
	sum += std::inner_product(std::begin(featureVector), std::end(featureVector), std::begin(parameters) + 1, 0.0);
	return sum;
}
```

This is how a typical test would look like to evaluate the implementation
```cpp
// Linear regression implementation using std::inner_product
TEST_METHOD(TestLinearRegressionEvaluation)
{
    // Parameter vector
    std::vector<double> parameters{ -6.867, 3.148, -1.656};
    // Feature vector
    std::vector<std::vector<double>> x = { 
                                            {60, 22},
                                            {62, 25},
                                            {67, 24},
                                            {70, 20},
                                            {71, 15},
                                            {72, 14},
                                            {75, 14},
                                            {78, 11}
    };
    // Estimated values for each feature vector sample. E.g., for sample {60, 22}, the esimate is -6.867 + 3.148 * 60 - 1.656 * 22 = 145.581
    std::vector<double> estimates = { 145.581, 146.909, 164.305, 180.373, 191.801, 196.605, 206.049, 220.461};
    auto estimate = begin(estimates);
    // Finally, calculate the estimate using the linear regression and compare is to the measured value (y). 
    // There should be no difference (slight difference is due to computation error).
    for (const auto& val : x) {
        Assert::AreEqual(*estimate++, LinearRegressionEvaluate(val, parameters), 1.0E-5);
    }
}
```
## Important Links

* [cost function for linear regression](./CostFunctionLinearRegression.md).
* EasyNN linear regression implementation [header](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/LinearRegression.h).
* EasyNN linear regression implementation [code](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/LinearRegression.cpp).
* EasyNN linear regression [test](https://github.com/azadwasan/neuralnetwork/blob/main/src/EasyNNTest/LinearRegressionTest.cpp).
* [Go back to Implementing Neural Networks in C++](./index.md)