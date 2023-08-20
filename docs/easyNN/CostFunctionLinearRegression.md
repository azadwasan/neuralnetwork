# Linear Regression Cost Function

## Fundamentals
The cost function for linear regression  can be defined as follows:

$\large{J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2}$   **(1)**

This is the form of equation that we will use for the implementation. However, for the sake of clarify we can expand $h_\theta(x^{(i)})$ 

$\large{J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\theta^Tx^{(i)} - y^{(i)})^2}$     **(2)**

Which can be expanded even further into individual feature components for $h_\theta(x^{(i)})$

$\large{J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\theta_0 x_0^{(i)} + \theta_1 x_1^{(i)} + ... + \theta_n x_n^{(i)} - y^{(i)})^2}$  **(3)**

This equation is useful for implementing without separately defining the hypothesis.

## Notations and Key Observations

$J(\theta)$ is the cost function

$m$ is the number of samples

$n$ is the number of features or linear regression model order

$x$ is a feature **vector** (and *not a single feature*)

$x^{(i)}$ is the feature vector of $i^{th}$ sample

$h_\theta(x^{(i)})$ is the linear regression hypothesis, [implemented here](./LinearRegression.md)

$y^{(i)}$ is the $i^{th}$ measured value

The superscript for $x$ and $y$ are used to denote the sample number and the subscripts (as it was pointed out [earlier](./LinearRegression.md)) denote a particular feature.

### Visualizing the parameters
Before we delve any further, here is a simple example to visualize these parameters (taken from Ng's lecture notes)

| Size ($feet^2$) | # of bedrooms     | # of floors | Price ($1000) |
|:----------------|:------------------|:------------|:--------------|
| 2104            | 5                 | 1           | 460           |
| 1416            | 3                 | 2           | 232           |
| 1534            | 3                 | 2           | 315           |
| 852             | 2                 | 1           | 178           |

Here, 

$m=4$ as there are 4 samples in total.

$n=3$ as there are 3 features (size, # of bedrooms, # of floors).

$x^{(2)}$ denotes the second feature vector with values [1416 3 2].

$y^{(2)}$ denotes the second measured value, i.e., 232.

## Implementation

Referring back to equation (1), implementing cost functional is trivial. It involves the following simple steps
* Iterate over the all the available samples, i.e., $i = 1... m$, which we call as feature matrix (nested nested loop) and extract the feature vector, $x^{(i)}$
* Compute hypothesis value for the extracted feature vector
* Compute the squared difference of hypothesis and the measured value
* Sum the squared differences for all the samples
* Normalize the sum by dividing it with $2m$

In order to implement the cost function would accept the following inputs
* A feature matrix, where each row of the matrix will represent a feature vector as shown above
* A measurement vector
* A parameters vector containing the model parameters to compute the cost
* Linear regression as hypothesis through the interface IRegression, as [implemented here](./LinearRegression.md). *Please read the software design section for details, as there we discuss that hypothesis will not be a required argument for evaluate(...) method.*


```cpp
// Cost function implementing using raw loops
double CostFunctionMSE::evaluate(const std::vector<std::vector<double>>& featuresMatrix, const std::vector<double>& measurementsVector, const std::vector<double>& parameters, const IRegression& hypothesis) const
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
double CostFunctionMSE::evaluate(const std::vector<std::vector<double>>& featuresMatrix, const std::vector<double>& measurementsVector, const std::vector<double>& parameters, const IRegression& hypothesis) const
{
   double mse = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
      std::plus<>(),
      [&](const std::vector<double>& featuresVector, double measurement) {return std::pow(hypothesis.evaluate(featuresVector, parameters) - measurement, 2); });
   
   auto m = measurementsVector.size();

   return mse / (2 * m) + regFactor;
}
```
We hav replaced the loops with std::transform_reduce and specified the summation operation through std::plus<>() to be performed for each feature vector. The operation of squared difference is specified through a lambda.

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

## Software Design - Generalizing the Cost Function

### Interface Implementation

Similar to linear regression, where we generalized the design of hypothesis through IRegression, we would be generalizing the design of the cost function by defining the following interface

```cpp
namespace EasyNN {
   class ICostFunction {
   public:
      virtual double evaluate(const std::vector<std::vector<double>>& featuresMatrix, const std::vector<double>& measurementsVector, const std::vector<double>& parameters, const IRegression& hypothesis) const = 0;
   };
}
```
Cost function implements the ICostFunction interface
```cpp
namespace EasyNN {
    class CostFunctionMSE : public ICostFunction
    {
    public:
        double evaluate(const std::vector<std::vector<double>>& featuresMatrix, const std::vector<double>& measurementsVector, const std::vector<double>& parameters, const IRegression& hypothesis) const override;
    };
}
```
### Cost function and Hypothesis

As we can observe in the interface, cost function requires the hypothesis to compute the cost. Later on we will also observe in the implementation of gradients descent and other algorithms that we will have to maintain the costs and hypothesis and keep track of them. Hence, EasyNN extends the ICostFunction interface to also hold an instance of hypothesis. This allows us to keep the overall design clean and flexible. The extended interafce looks as follows:
```cpp
namespace EasyNN {
   class ICostFunction {
   public:
      ICostFunction(std::unique_ptr<IRegression> hypo) : hypothesis{ std::move(hypo) } {
         if (hypothesis == nullptr) {
            throw std::runtime_error("Hypothesis is not allowed to be Null!");
         }
      }
      virtual double evaluate(const std::vector<std::vector<double>>& featuresMatrix, const std::vector<double>& measurementsVector, const std::vector<double>& parameters) const = 0;
      const IRegression& getHypothesis() const noexcept{
         return *hypothesis.get();
      }
   protected:
      std::unique_ptr<IRegression> hypothesis;
   };
}
```

The resulting change in the cost function implementation would be
```cpp
namespace EasyNN {
    class CostFunctionMSE : public ICostFunction
    {
    public:
        CostFunctionMSE(std::unique_ptr<IRegression> hypothesis) : ICostFunction(std::move(hypothesis)) {}
        double evaluate(const std::vector<std::vector<double>>& featuresMatrix, const std::vector<double>& measurementsVector, const std::vector<double>& parameters) const override;
    };
}
```

The hypothesis is now passed only directly at the time of creation of the cost function. evaluate(...) method doesn't require the hypothesis anymore. We also allow the instance of hypothesis to be retrievable by the user of cost function, as it would be needed to perform various other operations.

The final code for the computation of the cost function would change as follows, as we no longer need hypothesis as an argument for evaluate(...) method:

```cpp
double CostFunctionMSE::evaluate(const std::vector<std::vector<double>>& featuresMatrix, const std::vector<double>& measurementsVector, const std::vector<double>& parameters) const
{
   double mse = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
      std::plus<>(),
      [&](const std::vector<double>& featuresVector, double measurement) {return std::pow(hypothesis->evaluate(featuresVector, parameters) - measurement, 2); });
   
   auto m = measurementsVector.size();

   return mse / (2 * m) + regFactor;
}
```

### Design Tradeoff - Tight Coupling

ICostFunction holding an instance of IRegression creates tight coupling between the two. At the same time it allows to write relatively cleaner code and we don't need to hold and maintain two interfaces instances separately (it is specially tricky when we have multiple cost functions and multiple corresponding hypothesis representing different underlying cost operations and hypothesis). Hence, it is a compromise and different requirements may yield different design decisions. However, for EasyNN I have made this design decision with tight coupling, at least for now! However, it is also not too difficult to extend the interface even further to allow replacing the underlying hypothesis of the cost function.

## Important Links
* [Next: Logistic regression](./LogisticRegression.md).
* [Back: Linear regression](./LinearRegression.md).
* [Go back to Implementing Neural Networks in C++](./index.md)
* EasyNN linear regression cost function implementation [header](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/CostFunctionMSE.h).
* EasyNN linear regression cost function  implementation [code](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/CostFunctionMSE.cpp).
* EasyNN linear regression cost function [test](https://github.com/azadwasan/neuralnetwork/blob/main/src/EasyNNTest/CostFunctionMSETest.cpp).
