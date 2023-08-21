# Implementing Regularization

Regularization is performed to avoid overfitting. It is pretty straight forward to implement. We will be using our earlier implementation and make minor adjustments to add regularization.

## Regularized Linear Regression

### Fundamentals

Here is the equation to implement regularization for linear regression

$J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^n \theta_j^2 \right]$

Regularization add an additional summation term for all the model parameters, except $\theta_0$. Before we would implement it for linear regression, lets look how regularization is implemented for logistic regression.

## Regularized Logistic Regression

### Fundamentals

Regularized logistic regression is given as follow

$J(\theta) = -\frac{1}{m} \left[ \sum_{i=1}^m y^{(i)} \log h_{\theta}(x^{(i)}) + (1 - y^{(i)}) \log (1 - h_{\theta}(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2$

We observe it is exactly the same regularization term that has been added to the above equation.

## Implementation

We would implement regularization in the base class of cost function as it is common between the cost functions. 

This is how the based class would look like

```cpp
class ICostFunction {
public:
    ICostFunction(std::unique_ptr<IRegression> hypo, double lmda = 0.0) : hypothesis{ std::move(hypo) }, lambda{ lmda } {
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
```

In order to implement regularization, the constructor accepts an the regularization parameters $\lambda$ and getRegFactor() computes the regularization factor. The regularized logistic regression looks as follows:

```cpp
double CostFuntionLogistic::evaluate(const std::vector<std::vector<double>>& featuresMatrix, std::span<const double> measurementsVector, std::span<const double> parameters) const{
   auto cost = [&parameters, this](const std::vector<double>& x, double y) -> double {
      auto hTheta = hypothesis->evaluate(x, parameters);
      auto cost = y * log(hTheta) + (1 - y) * log(1 - hTheta);
      return cost;
   };

   double costSum = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
      std::plus<>(),
      cost);

   auto m = measurementsVector.size();
   double regFactor = getRegFactor(parameters, m);
   return -1.0 / m * costSum + regFactor;
}
```

The only change required to implement regularization is to compute the regularization factor by calling getRegFactor(...) and add it to the logistic regression computed value.

## Regularized Gradient Descent

## Fundamentals

Regularized gradient descent is given as follow

Repeat{

$\large{\theta_j := \theta_j (1 - \alpha \frac{\lambda}{m}) - \alpha \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)}}$

}

All we have to do to implement regularization for gradient descent is to multiply a factor $1 - \alpha \frac{\lambda}{m}$ with $\theta_j$. Hence, the change required in the code is also very simple

```cpp
void GradientDescent::evaluate(const std::vector<std::vector<double>>& featuresMatrix,
   const std::vector<double>& measurementsVector,
   std::vector<double>& parameters,
   const ICostFunction& costFunction,
   double alpha, double stopThreshold, const size_t maxIterations /*= 3000*/) {

   std::vector<double> parametersNew(parameters.size());

   auto oldCost = costFunction.evaluate(featuresMatrix, measurementsVector, parameters);
   auto i = 0;
   auto newCost = 0.0;
   double m = measurementsVector.size();
   double regFactor = (1 - alpha * costFunction.getLambda() / m);
   auto costDerivativeZero = [&](const auto& featuresVector, const auto& parameters, auto measurement, size_t index) {return costFunction.getHypothesis().evaluate(featuresVector, parameters) - measurement; };
   auto costDerivative = [&](const auto& featuresVector, const auto& parameters, auto measurement, size_t index) {return (costFunction.getHypothesis().evaluate(featuresVector, parameters) - measurement) * featuresVector[index - 1]; };

   for (; i < maxIterations; i++) {
      parametersNew[0] = parameters[0] - alpha * 1 / m *
         computeCost(featuresMatrix, measurementsVector, parameters, costDerivativeZero);
      for (size_t index = 1; index < parameters.size(); index++) {
         parametersNew[index] = parameters[index] * regFactor - alpha * 1 / m
            * computeCost(featuresMatrix, measurementsVector, parameters, costDerivative, index);
      }

      parameters = parametersNew;

      newCost = costFunction.evaluate(featuresMatrix, measurementsVector, parametersNew);

      if (abs(oldCost - newCost) < stopThreshold) {
         break;
      }
      oldCost = newCost;
   }
}
```

The regularization parameters $\lambda$ is already available to GD through the ICostFunction interface. We compute the regularization, factor regFactor = (1 - alpha * costFunction.getLambda() / m), and multiply it to the model parameter value when calculating the new parameters (except $\theta_0$).

EasyNN implementation for linear regression, logistic regression and gradient descent are regularized implementations. The code shown above is the current implementation of EasyNN.

## Important Links
* [Next: Gradient Descent Evaluation/Testing](./GradientDescentTest.md).
* [Back: Gradient Descent Implementation](./GradientDescent.md).
* [Go back to Implementing Neural Networks in C++](./index.md)