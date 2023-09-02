# Implementing Gradient Descent (GD)

## Fundamentals

We use gradient descent to find the best model parameters for which the change in the cost function w.r.t. the model parameters is minimal.

Hence, it is given as follows

Repeat{

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

}

simultaneously update for every $j=0,...,n$.

Whereas, cost function $J(\theta)$ is defined as the mean square error (MSE) between the estimated value through the hypothesis ($ht_\theta^{(i)}$) and the measured value $y^{(i)}$. Hence, cost function is given as follows:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Note that cost function is mean square error of $m$ samples. MSE is one of the measured that could be used for cost function. However, further discussion about choosing the cost function is out of the scope of this work.

Before we proceed any further, let us define various symbols

$n$ is the model order

$\theta_j$ is the jth model parameter.

$\alpha$ is the learning rate, which determines the step size in each iteration.

$m$ is the number of samples

$(h_{\theta}(x^{(i)})$ is the hypothesis for $i^{th}$ feature vector 

$y^{(i)}$ is the $i^{th}$ measured value.

$x_j^{(i)}$ is the $j^{th}$ feature of $i^{th}$ sample, or in order words this is the corresponding feature of the model parameter, i.e., for $\theta_1$ the corresponding feature is $x_1$ as in a typical model like $\theta_0 + \theta_1 x_1 + \theta_2 x_2 + ...$.

$x_0^{(i)} = 1$ as in a typical model like $\theta_0 + \theta_1 x_1 + \theta_2 x_2 + ...$.

$J(\theta)$ is the cost function.

### Partial Differentiation of Cost Function

The partial differential of the cost function would be:

$$ \frac{\partial}{\partial \theta_j} J(\theta) = \frac{\partial}{\partial \theta_j} \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 $$

Expanding the above equation by plugging in the definition of $h_{\theta}(x^{(i)})$, we get the following equation:

$$ \frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\theta_0 + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)} + ... + \theta_n x_n^{(i)} - y^{(i)})^2 $$

This equation is very easily differentiable, i.e., the power will reduce by 1 and from the definition of $h_\theta(x^{(i)})$ only $x_i^{(j)}$ will remain and rest will be zero. Hence, differentiating the above equation and plugging back $h_{\theta}(x^{(i)})$ we get the following equation:

$$ \frac{\partial}{\partial \theta_j} J(\theta) = \frac{\partial}{\partial \theta_j} \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

Hence, the updated equation for gradient descent would be as follows:

Repeat{

$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

}

simultaneously update $\theta_j$ for every $j=0,...,n$.

This can be expanded further for clarification as follows

$$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})$$

$$\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) x_1^{(i)}$$

$$\theta_2 := \theta_2 - \alpha \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) x_2^{(i)}$$

...

simultaneously update $\theta_j$ for every $j=0,...,n$.

## Key observations

$\theta_0$ will have to be calculated separately from the rest of the parameters, as its equation is slightly different from the rest.

The most important step that must be carried out correctly is that the parameters have to be updated simultaneously, i.e., the set of parameters has to be considered as a batch and a new batch of the parameters has to be computed based on the previous one and the previous batch would be *simultaneously* updated/replaced with the new batch. 

The gradient descent algorithm is agnostic to the type of hypothesis that we are trying to optimize, i.e., we could optimize either a linear regression problem or a logistic regression problem and gradient descent algorithm will not have to be changed. Thanks to EasyNN flexible design, our implementation of GD will also be agnostic to the hypothesis type.

## Stopping criteria

Various stopping criteria can be used to stop the iterative GD algorithm. EasyNN implements the following two mechanisms

* Maximum number of iterations: The algorithm simply stops after iterating for a certain number of times
* Cost function change: If the difference between the cost function value between two iterations is below a certain threshold, the algorithm is stopped.

There are also more advanced techniques like taking in a callback function to let the user of GD monitor the progression of the optimization and possibly stop it. This functionality is not currently implemented in EasyNN.

## Achieving optimization with GD

Whether the algorithm achieves the optimization or not depends on a lot of factors

* the hypothesis has to be correct. For example, a non-linear problem can't be fit with linear model, hence the optimization will fail.
* Depending on the problem GD might need more or less number of iterations
* Learning rate $\alpha$ could slow down the optimization if too small and make the algorithms run in optimization loop or even overshoot if $\alpha$ is too high

All of these are discussed in detail in Ng's course, hence we are not going to illustrate them here or go into any further details.

## Making sense of it all - How does GD actually work!

GD works by iteratively adjusting the model parameters to reduce the error between the predicted values and the actual values. It computes the gradient of the cost function with respect to each parameter and updates the parameters in the opposite direction of the gradient, effectively moving towards the minimum of the cost function.

It starts making a lot more sense, if we would simplify the GD equation and assume there is only one sample of data. The GD equation reduces to only the following

$$\theta_j := \theta_j - \alpha (h_{\theta}(x) - y) x_j$$

$h_{\theta}(x) - y$ is very easily to interpret, it is the error between the estimate and the measured value. Multiplying $x_j$ with the error term serves to amplify the error term proportional to the concerned feature value and show its contribution in the estimation error. Therefore, the large the value of the feature $x_j$ the more pronounced the error would be and vice versa. However, at the same time we don't want the amplified error to change parameter $\theta_j$ too much, as it could cause convergence issues. Hence, we attenuate the proportionally amplified error by $\alpha$. The respective $\theta_j$ value is updated accordingly.

Referring back to the original original equation, the following part becomes obvious

$$\frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) x_j$$

This represents the general case when we have multiple data points. Hence, we would like to find the differences of estimates with the measured values for all the available data points. Finally, we normally them to find an average error. This gives us an average error in estimation for all the available data points that is amplified by the respective feature value.

Hence, the informal intuitive explanation of the steps involved in GD are given as follows

* Compute the cost function based on current parameter values
* For each parameter in the model
    * Compute the error between estimate and the measured value
    * Amplify the error by the respective feature value
    * Repeat it for all the available samples and sum the amplified estimated errors 
    * Attenuate the amplified error sum factor by learning rate $\alpha$
    * Find the updated parameter value, but store this value in a separate vector, keep the original parameters vector unmodified
* Replace the parameters vector with the new parameters vector
* Compute the cost function based on the new parameters
* If previous and new cost function difference is greater than the maximum threshold or the maximum number of iterations have reached for the algorithm, then exist, otherwise repeat from step 3

This explanation should give a very good intuitive understanding of various operations taking place in GD and hopefully it will also help in understanding the actual algorithm that we explain next in implementation section.

## Implementation

GD would require the following parameters
* A feature matrix, where each row of the matrix will represent a feature vector.
* A measurement vector.
* A parameters vector containing the model parameters to compute the cost (these will finally contain the optimized parameter values, hence are also the output of the algorithm).
* Cost function to evaluate the cost $J(\theta)$ based on the parameters values. One of the stopping criteria is based on this. If the cost function difference between two iterations is below a certain threshold, we stop.
* $\alpha$, the learning rate of GD algorithm
* Stopping threshold for the cost function difference between the two iterations
* Maximum iterations, to stop the algorithm

It is important to note that the cost function in EasyNN derives from ICostFunction that also holds an instance of the IHypothesis. Hence, through cost function, GD will have access to the hypothesis.

Here are the steps of to implement GD

1. Compute the cost function $J(\theta)$ based on the current batch of parameter values.
2. Start iterating the GD algorithm until maximum iterations count is reached.
3. Iterate for all the parameters (nested loop) that we need to tune, which is equal to the model order, i.e., $n$.
4. Iterate over the all the available samples, i.e., $i = 1... m$, which we call as feature matrix (nested nested loop) and extract the feature vector, $x^{(i)}$.
5. Evaluate the hypothesis for the current feature vector ($h_{\theta}(x^{(i)}$) based on the *current parameter batch*.
6. Find the difference of hypothesis with measure value $y^{(i)}$ corresponding to the feature vector.
7. The difference is multiplied the the corresponding feature of the model parameters which is being tuned, i.e., $j^{th}$ feature of $i^{the}$ feature vector, i.e, $x_j^{(i)}$.
8. Sum all the values from step 4 to 7 and finally normalize by the number of samples $m$.
9. Compute new value of the model parameter by subtracting the values computed in step 8 multiplied by the learning rate $\alpha$. *Note these model values are stored in a separate vector and the earlier model values are not modified.*
10. The previous batch of model parameters are replaced with the new computed model parameters.
11. Compute the cost function $J(\theta)$ based on the new parameters.
12. If the difference between the two cost functions is less than the threshold, stop the optimization.
13. If not, the new cost function value is treated as the old cost function value and the new parameters are treated as old parameters and we repeat from step 3.

Step 4 - 8 are same as the computation of [linear regression cost function](./CostFunctionLinearRegression.md), except the computation is slightly different.

Here is how the basic implementation would look like

```cpp
1   void GradientDescent::evaluate(const std::vector<std::vector<double>>& featuresMatrix,
2      const std::vector<double>& measurementsVector,
3      const ICostFunction& costFunction,
4      double alpha, double stopThreshold,
5      std::vector<double>& parameters,
6      size_t maxIterations) {
7
8      std::vector<double> parametersNew(parameters.size());
9
10      auto oldCost = costFunction.evaluate(featuresMatrix, measurementsVector, parameters);
11      auto i = 0;
12      auto newCost = 0.0;
13      for (; i < maxIterations; i++) {
14         double differenceSumZero = 0.0;
15         for (size_t k = 0; k < featuresMatrix.size(); k++) {
16            differenceSumZero += costFunction.getHypothesis().evaluate(featuresMatrix[k], parameters) - measurementsVector[k];
17         }
18         parametersNew[0] = parameters[0] - alpha * 1 / measurementsVector.size() * differenceSumZero;
19
20         for (size_t index = 1; index < parameters.size(); index++) {
21            double differenceSum = 0.0;
22            for (size_t k = 0; k < featuresMatrix.size(); k++) {
23               differenceSum += (costFunction.getHypothesis().evaluate(featuresMatrix[k], parameters) - measurementsVector[k]) * featuresMatrix[k][index - 1];
24            }
25            parametersNew[index] = parameters[index] - alpha * 1 / measurementsVector.size() * differenceSum;
26         }
27         parameters = parametersNew;
28         newCost = costFunction.evaluate(featuresMatrix, measurementsVector, parametersNew);
29         if (abs(oldCost - newCost) < stopThreshold) {
30            break;
31         }
32         oldCost = newCost;
33      }
34   }
```

Considering all the lengthy details about how GD works, the code to implement the algorithm is fairly simple. We are being provided the current parameter values as method argument. The cost function is computed based on the parameters (line 8). We start iterating up until the maximum number of iterations. Next, we compute the difference sum, using the hypothesis, feature matrix, measured values and parameters. However, as we noted in the beginning, the computation of difference sum for $\theta_0$ is different from the rest. Hence, we need separate code for the two cases. For $\theta_0$ the the new parameter is computed from line 15 to 18 and for the rest, these are computed from line 20 to 26. Note that the new parameters are being stored in a separate vector and the original vector is still being maintained.

After all the parameters have been computed, the parameters are replaced with the new parameters (line 25). Next, the cost function is computed using the new parameter values (line 28). GD stops if the difference between the cost functions is less than the stopping threshold.

Though, it looks like the method doesn't return any values, however, this is a peculiarity of C++, where parameters is being passed by reference and it is being modified within the body of the method. Hence, when the method returns, the parameters will contains the final optimized values and the caller of the method can use the optimized values of the parameters.

We can improve the code above slightly by moving the new parameter calculation into separate methods

```cpp
void GradientDescent::evaluate(const std::vector<std::vector<double>>& featuresMatrix,
    const std::vector<double>& measurementsVector,
    const ICostFunction& costFunction,
    double alpha, double stopThreshold,
    std::vector<double>& parameters, size_t maxIterations) {

    std::vector<double> parametersNew(parameters.size());

    auto oldCost = costFunction.evaluate(featuresMatrix, measurementsVector, parameters);
    auto i = 0;
    auto newCost = 0.0;
    for (; i < maxIterations; i++) {
        parametersNew[0] = computeNewParameterZero(featuresMatrix, measurementsVector, parameters, costFunction, alpha);
        for (size_t index = 1; index < parameters.size(); index++) {
            parametersNew[index] = computeNewParameter(featuresMatrix, measurementsVector, parameters, costFunction, alpha, index);
        }

        parameters = parametersNew;

        newCost = costFunction.evaluate(featuresMatrix, measurementsVector, parametersNew);
        if (abs(oldCost - newCost) < stopThreshold) {
            break;
        }
        oldCost = newCost;
    }
    auto temp = costFunction.evaluate(featuresMatrix, measurementsVector, parametersNew);
}

double GradientDescent::computeNewParameterZero(const std::vector<std::vector<double>>& featuresMatrix,
    const std::vector<double>& measurementsVector,  const std::vector<double>& parameters,
    const ICostFunction& costFunction, double alpha) {

    double differenceSumZero = 0.0;
    for (size_t k = 0; k < featuresMatrix.size(); k++) {
        differenceSumZero += costFunction.getHypothesis().evaluate(featuresMatrix[k], parameters) - measurementsVector[k];
    }
    return parameters[0] - alpha * 1 / measurementsVector.size() * differenceSumZero;
}

double GradientDescent::computeNewParameter(const std::vector<std::vector<double>>& featuresMatrix,
    const std::vector<double>& measurementsVector, const std::vector<double>& parameters,
    const ICostFunction& costFunction, double alpha, size_t index) {

    double differenceSum = 0.0;
    for (size_t k = 0; k < featuresMatrix.size(); k++) {
        differenceSum += (costFunction.getHypothesis().evaluate(featuresMatrix[k], parameters) - measurementsVector[k]) * featuresMatrix[k][index - 1];
    }
    return parameters[index] - alpha * 1 / measurementsVector.size() * differenceSum;
}
```

This makes the code slightly more structured but still there is quite a bit of code duplication. We can improve this even further as follows:

```cpp
void GradientDescent::evaluate(const std::vector<std::vector<double>>& featuresMatrix,
   const std::vector<double>& measurementsVector, const ICostFunction& costFunction,
   double alpha, double stopThreshold,
   std::vector<double>& parameters, size_t maxIterations) {

   std::vector<double> parametersNew(parameters.size());

   auto oldCost = costFunction.evaluate(featuresMatrix, measurementsVector, parameters);
   auto i = 0;
   auto newCost = 0.0;
   double m = measurementsVector.size();

    auto differenceSumZero = [&](const auto& featuresVector, const auto& parameters, auto measurement, size_t index) {return costFunction.getHypothesis().evaluate(featuresVector, parameters) - measurement; };
    auto differenceSum = [&](const auto& featuresVector, const auto& parameters, auto measurement, size_t index) {return (costFunction.getHypothesis().evaluate(featuresVector, parameters) - measurement) * featuresVector[index - 1]; };

   for (; i < maxIterations; i++) {
      parametersNew[0] = parameters[0] - alpha * 1 / m *
         computeCost(featuresMatrix, measurementsVector, parameters, costFunction.getHypothesis(),    differenceSumZero);
      for (size_t index = 1; index < parameters.size(); index++) {
         parametersNew[index] = parameters[index] - alpha * 1 / m
            * computeCost(featuresMatrix, measurementsVector, parameters, costFunction.getHypothesis(), differenceSum, index);
      }

      parameters = parametersNew;

      newCost = costFunction.evaluate(featuresMatrix, measurementsVector, parametersNew);
      if (abs(oldCost - newCost) < stopThreshold) {
         break;
      }
      oldCost = newCost;
   }
}

double GradientDescent::computeCost(const std::vector<std::vector<double>>& featuresMatrix,
   const std::vector<double>& measurementsVector, const std::vector<double>& parameters,
   const IRegression& hypothesis, 
   std::function<double(const std::vector<double>&, const std::vector<double>&, double, size_t)> differenceSum, size_t index /*=0*/) {

   double differenceSum = std::transform_reduce(std::begin(featuresMatrix), std::end(featuresMatrix), std::begin(measurementsVector), 0.0,
      std::plus<>(),
      [&](const auto& featuresVector, auto measurement) {
         return differenceSum(featuresVector, parameters, measurement, index);
      }
   );
   return differenceSum;
}
```

We define a lambda to compute $\theta_0$ difference sum called differenceSumZero and another one for the rest of the parameters called differenceSum. We replace the two methods with just a single method that accepts a lambda to compute the difference sum and replace the loop with standard library std::transform_reduce, like we did in earlier codes, e.g., [Linear regression cost function](./CostFunctionLinearRegression.md). Rest of the code stays almost the same. This is also how it is currently implemented in EasyNN.

## Testing

Test GD is a separate topic in itself, hence we discuss it in detail [here](./GradientDescentTest.md).

## Important Links
* [Next: Regularization](./Regularization.md).
* [Back: Logistic Regression Cost Function](./CostFunctionLogisticRegression.md).
* [Go back to Implementing Neural Networks in C++](./index.md)
* EasyNN Gradient Descent implementation [header](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/GradientDescent.h).
* EasyNN Gradient Descent implementation [code](https://github.com/azadwasan/neuralnetwork/tree/main/src/EasyNN/GradientDescent.cpp).
* EasyNN Gradient Descent [test](https://github.com/azadwasan/neuralnetwork/blob/main/src/EasyNNTest/GradientDescentTest.cpp).

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


