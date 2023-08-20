# Implementing Gradient Descent (GD)

## Fundamentals

We use gradient descent to find the best model parameters for which the change in the cost function is mimimal. Hence, it is given as follows

Repeat{

$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$

}

simultaneously update for every $j=0,...,n$.

Repeat{

$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)}$

}

simultaneously update $\theta_j$ for every $j=0,...,n$.

This can be expanded further for clarify as follows

$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})$

$\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) x_1^{(i)}$

$\theta_2 := \theta_2 - \alpha \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) x_2^{(i)}$

...

simultaneously update $\theta_j$ for every $j=0,...,n$.

where

$n$ is the model order

$\theta_j$ are the model parameters

$\alpha$ is the learning rate

$m$ is the number of samples

$(h_{\theta}(x^{(i)})$ is the hypothesis for $i^{th}$ feature vector 

$y^{(i)}$ is the $i^{th}$ measured value

$x_j^{(i)}$ is the $j^{th}$ feature of $i^{th}$ sample, or in order words this is the corresponding feature of the model parameter, i.e., for $\theta_1$ the corresponding feature is $x_1$ as in a typical model like $\theta_0 + \theta_1 x_1 + \theta_2 x_2 + ...$.

$x_0^{(i)} = 1$ as in a typical model like $\theta_0 + \theta_1 x_1 + \theta_2 x_2 + ...$.

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

In order to implement GD, let us focus on the following part of the algorithm:

$\frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) x_j^$

This very similar to the cost function $J(\theta)$, also because it is the derivative of $J(\theta)$. Hence, from implementation point of view, its computation is also very similar to that of $J(\theta)$. 

It starts making a lot more sense, if we would simplify the GD equation and assume there is only one sample of data. The GD equation reduces to only the following

$\theta_j := \theta_j - \alpha (h_{\theta}(x) - y) x_j$

$(h_{\theta}(x) - y)$ is very easily to interpret, it simply the error between the estimate and the measured value. Multiplying $x_j$ with the error term serves to amplify the error term proportional to the concerned feature value and show its contribution in the estimation error. Therefore, the large the value of the feature $x_j$ the more pronounced the error would be and vice versa. However, at the same time we don't want the amplified error to change parameter $\theta_j$ too much, as it could cause convergence issue. Hence, we attenuate the proportionally amplified error by $\alpha$. The respective $\theta_j$ value is updated accordingly.

Referring back to the original original equation, the following part becomes obvious

$\frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) x_j^$

This simplifies the case when we have multiple data points. Hence, we would like to find the differences of estimates with the measured values, proportionally amplify them with the corresponding feature value to determine the contribution of this specific feature in the error and repeat this process for all the data points we have available. Finally, we normally them by finding an average. This gives us an average error in estimation for all the available data points that is amplified by the respective feature value.

Hence, the informal intuitive steps of GD are given as follows

* Compute the cost function based on current parameter values
* For each parameter in the model
    * Compute the error between estimate and the measured value
    * Amplify the error by the respective feature value
    * Repeat it for all the available samples and sum the amplified estimate errors 
    * Attenuate the amplified error sum factor by learning rate $\alpha$
    * Find the updated parameter value, but store this value in a separate vector, keep the original parameters vector unmodified
* Replace the parameters vector with the new parameters vector
* Compute the cost function based on the new parameters
* If previous and new cost function difference is greater than the maximum threshold or the maximum number of iterations have reached for the algorithm, then exist, otherwise repeat from step 3

These steps should give a very good intuitive understanding of various operations taking place in GD and hopefully it will also help in understanding the actual algorithm that we explain next in implementation section.

## Implementation

GD would require the following parameters
* A feature matrix, where each row of the matrix will represent a feature vector
* A measurement vector
* A parameters vector containing the model parameters to compute the cost
* Cost function to evaluate the cost $J(\theta)$ based on the parameters values. One of the stopping criteria is based on this. If the cost function difference between two iterations is below a certain threshold, we stop.
* $\alpha$, the learning rate of GD algorithm
* Stopping threshold for the cost function difference between the two iterations
* maximum iterations, to stop the algorithm

It is important to note that the cost function in EasyNN derives from ICostFunction and it also hold an instance of the IHypothesis. Hence, through cost function, GD will have access to the hypothesis.

Here are the steps of to implement GD

1 Compute the cost function $J(\theta)$ based on the current batch of parameter values.
2 Start iterating the GD algorithm until maximum iterations count is reached.
    3 Iterate for all the parameters (nested loop) that we need to tune, which is equal to the model order, i.e., $n$.

    Step 4 - 8 are same as the computation of [linear regression cost function](./CostFunctionLinearRegression.md), except the computation is slightly different.

    4 Iterate over the all the available samples, i.e., $i = 1... m$, which we call as feature matrix (nested nested loop) and extract the feature vector, $x^{(i)}$.
        5 Evaluate the hypothesis for the current feature vector ($h_{\theta}(x^{(i)}$) based on the *current parameter batch*.
        6 Find the difference of hypothesis with measure value $y^{(i)}$ corresponding to the feature vector.
        7 The difference is multiplied the the corresponding feature of the model parameters which is being tuned, i.e., $j^{th} feature of $  $x_j^{(i)}$.
        8 Sum all the values from step 4 to 7 and finally normalize by the number of samples $m$.
    9 Compute new value of the model parameter by subtracting the values computed in step 8 multiplied by the learning rate $\alpha$. *Note these model values are stored in a separate vector and the earlier model values are not modified.*
10 The previous batch of model parameters are replaced with the new computed model parameters.
11 Compute the cost function $J(\theta)$ based on the new parameters
12 If the difference between the two cost functions is less than the threshold, stop the optimization
13 If not, the new cost function value is treated as the old cost function value and the new parameters are treated as old parameters and we repeat from step 2

