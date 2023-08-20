# Implementing Gradient Descent

## Fundamentals

We use gradient descent to find the best model parameters for which the change in the cost function is mimimal. Hence, it is given as follows

$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$


\begin{equation}
\text{Repeat } \left\{
\begin{aligned}
    \theta_j &:= \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)}
\end{aligned}
\right.
\end{equation}
