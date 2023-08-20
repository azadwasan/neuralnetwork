# Implementing Gradient Descent

## Fundamentals

We use gradient descent to find the best model parameters for which the change in the cost function is mimimal. Hence, it is given as follows

$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$

$\text{Repeat\{}
Cost(h_{\theta}(x), y) = \begin{cases} -\log(h_{\theta}(x)), & \text{if } y =1.\\\\ -\log (1 - h_{\theta}(x)), & \text{if } y = 0. \end{cases}
\text{\}}
$


$\text{Repeat{}
Cost(h_{\theta}(x), y) = \begin{cases} -\log(h_{\theta}(x)), & \text{if } y =1.\\\\ -\log (1 - h_{\theta}(x)), & \text{if } y = 0. \end{cases}
\text{}}
$

Repeat{
$
Cost(h_{\theta}(x), y) = \begin{cases} -\log(h_{\theta}(x)), & \text{if } y =1.\\\\ -\log (1 - h_{\theta}(x)), & \text{if } y = 0. \end{cases}
\text{}
$
}