# Implementing Logistic Regression

## Fundamentals
I will summarize the various forms of logistic regression hypothesis as taught in Ng's course here

$h_{\theta}(x) = g(z) = g(\theta^Tx) = \frac{1}{1 + e^{-z}}$

$ h_{\theta}(x) = \frac{1}{1 + e^{-\theta^Tx}}$

We will be using the last equation above for the implementation.

## Key observations

From the above equations we have $z = \theta^Tx$, if we look closely this is the same as linear regression. 