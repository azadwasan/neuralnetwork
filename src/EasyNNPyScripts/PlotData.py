import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def PlotHypothesis(ax, x1_coords, x2_coords, thetas, y_values, plotColor='blue', plotAlpha=0.5):
    x1_range = np.arange(min(x1_coords)-5, max(x1_coords)+5, 1)
    x2_range = np.arange(min(x2_coords)-5, max(x2_coords)+5, 1)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    [theta0, theta1, theta2] = thetas
    ys = np.array([theta0 + theta1*x1 + theta2 *x2 for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
    Y = ys.reshape(X1.shape)
    ax.plot_surface(X1, X2, Y, color=plotColor, alpha=plotAlpha)
    ax.scatter(x1_coords, x2_coords, y_values)

    ax.set_xlabel('X1 - Feature 1')
    ax.set_ylabel('X2 - Feature 2')
    ax.set_zlabel('Y - Measurement')

def CompareHypothesis(X, y, easyNNTheta, tensorFlowTheta):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    PlotHypothesis(ax, [row[0] for row in X], [row[1] for row in X], easyNNTheta, y, plotColor='green')
    PlotHypothesis(ax, [row[0] for row in X], [row[1] for row in X], tensorFlowTheta, y, plotColor='blue')
    ## Create proxy artists for the legend
    proxy1 = Patch(facecolor='green', edgecolor='green', alpha = 0.5, label='EasyNN gradient descent fit')
    proxy2 = Patch(facecolor='blue', edgecolor='blue', alpha =0.5, label='Tensorflow gradient descent fit')

    ## Add a legend to the plot using the proxy artists
    ax.legend(handles=[proxy1, proxy2])

    plt.show()

def PlotScatterData(X, y):
    X = np.array(X)  # Convert X to a NumPy array for convenience

    # Extract the data points for each class using list comprehensions
    class_0_points = [X[i] for i in range(len(X)) if y[i] == 0]
    class_1_points = [X[i] for i in range(len(X)) if y[i] == 1]

    # Convert the lists to NumPy arrays for indexing
    class_0_points = np.array(class_0_points)
    class_1_points = np.array(class_1_points)

    plt.scatter(class_0_points[:, 0], class_0_points[:, 1], color='blue', label='Class 0')
    plt.scatter(class_1_points[:, 0], class_1_points[:, 1], color='orange', label='Class 1')
    # Add labels and legend
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

def PlotClassificationLine(X, y, theta, lineLabel, lineColor, lineStyle = '--', min = 0, max = 0):
    X = np.array(X)  # Convert X to a NumPy array for convenience

    bias = theta[0]
    coefficients = theta[1:]

    # Plot the decision boundary
    x_boundary = np.linspace(np.min(X[:, 0])-min, np.max(X[:, 0])+max, 100)
    y_boundary = -(coefficients[0] * x_boundary + bias) / coefficients[1]
    plt.plot(x_boundary, y_boundary, color=lineColor, linestyle=lineStyle, label=lineLabel)

    # Add labels and legend

def PlotClassificationData(X, y, theta, theta2 = None):
    ## Create a scatter plot of the data points
    plt.figure(figsize=(8, 6))

    PlotScatterData(X, y)
    PlotClassificationLine(X, y, theta, "Model 1", 'green')
    if theta2 is not None:
        PlotClassificationLine(X, y, theta2, "Model 2 (Reference TensorFlow)", 'blue', 'dotted')

    plt.title('Logistic Regression Decision Boundary')

    plt.legend()

    # Show the plot
    plt.show()
