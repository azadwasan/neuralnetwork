import matplotlib.pyplot as plt
import numpy as np
import DataGenerator
import LogisticRegression
import PlotData

def MyTestMethod(data):
    print("Received data:", data)
    result = [x * 100 for x in data]
    print("Result within Python:", result)
    return result

def PlotTestGradientDescentEvaluation2Features(fig):
    ax = fig.add_subplot(111, projection='3d')

    # Define the range of the function to be centered around the scattered points
    x1_coords = [60, 62, 67, 70, 71, 72, 75, 78]
    x2_coords = [22, 25, 24, 20, 15, 14, 14, 11]

    # My gradient descent solution parameters
    thetas = [0.013080267480039371, 3.0559138415398879, -1.6822470943097785]

    # Reference solution parameters
    thetas2 = [-6.867, 3.148, -1.656]

    y_values = [140, 155, 159, 179, 192, 200, 212, 215]

    x1_range = np.arange(min(x1_coords)-5, max(x1_coords)+5, 1)
    x2_range = np.arange(min(x2_coords)-5, max(x2_coords)+5, 1)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    [theta0, theta1, theta2] = thetas
    ys = np.array([theta0 + theta1*x1 + theta2 *x2 for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
    Y = ys.reshape(X1.shape)
    ax.plot_surface(X1, X2, Y, color='green')
    ax.scatter(x1_coords, x2_coords, y_values)

    ax.set_xlabel('X1 - Feature 1')
    ax.set_ylabel('X2 - Feature 2')
    ax.set_zlabel('Y - Measurement')

    x1_range = np.arange(min(x1_coords)-5, max(x1_coords)+5, 1)
    x2_range = np.arange(min(x2_coords)-5, max(x2_coords)+5, 1)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    [theta0, theta1, theta2] = thetas
    ys = np.array([theta0 + theta1*x1 + theta2 *x2 for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
    Y = ys.reshape(X1.shape)
    ax.plot_surface(X1, X2, Y, color='red')
    ax.scatter(x1_coords, x2_coords, y_values)

    ax.set_xlabel('X1 - Feature 1')
    ax.set_ylabel('X2 - Feature 2')
    ax.set_zlabel('Y - Measurement')

def fitLogisticRegerssion():
    X, y = DataGenerator.getClassificationData(100, 2, 0, 1, 42)
    params = LogisticRegression.FitLogisticRegression(X, y)
    PlotData.PlotClassificationData(X, y, params)

fitLogisticRegerssion()