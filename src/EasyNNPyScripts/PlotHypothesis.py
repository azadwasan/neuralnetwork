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
