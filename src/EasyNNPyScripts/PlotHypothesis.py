import numpy as np

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
