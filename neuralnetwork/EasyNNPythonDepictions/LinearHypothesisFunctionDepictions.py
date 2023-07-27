from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the range of the function to be centered around the scattered points
x1_coords = [60, 62, 67, 70, 71, 72, 75, 78]
x2_coords = [22, 25, 24, 20, 15, 14, 14, 11]
x1_range = np.arange(min(x1_coords)-5, max(x1_coords)+5, 1)
x2_range = np.arange(min(x2_coords)-5, max(x2_coords)+5, 1)
X1, X2 = np.meshgrid(x1_range, x2_range)

theta0 = -6.867
theta1 = 3.148
theta2 = -1.656
ys = np.array([theta0 + theta1*x1 + theta2 *x2 for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
Y = ys.reshape(X1.shape)

ax.plot_surface(X1, X2, Y)

# Plot the given coordinates and values
y_values = [140, 155, 159, 179, 192, 200, 212, 215]
ax.scatter(x1_coords, x2_coords, y_values)

ax.set_xlabel('X1 - Feature 1')
ax.set_ylabel('X2 - Feature 2')
ax.set_zlabel('Y - Measurement')

plt.show()

# The folllwing plot is exactly as the above, but with additiona 
# coordinate axis passing through (0,0) and a (X,Y) plane.


#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import numpy as np

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

## Define the range of the function to be centered around the scattered points
#x1_coords = [60, 62, 67, 70, 71, 72, 75, 78]
#x2_coords = [22, 25, 24, 20, 15, 14, 14, 11]
#x1_range = np.arange(min(x1_coords)-70, max(x1_coords)+5, 1)
#x2_range = np.arange(min(x2_coords)-20, max(x2_coords)+5, 1)
#X1, X2 = np.meshgrid(x1_range, x2_range)

#theta0 = -6.867
#theta1 = 3.148
#theta2 = -1.656
#ys = np.array([theta0 + theta1*x1 + theta2 *x2 for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
#Y = ys.reshape(X1.shape)

#ax.plot_surface(X1, X2, Y)

## Plot the given coordinates and values
#y_values = [140, 155, 159, 179, 192, 200, 212, 215]
#ax.scatter(x1_coords, x2_coords, y_values)

### Add axis lines that pass through the (0,0) coordinates
#ax.plot([min(x1_range), max(x1_range)], [0,0], [0,0], color='black')
#ax.plot([0,0], [min(x2_range), max(x2_range)], [0,0], color='black')
#ax.plot([0,0], [0,0], [min(ys), max(ys)], color='black')

### Add an x-y plane to indicate the positive and negative regions of the z axis
#XX, YY = np.meshgrid(x1_range, x2_range)
#ZZ = np.zeros_like(XX) 
#ax.plot_surface(XX, YY, ZZ, color='lightgray', alpha=0.5)

#ax.set_xlabel('X1 - Feature 1')
#ax.set_ylabel('X2 - Feature 2')
#ax.set_zlabel('Y - Measurement')

#plt.show()

