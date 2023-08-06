import tensorflow as tf
from sklearn.datasets import make_regression

def OptimizeGD(X_data, y_data, paramCount):
    # Define the model
    print("Staring OptimizeGD")
    X = tf.Variable(X_data, dtype=tf.float32)
    y = tf.Variable(y_data, dtype=tf.float32)
    theta = tf.Variable(tf.zeros([paramCount]))
    y_pred = tf.tensordot(tf.concat([tf.ones((X.shape[0], 1)), X], axis=1), theta, axes=1)

    # Define the cost function
    cost = tf.reduce_mean(tf.square(y_pred - y))

    # Define the optimizer
    optimizer = tf.optimizers.SGD(learning_rate=0.01)

    # Initialize a list to store the cost history
    cost_history = []

    print("OptimizeGD: Initiating loop")

    # Run the optimization algorithm
    for i in range(1000):
        with tf.GradientTape() as tape:
            y_pred = tf.tensordot(tf.concat([tf.ones((X.shape[0], 1)), X], axis=1), theta, axes=1)
            cost = tf.reduce_mean(tf.square(y_pred - y))
        grads = tape.gradient(cost, [theta])
        optimizer.apply_gradients(zip(grads, [theta]))
        # Record the cost at each iteration
        cost_history.append(cost.numpy())
    # Print the results
    print(f"Theta: {theta.numpy()}")
    #print(f"Cost history: {cost_history}")
    return theta.numpy().tolist()
