import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# A high-dimensional quadratic bowl.
ndims = 60
minimum = np.ones([ndims], dtype='float64')
scales = np.arange(ndims, dtype='float64') + 1.0

start = np.arange(ndims, 0, -1, dtype='float64')
A = tf.Variable(initial_value=start)
B = tf.Variable(initial_value=start)
# The objective function and the gradient.
def predict():
    res = (2*A, 3*B)
    return res

def loss(x):
    A.assign(x[0])
    B.assign(x[1])
    return tf.reduce_sum(scales * tf.math.squared_difference(predict(), [minimum, minimum]))

def quadratic_loss_and_gradient(x):
    return tfp.math.value_and_gradient(loss, x)

optim_results = tfp.optimizer.lbfgs_minimize(
    quadratic_loss_and_gradient,
    initial_position=[start, start],
    num_correction_pairs=10,
    tolerance=1e-8)
#  quadratic_loss_and_gradient,

# Check that the search converged
assert(optim_results.converged)
# Check that the argmin is close to the actual value.
np.testing.assert_allclose(optim_results.position, minimum)


m = tf.Module
m.A = tf.Variable(initial_value=1, trainable=True)
m.B = tf.Variable(initial_value=2, trainable=True)
m.trainable_variables

m.C = m.A+m.B

@tf.function
def C():
    return m.A+m.B

