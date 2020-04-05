import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

ndims=10
start = np.arange(ndims, 0, -1, dtype='float64')

A = tf.Variable(initial_value=start, name='A')
B = tf.Variable(initial_value=start, name='B')

# @tf.function
def Loss():
    return A * B

# myDict = {'A':A,'B':B}
@tf.function
def myLoss():
    print('Hello World')
    return tf.reduce_sum(Loss())

opt = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(100):
    opt.minimize(loss=myLoss, var_list=[B])

#print(A.numpy())
#print(B.numpy())
