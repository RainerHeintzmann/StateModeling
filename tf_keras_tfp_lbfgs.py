# ! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

# This code is modified from the above script

"""An example of using tfp.optimizer.lbfgs_minimize to optimize a TensorFlow model.
This code shows a naive way to wrap a tf.keras.Model and optimize it with the L-BFGS
optimizer from TensorFlow Probability.
Python interpreter version: 3.6.9
TensorFlow version: 2.0.0
TensorFlow Probability version: 0.8.0
NumPy version: 1.17.2
Matplotlib version: 3.1.1
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot

def function_factory(loss, var_list): # , normFactors=None
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(var_list)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    # if normFactors == "mean":
    #     normFactors = []
    #     for var in var_list:
    #         nf = tf.reduce_mean(var).numpy()  # norm factors are NOT part of optimizations!
    #         if np.abs(nf) < 1e-5:
    #             nf = 1.0
    #         normFactors.append(nf)
    # elif normFactors == "max":
    #     normFactors = []
    #     for var in var_list:
    #         nf = tf.reduce_max(var).numpy()  # norm factors are NOT part of optimizations!
    #         if np.abs(nf) < 1e-5:
    #             nf = 1.0
    #         normFactors.append(nf)
    # elif normFactors is not None:
    #     normFactors = np.array(normFactors)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            # if normFactors is None:
            #     var_list[i].assign(tf.reshape(param, shape))
            # else:
            #     var_list[i].assign(tf.reshape(param, shape) * normFactors[i])
            var_list[i].assign(tf.reshape(param, shape))

    # @tf.function
    # def correctGradients(grads):
    #     if normFactors is not None:
    #         ngrads = []
    #         for i, grad in enumerate(grads):
    #             ngrads.append(grad*normFactors[i])
    #     else:
    #         ngrads = grads
    #     return ngrads

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """
        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        assign_new_model_parameters(params_1d)
        with tf.GradientTape() as tape:
            # update the parameters in the model
            # calculate the loss
            loss_value = loss()

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, var_list)
        # grads = correctGradients(grads)  # adjusts for the influence of the norm factors
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters

    init_list = var_list
    # if normFactors is None:
    #     init_list = var_list
    # else:
    #     init_list = []
    #     for i, var in enumerate(var_list):
    #         init_list.append(var / normFactors[i])
    f.initParams = lambda: tf.dynamic_stitch(idx, init_list)

    return f


def plot_helper(inputs, outputs, title, fname):
    """Plot helper"""
    pyplot.figure()
    pyplot.tricontourf(inputs[:, 0], inputs[:, 1], outputs.flatten(), 100)
    pyplot.xlabel("x")
    pyplot.ylabel("y")
    pyplot.title(title)
    pyplot.colorbar()
    pyplot.savefig(fname)


if __name__ == "__main__":
    # use float64 by default
    tf.keras.backend.set_floatx("float64")

    # prepare training data
    x_1d = np.linspace(-1., 1., 11)
    x1, x2 = np.meshgrid(x_1d, x_1d)
    inps = np.stack((x1.flatten(), x2.flatten()), 1)
    outs = np.reshape(inps[:, 0] ** 2 + inps[:, 1] ** 2, (x_1d.size ** 2, 1))

    # prepare prediction model, loss function, and the function passed to L-BFGS solver
    pred_model = tf.keras.Sequential(
        [tf.keras.Input(shape=[2, ]),
         tf.keras.layers.Dense(64, "tanh"),
         tf.keras.layers.Dense(64, "tanh"),
         tf.keras.layers.Dense(1, None)])

    loss_fun = tf.keras.losses.MeanSquaredError()
    func = function_factory(pred_model, loss_fun, inps, outs)

    # convert initial model parameters to a 1D tf.Tensor
    init_params = tf.dynamic_stitch(func.idx, pred_model.trainable_variables)

    # train the model with L-BFGS solver
    results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func, initial_position=init_params, max_iterations=500)

    # after training, the final optimized parameters are still in results.position
    # so we have to manually put them back to the model
    func.assign_new_model_parameters(results.position)

    # do some prediction
    pred_outs = pred_model.predict(inps)
    err = np.abs(pred_outs - outs)
    print("L2-error norm: {}".format(np.linalg.norm(err) / np.sqrt(11)))

    # plot figures
    plot_helper(inps, outs, "Exact solution", "ext_soln.png")
    plot_helper(inps, pred_outs, "Predicted solution", "pred_soln.png")
    plot_helper(inps, err, "Absolute error", "abs_err.png")
    pyplot.show()
