# -*- coding: utf-8 -*-
import tensorflow as tf


class ObservableLayer(tf.keras.layers.Layer):
    """constructs the observable by multiplying the NN structure
    functions with the corresponding coefficients.

    Parameters
    ----------
    theory_grid: np.ndarray[bool]
        A multidimensional array of coefficients
    """

    def __init__(self, theory_grid, **kwargs):
        self.theory_grid = tf.keras.backend.constant(theory_grid)
        super().__init__(**kwargs)

    def call(self, inputs):
        result = tf.einsum("ijk,jk->ij", inputs, self.theory_grid)
        return result


class GenMaskLayer(tf.keras.layers.Layer):
    """Apply a mask onto a given layer.

    Paramerters
    -----------
    bool_mask: np.ndarray[bool]
        numpy array of boolean masks
    """

    def __init__(self, bool_mask, **kwargs):
        self.mask = bool_mask
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.boolean_mask(inputs, self.mask, axis=1)


class TheoryConstraint(tf.keras.layers.Layer):
    """Stack ones to the input kinematics in order to enforce the
    constraint NN(x)-NN(1)=0.

    Parameters:
    -----------
    inputs: tf.constant
        input kinematics (x, Q^2, A)
    """

    def call(self, inputs):
        unstacked_inputs = tf.unstack(inputs, axis=2)
        ones = tf.ones_like(unstacked_inputs[0])
        input_x_equal_one = tf.stack(
            [ones, unstacked_inputs[1], unstacked_inputs[2]], axis=2
        )
        return input_x_equal_one
