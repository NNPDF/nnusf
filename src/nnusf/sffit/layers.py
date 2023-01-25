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


class SmallXPreprocessing(tf.keras.layers.Layer):
    """Add small-x preprocessing to the output of the NN"""

    def __init__(self, seed, dic_specs, **kwargs):
        self.output_dim = len(dic_specs.keys())
        self.seed = seed
        self.kernel = []
        self.dic_specs = dic_specs
        super().__init__(**kwargs)

    def generate_weights(self, sf_type):
        lower = self.dic_specs[sf_type][0]
        upper = self.dic_specs[sf_type][1]

        init = tf.keras.initializers.RandomUniform(
            minval=lower,
            maxval=upper,
            seed=self.seed + 1,
        )
        wconstr = tf.keras.constraints.MinMaxNorm(
            min_value=lower,
            max_value=upper,
        )

        kernel = self.add_weight(
            name=f"alpha_{sf_type}",
            shape=(1,),
            initializer=init,
            trainable=True,
            # constraint=wconstr,
        )
        self.kernel.append(kernel)

    def build(self, input_shape):
        for sf_type in self.dic_specs.keys():
            self.generate_weights(sf_type)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = tf.unstack(inputs, axis=2)[0]
        sfs_list = []
        for i in range(self.output_dim):
            sfs_list.append(x ** (1 - self.kernel[i][0]))
        sfs_exdims = [tf.expand_dims(i, axis=-1) for i in sfs_list]
        return tf.concat(sfs_exdims, axis=-1)
