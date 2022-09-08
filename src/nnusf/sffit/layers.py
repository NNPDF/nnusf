# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_probability as tfp


class ObservableLayer(tf.keras.layers.Layer):
    def __init__(self, theory_grid, **kwargs):
        self.theory_grid = tf.keras.backend.constant(theory_grid)
        super().__init__(**kwargs)

    def call(self, inputs):
        result = tf.einsum("ijk,jk->ij", inputs, self.theory_grid)
        return result


class GenMaskLayer(tf.keras.layers.Layer):
    def __init__(self, bool_mask, **kwargs):
        self.mask = bool_mask
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.boolean_mask(inputs, self.mask, axis=1)


class TheoryConstraint(tf.keras.layers.Layer):
    def call(self, inputs):
        unstacked_inputs = tf.unstack(inputs, axis=2)
        ones = tf.ones_like(unstacked_inputs[0])
        input_x_equal_one = tf.stack(
            [ones, unstacked_inputs[1], unstacked_inputs[2]], axis=2
        )
        return input_x_equal_one


class FeatureScaling(tf.keras.layers.Layer):
    def __init__(self, sorted_tr_data, kin_equal_spaced_targets, **kwargs):
        self.sorted_tr_data = sorted_tr_data
        self.kin_equal_spaced_targets = kin_equal_spaced_targets
        super().__init__(**kwargs)

    def __call__(self, inputs):
        unstacked_inputs = tf.unstack(inputs, axis=2)
        scaled_inputs = []
        for enum, kin_tensor in enumerate(unstacked_inputs):
            scaled_kin = tfp.math.interp_regular_1d_grid(
                kin_tensor,
                self.sorted_tr_data[:, enum].min().astype("float32"),
                self.sorted_tr_data[:, enum].max().astype("float32"),
                self.kin_equal_spaced_targets[enum].astype("float32"),
                axis=-1,
                fill_value="extrapolate",
                grid_regularizing_transform=None,
            )
            scaled_inputs.append(scaled_kin)
        return tf.stack(scaled_inputs, axis=2)
