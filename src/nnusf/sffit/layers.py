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
    def __init__(self, map_from, map_to, **kwargs):
        self.map_from = map_from
        self.map_to = map_to

        @tf.function
        def interpolation(inputs):
            unstacked_inputs = tf.unstack(inputs, axis=2)

            scaled_inputs = []
            for input_grid, map_from, map_to in zip(
                unstacked_inputs, self.map_from, self.map_to
            ):
                small_cond = input_grid < map_from.min()
                large_cond = input_grid > map_from.max()

                inter_cond = []
                y_inter_coefs = []
                for num in range(len(map_from) - 1):
                    x1 = map_from[num]
                    x2 = map_from[num + 1]
                    y1 = map_to[num]
                    y2 = map_to[num + 1]

                    y_inter_coefs.append([x1, x2, y1, y2])

                    geq_cond = input_grid >= x1
                    less_cond = input_grid < x2

                    inter_cond.append(tf.math.logical_and(geq_cond, less_cond))

                def y_inter(x, coefs):
                    return coefs[2] + (x - coefs[0]) * (coefs[2] - coefs[3]) / (
                        coefs[0] - coefs[1]
                    )

                res = input_grid
                for num in range(len(map_from) - 1):
                    res = tf.where(
                        inter_cond[num], y_inter(res, y_inter_coefs[num]), res
                    )

                res = tf.where(small_cond, y_inter(res, y_inter_coefs[0]), res)
                res = tf.where(large_cond, y_inter(res, y_inter_coefs[-1]), res)

                scaled_inputs.append(res)

            return tf.stack(scaled_inputs, axis=2)

        self.interpolation = interpolation

        super().__init__(**kwargs)

    def call(self, inputs):
        return self.interpolation(inputs)
