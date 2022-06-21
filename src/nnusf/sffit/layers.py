import tensorflow as tf


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
