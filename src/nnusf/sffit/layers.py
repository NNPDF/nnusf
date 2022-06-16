import tensorflow as tf


class Chi2Layer(tf.keras.layers.Layer):
    def __init__(
        self,
        theory_grid,
        invcovmat,
        experimental_central_value,
        data_domain,
        training_data,
        **args
    ):
        super(Chi2Layer, self).__init__(**args)
        self.theory_grid = tf.keras.backend.constant(theory_grid)
        self.invcovmat = tf.keras.backend.constant(invcovmat)
        self.experimental_central_value = tf.keras.backend.constant(
            experimental_central_value
        )
        self.data_domain = data_domain
        self.training_data = training_data

    def call(self, inputs):
        if inputs.shape[1]:
            inputs = inputs[:, self.data_domain[0] : self.data_domain[1], :]
        predictions = tf.einsum("ijk,jk->j", inputs, self.theory_grid)
        distance = predictions - self.experimental_central_value
        tmp_dot = tf.tensordot(self.invcovmat, distance, axes=1)
        chi2 = tf.tensordot(distance, tmp_dot, axes=1)
        if self.training_data:
            tf.print("training")
            tf.print(chi2 / self.experimental_central_value.shape[0])
        else:
            tf.print("validation")
            tf.print(chi2 / self.experimental_central_value.shape[0])
        return chi2
