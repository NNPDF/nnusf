import tensorflow as tf

class Chi2Layer(tf.keras.layers.Layer):
    def __init__(self, theory_grid, invcovmat, experimental_central_value):
        super(Chi2Layer, self).__init__()
        self.theory_grid = tf.keras.backend.constant(theory_grid)
        self.invcovmat = tf.keras.backend.constant(invcovmat)
        self.experimental_central_value = tf.keras.backend.constant(
            experimental_central_value
        )

    def call(self, input):
        predictions = tf.einsum("ijk,jk->ij", input, self.theory_grid)
        distance = predictions - self.experimental_central_value
        tmp_dot = tf.tensordot(self.invcovmat, distance[0, :], axes=1)
        chi2 = tf.tensordot(distance[0, :], tmp_dot, axes=1)
        ndat = self.theory_grid.shape[0]
        tf.print(tf.math.reduce_mean(distance))
        return chi2 / ndat
