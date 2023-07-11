import tensorflow as tf


class InsertiveLayer(tf.keras.layers.Layer):
    '''Layer implementing latent units a.k.a. ZeroPadding0D'''

    def __init__(self, latent_units=1, name=None, dtype=None, **kwargs):
        super().__init__(False, name, dtype, **kwargs)
        self.pad = tf.constant([[0, 0], [0, latent_units]])

    def get_config(self):
        return {**super().get_config(), 'latent_units': self.pad[1,1]}

    def call(self, inputs, *args, **kwargs):
        return tf.pad(inputs, self.pad)


if __name__ == '__main__':
    print(InsertiveLayer(2).call(tf.constant(
        [[1, 1], [1, 1], [1, 1]], dtype=tf.float32)))
