import tensorflow as tf


class TaylorKronekerLayer(tf.keras.layers.Layer):
    '''Polynomial layer implementing Taylor mapping'''

    def __init__(self, order=2, weights_initializer='zeros', weights_regularizer: tf.keras.regularizers.Regularizer = None,
                 name=None, dtype=None, **kwargs):
        super().__init__(True, name, dtype, **kwargs)
        self.order = order
        self.initializer = weights_initializer
        self.regularizer = weights_regularizer

    def get_config(self):
        return {**super().get_config(), 'order': self.order, 'weights_initializer': self.initializer,
                'weights_regularizer': tf.keras.saving.serialize_keras_object(self.regularizer)}

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = [self.add_weight(None, (input_dim**i, input_dim), self.dtype, 
                            self.initializer, self.regularizer 
                            if i!=1 else lambda w: self.regularizer(w-tf.eye(input_dim)) #
                            ) for i in range(self.order + 1)]
        self.W[1].assign_add(tf.eye(input_dim)) #

    def call(self, inputs, is_last=False, *args, **kwargs): 
        outputs = self.W[0] + tf.keras.backend.dot(inputs, self.W[1]) # + inputs
        x_vectors = tf.expand_dims(inputs, -1)
        tmp = inputs
        for i in range(2, self.order + 1):
            xext_vectors = tf.expand_dims(tmp, -1)
            x_extend_matrix = tf.matmul(
                x_vectors, xext_vectors, adjoint_a=False, adjoint_b=True)
            tmp = tf.reshape(x_extend_matrix, [-1, inputs.shape[-1]**i])
            outputs = outputs + tf.keras.backend.dot(tmp, self.W[i])
        return outputs


if __name__ == '__main__':
    input = tf.constant([[1, 1], [1, 1], [1, 1]], dtype=tf.float32)
    layer = TaylorKronekerLayer(3, 'random_normal')
    layer.build(input.shape)
    print(layer.call(input))
