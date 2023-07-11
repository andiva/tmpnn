import tensorflow as tf


class LyapunovRegularizer(tf.keras.regularizers.Regularizer):
    '''Gaining the stability of zero solution in term of ode: sum(sigmoid(real(eigs(W[1]))))'''

    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return self.a * tf.reduce_sum(tf.sigmoid(tf.math.real(tf.linalg.eig(x)[0])))

    def get_config(self):
        return {'a': self.a}
