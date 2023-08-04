import tensorflow as tf

def get(identifier):
    if identifier=='lyapunov':
        return Lyapunov()
    return tf.keras.regularizers.get(identifier)

class Lyapunov(tf.keras.regularizers.L1):
    '''Special TMPNN regularizer, penalty large positive eigenvalues of the linear weights.
    L1(exp(-Real(Eigval(linear weight))))
    '''
    def __init__(self, alpha=0.001, **kwargs):
        super().__init__(alpha, **kwargs)

    def __call__(self, W):
        eigs = tf.exp(-tf.math.real(tf.eig(W[1 : 1 + W.shape[1]])[0]))
        return super().__call__(eigs)

    def get_config(self):
        return {'alpha': float(self.l1)}