import tensorflow as tf
from keras.regularizers import L1
from tensorflow.keras import backend as K
from tensorflow import linalg, math

# TODO: shutdown complex to real casting warnings

class Lyapunov1(L1):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(alpha, **kwargs)

    def __call__(self, W, x=None):
        eigs = math.sigmoid(math.real(linalg.eigvals(W[1])))
        return super().__call__(eigs)

    def get_config(self):
        return {"alpha": float(super().l1)}

class Lyapunov2(L1):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(alpha, **kwargs)

    def __call__(self, W, x):
        sx = tf.stack([x]*x.shape[1])
        sw = tf.transpose(tf.split(W[2], num_or_size_splits=x.shape[1], axis=0), [1,0,2])
        A = W[1] + linalg.diag_part(tf.transpose(K.dot(sx, sw), [1,3,0,2]))
        eigs = math.sigmoid(math.real(linalg.eigvals(A)))
        return super().__call__(eigs)

    def get_config(self):
        return {"alpha": float(super().l1)}