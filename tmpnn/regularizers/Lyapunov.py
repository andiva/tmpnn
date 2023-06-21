import tensorflow as tf
from keras.regularizers import L1 as base1, L2 as base2
from keras import backend as K
from tensorflow import linalg, math

# TODO: shutdown complex to real casting warnings

class Lyapunov1(base1):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(alpha, **kwargs)

    def __call__(self, W, x=None):
        eigs = K.sigmoid(math.real(linalg.eigvals(W[1]))) # - tf.eye(W[1].shape[0]
        return super().__call__(eigs)

    def get_config(self):
        return {"alpha": float(super().l1)}

class Lyapunov2(base1):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(alpha, **kwargs)

    def __call__(self, W, x):
        sx = tf.stack([x]*x.shape[1])
        sw = tf.transpose(tf.split(W[2], num_or_size_splits=x.shape[1], axis=0), [1,0,2])
        A = W[1] + linalg.diag_part(tf.transpose(K.dot(sx, sw), [1,3,0,2])) # - tf.eye(W[1].shape[0]) 
        eigs = K.sigmoid(math.real(linalg.eigvals(A)))
        return super().__call__(eigs)

    def get_config(self):
        return {"alpha": float(super().l1)}