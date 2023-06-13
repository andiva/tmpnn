import numpy as np


from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
import tensorflow as tf



class TaylorMap(Layer):
    '''Polynomial layers implementing Taylor mapping'''
    def __init__(self, output_dim, order=1, weights_regularizer = None, **kwargs):
        self.output_dim = output_dim
        self.order = order if order > 1 else 1 # first order at least
        self.weights_regularizer = weights_regularizer
        super(TaylorMap, self).__init__(**kwargs)
        return

    def initial_weights_zeros(self):
        ''' Returns [0, I, 0, 0, ...] '''
        n = 1
        self.num_monomials = [n]
        yield np.zeros((n, self.output_dim)) # return W0

        self.num_monomials.append(self.output_dim)
        yield np.eye(self.output_dim) # return W1

        for i in range(2, self.order+1):
            self.num_monomials.append(self.num_monomials[-1]*self.output_dim)
            yield np.zeros((self.num_monomials[-1], self.output_dim)) # return W2, ...


    def build(self, input_shape):
        input_dim = input_shape[1]
        if input_dim != self.output_dim:
            raise ValueError("input_dim and output_dim have to be equal for Taylor mapping")

        self.W = []
        for w in self.initial_weights_zeros():
            self.W.append(K.variable(w))

        self._trainable_weights = self.W
        return


    def call(self, x, mask=None):
        ans = self.W[0] + K.dot(x, self.W[1]) # first order at least
        x_vectors = tf.expand_dims(x, -1)
        tmp = x
        for i in range(2, self.order+1):
            xext_vectors = tf.expand_dims(tmp, -1)
            x_extend_matrix = tf.matmul(x_vectors, xext_vectors, adjoint_a=False, adjoint_b=True)
            tmp = tf.reshape(x_extend_matrix, [-1, self.num_monomials[i]])
            ans = ans + K.dot(tmp, self.W[i])

        if self.weights_regularizer:
            self.add_loss(self.weights_regularizer(self.W))

        return ans


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
