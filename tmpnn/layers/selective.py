import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class Selective(Layer):
    '''Polynomial layers implementing Taylor mapping'''
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Selective, self).__init__(**kwargs)
        return

    def initial_weights_zeros(self):
        yield


    def build(self, input_shape):
        return


    def call(self, x, mask=None):
        return x[:, -self.output_dim:]


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
