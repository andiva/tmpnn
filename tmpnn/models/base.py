import tensorflow as tf
import sklearn.base
from ..layers import InsertiveLayer, TaylorKronekerLayer, SelectiveLayer

class BaseTmPNN(sklearn.base.BaseEstimator, tf.keras.Model):
    def __init__(self, 
                 order=2, 
                 steps=7, 
                 latent_units=0,
                 weights_regularizer: tf.keras.regularizers.Regularizer=None,
                 solver='adam', 
                 batch_size=256, 
                 max_iter=1000, 
                 learning_rate_init=1e-3, 
                 verbose=0, 
                 plot=False
                 ) -> None:
        self.order = order
        self.steps = steps
        self.latent_units = latent_units
        self.regularizer = weights_regularizer

        self.solver = solver
        self.batch_size = batch_size
        self.epochs = max_iter
        self.learning_rate_init = learning_rate_init

        self.verbose = verbose
        self.plot = plot

    def build(self, num_input, num_output=1, output_indexes=None) -> None:
        '''Builds the TmPNN as a keras model

        Parameters
        ----------
        num_input : int
            number of features, unpacked input_shape fot tabular data
        num_output : int, optional
            number of targets, by default 1
        ouput_indexes : optional
            columns indexes to peak as targets after propagation, 
            if not specified or len(ouput_indexes) < num_output the rest of indexes set to latent spaces
        '''
        pass

    def compile(self, *args, **kwargs):
        '''Compiles the keras model'''
        pass

    def train(self, X, y, epochs):
        '''Train the keras model'''
        pass

    def call(self, X):
        pass