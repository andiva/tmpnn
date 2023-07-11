import tensorflow as tf
import numpy as np
from .base import BaseTmPNN
from sklearn.base import ClassifierMixin

class TmPNNCLassifier(ClassifierMixin, BaseTmPNN):
    def __init__(self, 
                 order=2, 
                 steps=7, 
                 latent_units=0, 
                 weights_regularizer=None, 
                 solver='adam', 
                 batch_size=256, 
                 max_iter=1000, 
                 learning_rate_init=0.001, 
                 verbose=0, 
                 plot=False) -> None:
        super().__init__(
            order, 
            steps, 
            latent_units, 
            weights_regularizer,
            solver, 
            batch_size, 
            max_iter, 
            learning_rate_init, 
            verbose, 
            plot)

    def build(self, num_input, num_output=1, output_indexes=None) -> None:
        base = super().build(num_input, num_output, output_indexes)
        self.nn = tf.keras.layers.Softmax()(base)

    def fit(self, X, y):
        self.build()
        self.compile()
        self.train(X, y)

    def predict_proba(self, X):
        return self.call(X)
    
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=-1)