import tensorflow as tf
import numpy as np
from .base import BaseTmPNN
from sklearn.base import RegressorMixin

class TmPNNRegressor(RegressorMixin, BaseTmPNN):
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

    def fit(self, X, y):
        self.build()
        self.compile()
        self.train(X, y)

    def predict(self, X):
        return self.call(X)