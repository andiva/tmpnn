import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, MultiOutputMixin
from .base import TMPNN

class TMPNNRegressor(TMPNN, BaseEstimator, RegressorMixin, MultiOutputMixin):
    '''Taylor-mapped polynomial neural network regressor'''
    def fit(self, X, y,
            batch_size=None,
            verbose=None,
            epochs=None,
            validation_split=0,
            validation_data=None,
            sample_weight=None,
            callbacks=None):
        return super().fit(X, y,
            batch_size,
            verbose,
            epochs,
            validation_split,
            validation_data,
            None,
            sample_weight,
            callbacks)

class TMPNNLogisticRegressor(TMPNN, BaseEstimator, ClassifierMixin, MultiOutputMixin): #TODO
    '''Taylor-mapped polynomial neural network multilabel logistic regressor - multioutput binary classifier'''
    def call(self, inputs):
        return tf.sigmoid(super().call(inputs))

class TMPNNClassifier(TMPNN, BaseEstimator, ClassifierMixin): #TODO
    '''Taylor-mapped polynomial neural network multiclass classifier'''
    def call(self, inputs):
        return tf.nn.softmax(super().call(inputs))

class TMPNNClassifierPL(TMPNN, BaseEstimator, ClassifierMixin): #TODO
    '''Taylor-mapped polynomial neural network multiclass classifier based on Picard-Lindelöf theorem'''
    def call(self, inputs):
        return super()._call_full(inputs)