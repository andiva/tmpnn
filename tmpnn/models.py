'''Task specifications for TMPNN

Regreessor, several Classifiers, metric learning Transformer
!!!In Progress!!!
'''

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, MultiOutputMixin, TransformerMixin

from .base import TMPNNEstimator


class TMPNNRegressor(TMPNNEstimator, RegressorMixin, MultiOutputMixin):
    '''Taylor-mapped polynomial neural network regressor'''
    def fit(self, X, y, batch_size=None, epochs=None, validation_split=0, validation_data=None, sample_weight=None) -> BaseEstimator:
        return super().fit(X, y, batch_size, epochs, validation_split, validation_data, None, sample_weight)


class TMPNNLogisticRegressor(TMPNNEstimator, ClassifierMixin, MultiOutputMixin):
    '''Taylor-mapped polynomial neural network multilabel logistic regressor - multioutput binary classifier'''

    _DEFAULT_LOSS='binary_crossentropy'
    _ACTIVATION=tf.keras.layers.Lambda(tf.keras.activations.sigmoid)

    def predict(self, X, batch_size=None, verbose=None):
        return np.asfarray(super().predict(X, batch_size, verbose) > 0.5)

    def predict_proba(self, X, batch_size=None, verbose=None):
        return super().predict(X, batch_size, verbose)

    def predict_log_proba(self, X, batch_size=None, verbose=None):
        return np.log(super().predict(X, batch_size, verbose))


class TMPNNClassifier(TMPNNEstimator, ClassifierMixin):
    '''Taylor-mapped polynomial neural network multiclass classifier'''

    _DEFAULT_LOSS='categorical_crossentropy'
    _ACTIVATION=tf.keras.layers.Softmax()

    def predict(self, X, batch_size=None, verbose=None):
        return np.take(self.classes_, np.argmax(super().predict(X, batch_size, verbose), -1))

    def predict_proba(self, X, batch_size=None, verbose=None):
        return super().predict(X, batch_size, verbose)

    def predict_log_proba(self, X, batch_size=None, verbose=None):
        return np.log(super().predict(X, batch_size, verbose))


class TMPNNPLTransformer(TMPNNEstimator, TransformerMixin): #TODO
    '''Taylor-mapped polynomial neural network multiclass classifier based on Picard-Lindelöf theorem'''