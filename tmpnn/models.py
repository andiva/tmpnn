'''Task specifications for TMPNN

Regreessor, several Classifiers, metric learning Transformer

!!!In Progress!!!
'''

import tensorflow as tf
import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin, MultiOutputMixin, TransformerMixin

from .base import TMPNN


class TMPNNRegressor(TMPNN, RegressorMixin, MultiOutputMixin):
    '''Taylor-mapped polynomial neural network regressor'''
    def fit(self, X, y,
            batch_size=None,
            epochs=None,
            verbose=None,
            callbacks=None,
            validation_split=0,
            validation_data=None,
            sample_weight=None):
        return super().fit(X, y,
            batch_size,
            epochs,
            verbose,
            callbacks,
            validation_split,
            validation_data,
            None,
            sample_weight)


class TMPNNLogisticRegressor(TMPNN, ClassifierMixin, MultiOutputMixin):
    '''Taylor-mapped polynomial neural network multilabel logistic regressor - multioutput binary classifier'''

    _DEFAULT_LOSS='binary_crossentropy'

    def call(self, inputs):
        return tf.sigmoid(super().call(inputs))

    def predict(self, X, batch_size=None, verbose=None):
        return np.asfarray(super().predict(X, batch_size, verbose) > 0.5)

    def predict_proba(self, X, batch_size=None, verbose=None):
        return super().predict(X, batch_size, verbose)

    def predict_log_proba(self, X, batch_size=None, verbose=None):
        return np.log(super().predict(X, batch_size, verbose))


class TMPNNClassifier(TMPNN, ClassifierMixin):
    '''Taylor-mapped polynomial neural network multiclass classifier'''

    _DEFAULT_LOSS='categorical_crossentropy'

    def call(self, inputs):
        return tf.nn.softmax(super().call(inputs))

    def predict(self, X, batch_size=None, verbose=None):
        return np.take(self.classes_, np.argmax(super().predict(X, batch_size, verbose), -1))

    def predict_proba(self, X, batch_size=None, verbose=None):
        return super().predict(X, batch_size, verbose)

    def predict_log_proba(self, X, batch_size=None, verbose=None):
        return np.log(super().predict(X, batch_size, verbose))


class TMPNNClassifierPL(TMPNN, TransformerMixin): #TODO
    '''Taylor-mapped polynomial neural network multiclass classifier based on Picard-Lindelöf theorem'''
    def call(self, inputs):
        return super()._call_full(inputs)