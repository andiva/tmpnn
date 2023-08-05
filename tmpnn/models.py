import tensorflow as tf
import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin, MultiOutputMixin

from .base import TMPNN


class TMPNNRegressor(TMPNN, RegressorMixin, MultiOutputMixin):
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


class TMPNNLogisticRegressor(TMPNN, ClassifierMixin, MultiOutputMixin):
    '''Taylor-mapped polynomial neural network multilabel logistic regressor - multioutput binary classifier'''

    _DEFAULT_LOSS='binary_crossentropy'

    def call(self, inputs):
        return tf.sigmoid(super().call(inputs))

    def fit(self, X, y,
            batch_size=None,
            verbose=None,
            epochs=None,
            validation_split=0,
            validation_data=None,
            class_weight=None,
            sample_weight=None,
            callbacks=None):
        '''
            y: 2D binary array
        '''
        self.classes_=[0,1]
        return super().fit(X, y,
            batch_size,
            verbose,
            epochs,
            validation_split,
            validation_data,
            class_weight,
            sample_weight,
            callbacks)

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

    def fit(self, X, y,
            batch_size=None,
            verbose=None,
            epochs=None,
            validation_split=0,
            validation_data=None,
            class_weight=None,
            sample_weight=None,
            callbacks=None):
        '''
            y: 1D array
        '''
        self.classes_, indices=np.unique(y, return_inverse=True)
        return super().fit(X, tf.keras.utils.to_categorical(indices, self.classes_.shape[0]),
            batch_size,
            verbose,
            epochs,
            validation_split,
            validation_data,
            class_weight,
            sample_weight,
            callbacks)

    def predict(self, X, batch_size=None, verbose=None):
        return np.take(self.classes_, np.argmax(super().predict(X, batch_size, verbose), -1))

    def predict_proba(self, X, batch_size=None, verbose=None):
        return super().predict(X, batch_size, verbose)

    def predict_log_proba(self, X, batch_size=None, verbose=None):
        return np.log(super().predict(X, batch_size, verbose))


class TMPNNClassifierPL(TMPNN, ClassifierMixin): #TODO
    '''Taylor-mapped polynomial neural network multiclass classifier based on Picard-Lindelöf theorem'''
    def call(self, inputs):
        return super()._call_full(inputs)