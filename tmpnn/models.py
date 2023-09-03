'''Task specifications for TMPNN

Regreessor, several Classifiers, metric learning Transformer
'''

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, MultiOutputMixin, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.utils.validation import column_or_1d

from ._triplet_loss import TripletSemiHardLoss
from .base import TMPNNEstimator


class TMPNNRegressor(TMPNNEstimator, RegressorMixin, MultiOutputMixin):
    '''Taylor-mapped polynomial neural network regressor'''

    def _more_tags(self):
        '''Get sklearn tags for the estimator'''
        return {
            "requires_y": True,
            "multioutput": True
        }

    def predict(self, X, batch_size=None, verbose=None):
        '''Predict values using the Taylor-mapped Polynomial Neural Network model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        batch_size : int, default=None
            Size of batches for vectorised computation.
            When set to None, 'auto' Keras batch_size applies.

        verbose : bool, int default=None
            local vebosity level for prediction.
            When set to None, global estimators verbose level applies.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs) or (n_samples,)
            The predicted values. ndim is the same as y's provided to fit.
        '''
        y = self._decision_function(X, batch_size, verbose)
        return np.reshape(np.array(y, np.float64), self._target_shape)


class TMPNNLogisticRegressor(TMPNNEstimator, ClassifierMixin, MultiOutputMixin):
    '''Taylor-mapped polynomial neural network multilabel logistic regressor - multioutput binary classifier'''

    _DEFAULT_LOSS = tf.keras.losses.BinaryCrossentropy(True)

    def _more_tags(self):
        '''Get sklearn tags for the estimator'''
        return {
            "binary_only": True,
            "multilabel": True,
            "requires_y": True,
            "multioutput": True
        }

    def _encode_y(self, y):
        out = np.zeros_like(y)
        if not hasattr(self, '_encoders') or not self.warm_start:
            self._encoders = []
            self.classes_ = []
            for i in range(y.shape[1]):
                lb = LabelBinarizer().fit(y[i])
                self._encoders.append(lb)
                self.classes_.append(lb.classes_)
                if lb.classes_.size > 2:
                    raise ValueError('Can manage only binary labels.')
        for i in range(y.shape[1]):
            out[i] = self._encoders[i].transform(y[i])
        return out

    def predict(self, X, batch_size=None, verbose=None):
        '''Predict labels using the Taylor-mapped Polynomial Neural Network model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        batch_size : int, default=None
            Size of batches for vectorised computation.
            When set to None, 'auto' Keras batch_size applies.

        verbose : bool, int default=None
            local vebosity level for prediction.
            When set to None, global estimators verbose level applies.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs) or (n_samples,)
            The predicted values. ndim is the same as y's provided to fit.
        '''
        y = self._decision_function(X, batch_size, verbose) > 0
        for i in range(y.shape[1]):
            y[i] = self._encoders[i].inverse_transform(y[i])
        return np.reshape(y, self._target_shape)

    def predict_proba(self, X, batch_size=None, verbose=None):
        y = self._decision_function(X, batch_size, verbose)
        y = np.array(tf.sigmoid(y))
        return np.reshape(y, self._target_shape)

    def predict_log_proba(self, X, batch_size=None, verbose=None):
        return np.log(self.predict_proba(X, batch_size, verbose))


class TMPNNClassifier(TMPNNEstimator, ClassifierMixin):
    '''Taylor-mapped polynomial neural network multiclass classifier'''

    _DEFAULT_LOSS = tf.keras.losses.CategoricalCrossentropy(True)

    def _more_tags(self):
        '''Get sklearn tags for the estimator'''
        return {
            "requires_y": True,
        }

    def _encode_y(self, y):
        column_or_1d(y, warn=True)
        if not hasattr(self, '_encoder') or not self.warm_start:
            self._encoder = OneHotEncoder()
            self._encoder.fit(y)
            self.classes_ = self._encoder.categories_
        return self._encoder.transform(y)

    def predict(self, X, batch_size=None, verbose=None):
        '''Predict classes the Taylor-mapped Polynomial Neural Network model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        batch_size : int, default=None
            Size of batches for vectorised computation.
            When set to None, 'auto' Keras batch_size applies.

        verbose : bool, int default=None
            local vebosity level for prediction.
            When set to None, global estimators verbose level applies.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs) or (n_samples,)
            The predicted values. ndim is the same as y's provided to fit.
        '''
        y = self._decision_function(X, batch_size, verbose)
        y = np.array(tf.nn.softmax(y)) > 0
        y = self._encoder.inverse_transform(y)
        return np.reshape(y, self._target_shape)

    def predict_proba(self, X, batch_size=None, verbose=None):
        y = self._decision_function(X, batch_size, verbose)
        y = np.array(tf.nn.softmax(y))
        return y

    def predict_log_proba(self, X, batch_size=None, verbose=None):
        return np.log(self.predict_proba(X, batch_size, verbose))


class TMPNNPLTransformer(TMPNNEstimator, TransformerMixin): #TODO
    '''Taylor-mapped polynomial neural network multiclass classifier based on Picard-Lindelöf theorem'''

    _DEFAULT_LOSS = TripletSemiHardLoss()
    _FULL_OUTPUT = True

    def _more_tags(self):
        '''Get sklearn tags for the estimator'''
        return {
            "requires_y": True
        }

    def transform(self, X, batch_size=None, verbose=None):
        '''Transforms X by Taylor-mapped Polynomial Neural Network model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        batch_size : int, default=None
            Size of batches for vectorised computation.
            When set to None, 'auto' Keras batch_size applies.

        verbose : bool, int default=None
            local vebosity level for prediction.
            When set to None, global estimators verbose level applies.

        Returns
        -------
        X : ndarray of shape (n_samples, n_outputs)
        '''
        return np.array(self._decision_function(X, batch_size, verbose), np.array(X).dtype)