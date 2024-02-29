'''Base classes and subclasses for TMPNNEstimator

Contains intercepr layers, taylormap layers,
keras realisation and sklearn wrapper.
'''
from dataclasses import dataclass, InitVar
from typing import Any, List, Dict, Tuple, Optional, Union
from collections.abc import Iterable
from enum import IntEnum

import numpy as np
import tensorflow as tf
from keras.saving import (
    serialize_keras_object,
    deserialize_keras_object,
    register_keras_serializable
)

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.special import comb
from scipy.linalg import pascal

from ._utils import tensorflow_random_state, check_sample_weight


# ------------------------------------------------------------------------------
# Estimating intercepts - initial values for integration
class InterceptMode(IntEnum):
    CONST = 0 # untrainable vector intercept
    GLOBAL = 1 # trainable vector intercept
    LOCAL = 2 # a trainable matrix, sample-denepdant intercept
    LOCAL_REG = 3 # a trainable matrix with regularized varience
    _REG_STRENGTH = 0.1 # evristic regularization strength for LOCAL_REG

class Intercept():
    '''Trainable initial integration value'''
    def __init__(self, size, guess = 0, mode = InterceptMode.CONST):
        if np.isscalar(guess):
            guess = [guess]*size
        elif np.shape(guess)[-1] > size:
            raise ValueError("Iintercept's size couldn't be",
                            "less than last dim of guess.")
        self.size = size
        self.guess = guess
        self.mode = mode
        self._weight = None

    def pop(self):
        '''Chechks size and return extra intercept if size > guess shape'''
        r_size = self.size - np.shape(self.guess)[-1]
        out = [self]
        if r_size > 0:
            self.size -= r_size
            out.append(Intercept(r_size))
        return out

    def build_weight(self, layer: tf.keras.layers.Layer):
        n_samples = layer._build_input_shape[0] if self.mode >= InterceptMode.LOCAL else 1
        weight = layer.add_weight(
            shape = (n_samples, self.size),
            initializer = lambda shape, dtype: tf.reshape(tf.constant([self.guess]*n_samples, dtype), shape),
            regularizer = None if self.mode != InterceptMode.LOCAL_REG else
                lambda i: InterceptMode._REG_STRENGTH *
                            tf.reduce_mean(tf.math.reduce_variance(i, 0)),
            trainable = self.mode
        )
        self._weight = weight
        return weight

    def detach(self):
        '''Return detuched intercept with value of current weight'''
        guess = self.guess if self._weight is None else \
            tf.constant(self._weight).numpy()
        return Intercept(self.size, guess, self.mode)

@register_keras_serializable('tmpnn')
class InterceptLayer(tf.keras.layers.Layer):
    '''Fits and adds initial intergration values
    schema : Dict :
        keys are in ['target','latent'],
        values are Lists of Intercepts(size, guess, mode),
            with `trainable = mode>=1` and `fit_local = mode>=2`.
        Intercepts from 'target' are stacked to the beggining,
                   from 'latent' - to the end.
        Features intercept is inbetween as (size, 0, InterceptMode.CONST).
    '''
    def __init__(self, schema: Dict[str, List[Intercept]], dtype=None) -> None:
        super().__init__(True, 'InterceptEstimator', dtype)
        self.schema = schema or {}
        self._intercepts = []
        self._fit_local = False
        # validate schema, compute sizes and self._fit_local
        self._target_schema, self._target_size = self._check_schema('target')
        self._latent_schema, self._latent_size = self._check_schema('latent')

    def _check_schema(self, key) -> \
        Tuple[List[Intercept], int]:
        '''Validate schema, compute its size and adjust self._locality.'''
        checked_schema, schema_size  = [], 0
        if key in self.schema:
            for intercept in self.schema[key]:
                schema_size += intercept.size
                self._fit_local |= intercept.mode >= InterceptMode.LOCAL
                checked_schema.extend(intercept.pop())
        return checked_schema, schema_size

    def build(self, input_shape):
        super().build(input_shape)
        # check shape
        n_features = input_shape[1] - self._target_size - self._latent_size
        if n_features < 0:
            raise ValueError("Intercept schema: ",
                "overall size of shema is bigger than feature dimension")
        # zero intercept on features
        features = [Intercept(n_features)]
        # build weights
        for intercept in self._target_schema + features + self._latent_schema:
            self._intercepts.append(intercept.build_weight(self))

    def call(self, inputs, training=False):
        local_caster = tf.ones((tf.shape(inputs)[0], 1), dtype=self.dtype)
        intercept = tf.concat([i * local_caster for i in self._intercepts], -1)
        if not training:
            intercept = tf.reduce_mean(intercept, 0)
        return inputs + intercept

    def get_intercept(self) -> Dict[str, List[Intercept]]:
        '''Returns a fitted schema of self.schema format'''
        schema = {}
        if self._target_schema:
            schema['target'] = [i.detach() for i in self._target_schema]
        if self._latent_schema:
            schema['latent'] = [i.detach() for i in self._latent_schema]
        return schema

    def _get_intercept_config(self, intercept: Intercept):
        i = intercept.detach()
        config = {
            'size': i.size,
            'guess': i.guess,
            'mode': i.mode
        }
        return config

    def get_config(self):
        schema = {}
        if self._target_schema:
            schema['target'] = [self._get_intercept_config(i) for i in self._target_schema]
        if self._latent_schema:
            schema['latent'] = [self._get_intercept_config(i) for i in self._latent_schema]
        config = {
            'schema': schema,
            'dtype': self.dtype
        }
        return config

    @classmethod
    def from_config(cls, config):
        schema = {}
        for k, v in config['schema'].items():
            _schema = []
            for i in v:
                _schema.append(Intercept(**i))
            schema[k] = _schema
        return cls(schema, config['dtype'])

# ------------------------------------------------------------------------------
# Stability regularizer for Taylor Map
@register_keras_serializable('tmpnn')
class Lyapunov(tf.keras.regularizers.Regularizer):
    '''Special TMPNN regularizer, ensure the equilibrium solutions stability.

    penalty = sum(exp(-Real(Eigval(linear weight))))
    '''
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, W):
        eigs = tf.exp(tf.math.real(tf.eig(W[1 : 1 + W.shape[1]] - tf.eye(W.shape[1]))[0]))
        return self.alpha * tf.reduce_sum(eigs)

    def get_config(self):
        return {'alpha': self.alpha}

# Taylor Maps
@register_keras_serializable('tmpnn')
class TaylorMap(tf.keras.layers.Layer):
    '''Polynomial layer implementing Taylor mapping.

    TaylorMap(x) = x*residual + poly(x) @ W

    Note, this class is slow.
    '''
    def __init__(self,
            degree,
            regularizer=None,
            initializer=None,
            residual=True,
            guess=None,
            dtype=None):
        super().__init__(True, 'TaylorMap', dtype)
        self.degree = degree
        self.initializer = initializer or 'zeros'
        self.regularizer = \
            Lyapunov() if str.lower(str(regularizer))=='lyapunov' \
            else regularizer
        self.residual = residual
        self.guess = guess

    def _build_shape(self, n_features):
        n_poly_features = comb(n_features + self.degree, self.degree, True)
        self._pascal = pascal(max(self.degree, n_features)) \
            [1 : self.degree, :n_features].tolist()
        return (n_poly_features, n_features)

    def build(self, input_shape):
        super().build(input_shape)
        n_features = input_shape[-1]
        weight_shape = self._build_shape(n_features)

        self._W = self.add_weight('W',
            shape=weight_shape,
            initializer=self.initializer if self.guess is None else
                lambda shape, dtype: tf.reshape(tf.constant(self.guess, dtype), shape),
            regularizer=self.regularizer
        )
        if not self.residual:
            self._I = tf.concat((
                tf.zeros((1, n_features)),
                tf.eye(n_features),
                tf.zeros((weight_shape[0] - n_features - 1, n_features))
            ), 0)
            self._W.assign_add(self._I)

    def _poly(self, x):
        '''Tensorflow implementation of PolynomialFeatures.'''
        n_samples = tf.shape(x)[0]
        xP = [tf.ones((n_samples, 1), self.dtype), x] # degrees 0 and 1
        for indices in self._pascal: # iterate through degrees >= 2
            xP.append(tf.concat([
                x[:, feature_idx : feature_idx + 1] * xP[-1][:, :monomials_idx]
                for feature_idx, monomials_idx in enumerate(indices)
            ], -1))
        return tf.concat(xP, -1)

    def call(self, inputs):
        return inputs * self.residual + self._poly(inputs) @ self._W

    def get_coef(self):
        W = self._W if self.residual else self._W - self._I
        return tf.convert_to_tensor(W).numpy()

    def get_config(self):
        config = {
            'degree': self.degree,
            'regularizer': serialize_keras_object(self.regularizer),
            'initializer': serialize_keras_object(self.initializer),
            'residual': self.residual,
            'guess': self.guess,
            'dtype': self.dtype
        }
        return config

class FastTaylorMap(TaylorMap):
    '''Faster TaylorMap via duplicated monomials'''
    def _build_shape(self, n_features):
        n_poly_features = tf.math.reduce_sum(
            tf.pow([n_features]*(self.degree+1), tf.range(self.degree+1)))
        return (n_poly_features, n_features)

    def _poly(self, x):
        n_samples = tf.shape(x)[0]
        xP = [tf.ones((n_samples, 1), self.dtype), x] # degrees 0 and 1
        x = tf.expand_dims(x, -1)
        for _ in range(2, self.degree + 1): # iterate through degrees >= 2
            xP.append(tf.reshape(x @ tf.expand_dims(xP[-1],1), (n_samples,-1)))
        return tf.concat(xP, -1)

class ScaledTaylorMap(FastTaylorMap):
    '''FastTaylorMap with feature scaling before mul to avoid Nan. Slow.'''
    def _poly(self, x):
        n_samples = tf.shape(x)[0]
        xP = [tf.ones((n_samples, 1), self.dtype), x] # degrees 0 and 1
        x = tf.expand_dims(x, -1)
        for degree in range(2, self.degree + 1): # iterate through degrees >= 2
            x_ = tf.pow(tf.abs(x), 1/degree) * tf.sign(x)
            xP_1_ = tf.pow(tf.abs(xP[-1]), 1-1/degree) * tf.sign(xP[-1])
            xP.append(tf.reshape(x_ @ tf.expand_dims(xP_1_,1), (n_samples,-1)))
        return tf.concat(xP, -1)

# ------------------------------------------------------------------------------
# Taylor Mapped Polynomial Neural Network
@register_keras_serializable('tmpnn')
class TMPNN(tf.keras.Model):
    '''Keras-style TMPNN core class. Does NOT preprocess features.

    Note, TaylorMap weight = guess_coef / steps. Guess_coef stans for ODE coef.
    '''
    def __init__(self,
            n_targets=1,
            steps=7,
            # intercept
            intercept_schema: Dict[str, List[Intercept]] = None,
            # interlayer
            interlayer=None,
            # taylormap
            scaled=False,
            degree=2,
            regularizer=None,
            initializer=None,
            residual=True,
            guess_coef=None,
            # experiment
            custom_map=None,
            # activation
            activation=None,
            # tensorflow type
            dtype=None
            ):
        super().__init__(dtype=dtype, name='TMPNN')
        self.n_targets = n_targets
        self.steps = steps
        # intercept
        self._intercept = InterceptLayer(intercept_schema, dtype)
        # interlayer
        interlayer = interlayer or tf.keras.layers.Identity()
        clsname = interlayer.__class__.__name__
        config = interlayer.get_config()
        self._interlayers = [tf.keras.layers.deserialize(
                {'class_name':clsname,'config':config}
            ) for _ in range(self.steps)]
        # taylor map
        guess = None if guess_coef is None else guess_coef / steps
        tm_args = (degree, regularizer, initializer, residual, guess, dtype)
        self._taylormap = \
            ScaledTaylorMap(*tm_args) if scaled else FastTaylorMap(*tm_args) \
            if not custom_map else custom_map
        # activation
        self._activation = activation or tf.keras.layers.Identity()

    def call(self, inputs, training=False):
        x = self._intercept(inputs, training=training)
        for interlayer in self._interlayers:
            x = interlayer(x, training=training)
            x = self._taylormap(x)
        x = x[:,:self.n_targets]
        x = self._activation(x, training=training)
        return x

    def get_coef(self):
        return self._taylormap.get_coef() * self.steps

    def get_intercept(self):
        return self._intercept.get_intercept()

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_targets': self.n_targets,
            'steps': self.steps,
            'dtype': self.dtype,
            'intercept': serialize_keras_object(self._intercept),
            'taylormap': serialize_keras_object(self._taylormap),
            'activation': serialize_keras_object(self._activation)
        })
        config.update({
            'interlayer'+str(i): serialize_keras_object(
                self._interlayers[i]) for i in range(self.steps)
        })
        return config

    @classmethod
    def from_config(cls, config):
        obj = cls(config['n_targets'], config['steps'], dtype=config['dtype'])
        obj._intercept = deserialize_keras_object(config['intercept'])
        obj._taylormap = deserialize_keras_object(config['taylormap'])
        obj._activation = deserialize_keras_object(config['activation'])
        for i in range(obj.steps):
            obj._interlayers[i] = deserialize_keras_object(config['interlayer'+str(i)])
        return obj

@dataclass
class TMPNNPrepocessor(BaseEstimator, TransformerMixin):
    '''Feature preprocessor for TMPNN
    Pads inputs with estimators outputs and needed amount of zeros.
    Estimators normally should represent feature engineering or estimation intercepts.
    '''
    target_features: None | List[int] | Tuple[int] = None
    latent_units: int = 0
    estimators: Any = None # can't be pickled if functions

    def _call(self, estimator, X):
        if hasattr(estimator, 'predict'):
            estimated = getattr(estimator, 'predict')(X)
        elif hasattr(estimator, 'transform'):
            estimated = getattr(estimator, 'transform')(X)
        elif callable(estimator):
            estimated = estimator(X)
        else:
            raise ValueError("Estimator can't be called.")
        return np.reshape(estimated, (np.shape(X)[0], -1))

    def fit(self, X, y):
        # shapes
        self.n_features_in_ = n_featuers = np.shape(X)[1]
        r_targets = np.shape(y)[1] - len(self.target_features or [])
        # padding
        self._delegate = []
        n_estimated = 0
        if self.estimators:
            self._estimators = self.estimators if isinstance(self.estimators, Iterable) else [self.estimators]
            for estimator in self._estimators:
                n_estimated += np.shape(self._call(estimator, X[:10]))[1]
        else:
            self._estimators = []
        self.n_zeros = r_targets + self.latent_units - n_estimated
        # ordering
        targets = np.array([n_featuers + i for i in range(r_targets)] + (self.target_features or []))
        self.n_outputs_ = n_featuers + r_targets + self.latent_units
        self._permutation = np.hstack([targets, np.setdiff1d(np.arange(self.n_outputs_), targets)])

        self._rt_ne = r_targets - n_estimated # feature for intercepts initialization
        return self

    def transform(self, X):
        if np.shape(X)[1] != self.n_features_in_:
            raise ValueError("Number of features is different from that in fit")
        Z = [X] + [self._call(estimator, X) for estimator in self._estimators]
        if self.n_zeros > 0:
            Z.append(np.zeros((np.shape(X)[0], self.n_zeros)))
        return np.hstack(Z)[:, self._permutation]

@dataclass
class TMPNNEstimator(BaseEstimator):
    '''Estimator wrapping for Taylor Map Polynomial Neural Network

    This model optimizes the squared error using LBFGS or stochastic gradient
    descent.

    Parameters
    ----------
    degree : int, defualt=2

    steps : int, defualt=7

    latent_units: int, defualt=0

    target_features: list of int, defualt=None
        Features indices to chose as target initial values.

    guess_target_intercept : int, array, defualt=0
        value to initialize intercept for targets

    fit_target_intercept : bool, int, defualt=0
        0 - const, 1 - fit global value,
        2 - fit local values, 3 - fit local shrinked values.

    guess_latent_intercept : int, array, defualt=0
        value to initialize intercept for targets

    fit_latent_intercept : bool, int, defualt=0
        0 - const, 1 - fit global value,
        2 - fit local values, 3 - fit local shrinked values.

    intercept_estimators: list of estimators, default=None
        Estimators or callables to estimate intercepts from X.

    scaled: bool, default=False
        Prescale features before compute polynomials
        to avoid Nans, change family of approximator functions.

    regularizer: Keras regulariszer, default=None
        Regularizer for TaylorMap weight and final coef.

    initializer: Keras initializer, default=None
        When set to None, zeros initialization is used for coef.

    residual: bool, default=True
        Compute TaylorMap as residual layer if true,
        Init TaylorMap.weight with Eye if false.
        Influence only on regularization.

    guess_coef: array, default=None
        Initial value for coef.
        When set to None, initializer is used.

    interlayer: Keras layer, default=None
        Layer to go through before TaylorMap.
        BatchNorm, Dropout, any other norm, etc.
        Whet set to None, Identity is used.

    intercept_schema: dict of lists of Intercept objects, default=None
        Is used to initialize intercept directly.
        When set, guess_..._intercept and fit_..._intercept is ignored.

    custom_map : Keras Layer, default=None
        Layer to replace taylormap for experiments.

    optimiser : Keras optimizer, default='adamax'

    max_epochs : int, default=100
        Maximum number of iterations. This determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.

    verbose : bool, int, str, default=False
        Whether to print progress messages to stdout.

    loss : Keras loss, defualt=None
        Custom loss. When set to None, task-dependant default value.

    metrics : Keras metrics, default=None
        Metrics to compute during training

    weighted_metrics : Keras weighted metrics, default=None
        Metrics weigthed with sample_weight

    shuffle : bool, default=True
        Whether to shuffle samples in each iteration.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the
        previous solution.

    patience : int, default=None
        Maximum number of epochs to not meet improvement before earlystopping.
        If set to None, equal to max_epochs

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias
        initialization, validation split, and batch
        sampling.
        Pass an int for reproducible results across multiple function calls.

    dtype: str, np.dtype, tf.dtype, default=None
        dtype of model's weights

    Attributes
    ----------
    history_ : dict with losses and metrics history both training and validation:
        'loss', 'val_loss', etc. each histury is list of shape (n_epoches,)

    coef_ : array of shape (n_poly_features, n_features_in_)
        Represents the coef of an internal ODE, which depends of TaylorMap weight
        as coef_ = (Taylor map weight - Eye * residual) * steps

    intercept_schema_ : dict of lists of Intercepts for target dimensions and
        latent dimensions. Intercept.guess is its value.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    TMPNNRegressor : Taylor-mapped PNN regressor.
    TMPNNLogisticRegressor : Multioutput Taylor-mapped PNN binary classifier.
    TMPNNClassifier : Taylor-mapped PNN multiclass classifier.
    TMPNNPLTransformer : Taylor-mapped PNN transformer for classifiers
        based on Picard-Lindelöf theorem.
    Lyapunov : regularizer penalting positive eigenvalues of linear weight,
        increases stability of equallibrium solutions

    Notes
    -----
    TMPNN trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.

    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.

    This implementation works with data represented as dense numpy
    arrays of floating point values.

    '''
    # major
    degree: int = 2
    steps: int = 7
    # prepocessor and intercept
    latent_units: int = 0
    target_features: Optional[List[int]] = None
    guess_target_intercept: Any = 0
    fit_target_intercept: int = 0
    guess_latent_intercept: Any = 0
    fit_latent_intercept: int = 0
    intercept_estimators: Any = None
    # taylormap
    scaled: bool = False
    regularizer: Any = None
    initializer: Any = None
    residual: bool = True
    guess_coef: Any = None
    interlayer: Any = None
    activation: Any = None
    # other
    intercept_schema: Dict[str, List[Intercept]] = None
    custom_map: Any = None
    optimizer: Any = None
    loss: Any = None
    metrics: Any = None
    weighted_metrics: Any = None
    max_epochs: int = 100
    verbose: str|int = 0
    shuffle: bool = True
    warm_start: bool = False
    patience: Optional[int] = None
    random_state: Optional[Union[int,np.random.RandomState]] = None
    dtype: Any = None
    # default class params to specify task
    _DEFAULT_LOSS = 'mse'
    _FULL_OUTPUT = False

    def _more_tags(self):
        '''Get sklearn tags for the estimator'''
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": "TMPNN as Deep Learning model isn't invariant to data shuffling.",
                "check_estimators_pickle": "Optimizers fail to pickle in darwin OS.",
                "check_methods_sample_order_invariance": "Numerical error is to big due to high polynomial order.",
                "check_methods_subset_invariance": "Numerical error is to big due to high polynomial order.",
                "check_classifiers_one_label": "To be handled."
            }
        }

    def _init(self, X, y):
        '''Initialise inner TMPNN and preprocessor'''
        self.n_features_in_ = np.shape(X)[1]
        self.n_target_in_ = np.shape(y)[1]

        self._preprocessor = TMPNNPrepocessor(
            target_features=self.target_features,
            latent_units=self.latent_units,
            estimators=self.intercept_estimators).fit(X, y)

        intercept_schema = self.intercept_schema or {
            'target': [Intercept(max(0, self._preprocessor._rt_ne),
                self.guess_target_intercept, self.fit_target_intercept)],
            'latent': [Intercept(self.latent_units + min(0, self._preprocessor._rt_ne),
                self.guess_latent_intercept, self.fit_latent_intercept)]
        }
        self._model = TMPNN(
            n_targets=self._preprocessor.n_outputs_ if self._FULL_OUTPUT else self.n_target_in_,
            steps=self.steps,
            intercept_schema=intercept_schema,
            interlayer=self.interlayer,
            scaled=self.scaled,
            degree=self.degree,
            regularizer=self.regularizer,
            initializer=self.initializer,
            residual=self.residual,
            guess_coef=self.guess_coef,
            custom_map=self.custom_map,
            activation=self.activation,
            dtype=self.dtype)
        self._model.compile( # TODO: recompile if warm_start and losses or optimizers changed
            optimizer = self.optimizer or 'adamax',
            loss = self.loss or self._DEFAULT_LOSS,
            metrics = self.metrics,
            weighted_metrics = self.weighted_metrics,
        )

    def _fit_tmpnn(self, X, y,
            batch_size=None,
            epochs=None,
            validation_split=0,
            validation_data=None,
            class_weight=None,
            sample_weight=None
            ):
        '''Adjust fitting params and fits TMPNN with uncontrolled randomness'''
        X, y, sample_weight = check_sample_weight(X, y, sample_weight)

        callbacks = [tf.keras.callbacks.TerminateOnNaN()]
        if validation_data or validation_split:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                patience=self.patience if self.patience is not None else self.max_epochs,
                restore_best_weights=True))
        # if hasattr(self, '_history_callback') and self.warm_start:
        #     callbacks.append(self._history_callback)

        if validation_data:
            X_vl, y_vl = validation_data
            X_vl = self._preprocessor.transform(X_vl)
            validation_data = (X_vl, y_vl)

        self._history_callback = self._model.fit(
            self._preprocessor.transform(X), y,
            batch_size=batch_size if not self._model._intercept._fit_local else X.shape[0],
            epochs=epochs or self.max_epochs,
            verbose=self.verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=self.shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight
        )
        return self._history_callback.history

    def _encode_y(self, y):
        return check_array(y)

    def fit(self, X, y,
            batch_size=None,
            epochs=None,
            validation_split=0,
            validation_data=None,
            class_weight=None,
            sample_weight=None
            ):
        '''Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        batch_size : int, default=None
            Size of minibatches for stochastic optimizers.
            When set to None, 'auto' Keras batch_size applies.

        epochs : int, default=None
            Number of iterations. This determines the number of epochs
            (how many times each data point will be used), not the number of
            gradient steps.
            When set to None, self.max_epochs is used.

        validation_split : float, default=0
            The proportion of training data to set aside as validation set for
            early stopping. Must be between 0 and 1.

        validation_data : tuple(X, y), default=None
            Validation samples.

        class_weight : array, default=None
            Loss weights for classes.

        sample_weight : array, default=None
            Loss weights for samples.

        Returns
        -------
        self : object
            Returns a trained TMPNNEstimator.
        '''
        # check inputs
        X, y = check_X_y(X, y, multi_output=True)
        y_shape = list(np.shape(y))
        y_shape[0] = -1
        self._target_shape = y_shape
        y_to_shape = (-1, 1 if len(y_shape) == 1 else y_shape[1])
        y = np.reshape(y, y_to_shape)
        y = self._encode_y(y)

        if validation_data:
            X_vl, y_vl = check_X_y(*validation_data, multi_output=True, y_numeric=True)
            y_vl = np.reshape(y_vl, y_to_shape)
            y_vl = self._encode_y(y_vl)
            validation_data = (X_vl, y_vl)

        # initialize inner models
        if not hasattr(self, '_model') or not self.warm_start:
            self._init(X, y)
        else:
            pass # TODO: check if X y shapes matches fitted shapes

        # fit inner models with controlled randomness
        if self.random_state is not None:
            with tensorflow_random_state(self.random_state):
                history = self._fit_tmpnn(X, y, batch_size, epochs,
                    validation_split, validation_data,
                    class_weight, sample_weight)
        else:
            history = self._fit_tmpnn(X, y, batch_size, epochs,
                validation_split, validation_data,
                class_weight, sample_weight)

        self.y_mean_ = np.mean(y, 0)
        self.history_ = history
        self.coef_ = self._model.get_coef()
        self.intercept_schema_ = self._model.get_intercept()

        return self

    def _decision_function(self, X, batch_size=None, verbose=None):
        check_is_fitted(self, 'history_')
        X = check_array(X)
        X = self._preprocessor.transform(X)
        y = self._model.predict(X,
            batch_size if not self._model._intercept._fit_local else X.shape[0],
            verbose or self.verbose)
        #fillna
        inds = np.where(np.isnan(y))
        y[inds] = np.take(self.y_mean_, inds[1])
        return y