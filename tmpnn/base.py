import tensorflow as tf
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_sample_weight, check_X_y, check_array, check_is_fitted

from scipy.special import comb
from scipy.linalg import pascal

from . import regularizers


class TaylorMap(tf.keras.layers.Layer):
    '''Polynomial layer implementing Taylor mapping.

    TaylorMap(x) = x + poly(x) @ W

    Note, this class is slow.
    '''
    def __init__(self,
            degree,
            regularizer=None,
            initializer=None,
            dtype=None,
            name='TaylorMap',
            **kwargs):
        super().__init__(True, name, dtype, **kwargs)
        self.degree = degree
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        super().build(input_shape)
        n_features = input_shape[-1]
        n_poly_features = comb(n_features + self.degree, self.degree, True)
        self._pascal = pascal(max(self.degree, n_features))[1 : self.degree, :n_features].tolist()
        self.W = self.add_weight('W',
            shape=(n_poly_features, n_features),
            initializer=self.initializer,
            regularizer=regularizers.get(self.regularizer)
        )

    @tf.function
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

    @tf.function
    def call(self, x):
        '''x + poly(x) @ W'''
        return x + self._poly(x) @ self.W

class FastTaylorMap(TaylorMap):
    '''Faster TaylorMap via duplicated monomials'''
    def build(self, input_shape):
        super().build(input_shape)
        n_features = input_shape[-1]
        n_poly_features = tf.math.reduce_sum(tf.pow([n_features]*(self.degree+1), tf.range(self.degree+1)))
        self.W = self.add_weight('W',
            shape=(n_poly_features, n_features),
            initializer=self.initializer,
            regularizer=regularizers.get(self.regularizer)
        )

    @tf.function
    def _poly(self, x):
        n_samples = tf.shape(x)[0]
        xP = [tf.ones((n_samples, 1), self.dtype), x] # degrees 0 and 1
        x = tf.expand_dims(x, -1)
        for _ in range(2, self.degree + 1): # iterate through degrees >= 2
            xP.append(tf.reshape(x @ tf.expand_dims(xP[-1], 1), (n_samples, -1)))
        return tf.concat(xP, -1)

class ScaledTaylorMap(FastTaylorMap):
    '''FastTaylorMap with feature scaling before mul to avoid Nan. Slow though.'''
    @tf.function
    def _poly(self, x):
        n_samples = tf.shape(x)[0]
        xP = [tf.ones((n_samples, 1), self.dtype), x] # degrees 0 and 1
        x = tf.expand_dims(x, -1)
        for degree in range(2, self.degree + 1): # iterate through degrees >= 2
            x_ = tf.pow(tf.abs(x), 1/degree) * tf.sign(x)
            xP_1_ = tf.pow(tf.abs(xP[-1]), 1-1/degree) * tf.sign(xP[-1])
            xP.append(tf.reshape(x_ @ tf.expand_dims(xP_1_, 1), (n_samples, -1)))
        return tf.concat(xP, -1)


def _callable_intercept(value):
    '''Returns call/predict/transform method or None'''
    if hasattr(value, 'predict'):
        return getattr(value, 'predict')
    elif hasattr(value, 'transform'):
        return getattr(value, 'transform')
    elif callable(value):
        return value
    return None

def _const_intercept_initializer(value):
    '''Returns constant scalar/tensor initializer'''
    if np.isscalar(value):
        return tf.keras.initializers.Constant(value)
    else: # list, array, tensor
        def f(shape, dtype=None):
            v = tf.constant(value, dtype)
            if shape==v.shape:
                return v
            raise ValueError(f'guess_intercept shape {v.shape}. Expected {shape}.')
        return f

def _validate_sample_weight(X, y, sample_weight):
    '''Validate that the passed sample_weight and ensure it is a Numpy array.'''
    sample_weight = _check_sample_weight(sample_weight, X)
    if np.all(sample_weight == 0):
        raise ValueError(
            "No training samples had any weight; only zeros were passed in sample_weight."
            " That means there's nothing to train on by definition, so training can not be completed."
        )
    # drop any zero sample weights
    # this helps mirror the behavior of sklearn estimators
    # which tend to have higher precisions
    not_dropped_samples = sample_weight != 0
    return (X[not_dropped_samples], y[not_dropped_samples], sample_weight[not_dropped_samples])


class TMPNN(tf.keras.Model, BaseEstimator):
    '''Taylor-mapped polynomial neural network

    Steps times apply shared TaylorMap of given degree,
    Equally propagete features through fitted autonomous evolutionary operator,
    or integrates the internal ODE with Euler method.
    Results into a polynomial regreesion of order degree^steps.

    Note, TaylorMap weight = internal ODE coef / steps.
    '''

    def _more_tags(self):
        '''Get sklearn tags for the estimator'''
        return {
            "_xfail_checks": {
                "check_no_attributes_set_in_init": "Keras Model has default parameters.",
                "check_dict_unchanged": "Keras mutates parameters during predict.",
                "check_dont_overwrite_parameters": "Keras add some public paramentrs in fit.",

                "check_sample_weights_invariance": "Random state is not implemented.",
                "check_fit_idempotent": "Random state is not implemented.",

                "check_estimators_pickle": "Not Implemented."
            }
        }

    def __init__(self,
            degree=2,
            steps=7,
            latent_units=0,
            guess_latent_intercept=0, #TODO: callable
            fit_latent_intercept=False,
            target_features=None,
            guess_target_intercept=0, #TODO: callable
            fit_target_intercept=False,
            scaled_poly=False,
            guess_coef=None,
            initializer='zeros',
            regularizer=None,
            interlayer=None, #experimentral
            verbose='auto',
            max_epochs=1000,
            patience=None,
            warm_start=False,
            shuffle=True,
            solver='adamax',
            loss='mse',
            tftype=None):
        '''Init the parameters.

        Contatins no logic, only saves parametrs.

        Parameters
        ----------
        degree: Taylor mapping degree of the internal ODE.
            The model is linear if degree==1.

        steps: Depth of the network, #steps for integrating the internal ODE.
            Classical Polynomial model if steps==1.

        latent_units: Extra dimensions for overparametrization **excluding** targets.

        guess_latent_intercept:  Scalar | Vector | Estimator | Callable : Initial value for latent dimensions.
            If not scalar, should have shape (latent_units) #TODO: change to arb shape.

        fit_latent_intercept: If initial value for latent dimensions are trainable variables or not.
            if >= 2, intercept will be fitted locally for each sample and averaged for prediction.

        target_features: Indices to initial target values among features.
            If len(target_features) < n_targets extra dimensions will be appended and chose as initial targets.
            Note, feature columns specified as targets would be permuted as follows:
            (target_features, extra targets, features, latent units).
            Only first len(target_features) tergats can be specified with target_features.
            Otherwise permute columns in y by hand to move required targets to the beggining of the feature vector.

        guess_target_intercept: Scalar | Vector | Estimator | Callable : Initial value for extra target dimensions.
            If not scalar, should have shape (n_targets - len(target_features)).

        fit_target_intercept: If initial value for extra target dimensions are trainable variables or not.
            if >= 2, intercept will be fitted locally for each sample and averaged for prediction.

        scaled_poly: Scale X to sign(X)*pow(X, 1/degree) before calculating degree monomials.
            Helps to avoid Nans, but slow.

        guess_coef: Initial guess for coef of the internal ODE. If None, initializer will be used.

        initializer: Keras initializer for TaylorMap weight.

        regularizer: Keras regularizer for TaylorMap weight.

        interlayer: Keras BatchNorm or LayerNorm or any other layer. Identity if None. Experimental.

        verbose: Global verbosity level for the inner Keras model.

        max_epochs: Global maximum training rounds.

        patience: Patience for early stopping if validation data or split provided. max_epochs if None.

        warm_start: True will reuse results of previous fit runs. If False, weights will be reset to guess values or reinitialized.

        shuffle: True will shuffle training samples.

        solver: Keras optimizer. Learning rate and schedule can be specified in an optimizer constructor.
            #TODO: imlement separately

        loss: Keras loss.

        tftype: tf.dtype.
        '''
        self.degree=degree
        self.steps=steps
        self.latent_units=latent_units
        self.guess_latent_intercept=guess_latent_intercept
        self.fit_latent_intercept=fit_latent_intercept
        self.target_features=target_features
        self.guess_target_intercept=guess_target_intercept
        self.fit_target_intercept=fit_target_intercept
        self.scaled_poly=scaled_poly
        self.guess_coef=guess_coef
        self.initializer=initializer
        self.regularizer=regularizer
        self.interlayer=interlayer#experimentral
        self.verbose=verbose
        self.max_epochs=max_epochs
        self.patience=patience
        self.warm_start=warm_start
        self.shuffle=shuffle
        self.solver=solver
        self.loss=loss
        self.tftype=tftype

    def build(self, input_shape, output_shape):
        '''Build all inner params and build the keras model.

        Adjusts latents units and targets indices with given training data shapes,
        builds weights and sublayers at the beginning of the fitting routine,
        thus model shouldn't be called directly, only with fit().
        '''
        # get shapes
        n_samples = input_shape[0]
        self.n_features_in_ = input_shape[1]
        self.n_targets_, self._target_shape = (1, (-1,)) \
                if len(output_shape) < 2 else (output_shape[1], (-1, output_shape[1]))
        r_targets = self.n_targets_ - len(self.target_features or [])
        n_states = self.n_features_in_ + r_targets + self.latent_units
        # handle target dimensions ordering
        targets = np.array((self.target_features or []) + [self.n_features_in_ + i for i in range(r_targets)])
        self._permutation = np.hstack([targets, np.setdiff1d(np.arange(n_states), targets)])

        # get estimators' callables TODO: check shapes
        estimator_target_intercept = _callable_intercept(self.guess_target_intercept)
        estimator_latent_intercept = _callable_intercept(self.guess_latent_intercept)
        # create and compile self._pre_call(inputs) function
        self._pre_call_delegate=[]
        if estimator_target_intercept:
            self._pre_call_delegate.append(estimator_target_intercept)
        free_pad = \
            (0 if estimator_target_intercept else r_targets) + \
            (0 if estimator_latent_intercept else self.latent_units)
        if free_pad:
            paddings = np.zeros((free_pad))
            self._pre_call_delegate.append(lambda x: np.ones((x.shape[0], 1)) * paddings)
        if estimator_latent_intercept:
            self._pre_call_delegate.append(estimator_latent_intercept)

        # create weights for intercepts
        self._intercept = [tf.zeros((self.n_features_in_), self.dtype)]
        self._local_intecepts = [tf.zeros(input_shape, self.dtype)]
        if r_targets > 0:
            self._intercept.append(self.add_weight('target_intercept',
                (r_targets),
                initializer=_const_intercept_initializer(0 if estimator_target_intercept else self.guess_target_intercept),
                trainable=self.fit_target_intercept
            ))
            self._local_intecepts.append(self.add_weight('local_target_intercept',
                (n_samples, r_targets),
                initializer='zeros',
                regularizer='l2',
                trainable=self.fit_target_intercept>=2
            ))
        if self.latent_units > 0:
            self._intercept.append(self.add_weight('lanent_intercept',
                (self.latent_units),
                initializer=_const_intercept_initializer(0 if estimator_latent_intercept else self.guess_latent_intercept),
                trainable=self.fit_latent_intercept
            ))
            self._local_intecepts.append(self.add_weight('local_latent_intercept',
                (n_samples, self.latent_units),
                initializer='zeros',
                regularizer='l2',
                trainable=self.fit_latent_intercept>=2
            ))

        # build core layer
        taylor_map_args = (self.degree, self.regularizer, self.initializer, self.dtype)
        self._taylormap = ScaledTaylorMap(*taylor_map_args) if self.scaled_poly else FastTaylorMap(*taylor_map_args)
        if self.guess_coef:
            self._taylormap.W.assign(self.guess_coef/self.steps)
        # build interlayers - experimentral
        interlayer = self.interlayer or tf.keras.layers.Identity()
        clsname = interlayer.__class__.__name__
        config = interlayer.get_config()
        self._interlayers = [tf.keras.layers.deserialize({'class_name':clsname,'config':config}) for _ in range(self.steps)]
        # build the whole model
        self._build_input_shape = (n_samples, n_states)
        self.built = True

    def _pre_call(self, inputs):
        '''Perform untrainable feature preprocessing.
        Pads extra target and latent dimensions, calls callable intercept initializers,
        permutes state vector to move targets to the beginning.
        '''
        x = np.hstack([inputs] + [func(inputs) for func in self._pre_call_delegate])
        x = x[:,self._permutation]
        return x

    def call(self, inputs, training=False):
        '''Perform trainable actions and return prediction.
        _pre_call should be called before call.
        '''
        x = inputs + tf.concat(self._intercept, -1)
        if training and self._fit_local_intercept:
            x = x + tf.concat(self._local_intecepts, -1)
        for step in range(self.steps):
            x = self._interlayers[step](x, training=training)#experimentral
            x = self._taylormap(x)
        return x[:,:self.n_targets_]

    def fit(self, X, y,
            batch_size=None,
            epochs=None,
            verbose=None,
            callbacks=None,
            validation_split=0,
            validation_data=None,
            class_weight=None,
            sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)

        # (re)build network
        if not hasattr(self, 'history_') or not self.warm_start:
            super().__init__(dtype=self.tftype, name='TMPNN')
            self.build(X.shape, y.shape)
            self.compile(self.solver, self.loss)
        else:
            pass # TODO:check shapes

        # adjust args
        self._fit_local_intercept = self.fit_latent_intercept>=2 or self.fit_target_intercept>=2
        if self._fit_local_intercept:
            batch_size=X.shape[0]
            if verbose:
                print(f"Batch size set to {batch_size} to fit local intercepts")
        verbose = verbose if verbose is not None else self.verbose
        epochs = epochs or self.max_epochs
        if validation_data or validation_split:
            earlystopping = tf.keras.callbacks.EarlyStopping(
                patience=self.patience or epochs, restore_best_weights=True)
            callbacks = [earlystopping, tf.keras.callbacks.TerminateOnNaN()] + (callbacks or [])
        elif verbose:
            print("No early stopping will be performed, last training weights will be used.")
        X, y, sample_weight = _validate_sample_weight(X, y, sample_weight)

        # fit
        self.history_ = super().fit(
            self._pre_call(X),
            tf.reshape(y, (-1, self.n_targets_)),
            batch_size,
            epochs,
            verbose,
            callbacks,
            validation_split,
            validation_data,
            self.shuffle,
            class_weight,
            sample_weight
        ).history

        if (validation_data or validation_split) and verbose:
            print(f"Best weights from {earlystopping.best_epoch} epoch are restored.")

        # set public params
        self.coef_ = self._taylormap.W.numpy() * self.steps
        self.intercept_ = [self._intercept[i].numpy() for i in range(1, len(self._intercept))]
        self.local_intercepts_ = [(self._intercept[i] + self._local_intecepts[i]).numpy() for i in range(1, len(self._intercept))]

        return self

    def predict(self, X,
            batch_size=None,
            verbose=None):
        check_is_fitted(self)
        X = check_array(X)
        y = super().predict(
            self._pre_call(X),
            batch_size,
            verbose if verbose is not None else self.verbose
        )
        return np.reshape(y, self._target_shape)