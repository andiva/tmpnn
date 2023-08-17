import tensorflow as tf
from sklearn.base import BaseEstimator
import numpy as np

from scipy.linalg import pascal

# from . import regularizers


class TaylorMap(tf.keras.layers.Layer):
    '''Polynomial layer implementing Taylor mapping'''
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
        n_poly_features = tf.math.reduce_sum(tf.pow([n_features]*(self.degree+1), tf.range(self.degree+1)))
        # from scipy.special import comb
        # n_poly_features = comb(n_features + self.degree, self.degree, True)
        self._pascal = pascal(max(self.degree, n_features))[1 : self.degree, :n_features].tolist()

        self.W = self.add_weight('W',
            shape=(n_poly_features, n_features),
            initializer=self.initializer,
            regularizer=regularizers.get(self.regularizer)
        )
        # self.I = tf.concat([
        #     tf.zeros((1, n_features)),
        #     tf.eye(n_poly_features-1, n_features)
        # ], 0)
        # self.W.assign_add(self.I)

    @tf.function
    def _poly(self, X):
        '''Tensorflow implementation of PolynomialFeatures.'''
        n_samples = tf.shape(X)[0]
        XP = [tf.ones((n_samples, 1)), X] # degrees 0 and 1
        for indices in self._pascal: # iterate through degrees >= 2
            XP.append(tf.concat([
                X[:, feature_idx : feature_idx + 1] * XP[-1][:, :monomials_idx]
                for feature_idx, monomials_idx in enumerate(indices)
            ], -1))
        return tf.concat(XP, -1)

    @tf.function
    def _poly_full(self, X):
        n_samples = tf.shape(X)[0]
        XP = [tf.ones((n_samples, 1)), X] # degrees 0 and 1
        X = tf.expand_dims(X, -1)
        for _ in range(2, self.degree + 1): # iterate through degrees >= 2
            XP.append(tf.reshape(X @ tf.expand_dims(XP[-1], 1), (n_samples, -1)))
        return tf.concat(XP, -1)

    @tf.function
    def call(self, X):
        return X + self._poly_full(X) @ self.W


class TMPNN(tf.keras.Model, BaseEstimator):
    '''Basic class for Taylor-mapped polynomial neural networks'''
    _DEFAULT_LOSS='mse'

    def __init__(self,
            degree=2,
            steps=7,
            latent_units=0,
            guess_latent_intercept=0, #TODO: callable
            fit_latent_intercept=False,
            target_features=None,
            guess_target_intercept=0, #TODO: callable
            fit_target_intercept=False,
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
            loss=None,
            dtype=None):
        '''
            Attention: TaylorMap weight = internal ODE coef / steps.

            Parameters
            ----------
            degree: Taylor mapping degree of the internal ODE.
                The model is linear if degree==1.

            steps: Depth of the network, #steps for integrating the internal ODE.
                Classical Polynomial model if steps==1.

            latent_units: Extra dimensions for overparametrization **excluding** targets.

            guess_latent_intercept:  Scalar | Vector | Estimator | Callable : Initial value for latent dimensions.
                If not scalar, should have shape (latent_units) #TODO: change to arb shape

            fit_latent_intercept: If initial value for latent dimensions are trainable variables or not.
                if >= 2, intercept will be fitted locally for each sample and averaged for prediction.

            target_features: Indices to initial target values among features.
                If len(target_features) < n_targets extra dimensions will be appended and chose as initial targets.
                Note, feature columns specified as targets would be permuted as follows:
                    target_features, extra targets, features, latent units
                Only first len(target_features) tergats can be specified with target_features.
                Otherwise permute columns in y by hand to move required targets to the beggining of the feature vector.

            guess_target_intercept: Scalar | Vector | Estimator | Callable : Initial value for extra target dimensions.
                If not scalar, should have shape (n_targets - len(target_features))

            fit_target_intercept: If initial value for extra target dimensions are trainable variables or not.
                if >= 2, intercept will be fitted locally for each sample and averaged for prediction.

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

            loss: Custom loss function. If None, task-dependant default value will be used.

            dtype: tf.dtype
        '''
        super().__init__(dtype=dtype, name='TMPNN')
        self.degree=degree
        self.steps=steps

        self.latent_units=latent_units
        self.guess_latent_intercept=guess_latent_intercept
        self.fit_latent_intercept=fit_latent_intercept

        self.target_features=target_features
        self.guess_target_intercept=guess_target_intercept
        self.fit_target_intercept=fit_target_intercept

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

    def build(self, input_shape, output_shape):
        '''
            Adjusts latents units and targets indices with given training data shapes,
            builds weights and sublayers at the beginning of the fitting routine,
            thus model shouldn't be called directly, only with fit().
        '''
        # get shapes
        n_samples = input_shape[0]
        self.n_features_ = input_shape[-1]
        self.n_targets_ = output_shape[-1]
        # handle target dimensions ordering
        r_targets = self.n_targets_ - len(self.target_features or [])
        n_states = self.n_features_ + r_targets + self.latent_units
        targets = np.array((self.target_features or []) + [self.n_features_ + i for i in range(r_targets)])
        self._permutation = np.hstack([targets, np.setdiff1d(np.arange(n_states), targets)])
        # get estimators' callables
        estimator_target_intercept = TMPNN._callable_intercept(self.guess_target_intercept)
        estimator_latent_intercept = TMPNN._callable_intercept(self.guess_latent_intercept)

        # create and compile self._pre_call(inputs) function
        self._pre_call_blocks=[]
        if estimator_target_intercept:
            self._pre_call_blocks.append(estimator_target_intercept)
        free_pad = \
            (0 if estimator_target_intercept else r_targets) + \
            (0 if estimator_latent_intercept else self.latent_units)
        if free_pad:
            paddings = tf.zeros(free_pad)
            self._pre_call_blocks.append(lambda x: tf.ones((x.shape[0], 1)) * paddings)
        if estimator_latent_intercept:
            self._pre_call_blocks.append(estimator_latent_intercept)

        # create weights for intercepts
        self._intercept = [tf.zeros((self.n_features_))]
        self._local_intecepts = [tf.zeros(input_shape)]
        if r_targets > 0:
            self._intercept.append(self.add_weight('target_intercept',
                (r_targets),
                initializer=TMPNN._const_intercept_initializer(0 if estimator_target_intercept else self.guess_target_intercept),
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
                initializer=TMPNN._const_intercept_initializer(0 if estimator_latent_intercept else self.guess_latent_intercept),
                trainable=self.fit_latent_intercept
            ))
            self._local_intecepts.append(self.add_weight('local_latent_intercept',
                (n_samples, self.latent_units),
                initializer='zeros',
                regularizer='l2',
                trainable=self.fit_latent_intercept>=2
            ))

        # build layers
        self._taylormap = TaylorMap(self.degree, self.regularizer, self.initializer, self.dtype)
        if self.guess_coef:
            self._taylormap.W.assign(self.guess_coef/self.steps)
        self._build_interlayer()#experimentral
        # build whole model
        super().build((n_samples, n_states))

    def _build_interlayer(self):#experimentral
        interlayer = self.interlayer or tf.keras.layers.Identity()
        clsname = interlayer.__class__.__name__
        config = interlayer.get_config()
        self._interlayers = [tf.keras.layers.deserialize({'class_name':clsname,'config':config}) for _ in range(self.steps)]

    @tf.function
    def _pre_call(self, inputs, training=False):
        '''Perform untrainable feature preprocessing, thus can be called before fit.'''
        x = tf.concat([inputs] + [block(inputs) for block in self._pre_call_blocks], -1)
        return tf.gather(x, self._permutation, axis=-1)

    @tf.function
    def _call_full(self, inputs, training=False):
        '''Return the whole final state.'''
        x = inputs + tf.concat(self._intercept, -1)
        if training and self._fit_local_intercept:
            x = x + tf.concat(self._local_intecepts, -1)
        for step in range(self.steps):
            x = self._interlayers[step](x, training=training)#experimentral
            x = self._taylormap(x)
        return x

    def call(self, inputs, training=False):
        '''Perform trainable actions and return prediction.
        _pre_call should be called before call.
        '''
        return self._call_full(inputs, training)[:,:self.n_targets_]

    def fit(self, X, y,
            batch_size=None,
            verbose=None,
            epochs=None,
            validation_split=0,
            validation_data=None,
            class_weight=None,
            sample_weight=None,
            callbacks=None):
        # (re)build network
        if not self.built or not self.warm_start:
            #TODO: random state, warmup, schedules
            self.build(X.shape, y.shape)
            self.compile(self.solver, self.loss or self._DEFAULT_LOSS)

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

        # fit
        self.history_ = super().fit(self._pre_call(X), y,
            batch_size,
            epochs,
            verbose,
            callbacks,
            validation_split,
            validation_data,
            self.shuffle,
            class_weight,
            sample_weight
        )

        if (validation_data or validation_split) and verbose:
            print(f'Best weights from {earlystopping.best_epoch} epoch are restored.')

        # set public params
        self.coef_ = self._taylormap.W.numpy() * self.steps
        self.intercept_ = tf.concat(self._intercept, -1).numpy()
        self.local_intercepts_ = [(self._intercept[i] + self._local_intecepts[i]).numpy() for i in range(1, len(self._intercept))]

        return self

    def predict(self, X,
            batch_size=None,
            verbose=None):
        return super().predict(self._pre_call(X),
            batch_size,
            verbose if verbose is not None else self.verbose
        )

    def change_steps(self, new_steps):
        '''Changes network's depth/steps with proper weight transfer.'''
        self._taylormap.W.assign(self._taylormap.W * (self.steps/new_steps))
        self.steps = new_steps
        self._build_interlayer()


if __name__=='__main__': # to run this test comment row 6 with relative import
    import regularizers
    tf.keras.utils.set_random_seed(0)
    model = TMPNN(
        solver=tf.keras.optimizers.legacy.Adamax(tf.keras.optimizers.schedules.CosineDecay(1e-4,100,0,None,1e-3,10)),
        interlayer=None,#tf.keras.layers.BatchNormalization(),
        latent_units=0,
        fit_target_intercept=2,
        fit_latent_intercept=0
    )
    model.fit(tf.eye(10), tf.ones((10,1)), epochs=2, verbose=0)

    import time
    start_time=time.time()
    # model.predict(tf.ones((10000,10)),10000)
    model.call(tf.ones((10000,11)))
    end_time=time.time()
    print(f'`TMPNN().predict(x=tf.ones((10000,10)), batch_size=1000)` takes {(end_time - start_time):3.3f}s')