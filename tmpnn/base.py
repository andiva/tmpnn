import tensorflow as tf
from scipy.linalg import pascal
from . import regularizers


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


class TMPNN(tf.keras.Model):
    '''Basic class for Taylor-mapped polynomial neural networks'''
    def __init__(self,
            degree=2,
            steps=7,
            latent_units=0,
            latent_init=0, #TODO: list of inits
            latent_init_trainable=False, #TODO: boolean mask
            target_features=[],
            target_init=0, #TODO: list of inits
            target_init_trainable=False, #TODO: boolean mask
            verbose='auto',
            max_epochs=1000,
            warmup_epochs=10,
            patience=None,
            warm_start=False,
            shuffle=True,
            solver='adamax',
            loss='mse',
            regularizer=None,
            initializer='zeros',
            dtype=None):
        '''
            degree: Taylor mapping degree of the internal ODE.
                The model is linear if degree==1.

            steps: Depth of the network, #steps for integrating the internal ODE.
                Classical Polynomial model if steps==1.

            target_features: Indices to initial target values among features.
                If #target_features < #targets extra dimensions will be appended.

            target_init: Initial value for extra target dimensions.

            target_init_trainable: If initial value for extra target dimensions are trainable variables or not.

            latent_units: Extra dimensions for overparametrization **excluding** targets.

            latent_init: Initial value for latent dimensions.

            latent_init_trainable: If initial value for latent dimensions are trainable variables or not.
        '''
        super().__init__(dtype=dtype, name='TMPNN')
        self.degree=degree
        self.steps=steps
        self.latent_units=latent_units
        self.latent_init=latent_init
        self.latent_init_trainable=latent_init_trainable
        self.target_features=target_features
        self.target_init=target_init
        self.target_init_trainable=target_init_trainable
        self.verbose=verbose
        self.max_epochs=max_epochs
        self.warmup_epochs=warmup_epochs
        self.patience=patience
        self.warm_start=warm_start
        self.shuffle=shuffle
        self.solver=solver
        self.loss=loss
        self.regularizer=regularizer
        self.initializer=initializer

    def build(self, input_shape, output_shape):
        '''
            Adjusts latents units and targets indices with given training data shapes,
            builds weights and sublayers at the beginning of the fitting routine,
            thus model shouldn't be called directly, only with fit/predict.
        '''
        n_features, n_targets = input_shape[-1], output_shape[-1]

        r_targets = n_targets - len(self.target_features)
        self._targets = self.target_features + [n_features + i for i in range(r_targets)]

        self._paddings = tf.constant([[0, 0], [0, r_targets + self.latent_units]])

        self._inits = [tf.zeros((n_features))]
        if r_targets > 0:
            t = self.add_weight('target_init',
                r_targets,
                initializer=tf.keras.initializers.Constant(self.target_init),
                trainable=self.target_init_trainable
            )
            self._inits.append(t)
        if self.latent_units > 0:
            l = self.add_weight('lanent_init',
                (self.latent_units),
                initializer=tf.keras.initializers.Constant(self.latent_init),
                trainable=self.latent_init_trainable
            )
            self._inits.append(l)

        self._taylormap = TaylorMap(self.degree, self.regularizer, self.initializer, self.dtype)
        super().build(input_shape)

    @tf.function
    def _call_full(self, inputs):
        x = tf.pad(inputs, self._paddings) + tf.concat(self._inits, -1)
        for _ in range(self.steps):
            x = self._taylormap(x)
        return x

    def call(self, inputs):
        return tf.gather(self._call_full(inputs), self._targets, axis=-1)

    def change_steps(self, new_steps):
        '''Changes network's depth/steps with proper weight transfer.'''
        self._taylormap.W.assign(self._taylormap.W * (self.steps/new_steps))
        self.steps = new_steps

    def fit(self, X, y,
            batch_size=None,
            verbose=None,
            epochs=None,
            validation_split=0,
            validation_data=None,
            class_weight=None,
            sample_weight=None,
            callbacks=None):
        if not self.built or (self.built and not self.warm_start):
            self.build(X.shape, y.shape)
            self.compile(self.solver, self.loss)

        epochs = epochs or self.max_epochs
        verbose = verbose if verbose is not None else self.verbose

        if validation_data or validation_split:
            earlystopping = tf.keras.callbacks.EarlyStopping(
                patience=self.patience or epochs, restore_best_weights=True)
            callbacks = [earlystopping, tf.keras.callbacks.TerminateOnNaN()] + (callbacks or [])
        elif verbose:
            print("No early stopping will be performed, last training weights will be used.")

        self.history_ = super().fit(X, y,
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

        return self

    def predict(self, X,
            batch_size=None,
            verbose=None):
        return super().predict(X,
            batch_size,
            verbose if verbose is not None else self.verbose
        )


if __name__=='__main__':
    import regularizers
    tf.keras.utils.set_random_seed(0)
    model = TMPNN().fit(tf.eye(10), tf.ones((10,1)), epochs=3, verbose=0)

    import time
    start_time=time.time()
    model.predict(tf.ones((10000,10)),1000)
    end_time=time.time()
    print(f'```TMPNN().predict(x=tf.ones((10000,10)), batch_size=1000)``` takes {(end_time-start_time):3.3f}s')