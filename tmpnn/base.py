import tensorflow as tf
from scipy.special import comb
from scipy.linalg import pascal


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
        self._monomial_indices = []
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        n_features = input_shape[-1]
        n_poly_features = tf.math.reduce_sum(tf.pow([n_features]*(self.degree+1), tf.range(self.degree+1)))
        # n_poly_features = comb(n_features + self.degree, self.degree, True)
        self._pascal = pascal(max(self.degree, n_features))[1 : self.degree, :n_features].tolist()

        self.W = self.add_weight('W',
            shape=(n_poly_features, n_features),
            initializer=self.initializer,
            regularizer=self.regularizer
        )
        # self.I = tf.concat([
        #     tf.zeros((1, n_features)), 
        #     tf.eye(n_poly_features-1, n_features)
        # ], 0)
        # self.W.assign_add(self.I)
    
    @tf.function
    def _poly(self, X):
        '''Tensorflow implementation of PolynomialFeatures'''
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
            lanent_init_trainable=False, #TODO: boolean mask
            target_features=None,
            target_init=0, #TODO: list of inits
            target_init_trainable=False, #TODO: boolean mask
            verbose='auto',
            max_epochs=100,
            regularizer=None, 
            initializer='zeros', 
            dtype=None, 
            **kwargs):
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

        latent_inits: Initial value for latent dimensions.

        latent_init_trainable: If initial value for latent dimensions are trainable variables or not.
        '''
        super().__init__(dtype=dtype, **kwargs)
        self.degree = degree
        self.steps = steps
        self.latent_units = latent_units
        self.targets = target_features or []

        self.latent_init = latent_init
        self.lanent_init_trainable = lanent_init_trainable
        self.target_init = target_init
        self.target_init_trainable = target_init_trainable

        self.verbose = verbose
        self.max_epochs = max_epochs

        self.built = False
        self.taylormap = TaylorMap(degree, regularizer, initializer, dtype)

    def _build(self, input_shape, output_shape):
        '''Adjusts latents units and targets indices with given training data shapes, 
        builds weights and sublayers at the beginning of the fitting routine.'''
        if not self.built:
            extra_target_units = output_shape[-1] - len(self.targets)
            self.targets.extend([input_shape[-1] + i for i in range(extra_target_units)])

            self.extra_units = extra_target_units + self.latent_units
            self.paddings = [[0, 0], [0, self.extra_units]]
            inits = [tf.zeros((input_shape[-1]))]
            if extra_target_units > 0:
                t = self.add_weight('target_init', 
                    (extra_target_units),
                    initializer=tf.keras.initializers.Constant(self.target_init), 
                    trainable=self.target_init_trainable
                )
                inits.append(t)
            if self.latent_units > 0:
                l = self.add_weight('lanent_init', 
                    (self.latent_units),
                    initializer=tf.keras.initializers.Constant(self.latent_init), 
                    trainable=self.lanent_init_trainable
                )
                inits.append(l)
            self.inits = inits
            self.built = True

    @tf.function
    def _call_full(self, inputs):
        x = tf.pad(inputs, self.paddings) + tf.concat(self.inits, -1) #TODO: optimize
        for _ in range(self.steps):
            x = self.taylormap(x)
        return x
    
    @tf.function
    def call(self, inputs):
        return tf.gather(self._call_full(inputs), self.targets, axis=-1)
    
    def change_steps(self, new_steps):
        '''Changes network's depth/steps with proper weight transfer.'''
        self.taylormap.W.assign(
            self.taylormap.W * (self.steps/new_steps) # + self.taylormap.I * ((new_steps - self.steps)/new_steps)
        )
        self.steps = new_steps

    def fit(self, x=None, y=None, 
            batch_size=None, 
            epochs=None, 
            verbose=None, 
            callbacks=None, 
            validation_split=0, 
            validation_data=None, 
            shuffle=True, 
            class_weight=None, 
            sample_weight=None, 
            initial_epoch=0, 
            steps_per_epoch=None, 
            validation_steps=None, 
            validation_batch_size=None, 
            validation_freq=1, 
            max_queue_size=10, 
            workers=1, 
            use_multiprocessing=False):
        self._build(x.shape, y.shape)
        if validation_data or validation_split:
            callbacks = [tf.keras.callbacks.EarlyStopping(restore_best_weights=True)] + (callbacks or [])
        return super().fit(x, y, 
            batch_size, 
            epochs or self.max_epochs, 
            verbose if verbose is not None else self.verbose, 
            callbacks, 
            validation_split,
            validation_data, 
            shuffle, 
            class_weight, 
            sample_weight, 
            initial_epoch, 
            steps_per_epoch, 
            validation_steps, 
            validation_batch_size, 
            validation_freq, 
            max_queue_size, 
            workers, 
            use_multiprocessing
        )
    
    def predict(self, 
            x, 
            batch_size=None, 
            verbose=None, 
            steps=None, 
            callbacks=None, 
            max_queue_size=10, 
            workers=1, 
            use_multiprocessing=False):
        if not self.built:
            raise #TODO
        return super().predict(x, 
            batch_size, 
            verbose if verbose is not None else self.verbose, 
            steps, 
            callbacks, 
            max_queue_size, 
            workers, 
            use_multiprocessing
        )


if __name__=='__main__':
    tf.keras.utils.set_random_seed(0)
    model = TMPNN(target_init=-1, target_init_trainable=True)
    model.compile('adam', 'mse')
    model.fit(tf.eye(10), tf.ones((10,1)), epochs=3, verbose=0)

    import time
    start_time=time.time()
    model.predict(tf.ones((10000,10)),1000)
    end_time=time.time()
    print(f'```model.predict(tf.ones((10000,10)), batch_size=1000)``` takes {end_time-start_time:3.3f}s')