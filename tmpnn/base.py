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
        n_features = n_features = input_shape[-1]
        n_poly_features = comb(n_features + self.degree, self.degree, True)

        self.n_features = n_features
        mat_pascal = pascal(max(self.degree, n_features))[1 : self.degree, :n_features]
        for n_monomials in mat_pascal.tolist():
            self._monomial_indices.append(tf.ragged.range(n_monomials))

        self.W = self.add_weight('W',
            shape=(n_poly_features, n_features),
            initializer=self.initializer,
            regularizer=self.regularizer
        )
        self.W.assign_add(tf.concat([
            tf.zeros((1, n_features)), 
            tf.eye(n_poly_features-1, n_features)
        ], 0))

    @tf.function
    def poly(self, X):
        '''Tensorflow implementation of PolynomialFeatures'''
        n_samples = tf.shape(X)[0]
        XP = [tf.ones((n_samples, 1)), X]
        X = tf.expand_dims(X, -1)
        for indices in self._monomial_indices:
            XP.append(tf.reshape(X * tf.gather(XP[-1], indices, axis=-1), (n_samples, -1)))
        return tf.concat(XP, -1)

    def call(self, X):
        return tf.matmul(self.poly(X), self.W)
    
    @tf.function
    def _poly(self, X):
        '''Tensorflow implementation of PolynomialFeatures'''
        # bias term
        XP = [tf.ones_like(X[:, 0:1])]
        # linear term
        XP += [X[:, i : i+1] for i in range(self.n_features)]
        # degree >= 2 terms
        for _ in range(2, self.degree + 1):
            for feature_idx in range(self.n_features):
                XP.append(tf.multiply(
                    tf.concat(XP[-self.n_features : -feature_idx] or XP[-self.n_features:], -1), 
                    X[:, feature_idx : feature_idx + 1]
                ))
        return tf.concat(XP, -1)
    

class TMPNN(tf.keras.Model):
    '''Basic class for Taylor-mapped polynomial neural networks'''
    def __init__(self, 
                 degree=2, 
                 steps=7, 
                 latent_units=0,
                 target_features=None,
                 regularizer=None, 
                 initializer='zeros', 
                 dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.steps = steps
        self.latent_units = latent_units
        self.targets = target_features or []

        self.taylormap = TaylorMap(degree, regularizer, initializer, dtype)

    def change_steps(self, new_steps):
        self.taylormap.W *= self.steps/new_steps
        self.steps = new_steps

    def _prepare_shape(self, n_features, n_targets):
        r_targets = n_targets - len(self.targets)
        self.latent_units += r_targets
        self.targets.extend([n_features + i for i in range(r_targets)])

    def call(self, inputs):
        x = tf.pad(inputs, [[0, 0], [0, self.latent_units]])
        for _ in range(self.steps):
            x = self.taylormap(x)
        return tf.gather(x, self.targets, axis=-1)
    
    def fit(self, x=None, y=None, 
            batch_size=None, 
            epochs=1, 
            verbose="auto", 
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
        self._prepare_shape(x.shape[-1], y.shape[-1])
        return super().fit(x, y, 
            batch_size, 
            epochs, 
            verbose, 
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
            use_multiprocessing)


if __name__=='__main__':
    # from tmpnn import Regression
    # model = Regression(10,1,2,7)
    model = TMPNN()
    model.compile(tf.keras.optimizers.legacy.Adamax(1e-3), 'mse')
    model.fit(tf.eye(10), tf.ones((10,)), epochs=5, verbose=0)
    
    import time
    start_time=time.time()
    model.predict(tf.ones((10000,10)),1000)
    end_time=time.time()
    print(end_time-start_time)