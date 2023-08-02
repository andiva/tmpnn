import tensorflow as tf
from math import comb

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
        self.n_features = input_shape[-1]
        self.n_poly_features = comb(self.n_features + self.degree, self.degree)

        self.W = self.add_weight('W',
            shape=(self.n_poly_features, self.n_features),
            initializer=self.initializer,
            regularizer=self.regularizer
        )
        # self.W.assign_add(tf.concat([tf.zeros((n_features,1)), tf.eye(n_features, n_poly_features-1)], -1))

    def poly(self, X):
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
    
    def call(self, X, indeces=None):
        W = self.W if not indeces else tf.gather(self.W, indeces, axis=-1)
        return tf.matmul(self.poly(X), W) + X #
    

class TMPNN(tf.keras.Model):
    '''Basic class for Taylor-mapped polynomial neural networks'''
    def __init__(self, 
                 degree=2, 
                 steps=7, 
                 latent_units=1,
                 target_indeces=None,
                 regularizer=None, 
                 initializer='zeros', 
                 dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.steps = steps
        self.latent_units = latent_units
        self.target_indeces = target_indeces

        self.taylormap = TaylorMap(degree, regularizer, initializer, dtype)

    def build(self, input_shape):
        if self.target_indeces is None:
            self.target_indeces = [input_shape[-1]]

    def call(self, inputs):
        x = tf.pad(inputs, [[0, 0], [0, self.latent_units]])
        for _ in range(self.steps - 1):
            x = self.taylormap(x)
        return self.taylormap(x, indeces=self.target_indeces)

    def change_steps(self, new_steps):
        self.taylormap.W *= self.steps/new_steps
        self.steps = new_steps


if __name__=='__main__':
    import numpy as np
    model = TMPNN()
    model.train_step
    model.compile('adam', 'mse')
    model.fit(np.eye(5), np.ones((5,)), epochs=10000, verbose=0)