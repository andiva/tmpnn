import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adamax as Opt

from taylor import TaylorMap


class Regression:
    def __init__(self, num_features, num_targets, order=2, steps=10, learning_rate=1e-3, is_scale=True, alpha=0):
        self.order = order
        self.steps = steps
        self.is_scale = is_scale
        self._min = np.zeros(num_features+num_targets)
        self._ptp = np.ones(num_features+num_targets)

        self.num_features = num_features
        self.num_targets = num_targets
        self.alpha = alpha
        self.pnn, self.pnn_hidden = self.create_graph()
        self.set_learning_rate(learning_rate)

        self.X = None
        return

    def create_graph(self):
        inputDim = self.num_features + self.num_targets
        outputDim = self.num_features + self.num_targets

        input = Input(shape=(inputDim,))
        m = input
        tm = TaylorMap(output_dim = outputDim, input_shape = (inputDim,), order=self.order, 
            weights_regularizer=tf.keras.regularizers.L1(self.alpha/self.steps))

        outs = []
        for i in range(self.steps):
            m = tm(m)
            outs.append(m)

        model = Model(inputs=input, outputs=m)
        model_full = Model(inputs=input, outputs=outs)

        return model, model_full

    def custom_loss(self, y_true, y_pred):
        squared_error = (y_pred[:, -self.num_targets:] - y_true[:, -self.num_targets:])**2
        mse = K.mean(squared_error, axis=0)
        return K.mean(mse)

    def set_learning_rate(self, learning_rate):
        self.pnn.compile(loss=self.custom_loss, optimizer=Opt(learning_rate=learning_rate))
        return

    def scale(self, X, Y=None):
        if Y is None:
            return (X - self._min[:self.num_features])/self._ptp[:self.num_features]-0.5
        else:
            data = np.column_stack([X, Y])
            self._min = data.min(0)
            self._ptp = data.ptp(0)
            data = (data-self._min)/self._ptp-0.5
            return data[:, :self.num_features].reshape(-1, self.num_features), data[:, -self.num_targets:].reshape(-1, self.num_targets)

    def rescale(self, X_output):
        return (X_output+0.5)*self._ptp + self._min

    def fit(self, X, Y, epochs=1000, batch_size=256, verbose=1, validation_data=None):
        if self.is_scale:
            X, Y = self.scale(X, Y)
        X_input = np.hstack((X, np.zeros((X.shape[0], self.num_targets))))
        self.Z = X_input
        history = self.pnn.fit(X_input, Y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=validation_data)
        self.Z = None
        return history

    def predict(self, X, verbose=0):
        if self.is_scale:
            X = self.scale(X)
        X_input = np.hstack((X, np.zeros((X.shape[0], self.num_targets))))
        X_pred = self.pnn.predict(X_input, verbose)
        if self.is_scale:
            X_pred = self.rescale(X_pred)
        return X_pred[:,-self.num_targets:]