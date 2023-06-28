import numpy as np
import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras import Input, Model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers.legacy import Adamax as Opt
from keras import backend as K
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.optimizers.legacy import Adamax as Opt

from ..layers.taylor import TaylorMap


class Regression:
    def __init__(self, num_features, num_targets, order=2, steps=10, learning_rate=1e-3, scale=1, shift=True,regularizer=None):
        self.order = order
        self.steps = steps

        self.scale = scale
        self.shift = scale/2 * shift
        self._min = np.zeros(num_features+num_targets)
        self._ptp = np.ones(num_features+num_targets)

        self.num_features = num_features
        self.num_targets = num_targets
        self.pnn, self.pnn_hidden = self.create_graph(regularizer)
        self.set_learning_rate(learning_rate)
        return

    def create_graph(self, regularizer=None):
        inputDim = self.num_features + self.num_targets
        outputDim = self.num_features + self.num_targets

        input = Input(shape=(inputDim,))
        m = input
        tm = TaylorMap(output_dim = outputDim, input_shape = (inputDim,), order=self.order, 
            weights_regularizer=regularizer)

        outs = []
        for i in range(self.steps-1):
            m = tm(m,reg=False)
            outs.append(m)
        m = tm(m,reg=True)
        outs.append(m)

        model = Model(inputs=input, outputs=m)
        model_full = Model(inputs=input, outputs=outs)

        return model, model_full

    def _custom_loss(self, y_true, y_pred):
        squared_error = (y_pred[:, -self.num_targets:] - y_true[:, -self.num_targets:])**2
        mse = K.mean(squared_error, axis=0)
        return K.mean(mse)

    def set_learning_rate(self, learning_rate, loss=None, metrics=None):
        self.pnn.compile(loss=self._custom_loss if not loss else loss, optimizer=Opt(learning_rate=learning_rate), metrics=metrics)
        return

    def _scale(self, X, Y=None, fit=False):
        if Y is None:
            return (X - self._min[:self.num_features])/self._ptp[:self.num_features]-self.shift
        else:
            data = np.column_stack([X, Y])
            if fit:
                self._min = data.min(0)
                self._ptp = data.ptp(0)/self.scale
            data = (data-self._min)/self._ptp-self.shift
            return data[:, :self.num_features].reshape(-1, self.num_features), data[:, -self.num_targets:].reshape(-1, self.num_targets)
    
    def _rescale(self, X_output):
        return (X_output+self.shift)*self._ptp + self._min

    def fit(self, X, Y, epochs=1000, batch_size=256, verbose=1, 
            validation_data=None, stop_monitor=None, patience=10, init=None):
        callbacks=[]
        if stop_monitor:
            callbacks.append( EarlyStopping(monitor='val_'+stop_monitor if validation_data else stop_monitor, patience=patience) )

        if self.scale > 0:
            X, Y = self._scale(X, Y, fit=True) # can lead to misstraing if fitting different data regions sequentially TODO
        X_input = np.hstack((X, np.zeros((X.shape[0], self.num_targets)))) # if init is None else (init(X) if callable(init) else init)))
        
        if validation_data:
            X_val, Y_val = validation_data
            if self.scale > 0:
                 X_val, Y_val = self._scale(X_val, Y_val, fit=False)
            validation_data = (np.hstack((X_val, np.zeros((X_val.shape[0], self.num_targets)))), Y_val)

        history = self.pnn.fit(X_input, Y, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                               validation_data=validation_data, callbacks=callbacks, )
        return history

    def predict(self, X, verbose=0):
        if self.scale > 0:
            X = self._scale(X)

        X_input = np.hstack((X, np.zeros((X.shape[0], self.num_targets))))
        X_pred = self.pnn.predict(X_input, verbose)

        if self.scale > 0:
            X_pred = self._rescale(X_pred)
        return X_pred[:,-self.num_targets:]