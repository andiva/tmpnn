import numpy as np
import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras import Input, Model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers.legacy import Adamax as Opt
from keras import backend as K
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers.legacy import Adamax as Opt

from ..layers.taylor import TaylorMap
from ..layers.selective import Selective


class Classification:
    def __init__(self, num_features, num_targets, order=2, steps=10, learning_rate=1e-3, regularizer=None):
        self.order = order
        self.steps = steps

        self.num_features = num_features
        self.num_targets = num_targets
        self.pnn, self.pnn_hidden = self.create_graph(regularizer)
        self.set_learning_rate(learning_rate)
        return

    def create_graph(self, regularizer=None):
        inputDim = self.num_features + self.num_targets

        input = Input(shape=(inputDim,))
        m = input
        tm = TaylorMap(output_dim = inputDim, input_shape = (inputDim,), order=self.order, 
            weights_regularizer=regularizer)

        outs = []
        for i in range(self.steps-1):
            m = tm(m,reg=False)
            outs.append(m)
        m = tm(m,reg=True)
        outs.append(m)

        s = Selective(self.num_targets)
        m = s(m)
        outs.append(m)

        model = Model(inputs=input, outputs=m)
        model_full = Model(inputs=input, outputs=outs)

        return model, model_full

    def _custom_loss(self, y_true, y_pred):
        squared_error = (y_pred[:, -self.num_targets:] - y_true[:, -self.num_targets:])**2
        mse = K.mean(squared_error, axis=0)
        return K.mean(mse)

    def set_learning_rate(self, learning_rate, loss=None, metrics=None):
        self.pnn.compile(optimizer=Opt(learning_rate=learning_rate), 
                         loss=self._custom_loss if not loss else loss, 
                         metrics=metrics)
        return

    def fit(self, X, Y, epochs=1000, batch_size=256, verbose=1, 
            validation_data=None, stop_monitor=None, patience=10, 
            init=None, class_weight=None):
        
        callbacks=[]
        if stop_monitor:
            callbacks.append( EarlyStopping(monitor=stop_monitor, patience=patience) )

        X_input = np.hstack((X, np.zeros((X.shape[0], self.num_targets)))) # if init is None else (init(X) if callable(init) else init)))
        
        if validation_data:
            X_val, Y_val = validation_data
            validation_data = (np.hstack((X_val, np.zeros((X_val.shape[0], self.num_targets)))), Y_val)

        history = self.pnn.fit(X_input, Y, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                               validation_data=validation_data, callbacks=callbacks,
                               class_weight=class_weight, )
        return history

    def predict(self, X):
        X_input = np.hstack((X, np.zeros((X.shape[0], self.num_targets))))
        X_pred = self.pnn.predict(X_input)

        return X_pred