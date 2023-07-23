import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
# from keras import backend as K
# from keras.layers import Layer
# from keras import Input, Model
# from keras import optimizers
import tensorflow as tf

from ..layers.taylor import TaylorMap


class Regression:
    def __init__(self, num_features, num_targets, order=2, steps=10):
        self.order = order
        self.steps = steps

        self.num_features = num_features
        self.num_targets = num_targets
        self.pnn, self.pnn_hidden = self.create_graph()
        return

    def create_graph(self):
        inputDim = self.num_features + self.num_targets
        outputDim = self.num_features + self.num_targets

        input = Input(shape=(inputDim,))
        m = input
        tm = TaylorMap(output_dim = outputDim, input_shape = (inputDim,), order=self.order)

        outs = []
        for i in range(self.steps):
            m = tm(m)
            outs.append(m)

        model = Model(inputs=input, outputs=m)
        model_full = Model(inputs=input, outputs=outs)

        return model, model_full

    def custom_loss(self, y_true, y_pred):
        # return K.sum(K.square(y_true[:, -1] - y_pred[:, -1]))
        squared_error = (y_pred[:, -self.num_targets:] - y_true[:, -self.num_targets:])**2
        mse = K.mean(squared_error, axis=0)
        return K.mean(mse)

    def fit(self, X, Y, epochs, lr=1e-3, batch_size=256, verbose=1, eval_set=None, callbacks=[], opt=None):
        self.pnn.compile(loss=self.custom_loss, 
                         optimizer=opt or optimizers.legacy.Adamax(learning_rate=lr))
        
        if eval_set:
            ev_X, ev_Y = eval_set[0], eval_set[1]
            eval_set = (np.hstack((ev_X, np.zeros((ev_X.shape[0], self.num_targets)))), ev_Y)

        X_input = np.hstack((X, np.zeros((X.shape[0], self.num_targets))))
        return self.pnn.fit(X_input, Y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                            validation_data=eval_set,
                            callbacks = callbacks)

    def predict(self, X):
        X_input = np.hstack((X, np.zeros((X.shape[0], self.num_targets))))
        X_pred = self.pnn.predict(X_input,verbose=0)
        return X_pred[:,-self.num_targets:]