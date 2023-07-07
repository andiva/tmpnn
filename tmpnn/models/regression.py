from sklearn.base import BaseEstimator

import numpy as np
import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras import Input, Model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers.legacy import Adamax as Opt
from keras import backend as K
from keras import Input, Model
from keras.optimizers.legacy import Adamax as Opt

from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm.keras import TqdmCallback

from ..layers.taylor import TaylorMap
from ..layers.selective import Selective


class Regression(BaseEstimator):
    def __init__(self, num_features, num_targets, order=2, steps=10, verbose=0,
                 regularizer=None, learning_rate=1e-3, init=None):
        self.verbose = verbose
        self.order = order
        self.steps = steps

        self.num_features = num_features
        self.num_targets = num_targets

        self.pnn, self.pnn_hidden = self.create_graph(regularizer)
        self.set_learning_rate(learning_rate)
        self.init = init
        return
    
    def get_params(self, deep: bool = True) -> dict:
        params={'verbose':self.verbose,
                'order':self.order,
                'steps':self.steps,
                'num_features':self.num_features,
                'num_targets':self.num_targets,
                'init':self.init}
        if deep:
            params['pnn']=self.pnn
            params['pnn_hidden']=self.pnn_hidden
        return params
    
    def set_params(self, **params):
        self.verbose = params['verbose']
        self.order = params['order']
        self.steps = params['steps']
        self.num_features = params['num_features']
        self.num_targets = params['num_targets']
        self.init = params['init']
        if 'pnn' in params:
            self.pnn = params['pnn']
            self.pnn_hidden = params['pnn_hidden']
        return self

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

        s = Selective(self.num_targets, False)
        m = s(m)
        outs.append(m)

        model = Model(inputs=input, outputs=m)
        model_full = Model(inputs=input, outputs=outs)

        return model, model_full

    def set_learning_rate(self, learning_rate, loss=None, metrics=None):
        self.pnn.compile(optimizer=Opt(learning_rate=learning_rate), 
                         loss='mse' if not loss else loss, 
                         metrics=metrics)
        return

    def fit(self, X, Y, epochs=100, batch_size=256, verbose=None, 
            validation_data=None, stop_monitor=None, patience=10):
        
        callbacks=[]
        if stop_monitor:
            callbacks.append( EarlyStopping(monitor=stop_monitor, patience=patience) )
        if not verbose:
            verbose = self.verbose
        if verbose == 2:
            callbacks.append( TqdmCallback(verbose=0) )
            verbose = 0

        X_input = np.hstack((X, np.zeros((X.shape[0], self.num_targets)) if self.init is None 
                             else self.init(X) if callable(self.init) else self.init))
        
        if validation_data:
            X_val, Y_val = validation_data
            validation_data = (np.hstack((X_val, np.zeros((X_val.shape[0], self.num_targets)) if self.init is None 
                             else self.init(X_val) if callable(self.init) else self.init)), Y_val)

        history = self.pnn.fit(X_input, Y, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                               validation_data=validation_data, callbacks=callbacks)
        return history

    def predict(self, X):
        X_input = np.hstack((X, np.zeros((X.shape[0], self.num_targets)) if self.init is None 
                             else self.init(X) if callable(self.init) else self.init))
        X_pred = self.pnn.predict(X_input, verbose=0)

        return X_pred