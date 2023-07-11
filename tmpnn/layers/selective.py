import tensorflow as tf


class SelectiveLayer(tf.keras.layers.Layer):
    '''Layer for selection targets from state vector'''
    def __init__(self, target_indexes=-1, name=None, **kwargs):
        super().__init__(False, name, **kwargs)
        self.targets = target_indexes

    def call(self, inputs, *args, **kwargs):
        targets = tf.gather(inputs, self.targets, axis=-1)
        return targets
    
    def get_config(self):
        return {**super().get_config(), 'target_indices': self.targets}
