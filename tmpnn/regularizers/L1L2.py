from keras.regularizers import L1 as base1, L2 as base2

class L1(base1):
    def __init__(self, l1=0.01, **kwargs):
        super().__init__(l1, **kwargs)

    def __call__(self, W, x=None):
        loss = 0
        for w in W:
            loss += super().__call__(w)
        return loss
    
    def get_config(self):
        return super().get_config()
    
class L2(base2):
    def __init__(self, l2=0.01, **kwargs):
        super().__init__(l2, **kwargs)

    def __call__(self, W, x=None):
        loss = 0
        for w in W:
            loss += super().__call__(w)
        return loss
    
    def get_config(self):
        return super().get_config()