import os
import time
import tensorflow as tf
import numpy as np
from sklearn.utils.estimator_checks import check_estimator

import tmpnn

# simple test
model = tmpnn.base.TMPNN(verbose=0,max_epochs=10)
model.fit(np.eye(10),np.ones(10))
model.predict(np.eye(10))
del model

# test sklearn compatability
tf.get_logger().setLevel('ERROR')
print(check_estimator(tmpnn.base.TMPNN(verbose=0,max_epochs=10)))
# print(check_estimator(tmpnn.TMPNNRegressor(verbose=0,max_epochs=10)))
# print(check_estimator(tmpnn.TMPNNLogisticRegressor(verbose=0,max_epochs=10)))
# print(check_estimator(tmpnn.TMPNNClassifier(verbose=0,max_epochs=10)))