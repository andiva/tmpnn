# botrrowed from scikeras
import os
import random
from contextlib import contextmanager
from typing import Generator, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import config, ops

from sklearn.utils.validation import _check_sample_weight

DIGITS = frozenset(str(i) for i in range(10))


@contextmanager
def tensorflow_random_state(random_state: Union[int, np.random.RandomState]) -> Generator[None, None, None]:
    # Cast to int for tensorflow
    if isinstance(random_state, np.random.RandomState):
        state = random_state.get_state()
        r = np.random.RandomState()
        r.set_state(state)
        seed = r.randint(low=1)
    else:
        seed = random_state
    # Save values
    origin_gpu_det = os.environ.get("TF_DETERMINISTIC_OPS", None)
    orig_random_state = random.getstate()
    orig_np_random_state = np.random.get_state()
    if context.executing_eagerly():
        tf_random_seed = context.global_seed()
    else:
        tf_random_seed = ops.get_default_graph().seed

    # determism_enabled = config.is_op_determinism_enabled()
    # config.enable_op_determinism()

    # Set values
    # os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    yield

    # Reset values
    # if origin_gpu_det is not None:
    #     os.environ["TF_DETERMINISTIC_OPS"] = origin_gpu_det
    # else:
    #     os.environ.pop("TF_DETERMINISTIC_OPS")
    random.setstate(orig_random_state)
    np.random.set_state(orig_np_random_state)
    tf.random.set_seed(tf_random_seed)
    # if not determism_enabled:
    #     config.disable_op_determinism()


def check_sample_weight(X, y, sample_weight):
    '''Validate that the passed sample_weight and ensure it is a Numpy array.'''
    sample_weight = _check_sample_weight(sample_weight, X)
    if np.all(sample_weight == 0):
        raise ValueError(
            "No training samples had any weight; only zeros were passed in sample_weight."
            " That means there's nothing to train on by definition, so training can not be completed."
        )
    # drop any zero sample weights
    # this helps mirror the behavior of sklearn estimators
    # which tend to have higher precisions
    not_dropped_samples = sample_weight != 0
    return (X[not_dropped_samples], y[not_dropped_samples], sample_weight[not_dropped_samples])
