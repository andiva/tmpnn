import pytest
from sklearn.utils.estimator_checks import check_estimator

from tensorflow import get_logger
get_logger().setLevel('ERROR')

from tmpnn import TMPNNRegressor, TMPNNLogisticRegressor, TMPNNClassifier, TMPNNPLTransformer

params = dict(verbose=0, random_state=0)

@pytest.mark.parametrize(
    "estimator",
    [TMPNNRegressor(), TMPNNLogisticRegressor(), TMPNNClassifier(), TMPNNPLTransformer()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator.set_params(**params))