# TMPNN: High-Order Polynomial Regression Based on Taylor Map Factorization
This is a Tensorflow implementation of TMPNN, tabular neural network. TMPNN maps internal dynamical system with lower order polynomial and integrate it, resulting in high-order polynomial model with low complexity.

# Example
The default usage:
```
from tmpnn import TMPNNRegressor, TMPNNLogisticRegressor, TMPNNClassifier, TMPNNPLTransformer

tmpnn = TMPNNRegressor(random_state=0)
tmpnn.fit(x, y)
pred = tmpnn.predict(x)
score = tmpnn.score(x, y)
```

However, in most cases you schould scale x and optionally y:
```
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

tmpnn = TransformedTargetRegressor(
    regressor=Pipeline([
        ('mms', MinMaxScaler((-0.5,0.5))),
        ('est', TMPNNRegressor(random_state=0))
    ]),
    transformer=MinMaxScaler((-0.5,0.5))
)
```

TMPNN is sklearn-friednly, so you also can use it with cross validation or parameter search:
```
cv_scores = cross_val_score(tmpnn, x, y)
```

# Hyperparameters

TMPNN has two major parameters:

`degree`: polynomial order of the Taylor Map. Default value is 2

`steps`: number of iterations of applying the Taylor Map, integration stepd alternatevily. Default value is 7, it might be helpfil to search from 2 to 10 for each speciefic task

We recommend to use lover steps with higher degrees.

In case of small datsets (`n_samples < 1000`) one can also tune `regularizer`. All tensorflow regularizers and lyapnuov (`tmpnn.Lyapunov()`) are acceptable. Lyapunov regularizer tend to provide models robust to extra noise.

For bigger datasets one can increase number of optimization epochs `max_epochs`. Default value is 100.

# Citation
If you use this library for a scientific publication, please use the following BibTex entry to cite our work:
```
@misc{ivanov2023tmpnn,
      title={TMPNN: High-Order Polynomial Regression Based on Taylor Map Factorization},
      author={Andrei Ivanov and Stefan Maria Ailuro},
      year={2023},
      eprint={2307.16105},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```