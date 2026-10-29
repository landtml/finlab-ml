# `finlab.ensemble`: Bagging with sequential bootstrap (AFML ch. 6)

Ordinary bagging draws overlapping labels redundantly. `SequentialBootstrapBagging` draws
each bag with sequential bootstrap, so the trees see more distinct information.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
import pandas as pd
from finlab.cv import make_t1
from finlab.ensemble import SequentialBootstrapBagging

class Centroid:
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.c_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        return self
    def predict(self, X):
        return self.classes_[((X[:, None] - self.c_[None]) ** 2).sum(2).argmin(1)]

rng = np.random.default_rng(4)
idx = pd.date_range("2020-01-01", periods=200, freq="D")
y = rng.integers(0, 2, 200)
X = rng.normal(size=(200, 2)) + np.where(y[:, None] == 1, 1.5, -1.5)

bag = SequentialBootstrapBagging(Centroid, n_estimators=10, seed=0).fit(X, y, t1=make_t1(idx, 10))
print("train accuracy", (bag.predict(X) == y).mean().round(3))
```

## Out-of-bag score

With `oob_score=True`, `fit` also scores each row with the bags that did not draw it.
The base estimator needs `predict_proba`; without it, `fit` refuses by name before training.

```python
import numpy as np
import pandas as pd
from finlab.cv import make_t1
from finlab.ensemble import SequentialBootstrapBagging

class Proba:
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.c_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        return self
    def predict_proba(self, X):
        s = np.exp(-((X[:, None] - self.c_[None]) ** 2).sum(2))
        return s / s.sum(1, keepdims=True)

rng = np.random.default_rng(4)
idx = pd.date_range("2020-01-01", periods=200, freq="D")
y = rng.integers(0, 2, 200)
X = rng.normal(size=(200, 2)) + np.where(y[:, None] == 1, 1.5, -1.5)

bag = SequentialBootstrapBagging(Proba, n_estimators=20, seed=0, oob_score=True)
bag.fit(X, y, t1=make_t1(idx, 10))
print("oob accuracy", round(bag.oob_score_, 3))
print("oob_decision_function_", bag.oob_decision_function_.shape)
```

`oob_decision_function_` has shape `(n_samples, n_classes)`. Row i averages the
`predict_proba` of the bags whose draws exclude i. A row that every bag drew is NaN.
`oob_score_` is the accuracy of the argmax over the rows that have an estimate. With
`sample_weight`, that accuracy is weighted by the weights. `fit` warns with the number of
NaN rows.

Labels that overlap in time leak between in-bag and out-of-bag samples, so the OOB score
is optimistic for overlapping labels. It is no substitute for purged cross-validation
([cv](cv.md)). Claimed from the book only (not verified against the text).

## Use with scikit-learn

`SequentialBootstrapBagging` has `get_params`, `set_params`, `score` and the classifier tag, so
scikit-learn's `clone`, `cross_val_score` and `GridSearchCV` accept it. finlab does not import
scikit-learn. The block below needs scikit-learn installed, so the test suite does not run it.
Its `params=` keyword needs scikit-learn 1.4 or later.

```
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from finlab.cv import PurgedKFold, make_t1
from finlab.ensemble import SequentialBootstrapBagging

# X is a DataFrame indexed by event start, the same index as t1, so PurgedKFold can align them.
t1 = make_t1(X.index, horizon=10)
cv = PurgedKFold(n_splits=5, t1=t1)
bag = SequentialBootstrapBagging(DecisionTreeClassifier, n_estimators=10, seed=0)
scores = cross_val_score(bag, X, y, cv=cv, params={"t1": t1})
search = GridSearchCV(bag, {"n_estimators": [10, 20]}, cv=cv).fit(X, y, t1=t1)
```

The third argument of `fit` is `t1`, not `sample_weight`. `fit` refuses a `t1` whose values
are not datetimes or integer positions, so weights passed there, as in scikit-learn's
`fit(X, y, sample_weight)`, raise a `ValueError` that names `sample_weight`.

## Notes

Proof: [ensemble.md](../proofs/ensemble.md). Scope: the OOB score uses the draws of one fit
and does no purging of overlapping labels.

Full signatures: [API reference](../API.md).
