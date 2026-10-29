"""Fit a model with sample weights, sending them to a Pipeline's final step.

A scikit-learn ``Pipeline`` refuses a ``sample_weight`` keyword and takes weights as
``<step>__sample_weight``. AFML Snippet 9.2 ("MyPipeline") passes the weights to the last
step, which is the estimator that consumes them. A Pipeline is recognised by its class:
``sklearn.pipeline.Pipeline`` or a subclass, found by name and module, so scikit-learn is
never imported here.
"""

from __future__ import annotations

import sys
from typing import Any


def _metadata_routing_enabled() -> bool:
    # scikit-learn is read only if the caller already imported it, so no import happens here.
    sklearn = sys.modules.get("sklearn")
    if sklearn is None:
        return False
    return bool(sklearn.get_config().get("enable_metadata_routing", False))


def _is_pipeline(model: Any) -> bool:
    # A user class that merely has `steps` is fit like any other model.
    return any(
        cls.__name__ == "Pipeline" and cls.__module__ == "sklearn.pipeline"
        for cls in type(model).__mro__
    )


def _weight_key(pipeline: Any) -> str:
    # Nested Pipelines route through each final step: outer__inner__sample_weight.
    path = []
    step = pipeline
    while _is_pipeline(step):
        if not step.steps:
            raise ValueError("the Pipeline has no steps, so no estimator can take sample_weight.")
        name, step = step.steps[-1]
        if step is None or (isinstance(step, str) and step == "passthrough"):
            raise ValueError(
                f"the Pipeline's final step is {step!r}, so no estimator would receive "
                "sample_weight; end the Pipeline with an estimator that accepts sample_weight."
            )
        path.append(name)
    return "__".join(path + ["sample_weight"])


def fit_model(model: Any, X: Any, y: Any, sample_weight: Any = None) -> None:
    """Fit ``model`` on ``X`` and ``y``, passing ``sample_weight`` in the form its class takes.

    A model without weights is fit as ``fit(X, y)``. Any other model gets
    ``fit(X, y, sample_weight=w)``, except a scikit-learn Pipeline, which gets
    ``fit(X, y, <last step>__sample_weight=w)``, through every nested Pipeline that ends
    another. With scikit-learn metadata routing enabled, a Pipeline gets
    ``sample_weight=w`` as well, because routing refuses the ``step__param`` form.

    Raises
    ------
    ValueError
        If a Pipeline on the path to the final estimator has no steps, or ends in ``None``
        or ``"passthrough"``, since no estimator would receive the weights.
    """
    if sample_weight is None:
        # Some models (KNeighborsClassifier) take no sample_weight keyword at all.
        model.fit(X, y)
        return
    if not _is_pipeline(model):
        model.fit(X, y, sample_weight=sample_weight)
        return
    key = _weight_key(model)
    if _metadata_routing_enabled():
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y, **{key: sample_weight})
