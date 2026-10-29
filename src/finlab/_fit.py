"""Fit a model with sample weights, sending them to a Pipeline's final step.

A scikit-learn ``Pipeline`` refuses a ``sample_weight`` keyword and takes weights as
``<step>__sample_weight``. AFML Snippet 9.2 ("MyPipeline") passes the weights to the last
step, which is the estimator that consumes them. A Pipeline is recognised by duck typing
(``steps`` and ``named_steps``), so scikit-learn is never imported here.
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
    return isinstance(getattr(model, "steps", None), list) and hasattr(model, "named_steps")


def fit_model(model: Any, X: Any, y: Any, sample_weight: Any = None) -> None:
    """Fit ``model`` on ``X`` and ``y``, passing ``sample_weight`` in the form its class takes.

    A model without weights is fit as ``fit(X, y)``. Any other model gets
    ``fit(X, y, sample_weight=w)``, except a Pipeline, which gets
    ``fit(X, y, <last step>__sample_weight=w)``. With scikit-learn metadata routing
    enabled, a Pipeline gets ``sample_weight=w`` as well, because routing refuses the
    ``step__param`` form.

    Raises
    ------
    ValueError
        If ``model`` is a Pipeline with no steps, or whose final step is ``None`` or
        ``"passthrough"``, since no estimator would receive the weights.
    """
    if sample_weight is None:
        # Some models (KNeighborsClassifier) take no sample_weight keyword at all.
        model.fit(X, y)
        return
    if not _is_pipeline(model):
        model.fit(X, y, sample_weight=sample_weight)
        return
    if not model.steps:
        raise ValueError("the Pipeline has no steps, so no estimator can take sample_weight.")
    last_name, last = model.steps[-1]
    if last is None or (isinstance(last, str) and last == "passthrough"):
        raise ValueError(
            f"the Pipeline's final step is {last!r}, so no estimator would receive sample_weight; "
            "end the Pipeline with an estimator that accepts sample_weight."
        )
    if _metadata_routing_enabled():
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y, **{f"{last_name}__sample_weight": sample_weight})
