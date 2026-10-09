# `finlab.labeling`: Triple-barrier, meta-labels and trend-scanning labels (AFML ch. 3)

Labels that account for when an outcome resolves. Triple-barrier labels stop at a profit
target, a stop loss, or a vertical time barrier. Meta-labels say whether a primary side
prediction was right. `trend_scanning_labels` fits t-stats over several windows (from
Lopez de Prado 2019, not AFML ch. 3).

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
import pandas as pd
from finlab.labeling import get_daily_vol, get_events, get_bins, drop_labels

rng = np.random.default_rng(2)
close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))),
                  index=pd.date_range("2020-01-01", periods=400, freq="B"))
vol = get_daily_vol(close, span=20).dropna()
t_events = vol.index[:300]
side = pd.Series(1, index=t_events)

events = get_events(close, t_events, pt_sl=[1, 1], target=vol.loc[t_events],
                    min_ret=0.0, side=side)
bins = get_bins(events, close)
bins = drop_labels(bins, min_pct=0.05)
print(bins["bin"].value_counts().to_dict())
```

## Notes

Proof: [labeling.md](../proofs/labeling.md). Scope: trend-scanning is not in AFML ch. 3; its 1.96 threshold is a design choice.

Full signatures: [API reference](../API.md).
