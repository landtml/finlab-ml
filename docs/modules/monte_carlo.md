# `finlab.monte_carlo`: Seeded Monte Carlo trials and bootstrap uniqueness (AFML ch. 4)

`run_trials` runs a user function many times, giving each trial its own random stream.
The results do not depend on the number of worker processes, and a longer run extends a
shorter one with the same seed. `bootstrap_uniqueness_mc` uses it to compare the average
uniqueness of a standard bootstrap sample with a sequential bootstrap sample, on random
label sets.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
from finlab.monte_carlo import bootstrap_uniqueness_mc, run_trials

df = bootstrap_uniqueness_mc(n_iter=200, num_threads=1)
print(df.shape, "trials")
print("mean seq_u - std_u", round(float((df["seq_u"] - df["std_u"]).mean()), 3))


def coin_trial(rng, p):
    return {"heads": float(rng.random() < p)}


res = run_trials(coin_trial, 100, seed=0, p=0.5)
print(res.shape, round(float(res["heads"].mean()), 2))
```

## Design choices

- **Random streams.** Trial `i` draws from `SeedSequence(seed, spawn_key=(i,))`. Its
  stream depends only on `(seed, i)`, so `num_threads` and the chunking across workers
  do not change the output. `seed=None` draws fresh entropy once and records it in
  `result.attrs["seed"]`.
- **Label design (AFML Snippet 4.7).** `random_t1` draws `n_obs` start bars with
  replacement and a length in `1..max_h-1` for each, as `randint(1, maxH)` does in the
  book. A start drawn twice keeps the length of its last draw, as the book's
  `t1.loc[ix] = val` loop does. So a label set can have fewer than `n_obs` labels. Ends
  are not clipped.
- **Experiment (AFML Snippet 4.8).** The bar grid runs from 0 to the last label end. Each
  bootstrap sample has one draw per label. The standard sample is drawn with replacement,
  and the sequential sample uses the inverse-CDF rule of `finlab.weights.sequential_bootstrap`.
- **Random numbers.** The draws come from a seeded `numpy.random.Generator`, not numpy's
  global generator. So the same integer seed does not reproduce the book's random stream.
- **Draw order.** Each trial draws the start bars, then the lengths, then the standard
  draws, then one uniform per sequential draw. The order is fixed by the docstring of
  `bootstrap_uniqueness_trial`.
- **Defaults.** `n_obs=10`, `n_bars=100` and `max_h=5` are the values printed in Snippet 4.9.
  The book runs `numIters=1E6`. The default `n_iter` here is 10,000, for speed. Its medians
  match the medians the book states in the text after Figure 4.2 to one decimal place (see the proof note).
- **Parallel runs.** `num_threads > 1` uses a process pool through
  `finlab.parallel.mp_pandas_obj`, so the function must be defined at module level.

## Notes

Proof and measurements: [monte_carlo.md](../proofs/monte_carlo.md). The statistical gap
reported there is measured under these default sizes. It is not the same experiment as
the one in [weights.md](../proofs/weights.md), so the two numbers are not comparable.

Full signatures: [API reference](../API.md).
