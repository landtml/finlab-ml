# `quantowl.trials`: Trial registry for PBO and the deflated Sharpe ratio (not AFML)

Records each trial a research project runs, once, with its parameters and its return
series. `TrialRegistry` builds the performance matrix for PBO and the inputs of the
deflated Sharpe ratio from that record, so the number of trials N and the matrix are
never typed in by hand. The record is not an AFML construct; it feeds
[`quantowl.pbo`](pbo.md) (AFML ch. 11) and [`quantowl.stats`](stats.md) (AFML ch. 14).

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from quantowl.trials import TrialRegistry

rng = np.random.default_rng(3)
reg = TrialRegistry()
for window in (5, 10, 20, 40):
    returns = rng.normal(0.0002, 0.01, 64)   # one candidate's per-period returns
    reg.record(f"window={window}", {"window": window}, returns)

print(reg.n_trials)
print(round(reg.pbo(n_partitions=8).pbo, 3))
print(round(reg.deflated_sharpe(), 3))
print(TrialRegistry.from_json(reg.to_json()) == reg)
```

## Behaviour

- **Registering.** `record(name, params, returns)` validates one trial, registers it and
  returns the `TrialRecord`. The name is stripped of surrounding whitespace, must be
  non-empty, and must be unique after stripping. A repeated name raises `ValueError`, and
  a failed record is not kept (`test_duplicate_name_rejected_after_stripping`,
  `test_failed_record_is_not_registered`).
- **Returns.** A 1-D sequence of at least two finite real numbers. Strings and booleans are
  refused (`test_record_rejects_string_and_bool_returns`), and so are NaN and infinity
  (`test_record_validation_errors`). The record keeps a read-only float64 copy, so later
  changes to the input array do not reach it (`test_returns_are_copied_from_input`).
- **Parameters.** A mapping, or a sequence of (key, value) pairs. Keys are non-empty
  strings and must not repeat. Values are str, int, float (finite), bool or None. The
  record stores them sorted by key. Equality includes each value's type, so `10` and
  `10.0`, and `1` and `True`, are different parameters
  (`test_records_with_different_values_compare_unequal`).
- **Order.** Trials keep registration order. It sets the columns of the performance matrix
  and the tie-break in `deflated_sharpe`.
- **`performance_matrix()`** returns the (T, N) matrix. It needs at least two trials, all
  with the same number of returns (`test_performance_matrix_needs_two_trials_of_equal_length`).
- **`pbo(n_partitions=16)`** calls `probability_of_backtest_overfitting` on that matrix with
  N = `n_trials`. Its result equals a direct call exactly (`test_pbo_equals_direct_call_exactly`),
  and a new trial changes it (`test_pbo_uses_the_registry_count_as_n`). T must be divisible
  by `n_partitions` (`test_pbo_rejects_partitions_that_do_not_divide_t`).
- **`deflated_sharpe(n_obs=None)`** returns the deflated Sharpe ratio of the best trial, the
  one with the highest non-annualised Sharpe ratio. On a tie the earlier trial wins. `var_sr`
  is the ddof=1 variance of all trials' Sharpe ratios. Skewness and kurtosis come from the
  selected trial's returns, or are 0 and 3 when that trial is all zeros. `n_obs` replaces T
  in the probabilistic Sharpe ratio that the result is built on (`test_dsr_n_obs_override`). Trials may have different
  lengths here (`test_dsr_selects_best_trial_when_lengths_differ`). With one trial `var_sr`
  is 0, so the result is the probabilistic Sharpe ratio against zero skill
  (`test_dsr_single_trial_is_psr_against_zero`). The value matches a hand computation from
  the definition (`test_dsr_matches_hand_computation`). An all-zero trial is accepted
  (`test_dsr_accepts_identically_zero_trial_alongside_a_random_one`). A trial with zero
  variance and a nonzero mean raises `ValueError`, and so does an empty registry
  (`test_dsr_rejects_empty_registry_and_zero_variance_trial`).
- **Serialisation.** `to_dict()` and `to_json()` write `"format": 1`. `from_dict` and
  `from_json` check every trial again, and refuse another format number
  (`test_unknown_registry_format_is_rejected`), a missing key, a duplicate name
  (`test_from_dict_rejects_malformed_input`) and non-finite returns
  (`test_from_json_rejects_non_finite_returns`). Round trips are exact, down to the bit of
  each float (`test_json_round_trip_is_exact_and_stable`, `test_floats_round_trip_bit_for_bit`).
- **Equality.** Two records are equal when name, parameters (with types) and returns match,
  and equal records hash equal. `-0.0` equals `0.0` (`test_negative_zero_return_equals_positive_zero`).
  Two registries are equal only when they hold equal records in the same order
  (`test_registry_equality_depends_on_registration_order`). A registry is mutable, so it is
  not hashable (`test_registry_is_unhashable_because_it_is_mutable`).

## Notes

No proof note: the registry is bookkeeping. The estimators it feeds are proved in
[pbo.md](../proofs/pbo.md) and [stats.md](../proofs/stats.md). Tests: `tests/test_trials.py`.

Full signatures: [API reference](../API.md).
