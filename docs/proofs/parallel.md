# Proof notes: finlab.parallel (AFML ch. 20)

Module: `src/finlab/parallel.py`. Tests: `tests/test_parallel.py`.

## Proved here

**Q1 (`lin_parts` is an exact cover with near-equal chunks).** For `n >= 0`,
`T >= 1`, let `parts = ceil(linspace(0, n, T'+1))` with `T' = min(T, n)`. Then
`parts[0] = 0`, `parts[-1] = n`, `parts` is non-decreasing, and every chunk size
`parts[k+1] - parts[k]` lies in `{floor(n/T'), ceil(n/T')}`.
*Proof.* The unrounded points are `x_k = k n / T'` with spacing `d = n/T'`. For real
`a` and spacing `d`, `ceil(a + kd) - ceil(a + (k-1)d)` takes values in
`{floor(d), ceil(d)}`, which gives the size bound. Monotonicity and the endpoints
follow from `x_0 = 0`, `x_{T'} = n`. ∎

**Q2 (`mp_pandas_obj` equals the serial result).** Chunks are contiguous and
disjoint, and `pd.concat` in chunk order reassembles them in the input order, so for
a function applied independently to each chunk the parallel and serial results are
identical.
*Proof.* Chunks partition the index (Q1), and the process pool returns futures in
submission order which we consume in order. ∎ The test `test_serial_and_parallel_agree`
checks this equality.

**Q3 (`nested_parts` is an exact cover).** The boundaries are the sorted unique values of the
list `[0, b_1, ..., b_{T-1}, n]` after clipping to `[0, n]`, so the chunks are disjoint and
cover `[0, n)`.
*Proof.* `np.unique` removes duplicates and `np.clip` keeps values in `[0, n]`. ∎

## Claimed or not proved

- That `nested_parts` balances the quadratic cost to within a constant factor is not
  proved here; the test checks only the cover property. Balance depends on the
  cumulative-weight placement and is a heuristic from the book's snippet 20.6.
- Speed-up from multiprocessing depends on pickling costs and the machine; no
  general bound is claimed.

## Scope

Functions passed to `mp_pandas_obj` with `num_threads > 1` must be picklable, which
means module-level. Work that depends on other chunks (non-independent functions) is
not parallelisable by this helper.
