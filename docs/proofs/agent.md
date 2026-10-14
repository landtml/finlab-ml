# Proofs and derivations: `finlab.agent` (not AFML)

Code: `src/finlab/agent/server.py` (JSON-RPC over stdio) and `src/finlab/agent/tools.py`
(tool registry and argument checks). Tests: `tests/test_agent.py`. The page describes what
the server guarantees and names the test that checks each guarantee. The user-facing guide
is [AGENTS.md](../AGENTS.md).

Status labels used:

- **Proved here**: follows from the code as written. The page quotes the function it reads.
- **Checked by test** (the claim names its test function, for example `test_batch_array_gets_one_error_and_no_element_is_processed`): asserted in `tests/test_agent.py`.
- **Measured**: none on this page. `benchmarks/bench_agent.py` times the overhead of one call; its numbers are not repeated here.
- **Not verified**: a behaviour the code shows but no test checks. Listed in section 11.

No claim here comes from the book. The server is not an AFML construct.

---

## 1. Lifecycle and methods

- **`initialize`** returns protocol version `2024-11-05`, the capability `{"tools": {}}`, and
  server info with name `finlab` and the package version. **Checked by test**
  `test_initialize_reports_protocol_and_tools_capability`. The version equals the one in
  `pyproject.toml`: **Checked by test** `test_version_strings_match_pyproject`.
- **`ping`** returns an empty result whatever its `params` (the test covers `[]`, `0`, `""`,
  `false`, `"x"` and `[1]`). **Checked by test** `test_methods_that_ignore_params_accept_any_params`.
- **`tools/list`** returns the registry (section 9). Its `params` are ignored, as for `ping`.
  **Checked by test** `test_methods_that_ignore_params_accept_any_params`.
- **Any other method** gets error `-32601` (`method not found`). **Checked by test**
  `test_unknown_tool_and_method`. A non-string `method` gets `-32600`. **Checked by test**
  `test_non_string_method_is_invalid_request_not_crash`.

## 2. Notifications and client responses get no reply

- **A message without an `id` gets no reply.** **Checked by test** `test_notifications_get_no_reply`
  and, over stdio, `test_stdio_loop_handles_parse_errors_and_batches_of_lines` (three replies for
  four lines, the notification producing none).
- **A message with an `id` gets a reply, including a `notifications/*` method.** Such a
  method with an id gets `-32601`. **Checked by test** `test_notification_method_with_id_gets_a_reply`.
- **A client response gets no reply.** A message with an `id`, a `result` or `error`, and no
  `method` is a response to a request this server never sent. **Checked by test**
  `test_client_response_gets_no_reply` (parametrized over a `result` and an `error` response).
- **A batch made only of notifications gets no reply.** **Checked by test**
  `test_batch_of_only_notifications_gets_no_reply`.

## 3. Batches are refused with `-32600`

- **A JSON array that holds a request, or an empty array, gets one reply:** error `-32600`
  with message `batch requests are not supported` and id `null`. **Checked by test**
  `test_batch_array_gets_one_error_and_no_element_is_processed` (the `ping` and `tools/list`
  inside the batch produce no output of their own) and `test_empty_batch_array_gets_the_batch_error`.
- **The stream keeps going after a batch.** **Checked by test**
  `test_stdio_serves_ping_and_tools_list_around_batches_and_responses`.

## 4. Request ids

- **String, number and `null` ids are echoed.** **Checked by test** `test_server_still_echoes_valid_request_ids`.
- **Any other id type is refused with `-32600` and id `null`:** an object, an array, `true` or
  `false`. **Checked by test** `test_server_rejects_request_id_that_is_not_string_number_or_null`.
- **An id that parses to infinity is refused the same way** (`1e400`). The reply is strict JSON.
  **Checked by test** `test_id_that_parses_to_infinity_is_refused_and_reply_is_strict_json`.

## 5. Malformed input

Input lines are read with `json.loads(line, parse_constant=_reject_constant)`, so `NaN`,
`Infinity` and `-Infinity` raise instead of parsing.

- **A `NaN` literal in the arguments gives parse error `-32700` and keeps the request id.**
  The reply message is `parse error: non-standard constant NaN`. **Checked by test**
  `test_nan_in_arguments_parse_error_reply_keeps_request_id` (request id 7 comes back).
- **`Infinity` and `-Infinity` in the arguments take the same path.** **Checked by test**
  `test_non_finite_literal_in_arguments_takes_the_nan_parse_error_path`, parametrized over the three
  tokens, inside an array and as a scalar. The reply is `-32700` with the request id kept and the
  message `parse error: non-standard constant TOKEN`. **Proved here** from `main` and
  `_lenient_request_id`, which re-reads the line with every constant mapped to `None`.
- **A literal that overflows to infinity, such as `1e400`, is not a parse error.** Python reads it as
  inf, so the argument checks refuse it and the reply is a tool error that names the argument.
  **Checked by test** `test_literal_that_overflows_to_infinity_is_a_tool_error`.
- **A `NaN`, `Infinity` or `-Infinity` literal used as the id gives `-32700` with id `null`.**
  **Checked by test** `test_non_standard_constant_id_is_parse_error_and_reply_is_strict_json`
  (parametrized over the three tokens).
- **Malformed JSON gives `-32700`** and the stream continues. **Checked by test**
  `test_stdio_loop_handles_parse_errors_and_batches_of_lines`.
- **An integer literal over the decoder's digit limit gives `-32700`** and the next
  message is answered. **Checked by test** `test_oversized_integer_id_is_parse_error_and_server_survives`.
- **Nesting deeper than the decoder allows gives `-32700`** and the next message is answered.
  **Checked by test** `test_deeply_nested_line_is_parse_error_and_server_survives`.
- **A JSON array with no request in it**, such as `[1, 2]`, is refused as a batch with `-32600`
  and id `null`, and the stream continues. **Checked by test**
  `test_stdio_loop_survives_malformed_messages`.

## 6. Parameter errors

- **`tools/call` with `params` that is not an object** gets `-32602`
  (`invalid params: expected an object`). **Checked by test**
  `test_tools_call_params_that_are_not_an_object_are_invalid_params`.
- **`tools/call` without a string `name`** gets `-32602` (`invalid params: missing tool name`).
  **Checked by test** `test_tools_call_without_string_name_says_missing_tool_name`.
- **An exception raised while handling a message** gets `-32603` with the exception's type
  name, and the stream continues. A notification that raises gets no reply.
  **Checked by test** `test_stdio_loop_survives_unexpected_exception`.

## 7. Tool calls and tool errors

- **A successful call** returns `{"content": [{"type": "text", "text": ...}], "isError": false}`,
  where `text` is the result as compact JSON. **Checked by test** `test_clean_reply_has_no_warnings_key`.
- **A tool failure is a result with `isError: true`**, never a protocol error. The text is
  `{"error": message}`. **Checked by test** `test_bad_arguments_are_reported_not_raised`,
  `test_tool_error_is_reported_by_call_tool`, and `test_tool_overflow_is_reported_as_tool_error`,
  which shows the error text starts with the exception type name.
- **An unknown tool name** gives `unknown tool 'NAME'`. **Checked by test**
  `test_call_tool_unknown_name_returns_error_shape`.
- **Arguments that are not an object** give an error that names the object requirement.
  **Checked by test** `test_list_arguments_reported_as_tool_error`. Missing arguments count as
  `{}` (`test_tool_error_is_reported_by_call_tool`).
- **Unknown argument keys are refused.** The error names every unknown key and lists the
  declared arguments. A misspelt key is never dropped with the default used in its place.
  **Checked by test** `test_unknown_key_is_refused_by_name`. Declared keys keep working:
  `test_declared_keys_still_work`.
- **Types are checked strictly.** A boolean is not a number (`test_boolean_elements_are_rejected_as_numbers_and_symbols`,
  `test_matrix_bool_and_string_elements_are_rejected`). A string is not a boolean
  (`test_string_boolean_is_rejected_not_truthy`). A nested list is refused in a flat array
  (`test_nested_array_is_rejected_for_flat_input`). A non-numeric element is reported by index and
  type (`test_non_numeric_element_names_argument_index_and_type`).
- **Integers are exact.** An integer argument keeps its full value
  (`test_int_argument_keeps_large_integers_exact`). An integer too large for the argument's
  type, such as a 400-digit `seed` or `periods_per_year`, is a tool error that names the argument
  (`test_oversized_integers_are_tool_errors_naming_the_argument`).
- **Symbols must be strings or integers.** A fractional float is refused, and an integral float
  is accepted (`test_symbol_list_rejects_non_integer_floats`).
- **Matrices must be rectangular.** A ragged matrix gets the rectangular-matrix message, not
  numpy's text (`test_ragged_matrix_gets_the_named_rectangular_error`).
- **Range checks.** `frac_diff_ffd` requires `d` in [0, 1] and `thres` in (0, 1]
  (`test_bad_arguments_are_reported_not_raised`, `test_fracdiff_thres_out_of_range_is_rejected`).
  `onc_clusters` requires an integer `max_k` (`test_onc_max_k_must_be_integer`).
- **Non-finite floats become `null`.** Every float in a reply goes through `_clean`, which returns
  `None` for NaN and infinity. **Proved here** from `_clean`, which the tool handlers call for each
  float. **Checked by test** `test_hrp_and_bet_size_replies_turn_non_finite_values_into_null`, which
  replaces the output of `hrp_weights` and `prob_bet_size` with non-finite values. Bet sizes are
  never non-finite for an accepted input: `prob_bet_size` maps z = ±inf (p = 0 or 1) to ±1, and z is
  never NaN, because the 0/0 case needs p = 1/k and p in {0, 1} at once.

## 8. Size limits

- **An array above `MAX_ELEMENTS` (5,000,000 numbers) is refused.** **Checked by test**
  `test_size_limit_enforced`, which lowers the limit to 10 for the test.
  **Proved here**: `_vec` checks the length before `np.asarray`, and `_mat` sums the row lengths
  before the conversion, so the refusal comes before the array is allocated. No test measures
  memory.
- **CPCV summary bounds.** `n_samples` must be a positive integer no larger than
  `MAX_ELEMENTS`, and `horizon` must not be negative. **Checked by test**
  `test_cpcv_rejects_non_positive_n_samples`, `test_cpcv_rejects_negative_horizon`.
- **The number of splits is capped at `MAX_CPCV_SPLITS` (100,000).** **Checked by test**
  `test_cpcv_summary_refuses_split_count_beyond_cap` (C(20, 10) = 184,756 is refused).
  `test_cpcv_summary_huge_group_count_refused_without_computing_binomial` shows a group count of
  10^18 is refused. **Proved here**: `_binomial_exceeds` stops as soon as the running count passes
  the cap, so the full coefficient is never built.
- **`n_samples` times the number of splits is capped at `MAX_CPCV_WORK` (20,000,000).**
  **Checked by test** `test_cpcv_summary_bounds_samples_times_splits_not_splits_alone`
  (100,000 samples with 4,950 splits is refused, and the message names `n_samples`).
  `test_cpcv_split_summary_description_states_the_work_limit` checks that the tool description
  states the cap.

## 9. Warnings and the tool registry

**Warnings.** `call_tool` records every warning a library call raises and returns the distinct
messages in order under a `warnings` key of the result body. The key is absent when there are
none.

- **Checked by test** `test_pbo_equity_curve_warning_reaches_reply`: a cumulative equity curve
  given to `probability_of_backtest_overfitting` puts a warning that mentions "cumulative" or
  "equity" in the reply body.
- **Checked by test** `test_clean_reply_has_no_warnings_key`: a clean call has no `warnings` key.
- **Proved here**: the distinct messages come from `dict.fromkeys` in `call_tool`. No test
  checks the de-duplication.

**Registry.** `TOOLS` holds 14 tools. `tools/list` returns them in registry order.

- **Each entry has exactly `name`, `description` and `inputSchema`**, the schema has type `object`
  and a `properties` dict, and every name in `required` is a declared property. **Checked by test**
  `test_list_tools_entries_have_name_description_and_input_schema` and
  `test_tools_list_schemas_are_well_formed`.
- **The listing order is the registry order.** **Checked by test**
  `test_list_tools_entries_have_name_description_and_input_schema` (`[e["name"] ...] == list(TOOLS.keys())`).
- **Two descriptions state their units.** `plug_in_entropy` gives bits per symbol, not bits per
  word (`test_plug_in_entropy_tool_description_names_bits_per_symbol`). The probabilistic and
  deflated Sharpe tools say that `sr_hat` is per period (`test_psr_and_dsr_tool_descriptions_state_per_period_input`).

## 10. Values match the library

These tests compare a tool's output with a direct library call.

| Tool | Test | Check |
|---|---|---|
| `sharpe_ratio` | `test_sharpe_matches_library` | `rel=1e-12` |
| `probabilistic_sharpe_ratio` | `test_probabilistic_sharpe_matches_library` | `rel=1e-12` |
| `deflated_sharpe_ratio` | `test_deflated_sharpe_matches_library` | `rel=1e-12` |
| `min_track_record_length` | `test_min_track_record_length_matches_library` | `rel=1e-12` |
| `cusum_filter`, `frac_diff_ffd` | `test_cusum_and_fracdiff_match_library` | exact events; `rtol=1e-12` |
| `hrp_weights` | `test_hrp_matches_library` | `rtol=1e-12`; weights sum to 1 |
| `sadf` | `test_sadf_matches_library` | `rel=1e-12` on the maximum; point count exact |
| `bet_size_from_probability` | `test_bet_size_matches_library` | `rtol=1e-12`, two and three classes |
| `bet_size_from_probability` | `test_bet_size_endpoints_are_plus_or_minus_one` | p = 0 gives -1 and p = 1 gives 1 |
| `cpcv_split_summary` | `test_cpcv_summary_counts` | 15 splits, 5 paths for 6 groups, 2 test groups, horizon 5 |
| `probability_of_backtest_overfitting` | `test_clean_reply_has_no_warnings_key` | PBO 0.514 (36 of 70 splits, to 1e-6 relative) for a 240 x 12 matrix drawn with seed 21 |
| `plug_in_entropy` | `test_plug_in_entropy_tool_description_names_bits_per_symbol` | 0.975 for word length 2, to 1e-6 relative |

## 11. Not verified

- **Non-finite weights from `hrp_weights`.** Not verified whether any input gives a non-finite
  weight. The inputs tried (diagonals from 1e-320 to 1e308) gave finite weights or an error. Some
  large inputs give wrong weights from the library: `[[1e308, 0], [0, 1e308]]` gives `[1, 0]`, since
  the two cluster variances sum to infinity and the split factor becomes 1. The tool does not
  correct this; the library owns it.
- **Memory.** No test measures the allocation the size guards prevent (section 8).
