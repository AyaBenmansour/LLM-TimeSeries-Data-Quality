# Sensor data, calibration and clock experiments

## Preparing records

The long adapter accepts a DataFrame with timestamp and measurement columns. Set `adapter.timestamp_col`, `sensor_col`, `value_col`, and `id_col` to map the source schema. A single channel without a sensor column takes `sensor_name`. Each supplied record ID must be unique across the table. Without an ID column, the adapter assigns the input row number starting at zero. Put a meaningful DataFrame index into the ID column before injection. Other long-format columns are retained.

For wide tables, set `format: wide` and list measurement column names under `channels`. Each measurement cell becomes one record. Its private record ID encodes the channel and original row ID together. The source row ID becomes `source_record_id`. Extra columns outside the selected channels are copied to the corresponding records. An explicit NaN in a wide cell represents an existing record with a missing measurement. Use long format for a sensor that never reported at that instant.

Channel timestamps must already be strictly increasing in their original order. Unequal gaps and asynchronous channels are valid. Multiple channels may have simultaneous observations. Value and missingness transforms target finite clean measurements. The adapter preserves existing NaNs. Measurements use float64. Integer measurements that would lose precision on conversion raise a validation error.

Set `adapter.time_unit` for numeric clocks to `ns`, `us`, `ms`, `s`, `min`, `h` or `d`. The default is seconds. Datetimes preserve their zone and use nanosecond ticks. Set `adapter.timezone` to localize naive datetimes or convert aware datetimes to a specified zone. Ambiguous and nonexistent local times require input with an explicit offset. CSV timestamps with mixed UTC offsets across a DST transition also require an explicit adapter zone. `timestamp_type` accepts `auto`, `numeric` or `datetime`.

Offsets and rates use elapsed seconds after subtraction of a common time origin. Whole-tick integer shifts retain integer storage. A fractional shift can promote a numeric timestamp column to float64. For large integer epochs where promotion would lose precision, use datetime timestamps or `numeric_precision: integer_tick` to request rounding to native ticks. Post-encoding checks reject zero effects, forbidden collisions and ordering violations. Very long spans with sub-float64 relative-time separation can exceed the elapsed-seconds representation. Supply a shorter recording or an appropriate time precision in that case.

## Calibrating on clean data

The injector fits each channel independently. By default it uses the first `reference.train_fraction` of that channel. Those rows are excluded from the injection and evaluation pool. Set `reference.train_record_ids` to select other calibration rows. Pass `training_df=` with a separate, eligible clean recording to fit the reference while keeping the entire target recording available for injection. The caller selects healthy intervals and supplies sensor units and physical ranges.

| Setting | Interpretation |
| --- | --- |
| `scale`, `iqr_divisor` | Use the interquartile range divided by the configured divisor, or empirical population standard deviation with `std`. |
| `threshold_mode` | `tukey` uses Q1 and Q3 plus the configured IQR margins. `sigma` uses the healthy mean plus or minus `sigma_multiple × scale`. `quantile` uses the empirical endpoints in `quantiles`. |
| `quantile_method`, `rate_quantile` | NumPy quantile estimator and healthy absolute-gradient quantile. Each gradient uses the actual time gap of two consecutive eligible clean observations. |
| `local_neighbors`, `min_points` | Number of positions inspected on each side and minimum total finite observations. Context statistics use the clean surroundings of the proposed episode, omit its full interval, and exclude rate pairs crossing the omitted interval or missing data. |
| `zero_scale`, `local_zero_scale` | Choose `skip`, `std`, `resolution`, or `std_then_resolution`. A resolution fallback needs a positive `metadata.<sensor>.resolution`. In Tukey mode, when IQR is zero and a fallback scale exists, the observed clean minimum and maximum define the interval. |
| `smoother`, `smoothing_window`, `residual_scale` | Centered rolling mean or median over the requested number of observations. The fit groups consecutive finite rows within each calibration partition. Residual scale uses `iqr` or `std`. On irregular data, a fixed observation count may span different elapsed durations. |

Training estimates never consume test values. Glitch and contextual Burst use privileged clean neighbors at the candidate location. A reference with insufficient data carries unavailable statistics. A transform requiring a missing scale rejects that candidate. The default 1.349 divisor expresses a scale convention. Interpret thresholds in the selected mode and document the intended calibration population. [Shared calibration and thresholds, pp. 4–6](../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/specs/Final_Injection_Protocol_v1_%2826_Nov%29.pdf#page=4)

## Missing observations

Set `family: missing_values` and choose `params.representation` as `nan` or `drop_records`. The scheduler selects one continuous support per targeted channel. The mask retains the original IDs, values and true times, including physically deleted records. The resulting labels describe isolated, block, terminal, whole-channel and multi-channel patterns. A whole-channel episode requires that all target records can be injected. Use external training for calibration if the entire channel must disappear.

The observed injection fraction counts newly missing records divided by the original stream length. The private manifest also gives the longest injected gap in samples. Configure an occurrence count or target, duration, recurrence and scope independently. Bernoulli `q` controls starts. An absent event-driven record generally requires an expected observation process or independent reference to be identifiable. Supply channel metadata `expected_reporting` with the reporting rule when available and specify which context the evaluation provides. MCAR, MAR and MNAR generators remain future research extensions. [Compact framework, sections 3–5](../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/specs/FYP_Defect_Injection_Compact_Framework.md)

## Timestamp variants

Timestamp errors change the recorded time while preserving each source measurement. Each affected record has a private true timestamp and recorded timestamp. The `sign` choices are `positive`, `negative` and `random`. The options below belong under the corresponding subtype's `params`.

| Subtype | Transformation and controls |
| --- | --- |
| `fixed_shift` | Add a signed constant. `offset_mode: absolute` maps `offset` in `offset_unit`. `median_gap` uses `(round(steps) + fractional_offset) × typical_gap`. A zero fractional offset disables the half-step convention. Both modes work on irregular clocks. |
| `jitter` | Draw an independent centered normal or uniform displacement with pre-truncation STD `std_fraction × typical_gap`. Uniform uses half-width `sqrt(3) × STD`. Bound each displacement by a strict `local_gap_fraction × min(adjacent gaps)`, using the available adjacent gap at endpoints. Set `max_displacement_seconds` for an additional absolute bound. `bound_policy` chooses clipping or bounded redraws up to `redraw_limit`. Finite draws can have a nonzero sample mean. The default local fraction is 0.45. |
| `clock_drift` | Add `signed_rate × (true_time − onset)`. Choose `rate_unit: fraction` or `ppm`. The PDF preset interpolates a positive fractional rate in log space. The first observation has zero timing error. |
| `time_stretch` | Record `onset + factor × (true_time − onset)`. Enable explicitly for stress tests. |
| `duplicate`, `near_duplicate` | Create actual extra records with distinct IDs and source-record provenance. Exact copies retain their true time. Near copies use `sign × gap_fraction × min(adjacent gaps)`. Enable either variant explicitly. The record count or fraction target can select how many source records to copy. |
| `dst_shift` | Enable explicitly and provide `metadata.<sensor>.timezone` as an IANA zone. Add signed `offset_hours` to simulate a civil-time conversion error. The researcher selects the affected interval for the intended location and transition. |

The `allow_reorder` and `allow_collisions` flags apply to recorded times on the same channel, including episode boundaries. Exact duplicates enable collisions in their preset. A whole-channel constant shift preserves within-channel gaps, so measuring its sign and magnitude generally needs an independent clock. Fixed shifts on irregular data use the requested offset without any P95-gap floor. Unparseable timestamp strings and corrupted calendar fields are deferred. [Timestamp baseline, pp. 10–12](../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/specs/Final_Injection_Protocol_v1_%2826_Nov%29.pdf#page=10) [Fixed-shift correction and evaluation requirements, sections 2 and 6.10](../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/IMPLEMENTATION_TASK.md)
