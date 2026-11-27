# Value defect equations and controls

All equations compare a clean value `x` with its corrupted value `y` on a single channel. The scheduler supplies each channel's consecutive original indices. Its duration, recurrence, and spatial scope controls apply to every family. The transform returns proposed values, a validation report, and numerical measurements for each phase. The engine accepts or discards all channel phases together. [Implementation brief, §§4–7](../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/IMPLEMENTATION_TASK.md)

## Calibrating an experiment

`amplitude_multiplier` gives the bounds for `K`. Each applicable subtype has its own entry under `defects`. Numeric ranges accept a scalar, `{min, max}`, or ordered `{choices: [...]}`. Bounds may run in either direction. `direction` accepts `positive`, `negative`, or `random` for additive errors. Each targeted channel gets its own draw in random mode. Choose a direction explicitly to synchronize the signs across a multichannel episode. [Versioned preset](../configs/presets/pdf_v1.yaml)

The reference supplies a global scale `S`, healthy interval, and healthy rate threshold for each channel. Tukey fences are the default. The `sigma` and `quantile` reference modes change the definition of the acceptance interval. Glitches and contextual Bursts obtain their local statistics from clean neighbors outside the proposed episode. Contextual Bursts also exclude a reserved subsequent Bias. The scale never pools channels with different units. [Protocol, pp. 4–6](<../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/specs/Final_Injection_Protocol_v1_(26_Nov).pdf#page=4>)

For example, this plan entry gives a quadratic, upward Drift with independently selected magnitude bounds and a five-sample recovery. The ramp reaches the midpoint multiplier of 1.5 at its last scheduled observation. The recovery follows the next five observation times, including on irregular data.

```yaml
family: drift
severity: 0.5
duration: 12
duration_unit: samples
params:
  amplitude_multiplier: {min: 0.5, max: 2.5}
  direction: positive
  exponent: 2
  end_mode: recovery
  recovery_samples: 5
```

## Point Outlier

| Subtype | Equation for the one target observation | Acceptance and achieved effect |
| --- | --- | --- |
| Spike | `y = x + sign × K × S` | The original point lies in the global interval and the proposal crosses a boundary. The label records the signed perturbation, excess in channel units, and excess divided by the scale. |
| Glitch | `y = x + sign × K × S_local` | The proposal crosses its local interval while staying inside the global interval. Global and local checks use the selected threshold mode. |

Both require an observed input, exactly one scheduled sample, a usable scale, and a change above tolerance. [Protocol, pp. 5–6](<../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/specs/Final_Injection_Protocol_v1_(26_Nov).pdf#page=5>)

## Burst

For zero-based position `j` among `n` Burst observations, let `u = (j+1)/(n+1)`. Each generated error profile is divided by its maximum absolute value on the actual discrete grid. [Burst amendments](../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/specs/Burst_Injection_Protocol_Amendments.md)

| `shape` | Raw profile |
| --- | --- |
| `triangle` | `1 − abs(2u − 1)` |
| `half_sine` | `sin(πu)` |
| `fast_rise_slow_fall` | `exp(−decay × j/n)` |
| `slow_rise_fast_fall` | `(j+1)/n` |
| `alternating` | `(−1)^j` |

Set `shape` for a fixed profile or leave it null and configure the `shapes` list and `probabilities`. Equal probabilities apply when weights are null. The five amended equations define the shape constants. Duration controls discretization, `direction` controls polarity, and `decay` controls exponential fall. Rectangle, random, asymmetric variants of the triangle, and other experimental profiles are outside the core menu.

The equation is `y = x + sign × K × S_B × g`. In `scale_mode: value`, `S_B = S`. In `rate` mode, `S_B = max(S, v_star × median_gap)`. The healthy rate quantile uses absolute consecutive differences divided by their individual positive time gaps. Contextual mode uses local scale and local rates and skips a zero local IQR. Global and contextual modes have independent parameter entries.

The first corrupted value must produce `abs(y_first − x_previous) / actual_gap > v_star`. This strict benchmark rule requires an observed clean predecessor. At least `min_affected` consecutive errors must exceed the configured tolerance. Subtype validation checks every Burst value. Global requires a global interval exceedance. Contextual requires all values inside the global interval and at least one outside the local interval. The label includes the normalized profile, actual peak error, achieved onset rate, healthy threshold, and longest meaningful run.

The configurable `noise_min_samples` guard requires a Burst to be shorter than the stated Noise threshold. The preset is 20 samples. Set it to null to rely on the experiment's scheduled Burst durations. Two-sample triangles and half-sines discretize to a constant offset to numerical precision. The validator rejects these ambiguous candidates. Two-sample alternating and ramp-shaped candidates remain eligible.

With `end_mode: recovery` the Burst contributes zero error after its scheduled phase. An episode ending at the recording boundary has no observed post-Burst sample. With `end_mode: non_recovery`, a second, Bias-labelled phase holds the last signed Burst error times `residual_fraction`. Its `tail_samples` can be a positive integer or `to_end`. Choosing a residual fraction is an experiment assumption. A missing or zero-length requested tail makes the proposal infeasible. [Amended end behavior](../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/specs/Burst_Injection_Protocol_Amendments.md)

## Bias, Drift and Scaling

| Family | Equation | Controls and measurements |
| --- | --- | --- |
| Bias | `y = x + sign × K × S` | Constant additive error on the active segment. The measured offset retains the channel unit and sign. |
| Drift | `y = x + sign × K × S × ((t − t0)/L)^p` | `exponent` sets the ramp power. The ramp starts at zero and reaches the signed target at the last scheduled observation. `L` is that elapsed time in seconds. |
| Scaling | `y = center + (x − center) × gain` | Choose `amplify`, `attenuate`, or `random`. Set `gain_up`, `gain_down`, and `center` independently. `center` accepts `segment_mean` or a numeric baseline. The validator measures the achieved range ratio and requires clean variation above tolerance and `minimum_movement`. |

Drift `end_mode: reset` removes the error immediately after the ramp. `hold` adds a Bias phase with the attained offset. `recovery` adds a linear return to zero over `recovery_samples` within the same Drift label, using actual elapsed times. `continue` extends the same ramp equation to the end of the recording and can therefore exceed the magnitude reached at `L`. Schedule the ramp itself through the recording end to place the requested peak there. [Protocol, pp. 7–8](<../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/specs/Final_Injection_Protocol_v1_(26_Nov).pdf#page=7>)

## Noise

Let `q = target_ratio` and let `S_res` be the healthy training residual scale. The generator draws independent centered Gaussian errors with standard deviation `S_res × sqrt(q² − 1)`. A centered rolling mean or median supplies the baseline, with a configurable window and IQR or standard-deviation scale. The achieved ratio is the standard deviation of corrupted residuals divided by the standard deviation of clean residuals **on the same segment using the same clean baseline**. The expected variance addition assumes independent noise and a stable baseline. [Protocol, p. 8](<../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/specs/Final_Injection_Protocol_v1_(26_Nov).pdf#page=8>)

The clean segment must have measurable residual variation and at least `min_samples` observations. Accepted Noise increases the measured variance. A target of one produces zero added error and cannot earn a defect label. The achieved ratio and relative deviation are logged. Setting `ratio_relative_tolerance` adds a rejection check on that deviation. The preset uses durations of 0.2–0.5 reference windows for the default 100-sample reference window, satisfying the PDF's separate 20-sample lower bound.

## Flat Line

| Subtype | Generator | Acceptance |
| --- | --- | --- |
| Stuck | Repeat the last good value immediately before onset. | The held value must lie between the configured healthy `plausible_quantiles`. At least one clean segment value must depart from it by `minimum_movement_scale × S` and by more than the tolerance. |
| Outlier Flatline | Compute `H = x_first + sign × K × S` once, then hold that same `H`. | `H` must lie outside the global statistical interval. A hardware limit is optional metadata. `severity_magnitude` and `severity_duration` can be set separately. The legacy input alias is `hard_over`. |
| Saturation | Hard clipping at the selected lower, upper, or both caps. With `soft_coefficient = α`, values beyond a cap become `cap + α × (x − cap)`, where `0 ≤ α < 1`. | At least `min_capped_run` consecutive clean observations cross the same cap. In-range values still follow the clean signal, including after it recedes. Exact record masks include only values changed by the transfer function. |

Use `cap_source: hardware` with per-channel `lower_limit` or `upper_limit` metadata, or `cap_source: quantile` with `lower_quantile` / `upper_quantile` calculated on eligible healthy training observations. The cap side is configured with `side: lower`, `upper`, or `both`. No-crossing proposals are rejected. The label reports the cap values, source, coefficient, longest capped run, and capped fraction. [Protocol, pp. 8–9](<../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/specs/Final_Injection_Protocol_v1_(26_Nov).pdf#page=8>) · [Outlier Flatline correction](../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/IMPLEMENTATION_TASK.md)
