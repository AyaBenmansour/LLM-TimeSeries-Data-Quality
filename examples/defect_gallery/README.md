# Time-series defect gallery

Open the [illustrated comparison](gallery.pdf) to see the same synthetic measurements before and after each injection. Each case starts with a fresh clean stream. Missing Values cases use scheduled gaps.

The shaded band is the scheduled episode. Markers show only the readings whose final value, timestamp or presence differs from the clean reference. The grey line under each title lists the severity, scheduled versus changed counts, validation checks passed and the actual injected parameters.

The [before-and-after CSV](before_after.csv) aligns the two streams by acquisition ID. Filter by `case` to inspect an example and by `change` to find its actual effects. Unchanged records have empty defect labels, including unchanged ramp endpoints and samples inside clipping limits.

An `explicit_nan` retains its timestamp. A `record_removed` has no after row. A `time_changed` keeps its measured value. A `record_added` has its own ID and a `copied_from` source.

## Where the Drift label starts and ends

The quadratic ramp and recovery span 18 readings. The before/after comparison identifies 16 changed readings. These four rows show the two boundaries.

| Time (s) | Before (°C) | After (°C) | Ground truth |
| ---: | ---: | ---: | --- |
| 22 | 21.63733 | 21.63733 | Clean |
| 23 | 21.84801 | 21.88479 | Drift |
| 38 | 18.84026 | 19.58201 | Drift |
| 39 | 18.76290 | 18.76290 | Clean |

## Figures

- [Page 1 · Point Outlier](01-point-outlier.png) — Spike, Glitch
- [Page 2 · Burst](02-burst-1.png) — Global Burst · Triangle, Global Burst · Half-sine
- [Page 3 · Burst](03-burst-2.png) — Global Burst · Fast rise, slow fall, Global Burst · Slow rise, fast fall
- [Page 4 · Burst](04-burst-3.png) — Global Burst · Alternating, Contextual Burst
- [Page 5 · Burst](05-burst-4.png) — Burst followed by Bias
- [Page 6 · Bias and Drift](06-bias-and-drift-1.png) — Bias, Linear Drift
- [Page 7 · Bias and Drift](07-bias-and-drift-2.png) — Quadratic Drift with recovery, Drift followed by Bias
- [Page 8 · Bias and Drift](08-bias-and-drift-3.png) — Continuing Drift
- [Page 9 · Scaling and Noise](09-scaling-and-noise-1.png) — Gain amplification, Gain attenuation
- [Page 10 · Scaling and Noise](10-scaling-and-noise-2.png) — Noise
- [Page 11 · Flat Line](11-flat-line-1.png) — Stuck, Outlier Flatline
- [Page 12 · Flat Line](12-flat-line-2.png) — Saturation
- [Page 13 · Missing Values](13-missing-values-1.png) — Isolated missing reading, Missing block
- [Page 14 · Missing Values](14-missing-values-2.png) — Periodic gaps, Terminal gap
- [Page 15 · Missing Values](15-missing-values-3.png) — Whole-channel absence
- [Page 16 · Missing Values](16-missing-values-4.png) — Multi-channel gap
- [Page 17 · Timestamp](17-timestamp-1.png) — Fixed Shift · irregular sampling, Jitter · irregular sampling
- [Page 18 · Timestamp](18-timestamp-2.png) — Progressive Clock Drift · irregular sampling, Fixed Shift · regular sampling
- [Page 19 · Timestamp](19-timestamp-3.png) — Jitter · regular sampling, Progressive Clock Drift · regular sampling
- [Page 20 · Timestamp](20-timestamp-4.png) — Time Stretch, Exact Duplicate
- [Page 21 · Timestamp](21-timestamp-5.png) — Near-Duplicate, DST conversion error

## Cases

| Case | Family | Changed records | What to look for |
| --- | --- | ---: | --- |
| [spike](01-point-outlier.png) | point_outlier | 1 | One reading crosses the global healthy interval. |
| [glitch](01-point-outlier.png) | point_outlier | 1 | One reading exceeds its local interval while remaining globally plausible. |
| [burst_triangle](02-burst-1.png) | burst | 7 | The added error rises and falls linearly. |
| [burst_half_sine](02-burst-1.png) | burst | 7 | The added error follows a smooth half-wave. |
| [burst_fast_rise_slow_fall](03-burst-2.png) | burst | 7 | The error is largest at onset, then decays. |
| [burst_slow_rise_fast_fall](03-burst-2.png) | burst | 7 | The error builds through the episode and resets after its peak. |
| [burst_alternating](04-burst-3.png) | burst | 7 | Consecutive errors switch sign. |
| [burst_contextual](04-burst-3.png) | burst | 4 | The disturbance crosses its local healthy interval. All readings remain inside the global interval. |
| [burst_residual](05-burst-4.png) | burst | 14 | A rising Burst leaves a smaller constant offset. The follow-on phase has its own Bias label. |
| [bias](06-bias-and-drift-1.png) | bias | 12 | The sensor reads high by a constant additive amount. |
| [drift_linear](06-bias-and-drift-1.png) | drift | 11 | The offset grows linearly from zero and resets after the ramp. |
| [drift_quadratic](07-bias-and-drift-2.png) | drift | 16 | The offset grows quadratically, then returns to zero over six readings. |
| [drift_hold](07-bias-and-drift-2.png) | drift | 19 | The downward ramp settles into a separately labelled constant offset. |
| [drift_continue](08-bias-and-drift-3.png) | drift | 16 | The offset keeps growing beyond its initial ramp until the recording ends. |
| [scaling_amplify](09-scaling-and-noise-1.png) | scaling | 14 | Peaks and troughs move farther apart around the 20 °C baseline. Baseline crossings stay unchanged. |
| [scaling_attenuate](09-scaling-and-noise-1.png) | scaling | 14 | Peaks and troughs move closer to the 20 °C baseline. Baseline crossings stay unchanged. |
| [noise](10-scaling-and-noise-2.png) | noise | 30 | Independent Gaussian errors increase the residual variability. |
| [stuck](11-flat-line-1.png) | flat_line | 14 | The instrument holds its last good reading while the clean temperature moves. |
| [outlier_flatline](11-flat-line-1.png) | flat_line | 12 | A constant held reading lies outside the global healthy interval. |
| [saturation](12-flat-line-2.png) | flat_line | 3 | A simulated upper cap clips only temperatures that cross it. |
| [missing_isolated](13-missing-values-1.png) | missing_values | 1 | One scheduled reading becomes an explicit NaN. |
| [missing_block](13-missing-values-1.png) | missing_values | 6 | Six consecutive records are removed from an irregularly sampled stream. |
| [missing_periodic](14-missing-values-2.png) | missing_values | 6 | Three two-reading gaps recur every eleven samples. |
| [missing_terminal](14-missing-values-2.png) | missing_values | 8 | The final eight scheduled readings become NaN. |
| [missing_channel](15-missing-values-3.png) | missing_values | 60 | All records from probe A are removed. Probe B keeps reporting on its own clock. |
| [missing_multi](16-missing-values-4.png) | missing_values | 17 | Both probes lose readings in the same ten-second interval on independent sampling clocks. |
| [fixed_shift](17-timestamp-1.png) | timestamp | 12 | An irregularly sampled sensor records each affected measurement 0.6 seconds early. |
| [jitter](17-timestamp-1.png) | timestamp | 16 | Recorded times wander early or late within the local gap bounds. |
| [clock_drift](18-timestamp-2.png) | timestamp | 16 | A 1% fast clock accumulates error with elapsed time, starting from an unchanged time anchor. |
| [fixed_shift_regular](18-timestamp-2.png) | timestamp | 12 | On a 1 s grid, each affected reading is recorded 0.4 sample intervals early without reordering. |
| [jitter_regular](19-timestamp-3.png) | timestamp | 16 | On a 1 s grid, recorded times wander early or late within the local gap bounds. |
| [clock_drift_regular](19-timestamp-3.png) | timestamp | 16 | On a 1 s grid, a 1% fast clock accumulates error from an unchanged time anchor. |
| [time_stretch](20-timestamp-4.png) | timestamp | 14 | A simulated 1.5× time scale stretches the final interval. |
| [duplicate](20-timestamp-4.png) | timestamp | 1 | An extra record repeats the original time and measurement. Both occupy the same plot coordinates. |
| [near_duplicate](21-timestamp-5.png) | timestamp | 1 | An extra record repeats the measurement a small fraction of a sample interval later. |
| [dst_shift](21-timestamp-5.png) | timestamp | 6 | Around the Paris spring clock change, the simulated conversion records six readings one hour early. |

## Reproduce

From the repository root, run the following commands.

```sh
python -m pip install -e ".[gallery]"
python examples/make_defect_gallery.py --output examples/defect_gallery
```
