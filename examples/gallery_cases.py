"""Independent, seeded before-and-after experiments for the defect gallery."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tsdq import DefectInjector, InjectionResult


@dataclass(frozen=True)
class Case:
    key: str
    title: str
    group: str
    caption: str
    family: str
    subtype: str | None


def _sensor(times, values, name, acquisition):
    return pd.DataFrame({
        "record_id": [f"{acquisition}-{name}-{i}" for i in range(len(times))],
        "timestamp": times,
        "sensor_id": name,
        "value": values,
    })


def _streams():
    """Build evaluation and independent calibration acquisitions in degrees Celsius."""
    def temperature(t, rng):
        return (20 + 1.8 * np.sin((t - 12) * 2 * np.pi / 44)
                + .16 * np.sin(t * 2 * np.pi / 5) + rng.normal(0, .09, len(t)))

    datasets = {}
    for acquisition, size, start, seed in (("evaluation", 60, 0, 135), ("training", 300, -300, 290)):
        rng = np.random.default_rng(seed)
        t = np.arange(start, start + size)
        x = temperature(t, rng)
        regular = _sensor(t, x, "probe_A", acquisition)
        # A rounded clean cycle contains exact, observable crossings of 20 °C.
        cycle = _sensor(t, np.round(20 + 1.25 * np.sin(2*np.pi*t/16), 2), "probe_A", acquisition)
        t_a = start + np.r_[0., np.cumsum(rng.uniform(.8, 1.2, size - 1))]
        t_b = start + .35 + np.r_[0., np.cumsum(rng.uniform(1.2, 1.7, 2 * size // 3 - 1))]
        irregular = _sensor(t_a, temperature(t_a, rng), "probe_A", acquisition)
        second = _sensor(t_b, 2 + temperature(t_b + 3, rng), "probe_B", acquisition)
        asynchronous = pd.concat([irregular, second], ignore_index=True).sort_values("timestamp")
        date = "2024-03-29 10:00" if acquisition == "evaluation" else "2024-01-10 10:00"
        times = pd.date_range(date, periods=size, freq="2h", tz="Europe/Paris")
        civil_time = _sensor(times, x, "probe_A", acquisition)
        datasets[acquisition] = {"regular": regular, "cycle": cycle, "irregular": irregular,
                                 "asynchronous": asynchronous, "civil_time": civil_time}
    return datasets


def build_cases() -> list[tuple[Case, InjectionResult]]:
    """Run each configured intervention on a fresh clean copy and require acceptance."""
    data = _streams()
    cases = []
    metadata = {"probe_A": {"unit": "°C"}, "probe_B": {"unit": "°C"}}

    def add(key, title, group, caption, family, subtype=None, *, start=22, length=12,
            params=None, schedule=None, dataset="regular", severity=.8, reference=None):
        case = Case(key, title, group, caption, family, subtype)
        placement = {"max_attempts": 1, **(schedule or {})}
        if start is not None:
            placement["onset_index"] = start
        request = {"family": family, "severity": severity, "duration": length,
                   "duration_unit": "samples", "schedule": placement, "params": params or {}}
        if subtype:
            request["subtype"] = subtype
        if reference is not None:
            request["reference"] = reference
        if "onset_seconds" in placement:
            request["duration_unit"] = "seconds"
        meta = {"probe_A": {"unit": "°C", "timezone": "Europe/Paris"}} if dataset == "civil_time" else metadata
        result = DefectInjector(metadata=meta, seed=47).inject(
            data["evaluation"][dataset], plan=[request], training_df=data["training"][dataset])
        if result.failures or result.masks.empty:
            raise RuntimeError(f"Gallery case {key!r} produced no accepted comparison: {result.failures}")
        cases.append((case, result))

    positive = {"direction": "positive"}
    add("spike", "Spike", "Point Outlier", "One reading crosses the global healthy interval.",
        "point_outlier", "spike", length=1, params=positive)
    add("glitch", "Glitch", "Point Outlier", "One reading exceeds its local interval while remaining globally plausible.",
        "point_outlier", "glitch", length=1, params=positive)

    for shape, title, caption in (
        ("triangle", "Triangle", "The added error rises and falls linearly."),
        ("half_sine", "Half-sine", "The added error follows a smooth half-wave."),
        ("fast_rise_slow_fall", "Fast rise, slow fall", "The error is largest at onset, then decays."),
        ("slow_rise_fast_fall", "Slow rise, fast fall", "The error builds through the episode and resets after its peak."),
        ("alternating", "Alternating", "Consecutive errors switch sign.")):
        add("burst_" + shape, "Global Burst · " + title, "Burst", caption, "burst", "global",
            length=7, params={**positive, "shape": shape})
    add("burst_contextual", "Contextual Burst", "Burst",
        "The disturbance crosses its local healthy interval. All readings remain inside the global interval.",
        "burst", "contextual", length=4, params={**positive, "shape": "fast_rise_slow_fall"})
    add("burst_residual", "Burst followed by Bias", "Burst",
        "A rising Burst leaves a smaller constant offset. The follow-on phase has its own Bias label.",
        "burst", "global", length=6, params={**positive, "shape": "slow_rise_fast_fall", "end_mode": "non_recovery",
                                             "residual_fraction": .25, "tail_samples": 8})

    add("bias", "Bias", "Bias and Drift", "The sensor reads high by a constant additive amount.",
        "bias", params=positive)
    add("drift_linear", "Linear Drift", "Bias and Drift",
        "The offset grows linearly from zero and resets after the ramp.", "drift", params=positive)
    add("drift_quadratic", "Quadratic Drift with recovery", "Bias and Drift",
        "The offset grows quadratically, then returns to zero over six readings.", "drift",
        params={**positive, "exponent": 2, "end_mode": "recovery", "recovery_samples": 6})
    add("drift_hold", "Drift followed by Bias", "Bias and Drift",
        "The downward ramp settles into a separately labelled constant offset.", "drift",
        params={"direction": "negative", "end_mode": "hold", "tail_samples": 8})
    add("drift_continue", "Continuing Drift", "Bias and Drift",
        "The offset keeps growing beyond its initial ramp until the recording ends.", "drift",
        start=43, length=9, params={**positive, "end_mode": "continue"})

    for direction, title, caption in (
        ("amplify", "Gain amplification", "Peaks and troughs move farther apart around the 20 °C baseline."),
        ("attenuate", "Gain attenuation", "Peaks and troughs move closer to the 20 °C baseline.")):
        add("scaling_" + direction, title, "Scaling and Noise", caption + " Baseline crossings stay unchanged.",
            "scaling", dataset="cycle", start=16, length=17, params={"direction": direction, "center": 20.})
    add("noise", "Noise", "Scaling and Noise", "Independent Gaussian errors increase the residual variability.",
        "noise", start=17, length=30, severity=1.,
        reference={"smoother": "rolling_mean", "residual_scale": "std"})

    add("stuck", "Stuck", "Flat Line", "The instrument holds its last good reading while the clean temperature moves.",
        "flat_line", "stuck", dataset="cycle", start=17, length=16)
    add("outlier_flatline", "Outlier Flatline", "Flat Line", "A constant held reading lies outside the global healthy interval.",
        "flat_line", "outlier_flatline", length=12, params=positive)
    add("saturation", "Saturation", "Flat Line", "A simulated upper cap clips only temperatures that cross it.",
        "flat_line", "saturation", dataset="cycle", start=13, length=18, params={"upper_quantile": .7})

    add("missing_isolated", "Isolated missing reading", "Missing Values",
        "One scheduled reading becomes an explicit NaN.", "missing_values", length=1)
    add("missing_block", "Missing block", "Missing Values",
        "Six consecutive records are removed from an irregularly sampled stream.", "missing_values",
        dataset="irregular", length=6, params={"representation": "drop_records"})
    add("missing_periodic", "Periodic gaps", "Missing Values", "Three two-reading gaps recur every eleven samples.",
        "missing_values", start=17, length=2, schedule={"recurrence": "periodic", "period": 11,
                                                         "occurrence": {"mode": "fixed_count", "episode_count": 3}})
    add("missing_terminal", "Terminal gap", "Missing Values", "The final eight scheduled readings become NaN.",
        "missing_values", start=52, schedule={"persistence": "unresolved"})
    add("missing_channel", "Whole-channel absence", "Missing Values",
        "All records from probe A are removed. Probe B keeps reporting on its own clock.",
        "missing_values", dataset="asynchronous", start=None, params={"representation": "drop_records"},
        schedule={"channels": ["probe_A"], "persistence": "whole_channel"})
    add("missing_multi", "Multi-channel gap", "Missing Values",
        "Both probes lose readings in the same ten-second interval on independent sampling clocks.",
        "missing_values", dataset="asynchronous", start=None, length=10,
        schedule={"channels": "all", "onset_seconds": 22.})

    add("fixed_shift", "Fixed Shift · irregular sampling", "Timestamp",
        "An irregularly sampled sensor records each affected measurement 0.6 seconds early.", "timestamp", "fixed_shift",
        dataset="irregular", params={"offset_mode": "absolute", "offset_unit": "s", "offset": .6, "sign": "negative"})
    add("jitter", "Jitter · irregular sampling", "Timestamp", "Recorded times wander early or late within the local gap bounds.",
        "timestamp", "jitter", dataset="irregular", length=16)
    add("clock_drift", "Progressive Clock Drift · irregular sampling", "Timestamp",
        "A 1% fast clock accumulates error with elapsed time, starting from an unchanged time anchor.",
        "timestamp", "clock_drift", dataset="irregular", length=17, params={"sign": "positive", "rate": .01})
    add("fixed_shift_regular", "Fixed Shift · regular sampling", "Timestamp",
        "On a 1 s grid, each affected reading is recorded 0.4 sample intervals early without reordering.",
        "timestamp", "fixed_shift", params={"offset_mode": "median_gap", "steps": {"min": 0, "max": 0},
                                            "fractional_offset": .4, "sign": "negative"})
    add("jitter_regular", "Jitter · regular sampling", "Timestamp",
        "On a 1 s grid, recorded times wander early or late within the local gap bounds.",
        "timestamp", "jitter", length=16)
    add("clock_drift_regular", "Progressive Clock Drift · regular sampling", "Timestamp",
        "On a 1 s grid, a 1% fast clock accumulates error from an unchanged time anchor.",
        "timestamp", "clock_drift", length=17, params={"sign": "positive", "rate": .01})
    add("time_stretch", "Time Stretch", "Timestamp", "A simulated 1.5× time scale stretches the final interval.",
        "timestamp", "time_stretch", start=45, length=15, params={"enabled": True, "factor": 1.5})
    add("duplicate", "Exact Duplicate", "Timestamp",
        "An extra record repeats the original time and measurement. Both occupy the same plot coordinates.",
        "timestamp", "duplicate", length=1, params={"enabled": True})
    add("near_duplicate", "Near-Duplicate", "Timestamp",
        "An extra record repeats the measurement a small fraction of a sample interval later.", "timestamp",
        "near_duplicate", dataset="irregular", length=1, params={"enabled": True, "sign": "positive", "gap_fraction": .09})
    add("dst_shift", "DST conversion error", "Timestamp",
        "Around the Paris spring clock change, the simulated conversion records six readings one hour early.",
        "timestamp", "dst_shift", dataset="civil_time", start=20, length=6,
        params={"enabled": True, "sign": "negative", "offset_hours": 1})
    return cases
