"""Pure value-domain transformers. Indices always refer to the original channel."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from .config import map_severity
from .core import Candidate, Context, Phase, Reject, check
from .value_checks import burst_checks, clipping_checks, error_matches, noise_checks, range_checks


SHAPES = ("triangle", "half_sine", "fast_rise_slow_fall", "slow_rise_fast_fall", "alternating")


def _number(c: Context, key: str) -> float:
    return float(map_severity(c.severity, c.p[key]))


def _direction(c: Context, positive: str, negative: str) -> int:
    if c.p["direction"] == "random":
        return int(c.rng.choice([-1, 1]))
    if c.p["direction"] == positive:
        return 1
    if c.p["direction"] == negative:
        return -1
    raise ValueError(f"direction must be {positive}, {negative} or random")


def _amplitude(c: Context, reference, scale: float | None = None) -> tuple[float, dict]:
    scale = reference.scale if scale is None else scale
    if not np.isfinite(scale) or scale <= 0:
        raise Reject("unavailable_positive_scale", scale_source=reference.source)
    multiplier = _number(c, "amplitude_multiplier")
    sign = _direction(c, "positive", "negative")
    return sign * multiplier * scale, {
        "amplitude_multiplier": multiplier, "sign": sign, "scale": scale, "scale_source": reference.source,
    }


def _phase(c: Context, values: np.ndarray, *, metrics=None, idx=None) -> Phase:
    return Phase(c.family, c.subtype, c.idx if idx is None else idx, values=values, metrics=metrics or {})


def _tail(c: Context, setting, start: int) -> np.ndarray:
    length = len(c.x) - start if setting == "to_end" else int(setting)
    if length < 1 or length > len(c.x) - start:
        raise Reject("tail_outside_recording", requested_samples=length, available_samples=len(c.x) - start)
    idx = np.arange(start, start + length)
    if not np.all(np.isfinite(c.x[idx])):
        raise Reject("tail_requires_observed_clean_values")
    return idx


def _bias_tail(c: Context, idx: np.ndarray, offset: float) -> Phase:
    values = c.x[idx] + offset
    check("constant_bias_tail", error_matches(c.x[idx], values, offset, c.tolerance), offset=offset)
    check("bias_tail_meaningful", np.all(np.abs(values - c.x[idx]) > c.tolerance), offset=offset)
    return Phase("bias", None, idx, values=values, metrics={"offset": offset},
                 params={"phase": "held_offset", "follows_family": c.family, "offset": offset})


def _point(c: Context) -> Candidate:
    checks = [check("exactly_one_point", len(c.idx) == 1)]
    x = c.x[c.idx]
    local = c.reference.local(c.idx) if c.key == "glitch" else None
    if local is None:
        checks.append(check("clean_spike_target_in_range", c.stats.lower <= x[0] <= c.stats.upper,
                            clean_value=x[0], lower=c.stats.lower, upper=c.stats.upper))
    reference = c.stats if local is None else local
    offset, params = _amplitude(c, reference)
    values = x + offset
    intervals, metrics = range_checks(c, values, local)
    metrics["perturbation"] = float(values[0] - x[0])
    metric = "global_exceedance" if local is None else "local_exceedance"
    metrics["normalized_exceedance"] = metrics[metric] / reference.scale
    return Candidate([_phase(c, values, metrics=metrics)], params, checks + intervals)


def burst_shape(name: str, n: int, decay: float) -> np.ndarray:
    """Five amended error profiles, normalized *after* discretization."""
    if n < 2:
        raise Reject("burst_requires_multiple_samples", samples=n)
    j = np.arange(n)
    u = (j + 1) / (n + 1)
    if name == "triangle":
        raw = 1 - np.abs(2 * u - 1)
    elif name == "half_sine":
        raw = np.sin(np.pi * u)
    elif name == "fast_rise_slow_fall":
        raw = np.exp(-decay * j / n)
    elif name == "slow_rise_fast_fall":
        raw = (j + 1) / n
    elif name == "alternating":
        raw = (-1.0) ** j
    else:
        raise ValueError(f"Unknown core Burst shape: {name!r}")
    peak = np.max(np.abs(raw))
    if not np.isfinite(raw).all() or peak == 0:
        raise Reject("invalid_burst_profile")
    return raw / peak


def _burst(c: Context) -> Candidate:
    tail = None
    reference_idx = c.idx
    if c.p["end_mode"] == "non_recovery":
        tail = _tail(c, c.p["tail_samples"], c.idx[-1] + 1)
        reference_idx = np.r_[c.idx, tail]
    local = c.reference.local(reference_idx) if c.key == "contextual" else None
    reference = c.stats if local is None else local
    if local is not None and (not np.isfinite(local.iqr) or local.iqr <= 0):
        raise Reject("zero_or_unavailable_contextual_iqr")
    scale = reference.scale
    if c.p["scale_mode"] == "rate":
        if not np.isfinite([scale, reference.rate, reference.gap]).all() or reference.gap <= 0:
            raise Reject("unavailable_burst_rate_scale")
        scale = max(scale, reference.rate * reference.gap)
    amplitude, params = _amplitude(c, reference, scale)
    shape = c.p["shape"] or str(c.rng.choice(c.p["shapes"], p=c.p["probabilities"]))
    decay = _number(c, "decay")
    profile = burst_shape(shape, len(c.idx), decay)
    values = c.x[c.idx] + amplitude * profile
    checks, metrics = burst_checks(c, values, profile, amplitude, reference.rate, local)
    params.update(shape=shape, normalized_profile=profile.tolist(), decay=decay,
                  scale_mode=c.p["scale_mode"], typical_gap_seconds=reference.gap, end_mode=c.p["end_mode"])
    phases = [_phase(c, values, metrics=metrics)]
    if tail is not None:
        residual = amplitude * profile[-1] * _number(c, "residual_fraction")
        phases.append(_bias_tail(c, tail, residual))
        params["residual_offset"] = residual
    after = c.idx[-1] + 1
    returns_to_clean = tail is None and after < len(c.x) and np.isfinite(c.x[after])
    checks.append(check("burst_end_mode", (len(phases) == 2) == (c.p["end_mode"] == "non_recovery"),
                        end_mode=c.p["end_mode"], recovery_visible=returns_to_clean))
    return Candidate(phases, params, checks)


def _bias(c: Context) -> Candidate:
    amplitude, params = _amplitude(c, c.stats)
    values = c.x[c.idx] + amplitude
    errors = values - c.x[c.idx]
    checks = [check("constant_additive_error", error_matches(c.x[c.idx], values, amplitude, c.tolerance),
                    offset=amplitude)]
    return Candidate([_phase(c, values, metrics={"offset": float(np.mean(errors))})], params, checks)


def _drift(c: Context) -> Candidate:
    idx = c.idx
    times = c.t[idx]
    checks = [check("drift_has_ramp", len(times) >= 2 and times[-1] > times[0])]
    amplitude, params = _amplitude(c, c.stats)
    exponent = _number(c, "exponent")
    span = float(times[-1] - times[0])
    mode = c.p["end_mode"]
    errors = amplitude * ((times - times[0]) / span) ** exponent
    checks.append(check("ramp_endpoints", errors[0] == 0 and np.isclose(errors[-1], amplitude),
                        elapsed_seconds=span, final_ramp_offset=errors[-1]))
    params.update(exponent=exponent, end_mode=mode, ramp_samples=len(idx), ramp_seconds=span)
    tail_phase = None
    if mode == "hold":
        tail_phase = _bias_tail(c, _tail(c, c.p["tail_samples"], idx[-1] + 1), amplitude)
    elif mode == "recovery":
        tail = _tail(c, c.p["recovery_samples"], idx[-1] + 1)
        since_peak = c.t[tail] - times[-1]
        errors = np.r_[errors, amplitude * (1 - since_peak / since_peak[-1])]
        idx = np.r_[idx, tail]
        params["recovery_samples"] = len(tail)
        checks.append(check("linear_recovery_to_zero", errors[-1] == 0))
    elif mode == "continue" and idx[-1] + 1 < len(c.x):
        tail = _tail(c, "to_end", idx[-1] + 1)
        idx = np.r_[idx, tail]
        errors = amplitude * ((c.t[idx] - times[0]) / span) ** exponent
    values = c.x[idx] + errors
    checks.append(check("drift_error_equation", error_matches(c.x[idx], values, errors, c.tolerance)))
    phases = [_phase(c, values, idx=idx, metrics={"peak_absolute_error": float(np.max(np.abs(errors))),
                                                 "final_offset": float(values[-1] - c.x[idx[-1]])})]
    if tail_phase is not None:
        phases.append(tail_phase)
    return Candidate(phases, params, checks)


def _scaling(c: Context) -> Candidate:
    x = c.x[c.idx]
    movement = float(np.ptp(x))
    checks = [check("dynamic_scaling_segment", movement > max(c.tolerance, c.p["minimum_movement"]),
                    clean_range=movement)]
    amplify = _direction(c, "amplify", "attenuate") > 0
    gain = _number(c, "gain_up" if amplify else "gain_down")
    center = float(np.mean(x)) if c.p["center"] == "segment_mean" else float(c.p["center"])
    values = (x - center) * gain + center
    observed_gain = float(np.ptp(values) / movement)
    roundoff = np.spacing(np.max(np.abs(values))) * 2 / movement
    direction_matches = observed_gain > 1 if amplify else 0 < observed_gain < 1
    checks.append(check("measured_gain", direction_matches and np.isclose(observed_gain, gain,
                                                                         rtol=1e-10, atol=roundoff),
                        requested_gain=gain, achieved_gain=observed_gain))
    return Candidate([_phase(c, values, metrics={"achieved_gain": observed_gain})],
                     {"gain": gain, "center": center, "direction": "amplify" if amplify else "attenuate"}, checks)


def _noise(c: Context) -> Candidate:
    checks = [check("noise_segment_length", len(c.idx) >= c.p["min_samples"],
                    actual_samples=len(c.idx), minimum=c.p["min_samples"])]
    ratio = _number(c, "target_ratio")
    scale = c.stats.residual_scale
    checks.append(check("healthy_residual_scale", np.isfinite(scale) and scale > 0, residual_scale=scale))
    sigma = scale * np.sqrt(ratio**2 - 1)
    clean = c.x[c.idx]
    values = clean + c.rng.normal(0, sigma, len(clean))
    achieved, metrics = noise_checks(c, values, ratio)
    return Candidate([_phase(c, values, metrics=metrics)],
                     {"target_ratio": ratio, "reference_residual_scale": scale, "sigma_injected": sigma},
                     checks + achieved)


def _flat(c: Context) -> Candidate:
    checks = [check("held_for_multiple_samples", len(c.idx) >= 2)]
    x = c.x[c.idx]
    if c.key == "stuck":
        start = c.idx[0]
        checks.append(check("last_good_value_available", start > 0 and np.isfinite(c.x[start - 1])))
        hold = c.x[start - 1]
        if not len(c.stats.values):
            raise Reject("stuck_requires_reference_values")
        low, high = np.quantile(c.stats.values, c.p["plausible_quantiles"], method=c.stats.quantile_method)
        checks.append(check("plausible_held_value", low <= hold <= high, lower=low, upper=high, held_value=hold))
        movement = float(np.max(np.abs(x - hold)))
        required = _number(c, "minimum_movement_scale") * c.stats.scale
        checks.append(check("clean_moves_under_freeze", movement > c.tolerance and movement >= required,
                            movement=movement, required=required))
        params, metrics = {}, {"clean_max_departure": movement}
    else:
        offset, params = _amplitude(c, c.stats)
        hold = x[0] + offset
        intervals, metrics = range_checks(c, np.asarray([hold]))
        checks.extend(intervals)
    values = np.full(x.shape, hold, dtype=float)
    checks.append(check("held_constant", np.all(values == hold), held_value=hold))
    params["held_value"] = hold
    return Candidate([_phase(c, values, metrics=metrics)], params, checks)


def _saturation(c: Context) -> Candidate:
    side, source = c.p["side"], c.p["cap_source"]
    caps = {"lower": -np.inf, "upper": np.inf}
    for name in ("lower", "upper"):
        if side not in (name, "both"):
            continue
        if source == "hardware":
            limit = c.meta.get(f"{name}_limit")
            if limit is None or not np.isfinite(limit):
                raise Reject("hardware_cap_unavailable", side=name)
            caps[name] = float(limit)
        else:
            if not len(c.stats.values):
                raise Reject("cap_requires_reference_values")
            caps[name] = float(np.quantile(c.stats.values, _number(c, f"{name}_quantile"),
                                          method=c.stats.quantile_method))
    lower, upper = caps["lower"], caps["upper"]
    checks = [check("ordered_caps", lower < upper)]
    x = c.x[c.idx]
    coefficient = _number(c, "soft_coefficient")
    clipped = np.clip(x, lower, upper)
    values = clipped + coefficient * (x - clipped)
    conditional, metrics = clipping_checks(c, values, lower, upper, coefficient)
    params = {"lower_cap": lower if np.isfinite(lower) else None,
              "upper_cap": upper if np.isfinite(upper) else None,
              "cap_source": source, "side": side, "soft_coefficient": coefficient}
    return Candidate([_phase(c, values, metrics=metrics)], params, checks + conditional)


TRANSFORMERS = {
    "spike": _point, "glitch": _point, "global": _burst, "contextual": _burst,
    "bias": _bias, "drift": _drift, "scaling": _scaling, "noise": _noise,
    "stuck": _flat, "outlier_flatline": _flat, "saturation": _saturation,
}


def validate_value_params(key: str, p: dict) -> None:
    """Validate full bounds, including unselected discrete choices, before scheduling."""
    amplitude = {"amplitude_multiplier", "direction"}
    burst = amplitude | {"scale_mode", "shape", "shapes", "probabilities", "decay", "min_affected",
                         "noise_min_samples", "end_mode", "tail_samples", "residual_fraction"}
    fields = {
        "spike": amplitude, "glitch": amplitude, "bias": amplitude, "outlier_flatline": amplitude,
        "global": burst, "contextual": burst,
        "drift": amplitude | {"exponent", "end_mode", "recovery_samples", "tail_samples"},
        "scaling": {"direction", "gain_up", "gain_down", "center", "minimum_movement"},
        "noise": {"target_ratio", "min_samples", "ratio_relative_tolerance"},
        "stuck": {"plausible_quantiles", "minimum_movement_scale"},
        "saturation": {"cap_source", "side", "upper_quantile", "lower_quantile", "min_capped_run",
                       "soft_coefficient"},
    }
    if set(p) != fields[key]:
        raise ValueError(f"Unexpected or missing {key} parameters: {sorted(set(p) ^ fields[key])}")
    domains = {
        "amplitude_multiplier": lambda x: x > 0, "decay": lambda x: x > 0,
        "residual_fraction": lambda x: x > 0, "exponent": lambda x: x > 0,
        "gain_up": lambda x: x > 1, "gain_down": lambda x: 0 < x < 1,
        "target_ratio": lambda x: x >= 1, "lower_quantile": lambda x: 0 <= x <= 1,
        "upper_quantile": lambda x: 0 <= x <= 1, "soft_coefficient": lambda x: 0 <= x < 1,
        "minimum_movement_scale": lambda x: x >= 0,
    }
    for field in p.keys() & domains.keys():
        raw = p[field]
        endpoints = [map_severity(0, raw), map_severity(1, raw)]
        if isinstance(raw, dict) and "choices" in raw:
            endpoints.extend(raw["choices"])
        if not all(isinstance(v, Real) and not isinstance(v, bool) and np.isfinite(v)
                   and domains[field](v) for v in endpoints):
            raise ValueError(f"{key}.{field} contains a value outside its domain")
    enums = {
        "scale_mode": ("rate", "value"),
        "direction": ("random", "amplify", "attenuate") if key == "scaling" else ("random", "positive", "negative"),
        "end_mode": ("reset", "hold", "recovery", "continue") if key == "drift" else ("recovery", "non_recovery"),
        "cap_source": ("quantile", "hardware"), "side": ("lower", "upper", "both"),
    }
    for field in p.keys() & enums.keys():
        if p[field] not in enums[field]:
            raise ValueError(f"Invalid {key}.{field}: {p[field]!r}")
    for field, minimum in {"min_samples": 2, "min_affected": 2, "noise_min_samples": 3,
                           "tail_samples": 1, "recovery_samples": 1, "min_capped_run": 1}.items():
        if field not in p:
            continue
        value = p[field]
        if (field == "tail_samples" and value == "to_end") or (field == "noise_min_samples" and value is None):
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{key}.{field} must be an integer >= {minimum}")
    for field in p.keys() & {"minimum_movement", "ratio_relative_tolerance"}:
        v = p[field]
        if field == "ratio_relative_tolerance" and v is None:
            continue
        if not isinstance(v, Real) or isinstance(v, bool) or not np.isfinite(v) or v < 0:
            raise ValueError(f"{key}.{field} must be finite and nonnegative")
    if key == "scaling" and p["center"] != "segment_mean":
        v = p["center"]
        if not isinstance(v, Real) or isinstance(v, bool) or not np.isfinite(v):
            raise ValueError("scaling.center must be segment_mean or a finite numeric baseline")
    if key == "stuck":
        q = p["plausible_quantiles"]
        if (not isinstance(q, (list, tuple)) or len(q) != 2
                or not all(isinstance(v, Real) and not isinstance(v, bool) for v in q)
                or not 0 <= q[0] < q[1] <= 1):
            raise ValueError("stuck.plausible_quantiles requires two increasing quantiles within [0,1]")
    if key in ("global", "contextual"):
        shapes = p["shapes"]
        if not isinstance(shapes, (list, tuple)) or not shapes or not all(name in SHAPES for name in shapes):
            raise ValueError("Burst.shapes must select from the five core profiles")
        if p["shape"] is not None and p["shape"] not in SHAPES:
            raise ValueError("Burst.shape must be a supported core profile")
        if p["probabilities"] is not None:
            weights = np.asarray(p["probabilities"], dtype=float)
            if (weights.shape != (len(shapes),) or np.any(weights < 0) or
                    not np.isfinite(weights).all() or not np.isclose(np.sum(weights), 1, rtol=0, atol=1e-10)):
                raise ValueError("Burst.probabilities must match shapes and sum to one")


def transform_value(c: Context) -> Candidate:
    """Construct and validate one channel, leaving all clean inputs unchanged."""
    idx = c.idx
    if (len(idx) == 0 or idx[0] < 0 or idx[-1] >= len(c.x) or not np.all(np.diff(idx) == 1)):
        raise Reject("requires_consecutive_clean_indices")
    if not np.all(np.isfinite(c.x[idx])):
        raise Reject("requires_observed_clean_values")
    candidate = TRANSFORMERS[c.key](c)
    for phase in candidate.phases:
        errors = phase.values - c.x[phase.idx]
        meaningful = np.abs(errors) > c.tolerance
        candidate.checks.append(check("finite_meaningful_value_change", np.all(np.isfinite(phase.values))
                                      and np.any(meaningful), phase=phase.family, tolerance=c.tolerance))
        phase.metrics.update(peak_absolute_error=float(np.max(np.abs(errors))),
                             mean_absolute_error=float(np.mean(np.abs(errors))),
                             meaningful_points=int(np.sum(meaningful)))
    return candidate
