"""Candidate missing records and clock errors on one channel's true time axis."""

from __future__ import annotations

import numpy as np
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import map_severity
from .core import Candidate, Context, Phase, Reject, check
from .data import unit_seconds


def _positive_range(param, name, *, allow_zero=False):
    samples = param["choices"] if isinstance(param, dict) and "choices" in param else [
        map_severity(0, param), map_severity(1, param)]
    if any(isinstance(v, bool) or not isinstance(v, (int, float, np.number))
           or not np.isfinite(v) or v < 0 or (not allow_zero and v == 0) for v in samples):
        raise ValueError(f"Invalid magnitude in {name}")


def validate_availability_params(key: str, p: dict) -> None:
    if key == "missing_values":
        if p["representation"] not in {"nan", "drop_records"}:
            raise ValueError("representation must be nan or drop_records")
        return
    if key in {"duplicate", "near_duplicate", "dst_shift", "time_stretch"} and p["enabled"] is not True:
        raise ValueError(f"Enable the optional {key} experiment explicitly")
    if key in {"fixed_shift", "clock_drift", "near_duplicate", "dst_shift"} and p["sign"] not in {"positive", "negative", "random"}:
        raise ValueError("sign must be positive, negative or random")
    for flag in ("allow_reorder", "allow_collisions"):
        if type(p[flag]) is not bool:
            raise ValueError(f"{flag} must be boolean")
    if key == "duplicate":
        return
    if key == "near_duplicate":
        _positive_range(p["gap_fraction"], "gap_fraction")
        bounds = p["gap_fraction"]
        values = bounds["choices"] if isinstance(bounds, dict) and "choices" in bounds else [
            map_severity(0, bounds), map_severity(1, bounds)]
        if max(values) >= 1:
            raise ValueError("gap_fraction must be smaller than one")
        return
    if key == "fixed_shift":
        if p["offset_mode"] not in {"absolute", "median_gap"}:
            raise ValueError("offset_mode must be absolute or median_gap")
        unit_seconds(p["offset_unit"])
        _positive_range(p["offset"], "offset")
        _positive_range(p["steps"], "steps", allow_zero=True)
        _positive_range(p["fractional_offset"], "fractional_offset", allow_zero=True)
    elif key == "clock_drift":
        if p["rate_unit"] not in {"fraction", "ppm"}:
            raise ValueError("rate_unit must be fraction or ppm")
        _positive_range(p["rate"], "rate")
    elif key == "jitter":
        if p["distribution"] not in {"normal", "uniform"} or p["bound_policy"] not in {"redraw", "clip"}:
            raise ValueError("Unsupported jitter distribution or bound policy")
        _positive_range(p["std_fraction"], "std_fraction")
        if p["max_displacement_seconds"] is not None:
            _positive_range(p["max_displacement_seconds"], "max_displacement_seconds")
        if not np.isfinite(p["local_gap_fraction"]) or p["local_gap_fraction"] <= 0:
            raise ValueError("local_gap_fraction must be positive and finite")
        if type(p["redraw_limit"]) is not int or p["redraw_limit"] < 1:
            raise ValueError("redraw_limit must be a positive integer")
    elif key == "time_stretch":
        _positive_range(p["factor"], "factor")
    elif key == "dst_shift":
        _positive_range(p["offset_hours"], "offset_hours")
    else:
        raise ValueError(f"Unrecognized timestamp variant {key!r}")


def _sign(c: Context) -> int:
    if c.p["sign"] == "random":
        return int(c.rng.choice([-1, 1]))
    return 1 if c.p["sign"] == "positive" else -1


def _typical_gap(c: Context) -> float:
    gap = c.stats.gap
    if not np.isfinite(gap) or gap <= 0:
        raise Reject("no_healthy_adjacent_timestamps")
    return gap


def _fixed_shift(c: Context):
    p = c.p
    if p["offset_mode"] == "absolute":
        delta = map_severity(c.severity, p["offset"]) * unit_seconds(p["offset_unit"])
        params = {"offset_mode": "absolute"}
    else:
        steps = int(np.floor(map_severity(c.severity, p["steps"]) + .5))
        fraction = map_severity(c.severity, p["fractional_offset"])
        gap = _typical_gap(c)
        delta = (steps + fraction) * gap
        params = {"steps": steps, "fractional_offset": fraction, "typical_gap_seconds": gap,
                  "offset_mode": "median_gap"}
    signed = _sign(c) * delta
    return np.full(len(c.idx), signed), {**params, "signed_offset_seconds": signed}


def _jitter(c: Context):
    p = c.p
    gap = _typical_gap(c)
    std = map_severity(c.severity, p["std_fraction"]) * gap
    gaps = np.diff(c.t)
    neighbors = np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf])[c.idx]
    bound = np.nextafter(p["local_gap_fraction"] * neighbors, 0.)
    if p["max_displacement_seconds"] is not None:
        bound = np.minimum(bound, map_severity(c.severity, p["max_displacement_seconds"]))

    def draw(count):
        if p["distribution"] == "normal":
            return c.rng.normal(0, std, count)
        radius = np.sqrt(3.) * std  # Equal pre-truncation variance for the two distributions.
        return c.rng.uniform(-radius, radius, count)

    offsets = draw(len(c.idx))
    if p["bound_policy"] == "clip":
        offsets = np.clip(offsets, -bound, bound)
    else:
        for _ in range(p["redraw_limit"]):
            bad = abs(offsets) > bound
            if not np.any(bad):
                break
            offsets[bad] = draw(int(bad.sum()))
        if np.any(abs(offsets) > bound):
            raise Reject("jitter_redraw_limit", remaining=int(np.sum(abs(offsets) > bound)))
    return offsets, {"distribution": p["distribution"], "bound_policy": p["bound_policy"],
                     "std_seconds": std, "typical_gap_seconds": gap, "local_gap_fraction": p["local_gap_fraction"]}


def _clock_drift(c: Context):
    rate = map_severity(c.severity, c.p["rate"])
    if c.p["rate_unit"] == "ppm":
        rate *= 1e-6
    signed_rate = _sign(c) * rate
    elapsed = c.t[c.idx] - c.t[c.idx[0]]
    return signed_rate * elapsed, {"signed_fractional_rate": signed_rate}


def _time_stretch(c: Context):
    factor = map_severity(c.severity, c.p["factor"])
    elapsed = c.t[c.idx] - c.t[c.idx[0]]
    return (factor - 1) * elapsed, {"stretch_factor": factor}


def _dst_shift(c: Context):
    zone = c.meta.get("timezone")
    if not zone:
        raise Reject("dst_requires_sensor_timezone")
    try:
        ZoneInfo(zone)
    except (TypeError, ValueError, ZoneInfoNotFoundError):
        raise ValueError(f"Unknown sensor metadata timezone {zone!r}") from None
    hours = _sign(c) * map_severity(c.severity, c.p["offset_hours"])
    return np.full(len(c.idx), hours * 3600.), {"timezone": zone, "signed_offset_hours": hours}


def _duplicate(c: Context) -> Candidate:
    offsets = np.zeros(len(c.idx))
    params = {"exact": c.key == "duplicate"}
    if c.key == "near_duplicate":
        if len(c.t) < 2:
            raise Reject("no_gap_for_near_copy")
        gaps = np.diff(c.t)
        local = np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf])[c.idx]
        fraction = map_severity(c.severity, c.p["gap_fraction"])
        offsets = _sign(c) * fraction * local
        params.update(gap_fraction=fraction, signed_direction=int(np.sign(offsets[0])))
    phase = Phase(c.family, c.subtype, c.idx.copy(), copies=offsets,
                  metrics={"added_count": len(c.idx), "channel_added_fraction": len(c.idx) / len(c.x)})
    return Candidate([phase], params, [check("finite_copy_times", np.isfinite(offsets).all())])


TIME_TRANSFORMS = {"fixed_shift": _fixed_shift, "jitter": _jitter, "clock_drift": _clock_drift,
                   "time_stretch": _time_stretch, "dst_shift": _dst_shift}


def transform_availability(c: Context) -> Candidate:
    validate_availability_params(c.key, c.p)
    if c.key in {"duplicate", "near_duplicate"}:
        return _duplicate(c)
    if c.key == "missing_values":
        rep = c.p["representation"]
        checks = [check("existing_measurements", np.isfinite(c.x[c.idx]).all())]
        if len(c.idx) == len(c.x):
            pattern = "whole_channel"
        elif c.idx[-1] == len(c.x) - 1:
            pattern = "terminal"
        else:
            pattern = "isolated" if len(c.idx) == 1 else "block"
        metrics = {"pattern": pattern, "missing_count": len(c.idx), "longest_gap_samples": len(c.idx),
                   "channel_missing_fraction": len(c.idx) / len(c.x)}
        phase = Phase(c.family, None, c.idx.copy(), missing=rep, metrics=metrics)
        return Candidate([phase], {"representation": rep}, checks)
    offsets, params = TIME_TRANSFORMS[c.key](c)
    checks = [check("finite_time_offsets", np.isfinite(offsets).all()),
              check("clock_effect", np.any(offsets != 0))]
    metrics = {"min_offset_seconds": float(np.min(offsets)), "max_offset_seconds": float(np.max(offsets)),
               "mean_offset_seconds": float(np.mean(offsets)), "max_absolute_offset_seconds": float(max(abs(offsets)))}
    phase = Phase(c.family, c.subtype, c.idx.copy(), offsets=offsets, metrics=metrics)
    return Candidate([phase], params, checks)
