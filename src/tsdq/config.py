"""Versioned experiment configuration and deterministic parameter mapping."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import files
import json
import math
from pathlib import Path

import numpy as np
import yaml


TAXONOMY = {
    "point_outlier": ("spike", "glitch"),
    "burst": ("global", "contextual"),
    "bias": (), "drift": (), "scaling": (), "noise": (),
    "flat_line": ("stuck", "outlier_flatline", "saturation"),
    "missing_values": (),
    "timestamp": ("fixed_shift", "jitter", "clock_drift", "time_stretch", "duplicate", "near_duplicate", "dst_shift"),
}
ALIASES = {"hard_over": "outlier_flatline", "progressive_clock_drift": "clock_drift"}


def merge(base: dict, update: Mapping) -> dict:
    """Deep-copy overlays. Scalars and sequences replace their defaults."""
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            # A direct or discrete parameter replaces a continuous range.
            if (value.keys() & {"choices", "value"} or
                    value.keys() & {"min", "max"} and result[key].keys() & {"choices", "value"}):
                result[key] = deepcopy(value)
            else:
                result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def severity_number(s) -> float:
    if isinstance(s, bool) or not isinstance(s, (int, float, np.number)) or not 0 <= s <= 1:
        raise ValueError("severity must be a finite number in [0, 1]")
    return float(s)


def map_severity(s: float, bounds, mode: str | None = None):
    """Map severity to a scalar, ordered discrete choice or linear/log range."""
    s = severity_number(s)
    if isinstance(bounds, (tuple, list)):
        if len(bounds) != 2:
            raise ValueError("a pair of bounds is required")
        bounds = dict(zip(("min", "max"), bounds))
    if not isinstance(bounds, Mapping):
        if isinstance(bounds, bool) or not isinstance(bounds, (int, float, np.number)) or not math.isfinite(bounds):
            raise ValueError("a direct parameter must be a finite number")
        return bounds
    if set(bounds) == {"value"}:
        return map_severity(s, bounds["value"])
    if set(bounds) == {"choices"}:
        choices = bounds["choices"]
        if not isinstance(choices, (tuple, list)) or not choices:
            raise ValueError("choices must be a nonempty ordered list")
        return choices[math.floor(s * (len(choices) - 1) + .5)]
    if not {"min", "max"} <= bounds.keys() or bounds.keys() - {"min", "max", "mode"}:
        raise ValueError(f"invalid range specification: {bounds}")
    lo, hi = (map_severity(s, bounds[k]) for k in ("min", "max"))
    interpolation = mode or bounds.get("mode", "linear")
    if interpolation == "log":
        if min(lo, hi) <= 0:
            raise ValueError("log bounds must be positive")
        if 0 < s < 1:
            return math.exp((1-s) * math.log(lo) + s * math.log(hi))
    elif interpolation != "linear":
        raise ValueError(f"unknown interpolation {interpolation!r}")
    return lo if s == 0 else hi if s == 1 else (1-s)*lo + s*hi


def read_document(path) -> dict:
    with path.open(encoding="utf-8") as stream:
        obj = json.load(stream) if path.name.endswith(".json") else yaml.safe_load(stream)
    if not isinstance(obj, dict):
        raise ValueError("configuration must be a mapping")
    return obj


def _checked_range(s, bounds, name, *, probability=False, allow_zero=False):
    """Validate all endpoints/choices even if this run selects an interior value."""
    endpoints = [map_severity(0, bounds), map_severity(1, bounds)]
    if isinstance(bounds, dict) and "choices" in bounds:
        endpoints.extend(bounds["choices"])
    for value in endpoints:
        if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value) or value < 0 or not allow_zero and value == 0
                or probability and value > 1):
            raise ValueError(f"invalid {name} bound: {value!r}")
    return map_severity(s, bounds)


def preset() -> dict:
    try:
        path = files("tsdq.presets").joinpath("pdf_v1.yaml")
    except ModuleNotFoundError:  # Running directly from the source tree
        path = Path(__file__).resolve().parents[2] / "configs/presets/pdf_v1.yaml"
    return read_document(path)


def known_keys(mapping, allowed, label):
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {sorted(unknown)}")


@dataclass(frozen=True)
class ExperimentConfig:
    settings: dict

    @classmethod
    def load(cls, source=None):
        if isinstance(source, cls):
            source = source.settings
        if isinstance(source, (str, Path)):
            source = read_document(Path(source))
        base = preset()
        override = {} if source is None else source
        known_keys(override, base, "experiment")
        for name in ("adapter", "reference", "validation", "composition", "defects", "output", "evaluation", "windows", "schedule"):
            if name in override:
                known_keys(override[name], base[name], name)
        for variant, opts in override.get("defects", {}).items():
            known_keys(opts, {"duration", "duration_unit", "params", "reference", "tolerance", "schedule"}, variant)
            known_keys(opts.get("params", {}), base["defects"][variant]["params"], variant)
            known_keys(opts.get("reference", {}), base["reference"].keys() - {"train_fraction", "train_record_ids"}, "defect reference")
        output = override.get("output", {})
        if output.get("format", "jsonl") not in {"csv", "jsonl", "parquet"}:
            raise ValueError("output.format must be csv, jsonl or parquet")
        settings = merge(base, override)
        if settings["version"] != base["version"]:
            raise ValueError("unsupported preset version")
        # An occurrence mode is a tagged choice. Overriding it replaces the
        # other mode's inherited parameter, while retaining explicit conflicts.
        if "schedule" in override:
            settings["schedule"] = merge_schedule(base["schedule"], override["schedule"])
        if type(settings["reference_window"]) is not int or settings["reference_window"] < 1:
            raise ValueError("reference_window must be a positive integer")
        for flag in [*settings["evaluation"].values(), settings["validation"]["use_sensor_resolution"]]:
            if type(flag) is not bool:
                raise ValueError("visibility and use_sensor_resolution flags must be boolean")
        if not 0 < settings["reference"]["train_fraction"] < 1:
            raise ValueError("reference.train_fraction must be strictly between zero and one")
        if settings["validation"]["tolerance"] < 0 or not math.isfinite(settings["validation"]["tolerance"]):
            raise ValueError("validation tolerance must be finite and nonnegative")
        stages = settings["composition"]["order"]
        if sorted(stages) != ["missing", "timestamp", "value"]:
            raise ValueError("composition.order must contain value, missing and timestamp once each")
        for pair in settings["composition"]["allow_overlap"]:
            if len(pair) != 2 or len(set(pair)) != 2 or not set(pair) <= set(stages):
                raise ValueError("allow_overlap accepts pairs of distinct stages")
        return cls(settings)


def merge_schedule(base: dict, update: dict) -> dict:
    result = merge(base, update)
    if "occurrence" in update and isinstance(update["occurrence"], dict):
        if (update["occurrence"].keys() & {"episode_count", "target_fraction", "target_records", "q"} or
                "mode" in update["occurrence"] and base.get("occurrence", {}).get("mode") != update["occurrence"]["mode"]):
            result["occurrence"] = deepcopy(update["occurrence"])
            result["occurrence"].setdefault("mode", base.get("occurrence", {}).get("mode", "fixed_count"))
    return result


@dataclass(frozen=True)
class Request:
    family: str
    subtype: str | None
    alias: str | None
    key: str
    severity: float
    magnitude: float
    duration_severity: float
    duration: float
    schedule: dict
    params: dict
    reference: dict
    tolerance: float

    @classmethod
    def resolve(cls, cfg: dict, entry: dict):
        known_keys(entry, {"family", "subtype", "severity", "severity_magnitude", "severity_duration",
                           "duration", "duration_unit", "schedule", "params", "reference", "tolerance"}, "plan")
        family = entry.get("family")
        if family not in TAXONOMY:
            raise ValueError(f"unknown defect family {family!r}")
        original = entry.get("subtype")
        subtype = ALIASES.get(original, original)
        if (TAXONOMY[family] and subtype not in TAXONOMY[family]) or (not TAXONOMY[family] and subtype is not None):
            raise ValueError(f"unsupported subtype {original!r} for {family}")
        key = subtype or family
        spec = cfg["defects"][key]
        s = severity_number(entry.get("severity", cfg["severity"]))
        magnitude = severity_number(entry.get("severity_magnitude", s))
        ds = severity_number(entry.get("severity_duration", s))
        schedule = merge_schedule(merge_schedule(cfg["schedule"], spec.get("schedule", {})), entry.get("schedule", {}))
        unit = entry.get("duration_unit", spec.get("duration_unit", "samples"))
        dur = _checked_range(ds, entry.get("duration", spec["duration"]), "duration")
        if not math.isfinite(dur) or dur <= 0 or unit not in {"samples", "seconds", "reference_fraction"}:
            raise ValueError("duration must be positive, in samples, seconds or reference_fraction")
        if unit == "reference_fraction":
            dur *= cfg["reference_window"]
            unit = "samples"
        if unit == "samples":
            dur = math.floor(dur + .5)
            if dur < 1:
                raise ValueError("duration rounds to zero samples")
        schedule["duration_unit"] = unit
        if family == "point_outlier" and (unit != "samples" or dur != 1):
            raise ValueError("point outliers require one sample")
        known_keys(entry.get("reference", {}), cfg["reference"].keys() - {"train_fraction", "train_record_ids"}, "reference")
        reference = merge(merge(cfg["reference"], spec.get("reference", {})), entry.get("reference", {}))
        for name in ("tukey_multiplier", "sigma_multiple", "rate_quantile", "iqr_divisor"):
            reference[name] = _checked_range(magnitude, reference[name], name,
                                             probability=name == "rate_quantile", allow_zero=name == "rate_quantile")
        for name in ("local_neighbors", "min_points", "smoothing_window"):
            if isinstance(reference[name], dict):
                reference[name] = math.floor(_checked_range(magnitude, reference[name], name) + .5)
        known_keys(entry.get("params", {}), spec["params"], key)
        params = merge(spec["params"], entry.get("params", {}))
        tolerance_spec = entry.get("tolerance", spec.get("tolerance", cfg["validation"]["tolerance"]))
        tolerance = float(_checked_range(magnitude, tolerance_spec, "tolerance", allow_zero=True))
        return cls(family, subtype, original if original != subtype else None, key, s, magnitude,
                   ds, dur, schedule, params, reference, tolerance)
