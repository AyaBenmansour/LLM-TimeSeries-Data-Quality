"""Lossless record identities and explicit time units for sensor tables."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype, is_complex_dtype, is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype,
)


TIME_UNITS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1., "min": 60., "h": 3600., "d": 86400.}


def unit_seconds(unit: str) -> float:
    if unit not in TIME_UNITS:
        raise ValueError(f"Unknown time unit {unit!r}. Choose from {list(TIME_UNITS)}")
    return TIME_UNITS[unit]


def _identifiers(series: pd.Series, name: str) -> None:
    if series.isna().any() or series.duplicated().any():
        raise ValueError(f"{name} must be unique and non-null")
    for value in series:
        if not isinstance(value, (str, int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise ValueError(f"{name} must contain text or numeric identifiers")
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError(f"{name} must contain finite identifiers")


def _time_column(series: pd.Series, config: dict) -> pd.Series:
    kind = config.get("timestamp_type", "auto")
    if kind not in {"auto", "numeric", "datetime"}:
        raise ValueError("timestamp_type must be auto, numeric or datetime")
    if series.isna().any() or is_bool_dtype(series):
        raise ValueError("Clean timestamps must be non-null numbers or datetimes")
    if kind == "numeric" or (kind == "auto" and is_numeric_dtype(series)):
        unit_seconds(config.get("time_unit", "s"))
        numeric = pd.to_numeric(series, errors="raise")
        if is_complex_dtype(numeric) or not np.isfinite(numeric).all():
            raise ValueError("Numeric timestamps must be finite")
        return numeric
    # Mixed UTC offsets can occur in a CSV spanning a DST transition. An explicit
    # zone supplies the civil-time convention for reconstructing these instants.
    zone = config.get("timezone")
    try:
        parsed = pd.to_datetime(series, errors="raise", format="mixed")
        if not is_datetime64_any_dtype(parsed):
            raise ValueError("Mixed UTC offsets")
    except ValueError:
        if not zone:
            raise
        instants = [pd.Timestamp(value) for value in series]
        if not all(instant.tzinfo is not None for instant in instants):
            raise ValueError("Mixed offsets require timezone-aware inputs and an adapter timezone")
        parsed = pd.to_datetime(series, errors="raise", format="mixed", utc=True)
    if zone:
        parsed = parsed.dt.tz_localize(zone) if parsed.dt.tz is None else parsed.dt.tz_convert(zone)
    return parsed.dt.as_unit("ns")


def adapt(df: pd.DataFrame, config: dict, metadata: dict | None = None) -> pd.DataFrame:
    """Return independent canonical records in input order within each channel."""
    if not isinstance(df, pd.DataFrame) or df.empty or not df.columns.is_unique:
        raise ValueError("Supply a nonempty table with unique column names")
    source = df.copy(deep=True).reset_index(drop=True)
    time = config.get("timestamp_col", "timestamp")
    sensor = config.get("sensor_col", "sensor_id")
    value = config.get("value_col", "value")
    identity = config.get("id_col", "record_id")
    if time not in source or len({time, sensor, value, identity}) != 4:
        raise ValueError("Check timestamp_col, sensor_col, value_col and id_col mappings")
    source[time] = _time_column(source[time], config)
    if identity not in source:
        source[identity] = np.arange(len(source))
    _identifiers(source[identity], identity)
    kind = config.get("format", "long")
    if kind == "long":
        if value not in source:
            raise ValueError(f"Missing value column {value!r}")
        if sensor not in source:
            source[sensor] = config.get("sensor_name", "sensor")
        mapping = {time: "timestamp", sensor: "sensor_id", value: "value", identity: "record_id"}
        result = source.rename(columns=mapping)
    elif kind == "wide":
        channels = config.get("channels") or [c for c in source if c not in {time, identity}]
        if (not channels or len(set(channels)) != len(channels)
                or any(c not in source or c in {time, identity} for c in channels)):
            raise ValueError("Wide format requires distinct, existing measurement columns")
        identifiers = [c for c in source if c not in channels]
        reserved = {"sensor_id", "value", "source_record_id", "record_id", "timestamp"}
        if reserved.intersection(set(identifiers) - {time, identity}):
            raise ValueError("Wide metadata columns collide with canonical fields")
        names = {identity: "source_record_id", time: "timestamp"}
        blocks = []
        for channel in channels:
            block = source[identifiers].rename(columns=names).copy()
            block["sensor_id"] = str(channel)
            block["value"] = source[channel]
            block["record_id"] = [json.dumps([str(channel), i], separators=(",", ":"))
                                  for i in block["source_record_id"].tolist()]
            blocks.append(block)
        result = pd.concat(blocks, ignore_index=True)
    else:
        raise ValueError("format must be long or wide")
    if not result.columns.is_unique:
        raise ValueError("Column mappings collide with existing canonical fields")
    if result["sensor_id"].isna().any():
        raise ValueError("sensor_id must be non-null")
    original_sensor_count = result["sensor_id"].nunique()
    result["sensor_id"] = result["sensor_id"].astype(str)
    if original_sensor_count != result["sensor_id"].nunique():
        raise ValueError("Channel identifiers collide after conversion to text")
    _identifiers(result["record_id"], "record_id")
    original = result["value"]
    numeric = pd.to_numeric(original, errors="raise")
    if is_complex_dtype(numeric):
        raise ValueError("Sensor measurements must be real-valued")
    converted = numeric.astype(float)
    check_values = numeric if is_integer_dtype(numeric) else original
    if any(isinstance(before, (int, np.integer)) and int(before) != int(after)
           for before, after in zip(check_values, converted)):
        raise ValueError("Integer measurement is not exactly representable as float64")
    result["value"] = converted
    if np.isinf(result["value"]).any():
        raise ValueError("Clean values may be missing but must not be infinite")
    for channel, group in result.groupby("sensor_id", sort=False):
        times = group["timestamp"]
        if not times.is_monotonic_increasing or times.duplicated().any():
            raise ValueError(f"Channel {channel!r} requires strictly increasing clean timestamps")
    canonical = ["record_id", "timestamp", "sensor_id", "value"]
    return result[canonical + [c for c in result if c not in canonical]]


def elapsed_seconds(timestamps: pd.Series, config: dict, origin: Any = None) -> np.ndarray:
    """Subtract an exact origin *before* converting a time difference to seconds."""
    times = pd.Series(timestamps)
    if times.empty:
        return np.empty(0)
    if origin is None:
        origin = times.min()
    if is_datetime64_any_dtype(times):
        return (times - pd.Timestamp(origin)).dt.total_seconds().to_numpy(dtype=float)
    if is_integer_dtype(times):
        # Avoid int64 overflow for signed differences across the entire numeric range.
        whole = int(origin)
        fraction = float(origin - whole)
        differences = np.fromiter((int(t) - whole for t in times), dtype=float, count=len(times)) - fraction
    else:
        differences = times.to_numpy(dtype=float) - origin
    return differences * unit_seconds(config.get("time_unit", "s"))


def shift_timestamps(timestamps: pd.Series, seconds: np.ndarray, config: dict) -> pd.Series:
    """Add physical-time offsets while preserving datetime zones and unchanged ticks."""
    times = pd.Series(timestamps)
    offsets = np.broadcast_to(np.asarray(seconds, dtype=float), (len(times),))
    if not np.isfinite(offsets).all():
        raise ValueError("Time offsets must be finite")
    if is_datetime64_any_dtype(times):
        delta = pd.to_timedelta(np.rint(offsets * 1e9), unit="ns")
        return times + delta
    ticks = offsets / unit_seconds(config.get("time_unit", "s"))
    precision = config.get("numeric_precision", "fractional")
    if precision not in {"integer_tick", "fractional"}:
        raise ValueError("numeric_precision must be fractional or integer_tick")
    if precision == "integer_tick":
        ticks = np.rint(ticks)
    if is_integer_dtype(times) and np.all(ticks == np.rint(ticks)):
        result = [int(t) + int(d) for t, d in zip(times, ticks)]
        limits = np.iinfo(getattr(times.dtype, "numpy_dtype", times.dtype))
        if any(t < limits.min or t > limits.max for t in result):
            raise ValueError("Timestamp shift exceeds the original integer storage range")
        return pd.Series(result, index=times.index, dtype=times.dtype, name=times.name)
    if is_integer_dtype(times) and any(abs(int(t)) >= 2**53 for t in times):
        raise ValueError("Fractional shift loses integer epoch precision. Select integer_tick or datetime.")
    return times.astype(float) + ticks


def fit_jitter_bounds(timestamps: pd.Series, offsets: np.ndarray, gap_bounds_seconds: np.ndarray,
                      max_displacement_seconds: float | None, config: dict) -> np.ndarray:
    """For clipped jitter, keep the encoded displacement inside each declared bound.

    Adding an offset close to a strict bound can place the *encoded* time on or
    beyond that bound, even when a floating-point offset passed the check.
    """
    times = pd.Series(timestamps)
    offsets = np.array(offsets, dtype=float, copy=True)
    gap_bounds = np.asarray(gap_bounds_seconds, dtype=float)
    shifted = shift_timestamps(times, offsets, config)
    if is_datetime64_any_dtype(times):
        encoded = (shifted - times).dt.total_seconds().to_numpy()
        quantum = 1e-9
    else:
        unit = unit_seconds(config.get("time_unit", "s"))
        if is_integer_dtype(shifted) and is_integer_dtype(times):
            encoded = np.fromiter((int(s)-int(t) for s, t in zip(shifted, times)), dtype=float) * unit
        else:
            encoded = (shifted.to_numpy(dtype=float) - times.to_numpy(dtype=float)) * unit
        if config.get("numeric_precision") == "integer_tick":
            quantum = unit
        else:
            quantum = np.spacing(np.maximum(abs(times.to_numpy(dtype=float)), abs(shifted.to_numpy(dtype=float)))) * unit
    bad = abs(encoded) >= gap_bounds
    limit = gap_bounds
    if max_displacement_seconds is not None:
        bad |= abs(encoded) > max_displacement_seconds
        limit = np.minimum(gap_bounds, max_displacement_seconds)
    # Use one encoding tick of clearance only for samples that failed after encoding.
    interior = np.maximum(0., np.nextafter(limit - quantum, 0.))
    offsets[bad] = np.sign(offsets[bad]) * interior[bad]
    return offsets
