"""Healthy calibration statistics and candidate-specific clean neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Stats:
    values: np.ndarray
    iqr: float
    scale: float
    source: str
    lower: float
    upper: float
    rate: float
    gap: float
    residual_scale: float
    quantile_method: str


def validate_reference_config(config: dict) -> None:
    menus = {
        "scale": {"iqr", "std"},
        "threshold_mode": {"tukey", "sigma", "quantile"},
        "smoother": {"rolling_mean", "rolling_median"},
        "residual_scale": {"iqr", "std"},
        "zero_scale": {"skip", "std", "resolution", "std_then_resolution"},
        "local_zero_scale": {"skip", "std", "resolution", "std_then_resolution"},
    }
    for key, choices in menus.items():
        if config[key] not in choices:
            raise ValueError(f"{key} must be one of {sorted(choices)}")
    for key in ("local_neighbors", "smoothing_window", "min_points"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("tukey_multiplier", "sigma_multiple", "iqr_divisor"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    lo, hi = config["quantiles"]
    if not 0 <= lo < hi <= 1 or not 0 <= config["rate_quantile"] <= 1:
        raise ValueError("Reference quantiles must be ordered probabilities in [0, 1]")
    np.quantile([0, 1], .5, method=config["quantile_method"])


def _validate_series(x: np.ndarray, t: np.ndarray) -> None:
    if x.ndim != 1 or t.shape != x.shape:
        raise ValueError("Expected aligned one-dimensional clean values and times")
    if np.isinf(x).any() or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
        raise ValueError("Clean values must be finite or NaN and times strictly increasing")


def _residuals(x: np.ndarray, groups: np.ndarray, config: dict) -> np.ndarray:
    """Smooth within each contiguous finite segment of one partition."""
    valid = np.isfinite(x)
    edges = np.r_[0, np.flatnonzero((valid[1:] != valid[:-1]) | (groups[1:] != groups[:-1])) + 1, len(x)]
    result = np.full(len(x), np.nan)
    for start, end in zip(edges[:-1], edges[1:]):
        if start == end or not valid[start]:
            continue
        rolling = pd.Series(x[start:end]).rolling(config["smoothing_window"], min_periods=1, center=True)
        smooth = rolling.mean() if config["smoother"] == "rolling_mean" else rolling.median()
        result[start:end] = x[start:end] - smooth.to_numpy()
    return result


class Reference:
    """Global calibrations use training rows. Local checks use the clean target context."""

    def __init__(self, x, t, train_mask, config: dict, meta: dict, *, training=None):
        validate_reference_config(config)
        self.config = dict(config)
        self.meta = dict(meta)
        self.x = np.asarray(x, dtype=float).copy()
        self.t = np.asarray(t, dtype=float).copy()
        _validate_series(self.x, self.t)
        mask = np.asarray(train_mask)
        if mask.shape != self.x.shape or mask.dtype != np.bool_:
            raise ValueError("train_mask must contain one boolean per target record")
        self.residuals = _residuals(self.x, mask, self.config)
        if training is None:
            self.global_stats = self._stats(self.x, self.t, self.residuals, mask, local=False)
        else:
            fit_x, fit_t = (np.asarray(a, dtype=float) for a in training)
            _validate_series(fit_x, fit_t)
            all_fit = np.ones(len(fit_x), dtype=bool)
            residuals = _residuals(fit_x, all_fit, self.config)
            self.global_stats = self._stats(fit_x, fit_t, residuals, all_fit, local=False)
        for array in (self.x, self.t, self.residuals):
            array.setflags(write=False)

    def _scale(self, values, iqr, local):
        cfg = self.config
        scale = iqr / cfg["iqr_divisor"] if cfg["scale"] == "iqr" else float(np.std(values))
        if scale > 0:
            return scale, cfg["scale"]
        policy = cfg["local_zero_scale" if local else "zero_scale"]
        if policy in {"std", "std_then_resolution"}:
            std = float(np.std(values))
            if std > 0:
                return std, "std_fallback"
        if policy in {"resolution", "std_then_resolution"}:
            resolution = self.meta.get("resolution")
            if resolution is not None:
                if not np.isfinite(resolution) or resolution <= 0:
                    raise ValueError("Sensor resolution must be positive and finite")
                return float(resolution), "sensor_resolution"
        return np.nan, "zero_scale"

    def _stats(self, x, t, residuals, mask, *, local):
        cfg = self.config
        method = cfg["quantile_method"]
        valid = mask & np.isfinite(x)
        values = x[valid]
        values.setflags(write=False)
        adjacent = valid[:-1] & valid[1:]
        gaps = np.diff(t)[adjacent]
        gap = float(np.median(gaps)) if gaps.size else np.nan
        rate = np.nan
        if gaps.size:
            rates = abs(np.diff(x)[adjacent]) / gaps
            rate = float(np.quantile(rates, cfg["rate_quantile"], method=method))
        if values.size < cfg["min_points"]:
            return Stats(values, np.nan, np.nan, "insufficient_points", np.nan, np.nan,
                         rate, gap, np.nan, method)
        q1, q3 = np.quantile(values, [.25, .75], method=method)
        iqr = float(q3 - q1)
        scale, source = self._scale(values, iqr, local)
        mode = cfg["threshold_mode"]
        if mode == "tukey":
            margin = cfg["tukey_multiplier"] * iqr
            lo, hi = q1 - margin, q3 + margin
            if iqr == 0 and np.isfinite(scale):
                lo, hi = values.min(), values.max()
        elif mode == "sigma":
            center = float(np.mean(values))
            lo, hi = center - cfg["sigma_multiple"] * scale, center + cfg["sigma_multiple"] * scale
        else:
            lo, hi = np.quantile(values, cfg["quantiles"], method=method)
        residual = residuals[valid]
        if cfg["residual_scale"] == "std":
            noise_scale = np.std(residual)
        else:
            low, high = np.quantile(residual, [.25, .75], method=method)
            noise_scale = (high - low) / cfg["iqr_divisor"]
        return Stats(values, iqr, scale, source, float(lo), float(hi), rate, gap, float(noise_scale), method)

    def local(self, idx) -> Stats:
        positions = np.asarray(idx, dtype=int)
        if positions.size == 0 or positions.min() < 0 or positions.max() >= len(self.x):
            raise ValueError("Local reference requires target positions within the channel")
        start, stop = int(positions.min()), int(positions.max()) + 1
        width = self.config["local_neighbors"]
        left, right = max(0, start - width), min(len(self.x), stop + width)
        # Slice before computing rates. A candidate on a million-row stream still
        # needs only its immediate neighbors. Avoid allocating arrays for long tails.
        left_slice, right_slice = slice(left, start), slice(stop, right)
        x = np.r_[self.x[left_slice], np.nan, self.x[right_slice]]
        t = np.r_[self.t[left_slice], self.t[start], self.t[right_slice]]
        valid = np.isfinite(x)
        residual = _residuals(x, valid, self.config)
        return self._stats(x, t, residual, valid, local=True)
