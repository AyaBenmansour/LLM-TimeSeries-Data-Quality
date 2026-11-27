"""Checks on realized sensor values. A failed check discards the entire candidate."""

from __future__ import annotations

import numpy as np

from .core import Context, check


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest uninterrupted run of True entries."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return int(np.max(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1), initial=0))


def exceedance(values: np.ndarray, lower: float, upper: float) -> float:
    return float(max(0, np.max(lower - values), np.max(values - upper)))


def error_matches(clean: np.ndarray, values: np.ndarray, expected, tolerance: float) -> bool:
    """Compare in error units, accounting for rounding when adding to a large baseline."""
    roundoff = np.spacing(np.maximum(np.abs(values), np.abs(clean)))
    return bool(np.all(np.abs((values - clean) - expected) <= np.maximum(tolerance, roundoff)))


def range_checks(c: Context, values: np.ndarray, local=None) -> tuple[list, dict]:
    global_excess = exceedance(values, c.stats.lower, c.stats.upper)
    global_ok = np.isfinite([c.stats.lower, c.stats.upper]).all()
    checks = [check("global_interval_available", global_ok,
                    lower=c.stats.lower, upper=c.stats.upper)]
    metrics = {"global_exceedance": global_excess}
    if local is None:
        checks.append(check("outside_global_interval", global_excess > 0,
                            exceedance=global_excess))
    else:
        checks.append(check("inside_global_interval", global_excess == 0,
                            exceedance=global_excess))
        excess = exceedance(values, local.lower, local.upper)
        checks.append(check("outside_local_interval", np.isfinite([local.lower, local.upper]).all()
                            and excess > 0, lower=local.lower, upper=local.upper, exceedance=excess))
        metrics["local_exceedance"] = excess
    return checks, metrics


def burst_checks(c: Context, values: np.ndarray, profile: np.ndarray, amplitude: float,
                 rate_threshold: float, local=None) -> tuple[list, dict]:
    idx = c.idx
    errors = values - c.x[idx]
    min_affected = c.p["min_affected"]
    run = longest_run(np.abs(errors) > c.tolerance)
    checks = [check("consecutive_meaningful_changes", run >= min_affected, longest_run=run,
                    min_affected=min_affected, tolerance=c.tolerance)]
    start = idx[0]
    checks.append(check("unaffected_predecessor_available", start > 0
                        and np.isfinite(c.x[start - 1])))
    gap = c.t[start] - c.t[start - 1]
    checks.append(check("positive_onset_gap", gap > 0, gap_seconds=gap))
    rate = float(abs(values[0] - c.x[start - 1]) / gap)
    checks.append(check("strict_corrupted_onset", np.isfinite(rate_threshold) and rate > rate_threshold,
                        onset_rate=rate, healthy_rate_threshold=rate_threshold))
    checks.append(check("normalized_error_shape", np.max(np.abs(profile)) == 1
                        and error_matches(c.x[idx], values, amplitude * profile, c.tolerance),
                        peak_absolute_error=float(np.max(np.abs(errors)))))
    # A two-sample triangle or half-sine can discretize to an indistinguishable Bias.
    checks.append(check("nonconstant_error_profile", not np.allclose(profile, profile[0],
                                                                    rtol=1e-12, atol=0)))
    guard = c.p["noise_min_samples"]
    checks.append(check("burst_duration_guard", guard is None or len(idx) < guard,
                        samples=len(idx), noise_min_samples=guard))
    intervals, metrics = range_checks(c, values, local)
    metrics.update(onset_rate=rate, healthy_rate_threshold=rate_threshold,
                   meaningful_run=run, peak_absolute_error=float(np.max(np.abs(errors))))
    return checks + intervals, metrics


def noise_checks(c: Context, values: np.ndarray, target: float) -> tuple[list, dict]:
    residuals = c.reference.residuals[c.idx]
    clean_std = float(np.std(residuals))
    checks = [check("healthy_segment_residual_variation", np.isfinite(clean_std) and clean_std > 0,
                    clean_residual_std=clean_std)]
    # Keep the same clean smoother on both sides of the ratio.
    corrupt_std = float(np.std(residuals + (values - c.x[c.idx])))
    achieved = corrupt_std / clean_std
    checks.append(check("increased_residual_variability", achieved > 1,
                        achieved_ratio=achieved, target_ratio=target))
    tolerance = c.p["ratio_relative_tolerance"]
    deviation = abs(achieved - target) / target
    if tolerance is not None:
        checks.append(check("residual_ratio_tolerance", deviation <= tolerance,
                            relative_error=deviation, tolerance=tolerance))
    return checks, {"achieved_ratio": achieved, "target_ratio": target,
                    "relative_error": deviation, "clean_residual_std": clean_std,
                    "corrupt_residual_std": corrupt_std}


def clipping_checks(c: Context, values: np.ndarray, lower: float, upper: float,
                    coefficient: float) -> tuple[list, dict]:
    clean = c.x[c.idx]
    low, high = clean < lower, clean > upper
    # For two-sided clipping, alternating opposite extremes do not form one capped plateau.
    run = max(longest_run(low), longest_run(high))
    checks = [check("actual_capped_run", run >= c.p["min_capped_run"], longest_capped_run=run,
                    required=c.p["min_capped_run"])]
    crosses = low | high
    checks.append(check("in_range_samples_follow_input", np.array_equal(values[~crosses], clean[~crosses])))
    hard_caps = np.where(low[crosses], lower, upper)
    expected = hard_caps + coefficient * (clean[crosses] - hard_caps)
    checks.append(check("conditional_clipping_equation", np.allclose(values[crosses], expected,
                                                                     rtol=1e-12, atol=c.tolerance)))
    return checks, {"longest_capped_run": run, "clipped_count": int(np.sum(crosses)),
                    "clipped_fraction": float(np.mean(crosses))}
