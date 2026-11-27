"""Fixtures isolate the equations from candidate scheduling and location retries."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tsdq.config import merge, preset
from tsdq.core import Context, Reject
from tsdq.reference import Stats
from tsdq.values import SHAPES, burst_shape, transform_value, validate_value_params


def context(key, *, params=None, severity=1, x=None, t=None, idx=None, seed=23):
    if x is None:
        x = np.sin(np.arange(100) / 10)
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x), dtype=float) if t is None else np.asarray(t, dtype=float)
    idx = np.arange(20, 24) if idx is None else np.asarray(idx)
    family = next((name for name, keys in {
        "point_outlier": ("spike", "glitch"), "burst": ("global", "contextual"),
        "flat_line": ("stuck", "outlier_flatline", "saturation")}.items() if key in keys), key)
    subtype = key if key != family else None
    p = merge(preset()["defects"][key]["params"], params or {})
    validate_value_params(key, p)
    stats = Stats(np.linspace(-2, 2, 80), 1.349, 1., "iqr", -2., 2., 0.1, 1., .2, "linear")
    local = replace(stats, scale=.15, lower=.6, upper=1., rate=.03, iqr=.2)
    ref = SimpleNamespace(local=lambda positions: local,
                          residuals=np.random.default_rng(88).normal(0, .2, len(x)))
    return Context(family, subtype, key, p, severity, np.random.default_rng(seed), x, t, idx, {},
                   stats, ref, 1e-12)


@pytest.mark.parametrize("name", SHAPES)
@pytest.mark.parametrize("n", [3, 4, 9, 10])
def test_core_burst_shapes_have_exact_peak(name, n):
    assert np.max(np.abs(burst_shape(name, n, 3))) == 1


def test_triangle_discrete_normalization():
    np.testing.assert_allclose(burst_shape("triangle", 4, 3), [.5, 1, 1, .5])


@pytest.mark.parametrize("name", SHAPES)
def test_every_burst_shape_passes_by_its_observed_onset(name):
    c = context("global", params={"shape": name, "direction": "positive"})
    before = c.x.copy()
    result = transform_value(c)
    phase = result.phases[0]
    assert result.params["shape"] == name
    np.testing.assert_array_equal(c.x, before)
    np.testing.assert_array_equal(phase.idx, c.idx)
    assert phase.metrics["onset_rate"] > c.stats.rate
    assert phase.metrics["peak_absolute_error"] == pytest.approx(6)
    assert len(result.phases) == 1


def test_burst_onset_uses_corrupted_sensor_and_actual_gap_even_when_later_jump_is_large():
    x = np.zeros(8)
    x[2] = -6
    c = context("global", x=x, idx=[2, 3, 4], params={"shape": "alternating", "direction": "positive"})
    with pytest.raises(Reject, match="strict_corrupted_onset"):
        transform_value(c)
    long_gap = np.array([0, 1, 100, 101, 102, 103, 104, 105])
    c = replace(c, x=np.zeros(8), t=long_gap)
    with pytest.raises(Reject, match="strict_corrupted_onset"):
        transform_value(c)


def test_burst_valid_with_flat_internal_observed_gradient():
    c = context("global", params={"direction": "positive", "shape": "triangle"}, x=np.zeros(30),
                idx=[13, 14, 15, 16])
    result = transform_value(c)
    assert result.phases[0].values[1] == pytest.approx(result.phases[0].values[2])


def test_nonrecovering_burst_has_separate_bias_phase_and_disjoint_indices():
    c = context("global", params={"end_mode": "non_recovery", "shape": "triangle", "direction": "positive"})
    result = transform_value(c)
    burst, bias = result.phases
    assert burst.family == "burst" and bias.family == "bias"
    assert burst.idx[-1] + 1 == bias.idx[0] and bias.idx[-1] == len(c.x) - 1
    np.testing.assert_allclose(bias.values - c.x[bias.idx], burst.values[-1] - c.x[burst.idx[-1]])
    with pytest.raises(Reject, match="nonconstant_error_profile"):
        transform_value(replace(c, idx=np.array([20, 21])))


def test_contextual_burst_requires_all_points_globally_normal_and_one_locally_abnormal():
    c = context("contextual", params={"direction": "positive", "shape": "alternating"})
    result = transform_value(c)
    m = result.phases[0].metrics
    assert m["global_exceedance"] == 0 < m["local_exceedance"]
    # The first sample stays inside the global band while another sample escapes.
    modified_x = c.x.copy()
    modified_x[c.idx[1]] = -1.9
    with pytest.raises(Reject, match="inside_global_interval"):
        transform_value(replace(c, x=modified_x))


def test_contextual_burst_skips_zero_iqr_even_with_positive_fallback_scale():
    c = context("contextual")
    local = replace(c.stats, iqr=0., source="sensor_resolution")
    c = replace(c, reference=SimpleNamespace(local=lambda indices: local))
    with pytest.raises(Reject, match="zero_or_unavailable_contextual_iqr"):
        transform_value(c)


def test_rate_scaling_converts_rate_to_value_units():
    c = context("global", params={"shape": "alternating", "scale_mode": "rate", "direction": "positive"})
    c = replace(c, stats=replace(c.stats, rate=2, gap=3))
    result = transform_value(c)
    assert result.params["scale"] == 6
    assert result.phases[0].metrics["peak_absolute_error"] == pytest.approx(36)


@pytest.mark.parametrize("key,expected", [("spike", "global_exceedance"), ("glitch", "local_exceedance")])
def test_single_point_transform_changes_one_observation(key, expected):
    c = context(key, idx=[20], params={"direction": "positive"})
    result = transform_value(c)
    phase = result.phases[0]
    assert list(phase.idx) == [20] and phase.metrics[expected] > 0
    if key == "glitch":
        assert phase.metrics["global_exceedance"] == 0
    with pytest.raises(Reject, match="exactly_one_point"):
        transform_value(replace(c, idx=np.array([20, 21])))


def test_natural_outlier_is_ineligible_as_spike_origin():
    c = context("spike", x=np.r_[np.zeros(20), 50., 0.], idx=[20])
    with pytest.raises(Reject, match="clean_spike_target_in_range"):
        transform_value(c)


def test_bias_on_large_sensor_baseline_keeps_attained_rounded_effect():
    c = context("bias", x=1e12 + np.arange(20), idx=[5, 6, 7],
                params={"direction": "positive", "amplitude_multiplier": .1})
    phase = transform_value(c).phases[0]
    np.testing.assert_array_equal(phase.values, c.x[c.idx] + .1)
    assert phase.metrics["offset"] == phase.values[0] - c.x[c.idx[0]]


@pytest.mark.parametrize("key", ["spike", "bias", "outlier_flatline"])
@pytest.mark.parametrize("s", [0., .5, 1.])
@pytest.mark.parametrize("bounds", [(4., 8.), (5., 15.)])
def test_independent_limits_reach_actual_signal(key, s, bounds):
    c = context(key, idx=[10] if key == "spike" else [10, 11, 12],
                params={"amplitude_multiplier": bounds, "direction": "negative"}, severity=s)
    phase = transform_value(c).phases[0]
    assert phase.values[0] - c.x[c.idx[0]] == pytest.approx(-(bounds[0] + s * (bounds[1] - bounds[0])))
    if key == "outlier_flatline":
        assert len(np.unique(phase.values)) == 1


@pytest.mark.parametrize("exponent", [1, 2])
@pytest.mark.parametrize("mode", ["hold", "continue", "reset", "recovery"])
def test_drift_uses_elapsed_seconds_and_configured_end(exponent, mode):
    c = context("drift", t=[0., 1., 3., 13., 14., 17., 20.], idx=[1, 2, 3], x=[10, 9, 8, 7, 6, 5, 4],
                params={"direction": "positive", "exponent": exponent, "end_mode": mode, "recovery_samples": 2})
    result = transform_value(c)
    phase = result.phases[0]
    errors = phase.values - c.x[phase.idx]
    np.testing.assert_allclose(errors[:3], [0., 3. * (2. / 12.) ** exponent, 3.])
    if mode == "hold":
        assert result.phases[1].family == "bias"
        np.testing.assert_allclose(result.phases[1].values - c.x[4:], 3.)
    elif mode == "recovery":
        np.testing.assert_allclose(errors[3:], [2.25, 0.])
    elif mode == "continue":
        assert phase.idx[-1] == 6 and errors[-1] > 3.
    else:
        assert phase.idx[-1] == 3


@pytest.mark.parametrize("direction,gain", [("attenuate", .25), ("amplify", 3.)])
def test_scaling_baseline_and_observability(direction, gain):
    c = context("scaling", params={"direction": direction})
    phase = transform_value(c).phases[0]
    x = c.x[c.idx]
    np.testing.assert_allclose(phase.values, (x - x.mean()) * gain + x.mean())
    with pytest.raises(Reject, match="dynamic_scaling_segment"):
        transform_value(replace(c, x=np.ones_like(c.x)))


def test_noise_targets_total_clean_detrended_residual_ratio():
    c = context("noise", idx=np.arange(3000), x=np.linspace(0, 1000, 3000), params={"target_ratio": 2})
    result = transform_value(c)
    assert result.params["sigma_injected"] == pytest.approx(.2 * np.sqrt(3))
    m = result.phases[0].metrics
    assert 1.90 < m["achieved_ratio"] < 2.10 and m["relative_error"] > 0
    with pytest.raises(Reject, match="residual_ratio_tolerance"):
        transform_value(replace(c, p=dict(c.p, ratio_relative_tolerance=.00001)))


def test_stuck_uses_plausible_previous_value_and_checks_clean_movement():
    c = context("stuck", x=[0., .2, .3, .7, 1.3, 1.9, 1.1], idx=[2, 3, 4, 5])
    np.testing.assert_allclose(transform_value(c).phases[0].values, .2)
    with pytest.raises(Reject, match="clean_moves_under_freeze"):
        transform_value(replace(c, x=np.ones_like(c.x)))
    with pytest.raises(Reject, match="plausible_held_value"):
        transform_value(replace(c, x=np.array([0., 8., .3, .7, 1.3, 1.9, 1.1])))


@pytest.mark.parametrize("soft", [0, .6])
def test_saturation_tracks_receding_clean_signal_and_requires_crossings(soft):
    x = np.array([0., 2., 3., 4., .5, -2., -3., -4.])
    c = context("saturation", x=x, idx=np.arange(len(x)), params={
        "cap_source": "hardware", "side": "both", "soft_coefficient": soft})
    c = replace(c, meta={"lower_limit": -1., "upper_limit": 1.})
    result = transform_value(c)
    expected = np.clip(x, -1., 1.) + soft * (x - np.clip(x, -1., 1.))
    np.testing.assert_array_equal(result.phases[0].values, expected)
    assert result.phases[0].metrics["longest_capped_run"] == 3
    with pytest.raises(Reject, match="actual_capped_run"):
        transform_value(replace(c, x=np.zeros_like(x)))


def test_quantile_cap_uses_eligible_reference_values():
    c = context("saturation", x=[.5, 4., 4., 4., .5], idx=[0, 1, 2, 3, 4],
                params={"upper_quantile": .75})
    result = transform_value(c)
    assert result.params["upper_cap"] == pytest.approx(np.quantile(c.stats.values, .75))
    assert result.phases[0].metrics["clipped_count"] == 3


@pytest.mark.parametrize("key,invalid", [
    ("bias", {"amplitude_multiplier": {"choices": [1, -1, 2]}}), ("global", {"shape": "rectangle"}),
    ("global", {"probabilities": [1.0]}), ("noise", {"target_ratio": .5}),
    ("scaling", {"gain_down": 1.1}), ("stuck", {"plausible_quantiles": [.8, .7]}),
    ("saturation", {"soft_coefficient": 1}), ("drift", {"recovery_samples": 0})])
def test_invalid_domains_reject_configuration(key, invalid):
    with pytest.raises(ValueError):
        context(key, params=invalid)
