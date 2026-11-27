import numpy as np
import pandas as pd
import pytest

from tsdq.availability import transform_availability
from tsdq.config import merge, preset
from tsdq.core import Context, Reject
from tsdq.data import adapt, elapsed_seconds, shift_timestamps
from tsdq.reference import Reference
from tsdq import DefectInjector


@pytest.fixture
def regular_sensor():
    return pd.DataFrame({"timestamp": np.arange(12), "value": np.sin(np.arange(12))})


@pytest.fixture
def irregular_sensor():
    return pd.DataFrame({"timestamp": [0, 1, 1.2, 4, 10, 11, 50], "sensor_id": "a",
                         "record_id": list("abcdefg"), "value": [2, 4, 3, 1, 2, 2, 1]})


@pytest.fixture
def asynchronous(irregular_sensor):
    other = pd.DataFrame({"timestamp": [.5, 3.5, 7.], "sensor_id": "b", "record_id": list("xyz"),
                          "value": [1., np.nan, 3.]})
    return pd.concat([irregular_sensor, other], ignore_index=True)


def context(key, t, idx, params=None, s=.5, *, meta=None, x=None):
    config = preset()
    t = np.asarray(t, dtype=float)
    x = np.sin(t) if x is None else x
    reference = Reference(x, t, np.ones(len(t), bool), config["reference"], {})
    return Context("missing_values" if key == "missing_values" else "timestamp",
                   None if key == "missing_values" else key, key,
                   merge(config["defects"][key]["params"], params or {}), s,
                   np.random.default_rng(81), x, t, np.asarray(idx), meta or {},
                   reference.global_stats, reference, 0)


def test_adapters_preserve_asynchronous_rows_and_existing_nans(asynchronous):
    original = asynchronous.copy(deep=True)
    records = adapt(asynchronous.assign(quality="eligible"), {}, {})
    assert records["record_id"].tolist() == asynchronous["record_id"].tolist()
    assert records["value"].isna().sum() == 1
    assert list(records["quality"].unique()) == ["eligible"]
    assert len(records) == len(original)
    pd.testing.assert_frame_equal(original, asynchronous)
    with pytest.raises(ValueError, match="strictly increasing"):
        adapt(asynchronous.iloc[::-1], {}, {})


def test_wide_roundtrip_channel_identity():
    data = pd.DataFrame({"t": [1, 3], "id": ["r:0", "r:1"], "A": [2., np.nan], "B": [3., 4.],
                         "room": "lab"})
    records = adapt(data, {"format": "wide", "timestamp_col": "t", "id_col": "id",
                           "channels": ["A", "B"]}, {})
    assert records.record_id.is_unique and len(records) == 4
    assert records.source_record_id.tolist() == ["r:0", "r:1", "r:0", "r:1"]
    assert records.room.eq("lab").all()
    assert records.query("sensor_id == 'A'").value.isna().sum() == 1


def test_single_channel_numeric_precision(regular_sensor):
    assert adapt(regular_sensor, {}, {}).sensor_id.nunique() == 1
    times = pd.Series([2**60, 2**60+1, 2**60+3])
    np.testing.assert_allclose(elapsed_seconds(times, {"time_unit": "ns"}), [0, 1e-9, 3e-9], atol=0)
    advanced = shift_timestamps(times, np.full(3, 2e-9), {"time_unit": "ns"})
    assert advanced.tolist() == [2**60+2, 2**60+3, 2**60+5]
    with pytest.raises(ValueError, match="precision"):
        shift_timestamps(times, np.full(3, .25e-9), {"time_unit": "ns"})
    with pytest.raises(ValueError, match="float64"):
        adapt(pd.DataFrame({"timestamp": [0, 1], "value": [2**53, 2**53+1]}), {}, {})


def test_datetime_shift_preserves_zone_and_nanoseconds():
    times = pd.Series(pd.date_range("2026-01-01", periods=3, freq="ns", tz="Europe/Paris"))
    records = adapt(pd.DataFrame({"timestamp": times, "value": [1, 2, 3]}), {}, {})
    advanced = shift_timestamps(records.timestamp, [0, 2e-9, 0], {})
    assert str(advanced.dt.tz) == "Europe/Paris"
    assert advanced.iloc[1].value - times.iloc[1].value == 2
    assert advanced.iloc[0] == times.iloc[0]


def test_training_mask_excludes_values_rates_and_smoothing_across_partition():
    config = preset()["reference"]
    mask = np.array([True, True, False, False, True, True])
    x = np.array([0., 1., 11., 21., 100., 101.])
    t = np.arange(6)
    before = Reference(x, t, mask, config, {})
    x[~mask] = 1e8
    after = Reference(x, t, mask, config, {})
    assert after.global_stats.rate == before.global_stats.rate == 1
    assert after.global_stats.residual_scale == before.global_stats.residual_scale
    assert after.global_stats.upper == before.global_stats.upper
    context_stats = before.local([2, 3])
    assert context_stats.rate == 1
    np.testing.assert_array_equal(context_stats.values, [0., 1., 100., 101.])
    other = Reference(x, t, np.zeros(6, bool), config, {}, training=([9, 10, 11], [1, 2, 3]))
    np.testing.assert_array_equal(other.global_stats.values, [9, 10, 11])


@pytest.mark.parametrize("policy,expected", [("skip", "zero_scale"), ("resolution", "sensor_resolution"),
                                            ("std", "zero_scale")])
def test_constant_reference_and_local_shortage(policy, expected):
    cfg = {**preset()["reference"], "zero_scale": policy, "local_neighbors": 1}
    ref = Reference(np.ones(5), np.arange(5), np.ones(5, bool), cfg, {"resolution": .125})
    assert ref.global_stats.source == expected
    assert ref.global_stats.scale == .125 if policy == "resolution" else np.isnan(ref.global_stats.scale)
    assert ref.local([0, 1]).source == "insufficient_points"
    assert ref.local([2]).source == "zero_scale"


@pytest.mark.parametrize("mode", ["tukey", "sigma", "quantile"])
def test_threshold_modes_and_divisor(mode):
    x = np.array([0, 1, 2, 3, 11])
    cfg = {**preset()["reference"], "iqr_divisor": 2., "quantiles": [.2,.8],
           "threshold_mode": mode, "sigma_multiple": 2.}
    stats = Reference(x, np.arange(5), np.ones(5, bool), cfg, {}).global_stats
    assert stats.scale == 1.
    expected = {"tukey": (-2., 6.), "sigma": (x.mean()-2, x.mean()+2),
                "quantile": np.quantile(x, [.2,.8])}[mode]
    np.testing.assert_allclose([stats.lower, stats.upper], expected)


@pytest.mark.parametrize("sign,multiplier", [("positive",1), ("negative",-1)])
@pytest.mark.parametrize("bounds", [{"min": 10., "max": 30.}, {"min": 40., "max": 50.}])
@pytest.mark.parametrize("s", [0., .5, 1.])
def test_absolute_shift_ignores_large_irregular_gaps(irregular_sensor, sign, multiplier, bounds, s):
    c = context("fixed_shift", irregular_sensor.timestamp, [2,3,4], {"offset_mode": "absolute",
                "offset": bounds, "offset_unit": "ms", "sign": sign}, s)
    candidate = transform_availability(c)
    expected = multiplier * (bounds["min"] + s * (bounds["max"] - bounds["min"])) / 1000
    np.testing.assert_allclose(candidate.phases[0].offsets, expected)
    assert candidate.phases[0].values is None


def test_relative_regular_shift_and_optional_grid_fraction(regular_sensor):
    for fraction in (0, .5):
        c = context("fixed_shift", regular_sensor.timestamp, [2,3], {"steps": 3, "fractional_offset": fraction,
                                                                     "sign": "positive"})
        assert transform_availability(c).params["signed_offset_seconds"] == 3 + fraction


@pytest.mark.parametrize("distribution", ["normal", "uniform"])
@pytest.mark.parametrize("policy", ["clip", "redraw"])
def test_jitter_local_guards_and_draw_mean(distribution, policy):
    t = np.cumsum(np.tile([1., .6, 1.3], 3000))
    c = context("jitter", t, np.arange(1,len(t)-1), {"distribution": distribution, "bound_policy": policy,
                                                      "std_fraction": .15})
    offsets = transform_availability(c).phases[0].offsets
    assert np.all(abs(offsets) < .45 * np.minimum(np.diff(t)[:-1], np.diff(t)[1:]))
    assert abs(offsets.mean()) < .006
    t[c.idx] += offsets
    assert np.all(np.diff(t) > 0)


def test_clock_drift_real_time_and_ppm(irregular_sensor):
    idx = np.arange(1,6)
    for s, rate in [(0., 1e-4), (.5, .001), (1., .01)]:
        c = context("clock_drift", irregular_sensor.timestamp, idx, {"sign": "negative"}, s)
        expected = -rate * (c.t[idx] - c.t[idx[0]])
        np.testing.assert_allclose(transform_availability(c).phases[0].offsets, expected)
        ppm = context("clock_drift", c.t, idx, {"rate_unit": "ppm", "rate": rate * 1e6, "sign":"negative"})
        np.testing.assert_allclose(transform_availability(ppm).phases[0].offsets, expected)


@pytest.mark.parametrize("key", ["duplicate", "near_duplicate", "dst_shift", "time_stretch"])
def test_optional_clock_variants_require_enabling(key):
    with pytest.raises(ValueError, match="explicitly"):
        transform_availability(context(key, np.arange(7), [2,3]))


def test_stretch_copy_and_dst_parameters(irregular_sensor):
    t = irregular_sensor.timestamp.to_numpy()
    both = [2,3]
    c = context("time_stretch", t, both, {"enabled": True, "factor": 2})
    np.testing.assert_allclose(transform_availability(c).phases[0].offsets, [0, t[3]-t[2]])
    for key in ("duplicate", "near_duplicate"):
        c = context(key, t, both, {"enabled": True})
        copies = transform_availability(c).phases[0].copies
        if key == "duplicate":
            np.testing.assert_equal(copies, [0,0])
        else:
            assert np.all((copies > 0) & (copies < np.diff(t)[both]))
    with pytest.raises(Reject, match="timezone"):
        transform_availability(context("dst_shift", t, both, {"enabled": True}))
    c = context("dst_shift", t, both, {"enabled": True, "sign": "negative"}, meta={"timezone":"Europe/Paris"})
    np.testing.assert_equal(transform_availability(c).phases[0].offsets, [-3600.,-3600.])


@pytest.mark.parametrize("idx,pattern", [([1], "isolated"), ([2,3], "block"), ([4,5], "terminal"),
                                         (list(range(6)), "whole_channel")])
@pytest.mark.parametrize("representation", ["nan", "drop_records"])
def test_missing_patterns_and_representation(idx, pattern, representation):
    c = context("missing_values", np.arange(6), idx, {"representation": representation})
    phase = transform_availability(c).phases[0]
    assert phase.subtype is None and phase.missing == representation
    assert phase.metrics["pattern"] == pattern
    assert phase.metrics["channel_missing_fraction"] == len(idx) / 6
    c.x[idx[0]] = np.nan
    with pytest.raises(Reject, match="existing_measurements"):
        transform_availability(c)


@pytest.mark.parametrize("representation", ["nan", "drop_records"])
def test_async_missing_spans_preserve_ids_and_time(asynchronous, representation):
    reference = asynchronous.copy()
    # A two-second interval crosses channel A's observations at 1 and 1.2.
    # Channel B has an independent clock with a healthy sample at 1.5.
    data = asynchronous.copy()
    data.loc[data.record_id == "y", ["timestamp", "value"]] = [1.5, 9.]
    plan = [{"family": "missing_values", "duration": 2, "duration_unit": "seconds",
             "params": {"representation": representation},
             "schedule": {"channels": "all", "onset_seconds": 1}}]
    result = DefectInjector().inject(data, plan, training_df=reference)
    assert not result.failures
    assert len(result.episodes) == 1 and "multi_channel" in result.episodes[0]["patterns"]
    assert set(result.masks.record_id) == {"b", "c", "y"}
    assert result.manifest["missing"]["fraction_of_stream"] == 3/len(data)
    expected = data.set_index("record_id").sort_index()
    actual = result.corrupted.set_index("record_id").sort_index()
    if representation == "nan":
        assert actual.loc[["b","c","y"], "value"].isna().all()
    else:
        assert {"b", "c", "y"}.isdisjoint(actual.index)
        assert all(not detail["identifiable_given_context"]
                   for detail in result.episodes[0]["channel_details"].values())
    untouched = sorted(set(expected.index) - {"b", "c", "y"})
    pd.testing.assert_frame_equal(expected.loc[untouched, actual.columns], actual.loc[untouched])


@pytest.mark.parametrize("subtype", ["duplicate", "near_duplicate"])
def test_duplicate_creates_actual_extra_record(irregular_sensor, subtype):
    plan = [{"family":"timestamp", "subtype":subtype, "params":{"enabled":True},
             "duration":2, "schedule":{"onset_index":3}}]
    result = DefectInjector().inject(irregular_sensor, plan, training_df=irregular_sensor)
    assert not result.failures
    assert len(result.corrupted) == len(irregular_sensor) + 2
    assert result.corrupted.record_id.is_unique and result.corrupted.timestamp.is_monotonic_increasing
    assert set(result.masks.source_record_id) == {"d","e"}
    assert set(result.masks.record_id).isdisjoint(irregular_sensor.record_id)
    extra = result.corrupted[result.corrupted.record_id.isin(result.masks.record_id)]
    assert extra.value.tolist() == irregular_sensor.value.iloc[[3,4]].tolist()
    assert (extra.timestamp.tolist() == irregular_sensor.timestamp.iloc[[3,4]].tolist()) == (subtype == "duplicate")


def test_order_collision_rejection_is_atomic(regular_sensor):
    # Moving one interior observation forwards exactly one step collides with its successor.
    plan = [{"family":"timestamp", "subtype":"fixed_shift", "params":{"offset_mode":"absolute",
             "offset":1,"sign":"positive"}, "duration":1,"schedule":{"onset_index":6,"max_attempts":2}}]
    result = DefectInjector().inject(regular_sensor, plan)
    assert not result.episodes and result.masks.empty
    assert result.failures[0].reason == "timestamp_collisions"
    pd.testing.assert_frame_equal(result.clean, result.corrupted)


def test_whole_channel_shift_has_no_unassisted_timing_evidence(irregular_sensor):
    plan = [{"family":"timestamp", "subtype":"fixed_shift", "params":{"offset_mode":"absolute",
             "offset":.025,"sign":"negative"}, "schedule":{"persistence":"whole_channel"}}]
    result = DefectInjector().inject(irregular_sensor, plan, training_df=irregular_sensor)
    assert not result.failures and not result.episodes[0]["channel_details"]["a"]["identifiable_given_context"]
    np.testing.assert_allclose(result.corrupted.timestamp, irregular_sensor.timestamp-.025)
    pd.testing.assert_series_equal(result.corrupted.value, irregular_sensor.value.astype(float))


def test_jitter_validates_encoded_bound_over_long_fractional_clock():
    data = pd.DataFrame({"value": np.arange(100), "timestamp": np.arange(100.)})
    plan = [{"family":"timestamp","subtype":"jitter","duration":60,"schedule":{"onset_index":31}}]
    result = DefectInjector().inject(data, plan)
    assert not result.failures
    assert np.all(abs(result.masks.recorded_timestamp - result.masks.true_timestamp) < .45)
    assert result.corrupted.timestamp.is_monotonic_increasing


def test_unsigned_epoch_keeps_signed_ground_truth_displacement():
    data = pd.DataFrame({"value": np.arange(20), "timestamp": np.arange(20, dtype=np.uint64) + 2**60})
    plan = [{"family":"timestamp","subtype":"fixed_shift","duration":3,
             "params":{"offset_mode":"absolute","offset_unit":"ns","offset":1,"sign":"negative"},
             "schedule":{"persistence":"whole_channel"}}]
    result = DefectInjector({"adapter":{"time_unit":"ns"}}).inject(data, plan, training_df=data)
    assert not result.failures
    achieved = result.episodes[0]["channel_details"]["sensor"]["achieved"]
    np.testing.assert_allclose(achieved["actual_offset_seconds"], -1e-9, atol=0)
    assert result.corrupted.timestamp.tolist() == [int(t) - 1 for t in data.timestamp]
