"""Exercise materialized timestamps and record identity at the public API boundary."""
import numpy as np
import pandas as pd
import pytest

from tsdq import DefectInjector


def stream(times, ids=None):
    n = len(times)
    return pd.DataFrame({"record_id": range(n) if ids is None else ids, "timestamp": times,
                         "sensor_id": "meter", "value": np.arange(n, dtype=float) ** 2})


def clock(subtype, onset, duration=1, **params):
    return {"family": "timestamp", "subtype": subtype, "duration": duration, "params": params,
            "schedule": {"onset_index": onset, "max_attempts": 1}}


@pytest.mark.parametrize("subtype", ["duplicate", "near_duplicate", "dst_shift", "time_stretch"])
def test_optional_clock_generators_require_a_feature_gate(subtype):
    data = stream(np.arange(12))
    with pytest.raises(ValueError, match="explicitly"):
        DefectInjector().inject(data, [clock(subtype, 8)])


@pytest.mark.parametrize("ids", [[-1, *range(1, 12)], ["__tsdq_copy_1", *[f"s{i}" for i in range(1, 12)]]])
def test_exact_duplicate_adds_unique_records_and_preserves_originals(ids):
    data = stream(np.arange(12), ids)
    cfg = {"windows": {"lengths": [9], "scoring_basis": "visible"}}
    result = DefectInjector(cfg).inject(data, [clock("duplicate", 5, 2, enabled=True)])
    assert not result.failures
    assert result.corrupted.record_id.is_unique and len(result.corrupted) == len(data) + 2
    pd.testing.assert_frame_equal(result.clean, data)
    retained = result.corrupted.loc[result.corrupted.record_id.isin(ids)].reset_index(drop=True)
    pd.testing.assert_frame_equal(retained, data)
    for m in result.masks.itertuples():
        source = data.set_index("record_id").loc[m.source_record_id]
        added = result.corrupted.set_index("record_id").loc[m.record_id]
        assert m.record_id not in ids
        assert m.true_timestamp == m.recorded_timestamp == source.timestamp == added.timestamp
        assert m.clean_value == m.corrupt_value == source.value == added.value
    ep = result.episodes[0]
    assert ep["support_record_ids"] == ids[5:7]
    assert ep["record_ids"] == result.masks.record_id.tolist()
    assert result.windows[0]["visible_class"] == "single"
    assert set(result.windows[0]["episodes"][0]["visible_record_ids"]) == set(ep["record_ids"])


def test_negative_near_duplicate_is_inserted_before_source_with_real_gap_bound():
    times = [0., 1., 4., 6., 6.5, 10., 18., 18.2, 91., 110., 111., 120.]
    data = stream(times)
    result = DefectInjector().inject(data, [clock("near_duplicate", 7, enabled=True,
                                                   sign="negative", gap_fraction=.08)])
    assert not result.failures
    m = result.masks.iloc[0]
    assert m.source_record_id == 7 and m.record_id != 7
    assert m.recorded_timestamp == pytest.approx(times[7] - .08 * min(times[7] - times[6], times[8] - times[7]))
    assert result.corrupted.timestamp.is_monotonic_increasing and result.corrupted.timestamp.is_unique
    assert result.corrupted.iloc[7].record_id == m.record_id
    retained = result.corrupted[result.corrupted.record_id.isin(data.record_id)].reset_index(drop=True)
    pd.testing.assert_frame_equal(retained, data)


def test_rounded_jitter_must_change_stored_times_and_respect_declared_bound():
    # ±0.45 of a one-tick gap encodes to zero at integer precision.
    cfg = {"adapter": {"numeric_precision": "integer_tick"}}
    data = stream(np.arange(12))
    result = DefectInjector(cfg).inject(data, [clock("jitter", 5, 4, std_fraction=.44)])
    assert result.failures[0].reason == "materialized_time_change"
    assert result.masks.empty
    pd.testing.assert_frame_equal(result.corrupted, data)
    # On a three-tick grid the nearest permitted integer displacement is one tick.
    data = stream(np.arange(12) * 3)
    result = DefectInjector(cfg).inject(data, [clock("jitter", 5, 4, std_fraction=1e6,
                                                     local_gap_fraction=.55, max_displacement_seconds=1.6)])
    assert not result.failures
    np.testing.assert_array_equal(abs(result.masks.recorded_timestamp-result.masks.true_timestamp), [1]*4)


def test_clock_collision_and_reorder_are_distinct_policies():
    data = stream(np.arange(20))
    params = {"offset_mode": "absolute", "offset": 1., "sign": "positive"}
    collision = DefectInjector().inject(data, [clock("fixed_shift", 8, **params)])
    assert collision.failures[0].reason == "timestamp_collisions"
    assert collision.masks.empty
    params.update(offset=1.5)
    crossing = DefectInjector().inject(data, [clock("fixed_shift", 8, **params)])
    assert crossing.failures[0].reason == "timestamp_order"
    permitted = DefectInjector().inject(data, [clock("fixed_shift", 8, allow_reorder=True, **params)])
    assert not permitted.failures and permitted.masks.iloc[0].recorded_timestamp == 9.5


def test_shift_uses_physical_elapsed_hour_on_a_timezone_aware_dst_transition():
    times = pd.date_range("2026-03-28 12:00", periods=12, freq="2h", tz="Europe/Paris")
    data = stream(times)
    request = clock("dst_shift", 7, 2, enabled=True, sign="negative", offset_hours=1)
    unavailable = DefectInjector().inject(data, [request])
    assert unavailable.failures[0].reason == "dst_requires_sensor_timezone"
    result = DefectInjector(metadata={"meter": {"timezone": "Europe/Paris"}}).inject(data, [request])
    assert not result.failures
    assert (result.masks.recorded_timestamp - result.masks.true_timestamp == -pd.Timedelta(hours=1)).all()
    assert result.corrupted.timestamp.dt.tz == result.clean.timestamp.dt.tz
    assert result.corrupted.timestamp.is_monotonic_increasing
    pd.testing.assert_series_equal(result.corrupted.value, data.value)
    detail = result.episodes[0]["channel_details"]["meter"]
    assert detail["parameters"]["timezone"] == "Europe/Paris" and detail["parameters"]["signed_offset_hours"] == -1


def test_stretch_end_to_end_uses_elapsed_real_time():
    times = np.array([0, 1, 2, 3, 6, 10, 11, 13, 20, 23, 25, 29], dtype=float)
    data = stream(times)
    result = DefectInjector().inject(data, [clock("time_stretch", 7, 5, enabled=True, factor=2)])
    assert not result.failures
    expected = times.copy()
    expected[7:] = times[7] + 2 * (times[7:] - times[7])
    np.testing.assert_allclose(result.corrupted.timestamp, expected)
    assert result.masks.source_record_id.tolist() == [8, 9, 10, 11]
    pd.testing.assert_series_equal(result.corrupted.value, data.value)
