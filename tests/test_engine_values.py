"""Value families, composition and labels through the public injection API."""

import numpy as np
import pandas as pd
import pytest

from tsdq import DefectInjector


def test_burst_bias_phases_have_replayable_masks_through_main_api():
    n = 100
    clean = pd.DataFrame({"timestamp": np.arange(n), "value": np.sin(np.arange(n) / 3)})
    plan = [{"family": "burst", "subtype": "global", "duration": 4,
             "schedule": {"onset_index": 50},
             "params": {"shape": "triangle", "direction": "positive", "amplitude_multiplier": 10,
                        "end_mode": "non_recovery", "tail_samples": 6}}]
    result = DefectInjector(seed=42).inject(clean, plan)
    assert not result.failures
    burst, bias = result.episodes
    assert burst["subtype"] == "global" and bias["family"] == "bias"
    assert bias["parent_episode_id"] == burst["episode_id"]
    assert set(burst["record_ids"]) == set(range(50, 54))
    assert set(bias["record_ids"]) == set(range(54, 60))
    mask = result.masks.set_index("record_id")
    np.testing.assert_array_equal(result.clean.value, clean.value)
    np.testing.assert_array_equal(mask.corrupt_value, result.corrupted.iloc[mask.index].value)
    np.testing.assert_array_equal(mask.clean_value, clean.iloc[mask.index].value)
    np.testing.assert_array_equal(result.corrupted.iloc[60:].value, clean.iloc[60:].value)


def test_multichannel_value_transaction_rolls_back_if_one_channel_fails():
    t = np.arange(60)
    frame = pd.concat([pd.DataFrame({"timestamp": t, "value": x, "sensor_id": name})
                       for name, x in [("oscillating", np.sin(t)), ("constant", np.ones(len(t)))]],
                      ignore_index=True)
    result = DefectInjector().inject(frame, [{"family": "burst", "subtype": "global",
        "duration": 4, "schedule": {"channels": "all", "onset_index": 32, "max_attempts": 1},
        "params": {"amplitude_multiplier": 10, "direction": "positive", "shape": "alternating"}}])
    assert len(result.failures) == 1 and result.masks.empty and not result.episodes
    assert result.manifest["rejections"][0]["channel"] == "constant"
    np.testing.assert_array_equal(result.corrupted.value, frame.value)


def test_burst_cannot_use_an_already_corrupted_predecessor():
    t = np.arange(60)
    frame = pd.DataFrame({"timestamp": t, "value": np.sin(t)})
    result = DefectInjector().inject(frame, [
        {"family": "bias", "duration": 2, "duration_unit": "samples", "schedule": {"onset_index": 30}},
        {"family": "burst", "subtype": "global", "duration": 4, "schedule": {"onset_index": 32, "max_attempts": 1},
         "params": {"amplitude_multiplier": 10, "direction": "positive", "shape": "alternating"}}])
    assert result.failures[0].reason == "requires_unaffected_context"
    assert len(result.episodes) == 1
    np.testing.assert_array_equal(result.corrupted.iloc[32:].value, frame.iloc[32:].value)


def test_seven_value_families_survive_one_plan_and_leave_disjoint_masks():
    t = np.arange(720)
    x = 10 + np.sin(t / 16) + np.random.default_rng(19).normal(0, .15, len(t))
    clean = pd.DataFrame({"record_id": [f"r{i}" for i in t], "timestamp": t, "value": x})
    specs = [
        ("point_outlier", "spike", 310, 1, {"amplitude_multiplier": 10}),
        ("burst", "global", 340, 4, {"shape": "triangle", "amplitude_multiplier": 10}),
        ("bias", None, 360, 8, {}),
        ("drift", None, 382, 8, {"exponent": 2}),
        ("scaling", None, 410, 8, {"direction": "attenuate"}),
        ("noise", None, 440, 60, {"target_ratio": 4}),
        ("flat_line", "outlier_flatline", 525, 8, {"amplitude_multiplier": 10}),
    ]
    plan = [{"family": family, "subtype": subtype, "duration_unit": "samples", "duration": duration,
             "params": params, "schedule": {"onset_index": start, "max_attempts": 2}}
            for family, subtype, start, duration, params in specs]
    r = DefectInjector(seed=3).inject(clean, plan)
    assert not r.failures
    assert [(e["family"], e["subtype"]) for e in r.episodes] == [(s[0], s[1]) for s in specs]
    assert r.masks.record_id.is_unique
    changed = r.clean.value != r.corrupted.value
    assert set(r.masks.record_id) == set(r.clean.record_id[changed])
    for label, (_, _, start, duration, _) in zip(r.episodes, specs):
        assert label["support_record_ids"] == [f"r{i}" for i in range(start, start + duration)]
        detail = label["channel_details"]["sensor"]
        assert detail["extent"]["modified"] == len(label["record_ids"])
        assert all(check["passed"] for check in detail["checks"])
    # Drift's scheduled onset has zero error and stays outside its change mask.
    assert "r382" not in r.episodes[3]["record_ids"]
    assert r.episodes[5]["channel_details"]["sensor"]["achieved"]["achieved_ratio"] > 1


def test_local_subtypes_and_flat_line_subtypes_use_eligible_calibration_and_exact_ids():
    t = np.arange(200)
    x = np.where(t % 2 == 0, .2, -.2)
    x[:60] = np.linspace(-20, 20, 60)
    x[140:148] = [.1, 2., 3., 4., 2., .4, .2, .1]
    frame = pd.DataFrame({"timestamp": t, "value": x})
    plan = [
        {"family": "point_outlier", "subtype": "glitch", "schedule": {"onset_index": 75},
         "params": {"direction": "positive", "amplitude_multiplier": 8}},
        {"family": "burst", "subtype": "contextual", "duration": 4, "schedule": {"onset_index": 100},
         "params": {"shape": "alternating", "direction": "positive", "amplitude_multiplier": 8}},
        {"family": "flat_line", "subtype": "stuck", "duration": 4, "duration_unit": "samples",
         "schedule": {"onset_index": 125}, "params": {"minimum_movement_scale": 0.01}},
        {"family": "flat_line", "subtype": "saturation", "duration": 8, "schedule": {"onset_index": 140},
         "params": {"cap_source": "hardware"}},
    ]
    result = DefectInjector(metadata={"sensor": {"upper_limit": 1}}).inject(frame, plan)
    assert not result.failures
    for episode in result.episodes[:2]:
        m = episode["channel_details"]["sensor"]["achieved"]
        assert m["local_exceedance"] > 0 and m["global_exceedance"] == 0
    assert result.episodes[2]["record_ids"] == [125, 127]
    assert result.episodes[3]["record_ids"] == [141, 142, 143, 144]
    np.testing.assert_allclose(result.corrupted.value.iloc[145:148], frame.value.iloc[145:148])


@pytest.mark.parametrize("magnitude,duration_severity,expected_k,expected_count", [
    (0., 1., 6., 9), (1., 0., 12., 3), (.5, .5, 9., 6)])
def test_legacy_alias_supports_independent_severity_axes(magnitude, duration_severity, expected_k, expected_count):
    frame = pd.DataFrame({"timestamp": np.arange(80), "value": np.sin(np.arange(80) / 2)})
    plan = [{"family": "flat_line", "subtype": "hard_over", "severity_magnitude": magnitude,
             "severity_duration": duration_severity, "duration": {"min": 3, "max": 9},
             "duration_unit": "samples", "schedule": {"onset_index": 40},
             "params": {"direction": "positive", "amplitude_multiplier": {"min": 6, "max": 12}}}]
    r = DefectInjector().inject(frame, plan)
    assert not r.failures
    ep = r.episodes[0]
    assert ep["source_alias"] == "hard_over" and ep["subtype"] == "outlier_flatline"
    assert ep["severity_magnitude"] == magnitude and ep["severity_duration"] == duration_severity
    assert len(ep["record_ids"]) == expected_count
    detail = ep["channel_details"]["sensor"]
    observed = r.corrupted.value.iloc[40:40 + expected_count]
    assert observed.nunique() == 1
    assert observed.iloc[0] - frame.value.iloc[40] == pytest.approx(expected_k * detail["reference"]["scale"])


@pytest.mark.parametrize("intentional", [False, True])
@pytest.mark.parametrize("representation", ["nan", "drop_records"])
def test_missing_overlap_reports_concealed_burst_or_rejects_by_default(intentional, representation):
    t = np.arange(100)
    frame = pd.DataFrame({"timestamp": t, "value": np.sin(t / 3)})
    placement = {"onset_index": 50, "max_attempts": 1}
    plan = [
        {"family": "burst", "subtype": "global", "duration": 4, "schedule": placement,
         "params": {"direction": "positive", "shape": "alternating", "amplitude_multiplier": 10}},
        {"family": "missing_values", "duration": 4, "schedule": placement,
         "params": {"representation": representation}},
    ]
    cfg = {"composition": {"allow_overlap": [["value", "missing"]]}} if intentional else {}
    r = DefectInjector(cfg).inject(frame, plan)
    detail = r.episodes[0]["channel_details"]["sensor"]
    if intentional:
        assert not r.failures and len(r.episodes) == 2
        assert set(detail["hidden_record_ids"]) == set(r.episodes[0]["record_ids"])
        assert not detail["identifiable_given_context"]
        value_mask = r.masks[r.masks.kind == "value"]
        assert value_mask.final_value.isna().all() and value_mask.corrupt_value.notna().all()
        assert value_mask.finally_dropped.tolist() == [representation == "drop_records"] * 4
    else:
        assert len(r.episodes) == 1 and r.failures[0].reason == "overlap_or_separation"
        assert detail["identifiable_given_context"] and not detail["hidden_record_ids"]
        assert r.masks.final_value.notna().all()
