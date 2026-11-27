"""Exercise unchanged anchors, recovery, and clean holes through public APIs."""
import numpy as np
import pandas as pd

from tsdq import DefectInjector


def test_drift_anchor_and_recovered_endpoint_are_clean_context():
    t = np.arange(32)
    frame = pd.DataFrame({"timestamp": t, "value": np.sin(t)})
    calibration = pd.DataFrame({"timestamp": t, "value": np.cos(t)})
    plan = [{"family": "drift", "duration": 4, "duration_unit": "samples",
             "schedule": {"onset_index": 10}, "params": {"end_mode": "recovery", "recovery_samples": 2}}]
    cfg = {"windows": {"lengths": [1, 4, 6], "stride": 1}}
    result = DefectInjector(cfg).inject(frame, plan, training_df=calibration)
    assert not result.failures
    ep = result.episodes[0]
    ch = ep["channel_details"]["sensor"]
    assert (ep["scheduled_onset"], ep["scheduled_end"]) == (10, 15)
    assert ch["scheduled_support"] == {"onset": 10, "end": 15, "start_index": 10, "end_index": 15,
                                         "elapsed_seconds": 5}
    assert (ep["onset"], ep["end"]) == (11, 14)
    assert (ch["start_index"], ch["end_index"], ch["elapsed_seconds"]) == (11, 14, 3)
    assert (ch["context_before"], ch["context_after"]) == (11, 17)
    assert ep["record_ids"] == result.masks.record_id.tolist() == [11, 12, 13, 14]
    by_id = {w["window_id"]: w for w in result.windows}
    for rid in (10, 15):
        assert by_id[f"w1-{rid}"]["episodes"] == []
        assert by_id[f"w1-{rid}"]["class"] == "clean"
    containing = by_id["w6-10"]["episodes"][0]
    assert containing["support_record_ids"] == containing["affected_record_ids"] == ep["record_ids"]
    assert containing["position"] == "interior" and containing["episode_coverage"] == 1
    assert containing["window_coverage"] == 4 / 6
    assert containing["context"]["sensor"]["onset_visible"]
    assert by_id["w4-11"]["episodes"][0]["position"] == "edge"
    assert by_id["w1-11"]["episodes"][0]["episode_coverage"] == 1 / 4


def test_multichannel_clipping_localizes_each_actual_peak():
    frames = []
    for name, peaks in (("a", [0, 2, 2, 2, 0, 0, 2, 2, 2, 0]), ("b", [2, 2, 2, 0, 0, 0, 0, 0, 0, 0])):
        x = np.zeros(30)
        x[10:20] = peaks
        frames.append(pd.DataFrame({"record_id": [f"{name}{i}" for i in range(len(x))],
                                    "sensor_id": name, "value": x, "timestamp": np.arange(len(x))}))
    clean = pd.concat(frames, ignore_index=True)
    meta = {name: {"upper_limit": 1} for name in ("a", "b")}
    cfg = {"windows": {"lengths": [1, 10]}, "reference": {"train_fraction": 0.2}}
    req = {"family": "flat_line", "subtype": "saturation", "duration": 10,
           "schedule": {"channels": "all", "onset_index": 10}, "params": {"cap_source": "hardware"}}
    result = DefectInjector(cfg, metadata=meta).inject(clean, [req])
    ep = result.episodes[0]
    assert (ep["onset"], ep["end"]) == (10, 18)
    a, b = (ep["channel_details"][name] for name in ("a", "b"))
    assert (a["onset"], a["end"], b["onset"], b["end"]) == (11, 18, 10, 12)
    for window in result.windows:
        if window["start"] in (14, 15, 19) and window["length"] == 1:
            assert window["class"] == "clean" and not window["episodes"]
    episode_window = next(w for w in result.windows if w["start"] == 10 and w["length"] == 10)
    assert episode_window["episodes"][0]["window_coverage"] == 9 / 20


def test_clock_drift_anchor_stays_outside_timestamp_labels():
    clean = pd.DataFrame({"timestamp": np.arange(20), "value": np.ones(20)})
    req = {"family": "timestamp", "subtype": "clock_drift", "duration": 4,
           "schedule": {"onset_index": 10}, "params": {"rate": 0.01, "sign": "positive"}}
    result = DefectInjector({"windows": {"lengths": [1]}}).inject(clean, [req])
    assert not result.failures
    ep = result.episodes[0]
    assert (ep["onset"], ep["end"]) == (11, 13)
    assert ep["record_ids"] == [11, 12, 13]
    assert next(w for w in result.windows if w["start"] == 10)["episodes"] == []
    np.testing.assert_array_equal(result.clean.value, result.corrupted.value)
