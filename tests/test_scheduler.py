import numpy as np
import pandas as pd
import pytest

from tsdq.core import Reject
from tsdq.scheduler import make_schedule, overlaps
from tsdq.windows import slice_windows


def schedule(cfg, duration=1, times=None, eligible=None, seed=441):
    t = {"A": np.arange(100.)} if times is None else times
    ok = {name: np.ones(len(v), dtype=bool) for name, v in t.items()} if eligible is None else eligible
    return list(make_schedule(t, ok, cfg, duration, np.random.default_rng(seed)))


def place(opportunities, n=100, separation=0):
    occupied = np.zeros(n, dtype=bool)
    used = []
    for op in opportunities:
        for hit in op.candidates():
            idx = hit["A"]
            if not overlaps(idx, occupied, separation):
                used.append(idx.tolist())
                occupied[idx] = True
                break
    return used


def fixed(count):
    return {"mode": "fixed_count", "episode_count": count}


def test_fixed_count_recurrence_and_separation():
    settings = {"occurrence": fixed(5), "recurrence": "periodic", "period": 12, "onset_index": 5}
    starts = place(schedule(settings, duration=3))
    assert starts == [list(range(i, i+3)) for i in (5, 17, 29, 41, 53)]
    settings.update(recurrence="random")
    del settings["onset_index"]
    blocks = sorted(place(schedule(settings, duration=3), separation=4))
    assert len(blocks) == 5
    assert all(b[0] - a[-1] >= 5 for a, b in zip(blocks, blocks[1:]))
    assert blocks == sorted(place(schedule(settings, duration=3), separation=4))


def test_periodic_eligibility_holes_do_not_shift_phase():
    ok = np.ones(100, dtype=bool)
    ok[5] = False
    ops = schedule({"occurrence": fixed(3), "onset_index": 1, "recurrence": "periodic", "period": 4},
                   eligible={"A": ok})
    assert [next(op.candidates(), None) is not None for op in ops] == [True, False, True]
    assert place(ops) == [[1], [9]]


def test_clustered_groups_and_bernoulli_thinning():
    cfg = {"occurrence": fixed(6), "recurrence": "clustered", "cluster_width": 10, "cluster_size": 3}
    hits = np.array(place(schedule(cfg, duration=1))).ravel()
    assert len(set(hits)) == 6
    assert np.ptp(hits[:3]) < 10 and np.ptp(hits[3:]) < 10
    assert abs(hits[:3].mean() - hits[3:].mean()) > 10
    cfg["occurrence"] = {"mode": "bernoulli_start", "q": 1}
    hits = np.array(place(schedule(cfg))).ravel()
    assert max(np.unique(hits // 10, return_counts=True)[1]) <= 3


def test_bernoulli_over_independent_opportunities_monte_carlo():
    cfg = {"occurrence": {"mode": "bernoulli_start", "q": 0.1}}
    counts = [len(schedule(cfg, seed=i)) for i in range(1000)]
    assert abs(np.mean(counts) - 10.) < 0.4
    assert abs(np.var(counts) - 9.) < 1.6
    cfg.update(recurrence="periodic", period=10)
    cfg["occurrence"]["q"] = 1
    assert place(schedule(cfg)) == [[i] for i in range(0, 100, 10)]


def test_bernoulli_start_is_not_missing_fraction_and_conflicts_are_not_redrawn():
    cfg = {"occurrence": {"mode": "bernoulli_start", "q": 1}, "max_attempts": 3}
    ops = schedule(cfg, duration=4)
    assert len(ops) == 97
    for i, op in enumerate(ops):
        assert [d["A"][0] for d in op.candidates()] == [i]*3
    placed = place(ops)
    assert len(placed) == 25
    cfg["occurrence"]["q"] = 0.1
    placed = place(schedule(cfg, duration=4))
    assert sum(map(len, placed)) == 4 * len(placed)
    assert sum(map(len, placed)) / 100 > 0.1


def test_async_elapsed_window_range_and_exact_duration():
    t = {"A": np.array([0., 2.1, 4.1, 5.9, 9.]), "B": np.array([0.4, 3., 4., 5.2, 8.5])}
    cfg = {"channels": "all", "recurrence": "periodic", "period": 3, "recurrence_unit": "seconds",
           "duration_unit": "seconds", "onset_seconds": 1., "occurrence": fixed(2)}
    parts = [next(op.candidates()) for op in schedule(cfg, duration=2.5, times=t)]
    assert [p["A"].tolist() for p in parts] == [[1], [2, 3]]
    assert [p["B"].tolist() for p in parts] == [[1], [2, 3]]
    cfg.update(allowed_seconds=[0.5, 5.5])
    parts = schedule(cfg, duration=2.5, times=t)
    assert next(parts[1].candidates(), None) is None


def test_unresolved_whole_channel_and_train_exclusion():
    cfg = {"onset_index": 95, "persistence": "unresolved"}
    assert next(schedule(cfg)[0].candidates())["A"].tolist() == list(range(95, 100))
    cfg = {"persistence": "whole_channel"}
    assert len(next(schedule(cfg)[0].candidates())["A"]) == 100
    ok = np.ones(100, dtype=bool)
    ok[10] = False
    with pytest.raises(Reject, match="no_eligible_onsets"):
        schedule(cfg, eligible={"A": ok})


def test_targets_and_invalid_probability_combination():
    cfg = {"occurrence": {"mode": "fixed_count", "target_fraction": .08}}
    assert sum(map(len, place(schedule(cfg, duration=2)))) == 8
    cfg["occurrence"] = {"mode": "fixed_count", "target_records": 7}
    with pytest.raises(Reject, match="record_target_indivisible"):
        schedule(cfg, duration=2)
    for occurrence in ({"mode": "fixed_count", "episode_count": 2, "q": .5},
                       {"mode": "bernoulli_start", "episode_count": 2, "q": .5}):
        with pytest.raises(ValueError):
            schedule({"occurrence": occurrence})


def test_windows_visibility_context_and_stratification():
    clean = pd.DataFrame({"record_id": range(12), "timestamp": range(12), "sensor_id": "A",
                          "value": [2, 2, 16, 16, 16, 16, 1, 0, 0, 0, 0, 0]})
    corrupted = clean.query("record_id != 4").copy()
    corrupted.loc[corrupted.record_id == 7, "timestamp"] = 8
    episodes = [
        {"episode_id": 1, "family": "missing_values", "channels": ["A"],
         "support_record_ids": [4], "record_ids": [4]},
        {"episode_id": 2, "family": "timestamp", "channels": ["A"],
         "support_record_ids": [7], "record_ids": [7]},
        {"episode_id": 3, "family": "bias", "channels": ["A"],
         "support_record_ids": [9, 10, 11], "record_ids": [9, 10, 11]},
    ]
    masks = pd.DataFrame({"record_id": [4, 7, 9, 10, 11]})
    cfg = {"lengths": [4], "stride": 4, "natural_change_record_ids": [2]}
    windows = slice_windows(clean, corrupted, episodes, masks, cfg, np.random.default_rng(11))
    assert [w["class"] for w in windows] == ["clean", "mixed", "mixed"]
    assert windows[0]["natural_change_control_ids"] == [2]
    loss, shift = windows[1]["episodes"]
    assert loss["absent_record_ids"] == [4] and not loss["visible_record_ids"]
    assert shift["moved_out_record_ids"] == [7] and shift["context"]["A"]["before"] == 2
    moved_in = [label for label in windows[2]["episodes"] if label["family"] == "timestamp"][0]
    assert moved_in["position"] == "recorded_only" and moved_in["moved_in_record_ids"] == [7]

    cfg.update(stride=1, target_proportions={"clean": .25, "single": .5, "mixed": .25}, count=4)
    selected = slice_windows(clean, corrupted, episodes, masks, cfg, np.random.default_rng(3))
    labels = [w["class"] for w in selected]
    assert sorted(labels) == ["clean", "mixed", "single", "single"]
    all_windows = slice_windows(clean, corrupted, episodes, masks, {"lengths": [4]}, np.random.default_rng(3))
    cut = next(e for w in all_windows for e in w["episodes"] if e["position"] == "cut")
    assert 0 < cut["episode_coverage"] < 1
    cfg.update(count=1000)
    with pytest.raises(ValueError, match="insufficient windows"):
        slice_windows(clean, corrupted, episodes, masks, cfg, np.random.default_rng(3))

def test_window_identifiability_when_onset_is_cut_or_deleted_records_are_unobservable():
    clean = pd.DataFrame({"record_id": range(12), "timestamp": range(12), "sensor_id": "a", "value": 1.})
    corrupt = clean.query("record_id != 9").copy()
    corrupt.loc[corrupt.record_id.between(2, 7), "value"] = 2.
    shift = {"episode_id": "bias", "family": "bias", "channels": ["a"],
             "support_record_ids": list(range(2, 8)), "record_ids": list(range(2, 8)),
             "channel_details": {"a": {"onset": 2, "information_needed": "U/M", "identifiable_given_context": True}}}
    loss = {"episode_id": "loss", "family": "missing_values", "channels": ["a"],
            "support_record_ids": [9], "record_ids": [9],
            "channel_details": {"a": {"onset": 9, "information_needed": "U+", "identifiable_given_context": False}}}
    masks = pd.DataFrame({"record_id": [*range(2, 8), 9]})
    cfg = {"lengths": [4], "stride": 4, "scoring_basis": "visible", "natural_change_record_ids": [10]}
    labels = slice_windows(clean, corrupt, [shift, loss], masks, cfg, np.random.default_rng(1))
    assert [w["class"] for w in labels] == ["single", "clean", "clean"]
    assert [w["ground_truth_class"] for w in labels] == ["single"]*3
    assert labels[0]["episodes"][0]["context"]["a"]["onset_visible"]
    assert not labels[1]["episodes"][0]["context"]["a"]["onset_visible"]
    assert not labels[2]["natural_change_control_ids"]

    shift["reference_visible"] = True
    loss["channel_details"]["a"]["identifiable_given_context"] = True
    privileged = slice_windows(clean, corrupt, [shift, loss], masks, cfg, np.random.default_rng(1))
    assert [w["class"] for w in privileged] == ["single"]*3
    assert privileged[2]["episodes"][0]["visible_record_ids"] == []


def test_duplicate_window_support_tracks_true_and_recorded_coordinates_separately():
    clean = pd.DataFrame({"record_id": list("abcdef"), "timestamp": range(6), "sensor_id": "a", "value": 1.})
    copy = pd.DataFrame({"record_id": ["extra"], "timestamp": [3.1], "sensor_id": "a", "value": 1.})
    corrupt = pd.concat([clean, copy], ignore_index=True)
    ep = {"episode_id": "dup", "family": "timestamp", "subtype": "near_duplicate", "channels": ["a"],
          "support_record_ids": ["c"], "record_ids": ["extra"],
          "channel_details": {"a": {"onset": 2, "identifiable_given_context": True}}}
    masks = pd.DataFrame({"record_id": ["extra"], "source_record_id": ["c"]})
    cfg = {"lengths": [3], "stride": 3, "scoring_basis": "visible"}
    left, right = slice_windows(clean, corrupt, [ep], masks, cfg, np.random.default_rng(1))
    assert left["class"] == right["class"] == "clean"
    assert left["ground_truth_class"] == right["ground_truth_class"] == "single"
    assert left["episodes"][0]["affected_record_ids"] == ["extra"]
    assert left["episodes"][0]["support_record_ids"] == ["c"]
    assert right["episodes"][0]["moved_in_record_ids"] == ["extra"]
    both = slice_windows(clean, corrupt, [ep], masks, {"lengths": [6]}, np.random.default_rng(1))[0]
    assert both["visible_class"] == "single" and both["episodes"][0]["episode_coverage"] == 1.
    assert both["episodes"][0]["moved_in_record_ids"] == []


def test_multiple_channels_require_a_common_active_interval():
    disjoint = {"A": np.arange(10.), "B": np.arange(10.) + 20}
    for persistence in ("temporary", "unresolved", "whole_channel"):
        with pytest.raises(Reject, match="no_eligible_onsets"):
            schedule({"channels": "all", "persistence": persistence}, duration=2, times=disjoint)
    with pytest.raises(Reject, match="no_eligible_onsets"):
        schedule({"channels": "all", "persistence": "unresolved", "duration_unit": "seconds"},
                 duration=2, times=disjoint)
