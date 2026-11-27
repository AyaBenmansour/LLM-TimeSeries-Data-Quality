import json

import numpy as np
import pandas as pd
import pytest

from gallery_cases import build_cases
from make_defect_gallery import compare_records
from plot_gallery import describe, render_gallery
from tsdq.config import TAXONOMY


@pytest.fixture(scope="module")
def gallery():
    return build_cases()


def test_gallery_exercises_the_taxonomy_and_keeps_missingness_pattern_based(gallery):
    expected = {(family, subtype) for family, subtypes in TAXONOMY.items() for subtype in subtypes or (None,)}
    assert {(case.family, case.subtype) for case, _ in gallery} == expected
    assert len({case.key for case, _ in gallery}) == len(gallery)
    patterns = set()
    for case, result in gallery:
        assert not result.failures and not result.manifest["rejections"], case.key
        assert result.training is not None and set(result.training.record_id).isdisjoint(result.clean.record_id)
        if case.family == "missing_values":
            for episode in result.episodes:
                assert episode["subtype"] is None
                patterns.update(episode["patterns"])
    assert patterns == {"isolated", "block", "terminal", "whole_channel", "multi_channel"}


def test_every_gallery_mask_equals_the_actual_before_after_difference(gallery):
    for case, result in gallery:
        before = result.clean.set_index("record_id")
        after = result.corrupted.set_index("record_id")
        common = before.index.intersection(after.index)
        changed = before.index.symmetric_difference(after.index)
        different = ((before.loc[common, "value"] != after.loc[common, "value"])
                     | (before.loc[common, "timestamp"] != after.loc[common, "timestamp"]))
        expected = set(changed) | set(common[different])
        assert set(result.masks.record_id) == expected, case.key
        assert {rid for episode in result.episodes for rid in episode["record_ids"]} == expected, case.key


def test_gallery_has_visible_examples_of_unchanged_endpoints_and_interior_crossings(gallery):
    runs = {case.key: result for case, result in gallery}
    for key, untouched in {
        "scaling_amplify": [16, 24, 32],
        "scaling_attenuate": [16, 24, 32],
        "drift_quadratic": [22, 39],
        "clock_drift": [22],
        "time_stretch": [45],
    }.items():
        result = runs[key]
        excluded = result.clean.iloc[untouched].record_id
        assert set(excluded).isdisjoint(result.masks.record_id), key


def test_core_timestamp_defects_have_regular_and_irregular_examples(gallery):
    keys = {case.key for case, _ in gallery}
    for subtype in ("fixed_shift", "jitter", "clock_drift"):
        assert {subtype, f"{subtype}_regular"} <= keys
    for case, result in gallery:
        if case.family == "timestamp":
            clean = result.clean.set_index("record_id").value
            after = result.corrupted.set_index("record_id").value
            np.testing.assert_array_equal(clean.sort_index(), after.loc[clean.index].sort_index())


def test_every_accepted_gallery_check_passed_and_scheduled_support_contains_the_labels(gallery):
    for case, result in gallery:
        for episode in result.episodes:
            assert set(episode["record_ids"]) <= set(episode["support_record_ids"]) | set(
                result.masks.loc[result.masks.record_id != result.masks.source_record_id, "record_id"]), case.key
            for detail in episode["channel_details"].values():
                assert all(check["passed"] for check in detail["checks"]), case.key
        assert "checks" in describe(result) and "→ changed" in describe(result)


def test_labels_survive_export_and_reload(gallery, tmp_path):
    for case, result in gallery:
        folder = result.write(tmp_path / case.key) / "ground_truth"
        clean = pd.read_json(folder / "clean.jsonl", lines=True, dtype=False)
        corrupted = pd.read_json(folder / "corrupted.jsonl", lines=True, dtype=False)
        masks = pd.read_json(folder / "masks.jsonl", lines=True, dtype=False)
        before, after = clean.set_index("record_id"), corrupted.set_index("record_id")
        common = before.index.intersection(after.index)
        b, a = before.loc[common], after.loc[common]
        value_changed = ~((b.value == a.value) | (b.value.isna() & a.value.isna()))
        changed = set(before.index.symmetric_difference(after.index))
        changed |= set(common[value_changed | (b.timestamp != a.timestamp)])
        assert set(masks.record_id) == changed, case.key
        episodes = [json.loads(line) for line in (folder / "episodes.jsonl").read_text("utf-8").splitlines()]
        assert {rid for ep in episodes for rid in ep["record_ids"]} == changed, case.key


def test_gallery_renders_every_case_to_png(gallery, tmp_path):
    for case, result in gallery:
        compare_records(case, result)
    images = render_gallery(gallery, tmp_path)
    assert images and all(path.suffix == ".png" and path.stat().st_size > 10_000 for path in images)
    assert (tmp_path / "gallery.pdf").stat().st_size > 10_000
