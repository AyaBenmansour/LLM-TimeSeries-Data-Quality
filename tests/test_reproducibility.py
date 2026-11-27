"""Exercise calibration privacy, run replay and the serialized CLI boundary."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import tsdq
from tsdq import DefectInjector
from tsdq.cli import read_table
from tsdq.config import preset
from tsdq.data import adapt


@pytest.fixture
def measurements():
    n = 100
    origin = pd.Timestamp("2024-11-05T10:00:00.123456789Z")
    timestamps = origin + pd.to_timedelta(np.arange(n) * 3_000_000_007, unit="ns")
    x = np.sin(np.arange(n) / 2.31)
    x[53] = np.nan
    return pd.DataFrame({"record_id": [f"probe-{i:03d}" for i in range(n)], "timestamp": timestamps,
                         "sensor_id": "pressure", "value": x, "private_note": "calibration log"})


def replay_config(fmt="jsonl"):
    # The first clean point is provably too far from a boundary for this small
    # Spike. Its failed attempts must also replay exactly.
    return {"seed": 173, "output": {"format": fmt}, "windows": {"lengths": [12], "stride": 6},
            "plan": [
                {"family": "point_outlier", "subtype": "spike", "params": {"amplitude_multiplier": .001},
                 "schedule": {"onset_index": 40, "max_attempts": 3}},
                {"family": "bias", "duration": 6, "duration_unit": "samples",
                 "schedule": {"onset_index": 67}},
                {"family": "missing_values", "duration": 3, "schedule": {"onset_index": 80}}]}


def json_rows(path):
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines()]


def test_same_seed_replays_accepted_labels_and_rejected_candidates(measurements):
    cfg = replay_config()
    untouched = measurements.copy(deep=True)
    untouched_cfg = deepcopy(cfg)
    injector = DefectInjector(cfg)
    a = injector.inject(measurements)
    b = injector.inject(measurements)
    pd.testing.assert_frame_equal(measurements, untouched, check_exact=True)
    assert cfg == untouched_cfg
    for name in ("clean", "corrupted", "masks"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name), check_exact=True)
    assert a.manifest == b.manifest and a.episodes == b.episodes and a.windows == b.windows
    assert len(a.manifest["rejections"]) == 3 and a.failures == b.failures
    assert [r["attempt"] for r in a.manifest["rejections"]] == [1, 2, 3]
    assert len({r["seed"] for r in a.manifest["rejections"]}) == 3
    assert [ep["episode_id"] for ep in a.episodes] == ["e000001", "e000002"]
    assert a.manifest["code_version"]["package"]
    assert len(a.manifest["code_version"]["source_sha256"]) == 64
    changed_ids = set(a.masks.record_id)
    unaffected = ~a.clean.record_id.isin(changed_ids)
    pd.testing.assert_frame_equal(a.corrupted[unaffected], a.clean[unaffected], check_exact=True)
    different = DefectInjector(cfg, seed=174).inject(measurements)
    assert different.episodes[0]["seed"] != a.episodes[0]["seed"]


def test_result_views_and_reusable_configuration_are_independent(measurements):
    cfg = replay_config()
    injector = DefectInjector(cfg)
    cfg["plan"].clear()
    r = injector.inject(measurements)
    assert len(r.episodes) == 2
    original = r.clean
    clean_view = r.clean
    public_view = r.observations()
    clean_view.loc[:, "value"] = 88.
    public_view.loc[:, "value"] = -8.
    assert (clean_view.value == 88).all() and (public_view.value == -8).all()
    r.corrupted.loc[:, "value"] = 2.
    pd.testing.assert_frame_equal(r.clean, original, check_exact=True)
    repeated = injector.inject(measurements)
    assert repeated.clean.value.iloc[0] == measurements.value.iloc[0]
    assert not (repeated.corrupted.value == 2).all()


def test_noncontiguous_training_ids_stay_private_and_natural_change_remains_clean(measurements):
    training = measurements.record_id.iloc[[0, 2, 7, 10, 25]].tolist()
    natural_id = measurements.record_id.iloc[41]
    measurements.loc[41, "value"] = 80.
    config = {"reference": {"train_record_ids": training},
              "windows": {"lengths": [10], "stride": 5, "natural_change_record_ids": [natural_id]}}
    r = DefectInjector(config).inject(measurements, plan=[])
    assert r.manifest["training_source"] == "record_ids"
    assert r.manifest["training_record_ids"] == training
    assert r.masks.empty and not r.episodes and not r.failures
    pd.testing.assert_frame_equal(r.clean, r.corrupted, check_exact=True)
    obs = r.observations()
    expected = measurements.loc[~measurements.record_id.isin(training), ["timestamp", "sensor_id", "value"]]
    pd.testing.assert_frame_equal(obs, expected.reset_index(drop=True), check_exact=True)
    assert r.windows and all(w["class"] == "clean" for w in r.windows)
    for window in r.windows:
        ids = window["observation_record_ids"]
        assert not set(training).intersection(ids)
        assert len(r.observations(window["window_id"])) == len(ids)
    assert any(natural_id in w["natural_change_control_ids"] for w in r.windows)


@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_private_tables_roundtrip_and_saved_config_reconstructs_the_run(tmp_path, measurements, fmt):
    config = replay_config(fmt)
    r = DefectInjector(config).inject(measurements)
    folder = r.write(tmp_path / "run")
    private = folder / "ground_truth"
    manifest = json.loads((private / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in [("clean", r.clean), ("corrupted", r.corrupted)]:
        reloaded = adapt(read_table(private / f"{filename}.{fmt}"), manifest["config"]["adapter"])
        pd.testing.assert_frame_equal(reloaded, expected, check_exact=True)
    public = read_table(folder / f"observations.{fmt}")
    assert list(public) == ["timestamp", "sensor_id", "value"] and len(public) == 70
    np.testing.assert_array_equal(public.value.to_numpy(), r.observations().value.to_numpy())
    exact_times = pd.to_datetime(public.timestamp, format="mixed", utc=True).dt.as_unit("ns")
    pd.testing.assert_series_equal(exact_times, r.observations().timestamp)
    masks = read_table(private / f"masks.{fmt}")
    assert masks.record_id.tolist() == r.masks.record_id.tolist()
    np.testing.assert_array_equal(masks.corrupt_value, r.masks.corrupt_value)
    assert json_rows(private / "episodes.jsonl") == r.episodes
    assert json_rows(private / "windows.jsonl") == r.windows
    clean = adapt(read_table(private / f"clean.{fmt}"), manifest["config"]["adapter"])
    replay = DefectInjector(manifest["config"]).inject(clean)
    pd.testing.assert_frame_equal(replay.corrupted, r.corrupted, check_exact=True)
    assert replay.episodes == r.episodes and replay.manifest == manifest
    with pytest.raises(FileExistsError):
        r.write(folder)


@pytest.mark.parametrize("conf_ext,returncode", [("yaml", 0), ("json", 2)])
def test_module_cli_reports_failures_and_writes_portable_outputs(tmp_path, measurements, conf_ext, returncode):
    src = tmp_path / "device.csv"
    measurements.to_csv(src, index=False)
    path = tmp_path / f"experiment.{conf_ext}"
    if conf_ext == "json":
        path.write_text(json.dumps(replay_config()), encoding="utf-8")
    else:
        path.write_text("plan:\n  - family: bias\n    duration: 3\n    duration_unit: samples\n",
                        encoding="utf-8")
    out = tmp_path / "run"
    env = os.environ.copy()
    package_parent = str(Path(tsdq.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([package_parent, env.get("PYTHONPATH", "")])
    completed = subprocess.run([sys.executable, "-m", "tsdq", "--input", str(src), "--config", str(path),
                                "--output", str(out), "--seed", "91"], cwd=tmp_path, env=env,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == returncode, completed.stderr
    manifest = json.loads((out / "ground_truth/manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 91
    assert bool(manifest["failures"]) == (returncode != 0)
    assert manifest["config"]["version"] == preset()["version"]
    assert (out / "observations.jsonl").exists()


def test_preset_includes_corrected_taxonomy_and_loads_independent_defaults():
    settings = preset()
    assert settings["version"] == "fyp-v1"
    assert {"spike", "glitch", "bias", "drift", "scaling", "noise", "stuck", "outlier_flatline",
            "saturation", "global", "contextual", "missing_values", "fixed_shift", "clock_drift", "jitter"
            } <= set(settings["defects"])
    settings["defects"]["spike"]["params"]["amplitude_multiplier"]["max"] = 990
    assert preset()["defects"]["spike"]["params"]["amplitude_multiplier"]["max"] == 7.5


@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_file_boundary_preserves_lexically_distinct_identifiers(tmp_path, fmt):
    records = [{"record_id": rid, "sensor_id": "003", "timestamp": i, "value": i / 3}
               for i, rid in enumerate(["0007", "7", "NA"])]
    path = tmp_path / f"device.{fmt}"
    if fmt == "csv":
        pd.DataFrame(records).to_csv(path, index=False)
    else:
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    restored = read_table(path)
    assert restored.record_id.astype(str).tolist() == ["0007", "7", "NA"]
    assert restored.sensor_id.astype(str).tolist() == ["003"] * 3
