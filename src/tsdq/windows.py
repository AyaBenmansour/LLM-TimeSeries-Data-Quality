"""Private window-level scoring metadata derived after injection on the long stream."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .scheduler import _integer, _number


CLASSES = ("clean", "single", "mixed")


def _select(windows, config, rng):
    n = config.get("count")
    if n is not None:
        n = _integer(n, "window count", 0)
    weights = config.get("target_proportions")
    if weights is None:
        if n is None:
            return windows
        if n > len(windows):
            raise ValueError(f"requested {n} distinct windows, only {len(windows)} exist")
        indices = sorted(rng.choice(len(windows), size=n, replace=False))
        return [windows[i] for i in indices]

    if not isinstance(weights, dict) or set(weights) != set(CLASSES):
        raise ValueError("target_proportions requires clean, single and mixed fractions")
    w = np.array([_number(weights[c], c) for c in CLASSES])
    if min(w) < 0 or not np.isclose(sum(w), 1, atol=1e-10, rtol=0):
        raise ValueError("window proportions must be nonnegative and sum to one")
    pools = {c: [i for i, window in enumerate(windows) if window["class"] == c] for c in CLASSES}
    sizes = np.array([len(pools[c]) for c in CLASSES])
    if n is None:
        if np.any((sizes == 0) & (w > 0)):
            raise ValueError(f"no windows available for a requested class: {dict(zip(CLASSES, sizes))}")
        n = int(np.floor(np.min(sizes[w > 0] / w[w > 0])))
    ideal = w * n
    quotas = np.floor(ideal).astype(int)
    for idx in np.argsort(-(ideal - quotas), kind="stable")[:n - int(sum(quotas))]:
        quotas[idx] += 1
    if (sizes < quotas).any():
        raise ValueError(f"insufficient windows; classes {CLASSES}, requested {quotas}, available {sizes}")
    indices = []
    for c, size in zip(CLASSES, quotas):
        if size:
            indices.extend(rng.choice(pools[c], size=size, replace=False))
    return [windows[i] for i in sorted(indices)]


def _window_episode(ep, sources, changed, onsets, true_frame, obs_frame, observed_clean, original,
                    observed, copies, left_time, last_instant):
    in_true = true_frame.loc[true_frame.index.isin(sources)]
    in_obs = obs_frame.loc[obs_frame.index.isin(sources | changed)]
    truth = in_true.index.tolist()
    affected = [i for i in truth if i in changed]
    generated = [i for i in ep["record_ids"] if i in copies]
    affected.extend(i for i in generated if copies[i] in in_true.index)
    absent = [i for i in affected if i in original.index and i not in observed.index]
    visible = in_obs.loc[in_obs.index.isin(changed)]
    if ep["family"] not in {"missing_values", "timestamp"}:
        visible = visible.loc[visible["value"].notna()]
    if not truth:
        position = "recorded_only"
    elif len(affected) < len(changed):
        position = "cut"
    elif in_true["timestamp"].min() == left_time or in_true["timestamp"].max() == last_instant:
        position = "edge"
    else:
        position = "interior"

    context = {}
    for name in ep["channels"]:
        detail = ep.get("channel_details", {}).get(name, {})
        true_channel = in_true.loc[in_true["sensor_id"] == name]
        obs_channel = in_obs.loc[in_obs["sensor_id"] == name]
        effects = visible.loc[visible["sensor_id"] == name]
        evidence = obs_channel["timestamp"] if len(obs_channel) else true_channel["timestamp"]
        neighbors = observed_clean.loc[(observed_clean["sensor_id"] == name)
                                       & ~observed_clean.index.isin(sources), "timestamp"]
        before = int((neighbors < evidence.min()).sum()) if len(evidence) else 0
        after = int((neighbors > evidence.max()).sum()) if len(evidence) else 0
        onset_ids = true_channel.index[true_channel["timestamp"] == onsets.get(name)]
        onset_visible = bool(before and obs_channel.loc[
            obs_channel.index.isin(onset_ids), "value"].notna().any())
        has_evidence = bool(len(effects))
        reason = detail.get("identifiability_condition", "within-window observations")
        permitted = detail.get("identifiable_given_context", True)
        if generated:
            has_evidence = any(copies[i] in obs_frame.index for i in effects.index if i in copies)
            reason = "source and copy observed in the same window"
        if ep["family"] == "missing_values" and any(i in true_channel.index for i in absent):
            has_evidence |= detail.get("identifiable_given_context", False)
            reason = "expected observation process available for the missing interval"
        if detail.get("information_needed") == "U/M":
            permitted &= onset_visible or ep.get("reference_visible", False)
            reason = "visible onset with preceding clean context or an independent measurement"
        context[name] = {"before": before, "after": after, "onset_visible": onset_visible,
                         "identifiable_given_context": bool(permitted and has_evidence),
                         "identifiability_condition": reason}

    return {
        "episode_id": ep["episode_id"], "family": ep["family"], "subtype": ep.get("subtype"),
        "position": position, "support_record_ids": truth, "affected_record_ids": affected,
        "visible_record_ids": visible.index.tolist(), "absent_record_ids": absent,
        "moved_out_record_ids": [i for i in affected if i in observed.index and i not in obs_frame.index],
        "moved_in_record_ids": [i for i in visible.index if copies.get(i, i) not in true_frame.index],
        "episode_coverage": len(affected) / len(changed), "window_coverage": len(truth) / len(true_frame),
        "context": context, "identifiable_given_context": any(v["identifiable_given_context"] for v in context.values()),
    }


def _class(labels, scoring):
    count = len(labels) if scoring == "episodes" else len({e["family"] for e in labels})
    return CLASSES[min(count, 2)]


def slice_windows(clean: pd.DataFrame, corrupted: pd.DataFrame, episodes: list[dict],
                  masks: pd.DataFrame, config: dict, rng: np.random.Generator) -> list[dict]:
    """Slice by unique clean instants; identify recorded observations on the same intervals."""
    allowed = {"lengths", "stride", "include_partial", "target_proportions", "count",
               "scoring", "scoring_basis", "natural_change_record_ids"}
    if set(config) - allowed:
        raise ValueError(f"unknown window options: {sorted(set(config) - allowed)}")
    lengths = [_integer(v, "window length") for v in config.get("lengths", [])]
    if len(set(lengths)) != len(lengths):
        raise ValueError("window lengths must be distinct")
    if not lengths:
        return []
    stride = _integer(config.get("stride", 1), "stride")
    scoring = config.get("scoring", "episodes")
    if scoring not in {"episodes", "families"}:
        raise ValueError("window scoring must be episodes or families")
    basis = config.get("scoring_basis", "ground_truth")
    if basis not in {"ground_truth", "visible"}:
        raise ValueError("scoring_basis must be ground_truth or visible")
    if type(config.get("include_partial", False)) is not bool:
        raise ValueError("include_partial must be boolean")
    original = clean.sort_values("timestamp", kind="stable").set_index("record_id", drop=False)
    observed = corrupted.sort_values("timestamp", kind="stable").set_index("record_id", drop=False)
    timeline = pd.Index(original["timestamp"].drop_duplicates())
    original_t = pd.Index(original["timestamp"])
    observed_t = pd.Index(observed["timestamp"])
    natural = set(config.get("natural_change_record_ids", []))
    altered = set(masks["record_id"]) if len(masks) else set()
    copies = {}
    if "source_record_id" in masks:
        extra = masks.loc[masks["source_record_id"].notna() & ~masks["record_id"].isin(original.index)]
        copies = dict(zip(extra["record_id"], extra["source_record_id"]))

    by_id = defaultdict(list)
    by_episode = {}
    for ep in episodes:
        eid = ep["episode_id"]
        changed = set(ep["record_ids"])
        if not changed:
            continue
        # Work in clean acquisition coordinates for coverage, and recorded
        # coordinates for evidence. An added copy refers back to its source.
        sources = {copies.get(i, i) for i in changed}
        true_rows = original.loc[[rid for rid in sources if rid in original.index]]
        onsets = true_rows.groupby("sensor_id", sort=False)["timestamp"].min().to_dict()
        by_episode[eid] = (ep, sources, changed, onsets)
        for rid in sources | changed:
            by_id[rid].append(eid)
    windows = []
    for length in lengths:
        for offset in range(0, len(timeline), stride):
            end = min(offset + length, len(timeline))
            if end - offset < length and not config.get("include_partial", False):
                break
            left_time = timeline[offset]
            right_open = end < len(timeline)
            right_time = timeline[end] if right_open else timeline[end-1]
            true_frame = original.iloc[original_t.searchsorted(left_time):
                                       original_t.searchsorted(right_time, side="left" if right_open else "right")]
            obs_frame = observed.iloc[observed_t.searchsorted(left_time):
                                      observed_t.searchsorted(right_time, side="left" if right_open else "right")]
            touched = dict.fromkeys(e for rid in [*true_frame.index, *obs_frame.index] for e in by_id.get(rid, ()))
            observed_clean = obs_frame.loc[~obs_frame.index.isin(altered) & obs_frame["value"].notna()]
            labels = [
                _window_episode(*by_episode[eid], true_frame, obs_frame, observed_clean, original,
                                observed, copies, left_time, timeline[end-1])
                for eid in touched
            ]
            generating = _class([e for e in labels if e["affected_record_ids"] or e["visible_record_ids"]], scoring)
            visible = _class([e for e in labels if e["identifiable_given_context"]], scoring)
            kind = visible if basis == "visible" else generating
            windows.append({
                "window_id": f"w{length}-{offset}", "class": kind, "ground_truth_class": generating, "visible_class": visible,
                "start": left_time, "end": right_time, "end_exclusive": right_open,
                "length": length, "actual_length": end - offset, "clean_record_ids": true_frame.index.tolist(),
                "observation_record_ids": obs_frame.index.tolist(), "episodes": labels,
                "natural_change_control_ids": [rid for rid in true_frame.index if rid in natural] if generating == "clean" else [],
            })
    return _select(windows, config, rng)
