"""Stage and validate all channels and phases before committing one occurrence."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .availability import transform_availability
from .config import Request, map_severity
from .core import Candidate, Context, Reject, check
from .data import elapsed_seconds, fit_jitter_bounds, shift_timestamps, unit_seconds
from .scheduler import overlaps
from .values import transform_value


def stage(family):
    return {"timestamp": "timestamp", "missing_values": "missing"}.get(family, "value")


@dataclass
class Channel:
    name: str
    records: pd.DataFrame
    t: np.ndarray
    train: np.ndarray
    meta: dict
    external: tuple | None
    references: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x = self.records.value.to_numpy(dtype=float, copy=True)
        self.x.setflags(write=False)
        self.values = self.x.copy()
        self.offsets = np.zeros(len(self.x))
        self.dropped = np.zeros(len(self.x), dtype=bool)
        self.occupied = {name: np.zeros(len(self.x), dtype=bool) for name in ("value", "missing", "timestamp")}
        self.separation = {name: np.zeros(len(self.x), dtype=bool) for name in self.occupied}
        self.protected = np.zeros(len(self.x), dtype=bool)
        self.recorded = self.records.timestamp.copy()
        self.copies = []


@dataclass
class Prepared:
    channel: Channel
    candidate: Candidate
    protected: np.ndarray
    recorded: pd.Series
    reference: object
    copy_records: pd.DataFrame | None = None


def _context_indices(req, phases, size):
    first, last = phases[0].idx[0], phases[-1].idx[-1]
    before = after = 0
    if req.key in ("contextual", "glitch"):
        before = after = req.reference["local_neighbors"]
    elif req.family == "burst" or req.key == "stuck":
        before = 1
    if req.family == "burst" and req.params["end_mode"] == "recovery":
        after = max(after, 1)
    return np.r_[np.arange(max(0, first-before), first), np.arange(last+1, min(size, last+1+after))]


def prepare(c: Channel, idx, req: Request, ref, cfg, rng) -> Prepared:
    kind = stage(req.family)
    tolerance = req.tolerance
    if cfg["validation"]["use_sensor_resolution"]:
        tolerance = max(tolerance, c.meta.get("resolution", 0))
    ctx = Context(req.family, req.subtype, req.key, req.params, req.magnitude, rng,
                  c.x, c.t, idx, c.meta, ref.global_stats, ref, tolerance)
    transform = transform_value if kind == "value" else transform_availability
    candidate = transform(ctx)
    main_idx = candidate.phases[0].idx
    if not np.array_equal(main_idx[:len(idx)], idx):
        raise Reject("candidate_changed_scheduled_footprint")
    support = np.concatenate([p.idx for p in candidate.phases])
    checks = candidate.checks
    checks.append(check("eligible_contiguous_support", len(support) and np.all(np.diff(support) == 1)
                        and support[0] >= 0 and support[-1] < len(c.x) and not c.train[support].any()))
    for phase in candidate.phases:
        payloads = [a for a in (phase.values, phase.offsets, phase.copies) if a is not None]
        valid = len(payloads) + int(phase.missing is not None) == 1
        checks.append(check("aligned_phase_payload", valid and all(np.shape(a) == phase.idx.shape for a in payloads)))
    for name, coord in (("allowed_indices", support), ("allowed_seconds", c.t[support])):
        bounds = req.schedule.get(name)
        if bounds:
            checks.append(check(name, coord[0] >= bounds[0] and coord[-1] < bounds[1]))
    if kind != "timestamp":
        checks.append(check("observed_clean_support", np.isfinite(c.x[support]).all()))
    sharing = [set(pair) for pair in cfg["composition"]["allow_overlap"]]
    for prior, occupied in c.occupied.items():
        if {kind, prior} in sharing:
            continue
        guard = max(req.schedule["min_separation"], int(kind == prior == "missing"))
        if overlaps(support, occupied, guard) or c.separation[prior][support].any():
            raise Reject("overlap_or_separation", existing_stage=prior, min_separation=guard)
    if c.dropped[support].any() or kind == "value" and not np.isfinite(c.values[support]).all():
        raise Reject("target_already_missing")
    protected = _context_indices(req, candidate.phases, len(c.x))
    if c.protected[support].any() or any(mask[protected].any() for mask in c.occupied.values()):
        raise Reject("requires_unaffected_context")
    recorded = c.recorded
    copies = None
    if kind == "timestamp":
        phase = candidate.phases[0]
        offsets = c.offsets.copy()
        try:
            if req.key == "jitter":
                gaps = np.diff(c.t)
                neighbors = np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf])[support]
                bounds = req.params["local_gap_fraction"] * neighbors
                limit = req.params["max_displacement_seconds"]
                if limit is not None:
                    limit = map_severity(req.magnitude, limit)
                if req.params["bound_policy"] == "clip":
                    phase.offsets = fit_jitter_bounds(c.records.timestamp.iloc[support], phase.offsets,
                                                      bounds, limit, cfg["adapter"])
            if phase.copies is None:
                offsets[support] = phase.offsets
            recorded = shift_timestamps(c.records.timestamp, offsets, cfg["adapter"])
            if phase.copies is not None:
                src = c.records.iloc[support]
                copy_times = shift_timestamps(recorded.iloc[support], phase.copies, cfg["adapter"])
                copies = src.assign(timestamp=copy_times, value=c.values[support])
        except (ValueError, OverflowError) as exc:
            raise Reject("timestamp_precision", detail=str(exc)) from exc
        present = recorded[~c.dropped]
        existing_copies = [frame.timestamp for frame in c.copies]
        if copies is not None:
            existing_copies.append(copies.timestamp)
        if existing_copies:
            present = pd.concat([present, *existing_copies])
            present = present.iloc[np.lexsort((present.to_numpy(), present.index.to_numpy()))]
        checks.append(check("timestamp_order", req.params["allow_reorder"] or present.is_monotonic_increasing))
        # Existing explicitly requested duplicates may remain in another interval.
        proposed_times = copies.timestamp if copies is not None else recorded.iloc[support]
        collisions = present[present.duplicated(keep=False)].isin(proposed_times).any()
        checks.append(check("timestamp_collisions", req.params["allow_collisions"] or not collisions))
        baseline = c.records.timestamp.iloc[support]
        changed = proposed_times.to_numpy() != baseline.to_numpy()
        if req.key == "duplicate":
            checks.append(check("duplicate_preserves_timestamp", not changed.any()))
        else:
            require_all = req.key in {"fixed_shift", "near_duplicate", "dst_shift"}
            checks.append(check("materialized_time_change", changed.all() if require_all else changed.any()))
        if pd.api.types.is_integer_dtype(proposed_times) and pd.api.types.is_integer_dtype(baseline):
            actual = np.array([int(a) - int(b) for a, b in zip(proposed_times, baseline)], dtype=float)
            actual *= unit_seconds(cfg["adapter"]["time_unit"])
        elif pd.api.types.is_datetime64_any_dtype(baseline):
            actual = (proposed_times - baseline).dt.total_seconds().to_numpy()
        else:
            actual = elapsed_seconds(proposed_times - baseline, cfg["adapter"], origin=0)
        phase.metrics.update(actual_offset_seconds=actual.tolist(),
                             maximum_absolute_offset_seconds=float(max(abs(actual))))
        if req.key == "jitter":
            checks.append(check("encoded_jitter_within_local_gap", np.all(abs(actual) < bounds)))
            if limit is not None:
                checks.append(check("encoded_jitter_within_displacement", np.all(abs(actual) <= limit)))
    return Prepared(c, candidate, protected, recorded, ref.global_stats, copies)


def observability(family, subtype, whole, at_start, meta, params, cfg):
    view = cfg["evaluation"]
    if family == "timestamp":
        if subtype == "duplicate":
            return "U", True, "repeated timestamp and measurement"
        if whole and subtype == "fixed_shift":
            return "M", bool(view["timing_reference_visible"]), "independent time reference"
        return "U+", bool(view["timing_reference_visible"] or
                          view["metadata_visible"] and meta.get("expected_reporting")), "timing reference or reporting rule"
    if family == "missing_values" and params.get("representation") == "drop_records":
        return "U+", bool(view["timing_reference_visible"] or
                          view["metadata_visible"] and meta.get("expected_reporting")), "expected observation process"
    if family in {"bias", "scaling", "drift"}:
        return "U/M", bool(not at_start or view["reference_visible"]), "visible onset or independent measurement"
    if subtype == "saturation":
        return "U+", bool(view["metadata_visible"]), "cap context"
    return "U", True, "within-series evidence"


def commit(group: list[Prepared], req, labels, mask_rows, *, index, opportunity, seed, cfg, allocate_ids):
    """Group corresponding phases across channels under one absolute episode ID."""
    parent = None
    for j in range(len(group[0].candidate.phases)):
        episode_id = f"e{len(labels)+1:06d}"
        first = group[0].candidate.phases[j]
        label = {"episode_id": episode_id, "parent_episode_id": parent, "family": first.family,
                 "subtype": first.subtype, "source_alias": req.alias, "plan_index": index,
                 "opportunity": opportunity, "seed": seed, "preset_version": cfg["version"],
                 "severity": req.severity, "severity_magnitude": req.magnitude,
                 "severity_duration": req.duration_severity, "generating_family": req.family,
                 "generating_subtype": req.subtype, "requested_duration": req.duration if j == 0 else req.params.get("tail_samples"),
                 "schedule": req.schedule, "validation_mode": req.reference["threshold_mode"],
                 "channels": [], "channel_details": {}, "record_ids": [], "support_record_ids": []}
        for item in group:
            c, candidate = item.channel, item.candidate
            phase = candidate.phases[j]
            idx = phase.idx
            kind = stage(phase.family)
            records = c.records.iloc[idx]
            before = c.values[idx].copy()
            if phase.copies is not None:
                copies = item.copy_records.assign(record_id=allocate_ids(len(idx)))
                c.copies.append(copies)
                changed = np.ones(len(idx), dtype=bool)
            elif phase.values is not None:
                c.values[idx] = phase.values
                changed = phase.values != before
            elif phase.offsets is not None:
                c.offsets[idx] = phase.offsets
                c.recorded = item.recorded
                changed = records.timestamp.to_numpy() != c.recorded.iloc[idx].to_numpy()
            else:
                c.values[idx] = np.nan
                c.dropped[idx] = phase.missing == "drop_records"
                changed = np.ones(len(idx), dtype=bool)
            c.occupied[kind][idx] = True
            separation = req.schedule["min_separation"]
            if separation:
                c.separation[kind][max(0, idx[0]-separation):idx[-1]+separation+1] = True
            c.protected[item.protected] = True
            affected = idx[changed]
            ids = copies.record_id.tolist() if phase.copies is not None else c.records.record_id.iloc[affected].tolist()
            label["channels"].append(c.name)
            label["record_ids"].extend(ids)
            label["support_record_ids"].extend(records.record_id.tolist())
            stats = item.reference
            # The generator may use clean anchor or recovery observations to
            # define its equation. Only materialized changes bound the defect.
            start, end = int(affected[0]), int(affected[-1])
            params = phase.params if j else {**candidate.params, **phase.params}
            code, visible, condition = observability(phase.family, phase.subtype, len(affected) == len(c.x),
                                                    start == 0, c.meta, req.params, cfg)
            label["channel_details"][c.name] = {
                "onset": c.records.timestamp.iloc[start], "end": c.records.timestamp.iloc[end],
                "start_index": start, "end_index": end, "elapsed_seconds": c.t[end]-c.t[start],
                "scheduled_support": {"onset": records.timestamp.iloc[0], "end": records.timestamp.iloc[-1],
                                      "start_index": int(idx[0]), "end_index": int(idx[-1]),
                                      "elapsed_seconds": c.t[idx[-1]] - c.t[idx[0]]},
                "extent": {"support": len(idx), "modified": len(ids), "fraction": len(ids)/len(c.x)},
                "preceding_records": start, "following_records": len(c.x)-end-1,
                "reaches_record_end": end == len(c.x)-1, "parameters": params,
                "reference": {"scale": stats.scale, "source": stats.source, "lower": stats.lower,
                              "upper": stats.upper, "rate_threshold": stats.rate, "typical_gap": stats.gap},
                "achieved": phase.metrics, "generating_checks": candidate.checks if j else [],
                "checks": ([check("held_bias_offset", True, offset=params["offset"])] if j else candidate.checks),
                "information_needed": code,
                "identifiable_given_context": visible, "identifiability_condition": condition,
            }
            for pos, local_i, record_id in zip(np.flatnonzero(changed), affected, ids):
                recorded_time = copies.timestamp.iloc[pos] if phase.copies is not None else c.recorded.iloc[local_i]
                mask_rows.append({"episode_id": episode_id, "record_id": record_id,
                                  "source_record_id": c.records.record_id.iloc[local_i],
                                  "sensor_id": c.name, "kind": kind, "true_timestamp": records.timestamp.iloc[pos],
                                  "recorded_timestamp": recorded_time, "clean_value": c.x[local_i],
                                  "prior_value": before[pos], "corrupt_value": c.values[local_i],
                                  "dropped": c.dropped[local_i]})
        details = label["channel_details"].values()
        label["onset"] = min(d["onset"] for d in details)
        label["end"] = max(d["end"] for d in details)
        label["scheduled_onset"] = min(d["scheduled_support"]["onset"] for d in details)
        label["scheduled_end"] = max(d["scheduled_support"]["end"] for d in details)
        if first.family == "missing_values":
            patterns = sorted({d["achieved"]["pattern"] for d in details})
            label["patterns"] = patterns + (["multi_channel"] if len(group) > 1 else [])
        labels.append(label)
        parent = episode_id
