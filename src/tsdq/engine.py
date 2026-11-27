"""Coordinate calibration, placement, atomic validation and benchmark outputs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .availability import validate_availability_params
from .composition import Channel, commit, prepare, stage
from .config import ExperimentConfig, Request, merge, severity_number
from .core import InjectionNotFeasible, Reject
from .data import adapt, elapsed_seconds
from .reference import Reference, validate_reference_config
from .result import InjectionResult, portable
from .scheduler import make_schedule
from .values import validate_value_params
from .windows import slice_windows


MASK_COLUMNS = ["episode_id", "record_id", "source_record_id", "sensor_id", "kind", "true_timestamp",
                "recorded_timestamp", "clean_value", "prior_value", "corrupt_value", "dropped"]


def _channels(clean, external, cfg, metadata):
    adapter = cfg["adapter"]
    origin = clean.timestamp.min()
    training_ids = cfg["reference"]["train_record_ids"]
    if training_ids is not None:
        if external is not None:
            raise ValueError("choose either train_record_ids or a separate training table")
        if len(set(training_ids)) != len(training_ids) or set(training_ids) - set(clean.record_id):
            raise ValueError("train_record_ids must be distinct existing record IDs")
    channels = {}
    for name, records in clean.groupby("sensor_id", sort=False):
        train = np.zeros(len(records), dtype=bool)
        calibration = None
        if external is not None:
            fit = external[external.sensor_id == name]
            if fit.empty:
                raise ValueError(f"no external training rows for {name}")
            calibration = (fit.value.to_numpy(), elapsed_seconds(fit.timestamp, adapter))
        elif training_ids is not None:
            train = records.record_id.isin(training_ids).to_numpy()
        else:
            n = int(np.floor(len(records) * cfg["reference"]["train_fraction"]))
            train[:n] = True
        chan_meta = deepcopy(metadata.get(name, {}))
        if "resolution" in chan_meta and (not np.isfinite(chan_meta["resolution"]) or chan_meta["resolution"] <= 0):
            raise ValueError(f"Invalid sensor resolution for {name}")
        channels[name] = Channel(name, records, elapsed_seconds(records.timestamp, adapter, origin),
                                 train, chan_meta, calibration)
    return channels


def _reference(channel, req):
    key = json.dumps(req.reference, sort_keys=True)
    if key not in channel.references:
        channel.references[key] = Reference(channel.x, channel.t, channel.train, req.reference,
                                             channel.meta, training=channel.external)
    return channel.references[key]


def _version():
    digest = sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return {"package": "0.1.0", "source_sha256": digest.hexdigest(), "numpy": np.__version__,
            "pandas": pd.__version__, "python": platform.python_version()}


def _fingerprint(frame):
    values = pd.util.hash_pandas_object(frame[["record_id", "sensor_id", "timestamp", "value"]], index=False)
    return sha256(values.to_numpy().tobytes()).hexdigest()


def _finalize_labels(channels, labels, masks):
    """Score clean context and any explicit masking on the final observation stream."""
    altered_ids = set(masks.record_id)
    bad_indices = {name: np.flatnonzero(~np.isfinite(c.x) | c.records.record_id.isin(altered_ids).to_numpy() | c.train)
                   for name, c in channels.items()}
    for label in labels:
        for name, detail in label["channel_details"].items():
            c = channels[name]
            lo, hi = detail["start_index"], detail["end_index"]
            bad = bad_indices[name]
            left = np.searchsorted(bad, lo)
            right = np.searchsorted(bad, hi, side="right")
            detail["context_before"] = int(lo-bad[left-1]-1 if left else lo)
            detail["context_after"] = int(bad[right]-hi-1 if right < len(bad) else len(c.x)-hi-1)
            if stage(label["family"]) == "value":
                idx = np.arange(lo, hi+1)
                hidden = ~np.isfinite(c.values[idx]) & (c.x[idx] != c.values[idx])
                hidden_ids = set(c.records.record_id.iloc[idx[hidden]])
                changed_ids = set(c.records.record_id.iloc[idx]).intersection(label["record_ids"])
                detail["hidden_record_ids"] = sorted(hidden_ids.intersection(changed_ids), key=str)
                if changed_ids <= hidden_ids:
                    detail["identifiable_given_context"] = False
                    detail["identifiability_condition"] = "all modified values hidden in the final observations"
                if (label["family"] in {"bias", "drift", "scaling"} and not detail["context_before"]
                        and not label.get("parent_episode_id")):
                    detail["identifiable_given_context"] &= bool(label.get("reference_visible"))


def _run_request(index, req, channels, cfg, rng, labels, rows, failures, rejections, allocate_ids):
    clocks = {name: c.t for name, c in channels.items()}
    eligible = {name: ~c.train for name, c in channels.items()}
    if req.family != "timestamp":
        eligible = {name: mask & np.isfinite(channels[name].x) for name, mask in eligible.items()}
    opportunities = 0
    accepted = 0
    diagnostics = {}
    try:
        schedule = make_schedule(clocks, eligible, req.schedule, req.duration, rng, diagnostics=diagnostics)
        for opportunities, opportunity in enumerate(schedule, 1):
            last_reason = "no_eligible_onsets"
            attempt = 0
            for attempt, placements in enumerate(opportunity.candidates(), 1):
                seed = int(rng.integers(0, 2**63))
                candidate_rng = np.random.default_rng(seed)
                proposed = []
                try:
                    for name, idx in placements.items():
                        c = channels[name]
                        proposed.append(prepare(c, idx, req, _reference(c, req), cfg, candidate_rng))
                except Reject as exc:
                    last_reason = exc.reason
                    rejections.append({"plan_index": index, "opportunity": opportunities, "attempt": attempt,
                                       "seed": seed, "onset_indices": {n: int(i[0]) for n, i in placements.items()},
                                       "channel": name, "reason": exc.reason, "measurements": exc.measurements})
                    continue
                commit(proposed, req, labels, rows, index=index, opportunity=opportunities,
                       seed=seed, cfg=cfg, allocate_ids=allocate_ids)
                accepted += 1
                break
            else:
                failures.append(InjectionNotFeasible(index, req.family, req.subtype, opportunities, attempt, last_reason))
    except Reject as exc:
        failures.append(InjectionNotFeasible(index, req.family, req.subtype, opportunities, 0, exc.reason))
        rejections.append({"plan_index": index, "reason": exc.reason, "measurements": exc.measurements})
    return {"plan_index": index, "opportunities": opportunities, "accepted_occurrences": accepted, **diagnostics}


class DefectInjector:
    """Reusable specification. Each inject call creates fresh state and generators."""

    def __init__(self, config=None, *, metadata=None, seed=None):
        self.config = ExperimentConfig.load(config)
        self.metadata = merge(self.config.settings["metadata"], metadata or {})
        self.seed = self.config.settings["seed"] if seed is None else seed
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    def inject(self, clean_df, plan=None, *, training_df=None) -> InjectionResult:
        cfg = deepcopy(self.config.settings)
        cfg.update(seed=int(self.seed), metadata=deepcopy(self.metadata))
        if plan is not None:
            cfg["plan"] = deepcopy(plan)
        severity_number(cfg["severity"])
        if not isinstance(cfg["plan"], list):
            raise ValueError("plan must be a list of episode requests")
        clean = adapt(clean_df, cfg["adapter"], self.metadata)
        columns = cfg["output"]["observation_columns"]
        if not isinstance(columns, list) or not columns or len(set(columns)) != len(columns) or set(columns) - set(clean):
            raise ValueError("observation_columns must select distinct existing columns")
        external = adapt(training_df, cfg["adapter"], self.metadata) if training_df is not None else None
        channels = _channels(clean, external, cfg, self.metadata)
        requests = [Request.resolve(cfg, item) for item in cfg["plan"]]
        for req in requests:
            validate_reference_config(req.reference)
            validate = validate_value_params if stage(req.family) == "value" else validate_availability_params
            validate(req.key, req.params)
        order = cfg["composition"]["order"]
        ordered = sorted(enumerate(requests), key=lambda pair: order.index(stage(pair[1].family)))
        injection_seed, window_seed = np.random.SeedSequence(int(self.seed)).spawn(2)
        rng = np.random.default_rng(injection_seed)
        labels, rows, failures, rejections, schedule_results = [], [], [], [], []
        used_ids = set(clean.record_id)
        numeric_ids = pd.api.types.is_integer_dtype(clean.record_id)
        serial = 0

        def allocate_ids(count):
            nonlocal serial
            ids = []
            while len(ids) < count:
                serial += 1
                item = -serial if numeric_ids else f"__tsdq_copy_{serial}"
                if item not in used_ids:
                    ids.append(item)
                    used_ids.add(item)
            return ids

        for index, req in ordered:
            summary = _run_request(index, req, channels, cfg, rng, labels, rows, failures, rejections, allocate_ids)
            schedule_results.append(summary)
        parts, train_ids = [], []
        for c in channels.values():
            part = c.records.assign(value=c.values, timestamp=c.recorded)
            parts.append(part.loc[~c.dropped])
            parts.extend(c.copies)
            train_ids.extend(c.records.record_id[c.train].tolist())
        combined = pd.concat(parts)
        if any(c.copies for c in channels.values()):
            combined = combined.iloc[np.lexsort((combined.timestamp.to_numpy(), combined.index.to_numpy()))]
        else:
            combined = combined.sort_index(kind="stable")
        corrupted = combined.reset_index(drop=True)
        masks = pd.DataFrame(rows, columns=MASK_COLUMNS)
        lookup = corrupted.set_index("record_id")
        masks["final_value"] = masks.record_id.map(lookup.value)
        final_times = lookup.timestamp
        if pd.api.types.is_integer_dtype(final_times) and masks.dropped.any():
            final_times = final_times.astype("UInt64" if pd.api.types.is_unsigned_integer_dtype(final_times) else "Int64")
        masks["final_timestamp"] = masks.record_id.map(final_times)
        masks["finally_dropped"] = ~masks.record_id.isin(lookup.index)
        for label in labels:
            label["reference_visible"] = cfg["evaluation"]["reference_visible"]
        _finalize_labels(channels, labels, masks)
        eval_clean = clean[~clean.record_id.isin(train_ids)]
        eval_corrupted = corrupted[~corrupted.record_id.isin(train_ids)]
        windows = []
        if cfg["windows"]["lengths"]:
            windows = slice_windows(eval_clean, eval_corrupted, labels, masks, cfg["windows"],
                                    np.random.default_rng(window_seed))
        missing_count = sum(int(c.occupied["missing"].sum()) for c in channels.values())
        manifest = {"seed": int(self.seed), "config": cfg, "code_version": _version(),
                    "clean_sha256": _fingerprint(clean), "training_source": "external" if external is not None
                    else "record_ids" if cfg["reference"]["train_record_ids"] is not None else "chronological_prefix",
                    "training_record_ids": train_ids, "external_sha256": _fingerprint(external) if external is not None else None,
                    "resolved_requests": [asdict(req) for req in requests], "schedule_results": schedule_results,
                    "accepted_phases": len(labels), "rejections": rejections, "failures": [asdict(f) for f in failures],
                    "missing": {"record_count": missing_count, "fraction_of_stream": missing_count/len(clean),
                                "longest_gap_samples": max((d["achieved"]["longest_gap_samples"]
                                for ep in labels if ep["family"] == "missing_values"
                                for d in ep["channel_details"].values()), default=0)},
                    "channels": {name: {"records": len(c.x), "eligible_training_records": int(c.train.sum()) if external is None
                                        else len(c.external[0]), "existing_missing": int(np.isnan(c.x).sum()),
                                        "metadata": c.meta, "first_timestamp": c.records.timestamp.iloc[0],
                                        "last_timestamp": c.records.timestamp.iloc[-1]} for name, c in channels.items()},
                    "window_count": len(windows)}
        return InjectionResult(clean, corrupted, portable(labels), masks, portable(manifest), failures,
                               portable(windows), external)
