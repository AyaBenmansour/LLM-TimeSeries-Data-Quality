"""Schedule joint episodes in clean coordinates, independently of defect physics."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .core import Reject


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _interval(value, name, integers=False):
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} expects [lower, upper)")
    lo, hi = ((_integer(x, name, 0) if integers else _number(x, name)) for x in value)
    if lo >= hi:
        raise ValueError(f"empty {name}")
    return lo, hi


def overlaps(idx: np.ndarray, occupied: np.ndarray, separation: int = 0) -> bool:
    """Test consecutive support including a guard of untouched samples on both sides."""
    separation = _integer(separation, "min_separation", 0)
    return bool(len(idx) and occupied[max(0, idx[0] - separation):idx[-1] + separation + 1].any())


def _scope(channels, cfg, rng):
    requested = cfg.get("channels")
    explicit = requested is not None
    if requested == "all" or requested is None:
        pool = list(channels)
    elif isinstance(requested, list):
        pool = requested
    else:
        raise ValueError("channels must be a list, 'all' or null")
    if not pool or len(set(pool)) != len(pool) or set(pool) - channels.keys():
        raise ValueError("channels must refer to distinct existing sensors")
    count = cfg.get("channel_count")
    if count == "all" or (count is None and explicit):
        count = len(pool)
    elif count is None:
        count = 1
    count = _integer(count, "channel_count")
    if count > len(pool):
        raise ValueError("channel_count exceeds available scope")
    rule = cfg.get("channel_selection", "random")
    if rule == "first" or (rule == "random" and len(pool) == count):
        return pool[:count]
    if rule != "random":
        raise ValueError("channel_selection must be first or random")
    return [pool[i] for i in np.sort(rng.choice(len(pool), size=count, replace=False))]


@dataclass
class _PlacementTable:
    # Row i identifies the same proposed physical onset in every channel.
    starts: np.ndarray
    bounds: dict[str, tuple[np.ndarray, np.ndarray]]
    valid: np.ndarray

    def at(self, row):
        return {name: np.arange(lo[row], hi[row]) for name, (lo, hi) in self.bounds.items()}


@dataclass
class Opportunity:
    table: _PlacementTable
    rows: np.ndarray
    max_attempts: int
    rng: np.random.Generator

    def candidates(self) -> Iterator[dict[str, np.ndarray]]:
        """Validation retries may change the draw but stay inside this opportunity."""
        if not len(self.rows):
            return
        if len(self.rows) == 1:
            for _ in range(self.max_attempts):
                yield self.table.at(self.rows[0])
            return
        # Visit distinct locations before reusing one to redraw transform parameters.
        left = self.max_attempts
        while left:
            size = min(left, len(self.rows))
            for row in self.rng.choice(self.rows, size=size, replace=False):
                yield self.table.at(row)
            left -= size


def _table(channels, eligible, names, starts, cfg, duration):
    unit = cfg.get("duration_unit", "samples")
    persistence = cfg.get("persistence", "temporary")
    if persistence not in {"temporary", "unresolved", "whole_channel"}:
        raise ValueError("unknown persistence")
    if unit not in {"samples", "seconds"} or _number(duration, "duration") <= 0:
        raise ValueError("duration must be positive in samples or seconds")
    if unit == "samples" and duration != int(duration):
        raise ValueError("sample duration must already be rounded to an integer")
    indices = _interval(cfg.get("allowed_indices"), "allowed_indices", integers=True)
    seconds = _interval(cfg.get("allowed_seconds"), "allowed_seconds")
    valid = np.ones(len(starts), dtype=bool)
    if seconds:
        valid &= (starts >= seconds[0]) & (starts < seconds[1])
    bounds = {}
    common_start = np.full(len(starts), -np.inf)
    common_end = np.full(len(starts), np.inf)
    for name in names:
        t = channels[name]
        ok = np.asarray(eligible[name], dtype=bool).copy()
        if indices:
            ok[:indices[0]] = False
            ok[indices[1]:] = False
        if seconds:
            ok &= (t >= seconds[0]) & (t < seconds[1])
        lo = np.searchsorted(t, starts)
        if persistence == "whole_channel":
            lo[:] = 0
        if persistence != "temporary":
            hi = np.full(len(starts), len(t))
        elif unit == "samples":
            hi = lo + int(duration)
        else:
            hi = np.searchsorted(t, starts + duration)
            # Avoid claiming an interval that extends beyond observed follow-up.
            valid &= starts + duration <= t[-1]
        valid &= (lo < hi) & (hi <= len(t))
        # Prefix counts identify windows containing even one ineligible record.
        invalid_count = np.r_[0, np.cumsum(~ok)]
        valid &= invalid_count[np.minimum(hi, len(t))] == invalid_count[lo]
        bounds[name] = lo, hi
        if len(names) > 1 and (unit == "samples" or persistence != "temporary"):
            # A record-count episode on asynchronous streams must still refer
            # to intersecting time intervals. Ends use the next clean instant.
            times = np.r_[t, np.nextafter(float(t[-1]), np.inf)]
            common_start = np.maximum(common_start, times[lo])
            common_end = np.minimum(common_end, times[np.minimum(hi, len(t))])
    valid &= common_start < common_end
    return _PlacementTable(starts, bounds, valid)


def _count(occ, duration, scope_size, total, cfg):
    if occ.keys() == {"mode", "episode_count"}:
        return _integer(occ["episode_count"], "episode_count", 0)
    if set(occ) not in ({"mode", "target_records"}, {"mode", "target_fraction"}):
        raise ValueError("fixed_count requires exactly one of episode_count, target_records, target_fraction")
    if cfg.get("duration_unit", "samples") != "samples" or cfg.get("persistence", "temporary") != "temporary":
        raise ValueError("a record target requires temporary, sample-count episodes")
    if "target_fraction" in occ:
        fraction = _number(occ["target_fraction"], "target_fraction")
        if not 0 <= fraction <= 1:
            raise ValueError("target_fraction must be in [0,1]")
        target = int(np.floor(total * fraction + 0.5))
    else:
        target = _integer(occ["target_records"], "target_records", 0)
    k, remainder = divmod(target, int(duration) * scope_size)
    if remainder:
        raise Reject("record_target_indivisible", target=target, records_per_episode=int(duration)*scope_size)
    return k


def make_schedule(channels: dict[str, np.ndarray], eligible: dict[str, np.ndarray],
                  schedule: dict, duration: float, rng: np.random.Generator,
                  *, diagnostics: dict | None = None) -> Iterator[Opportunity]:
    """Plan onsets against clean, per-channel seconds measured from a common origin."""
    allowed = {"occurrence", "recurrence", "period", "cluster_size", "cluster_width", "recurrence_unit",
               "duration_unit", "channels", "channel_count", "channel_selection", "anchor_channel",
               "onset_index", "onset_seconds", "allowed_indices", "allowed_seconds", "persistence",
               "min_separation", "max_attempts"}
    if set(schedule) - allowed:
        raise ValueError(f"unknown schedule keys: {sorted(set(schedule) - allowed)}")
    if not channels or channels.keys() != eligible.keys():
        raise ValueError("supply a clock and eligibility mask for each channel")
    for name, t in channels.items():
        if not len(t) or np.shape(eligible[name]) != np.shape(t) or np.ndim(t) != 1:
            raise ValueError(f"invalid clock or eligibility array for {name}")
        if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError(f"channel {name} requires strictly increasing seconds")
    names = _scope(channels, schedule, rng)
    if diagnostics is not None:
        diagnostics["selected_channels"] = names
    anchor = schedule.get("anchor_channel") or names[0]
    if anchor not in channels:
        raise ValueError("unknown anchor_channel")
    t = channels[anchor]
    attempts = _integer(schedule.get("max_attempts", 32), "max_attempts")
    _integer(schedule.get("min_separation", 0), "min_separation", 0)
    occurrence = schedule.get("occurrence", {"mode": "fixed_count", "episode_count": 1})
    mode = occurrence.get("mode")
    if mode == "fixed_count":
        count = _count(occurrence, duration, len(names), sum(map(len, channels.values())), schedule)
    elif mode == "bernoulli_start" and set(occurrence) == {"mode", "q"}:
        q = _number(occurrence["q"], "start probability")
        if not 0 <= q <= 1:
            raise ValueError("q must be in [0,1]")
        count = None
    else:
        raise ValueError("use fixed_count or bernoulli_start with q exclusively")

    rec = schedule.get("recurrence", "random")
    unit = schedule.get("recurrence_unit", "samples")
    if rec not in {"random", "periodic", "clustered"} or unit not in {"samples", "seconds"}:
        raise ValueError("invalid recurrence or recurrence_unit")
    index = schedule.get("onset_index")
    onset = schedule.get("onset_seconds")
    if index is not None and onset is not None:
        raise ValueError("specify at most one onset coordinate")
    explicit = index is not None or onset is not None
    if index is not None:
        index = _integer(index, "onset_index", 0)
        if index >= len(t):
            raise Reject("onset_index_outside_channel", index=index, channel=anchor)
        onset = t[index]
    elif onset is not None:
        onset = _number(onset, "onset_seconds")
        index = np.searchsorted(t, onset)
    if rec == "clustered" and explicit:
        raise ValueError("use an allowed region to locate clusters")
    if schedule.get("persistence") == "whole_channel":
        if explicit or rec != "random":
            raise ValueError("whole_channel fixes onset to recording start and requires random recurrence")
        starts = np.array([min(channels[name][0] for name in names)])
    elif rec == "periodic":
        period = _number(schedule["period"], "period")
        if period <= 0 or (unit == "samples" and int(period) != period):
            raise ValueError("period must be positive and integral in sample units")
        if onset is None:
            base = _table(channels, eligible, names, t, schedule, duration)
            available = np.flatnonzero(base.valid)
            if not len(available):
                if count == 0 or count is None:
                    return
                raise Reject("no_eligible_periodic_origin")
            index = available[0]
            onset = t[index]
        if unit == "samples":
            if index >= len(t) or onset != t[index]:
                raise ValueError("sample periodicity requires an anchor-observation onset")
            starts = t[index::int(period)]
        else:
            number = max(0, int(np.floor((t[-1] - onset) / period)) + 1)
            if count is not None:
                number = min(count, number)
            starts = onset + period * np.arange(number)
    elif explicit:
        starts = np.array([onset])
    else:
        starts = t

    table = _table(channels, eligible, names, starts, schedule, duration)
    rows = np.flatnonzero(table.valid)
    if diagnostics is not None:
        diagnostics["eligible_starts"] = len(rows)
        diagnostics["requested_occurrences"] = count
    if count == 0:
        return
    if count is not None and not len(rows):
        raise Reject("no_eligible_onsets")
    if rec == "periodic":
        if count is not None:
            for row in range(count):
                choices = [row] if row < len(starts) and table.valid[row] else []
                yield Opportunity(table, np.asarray(choices, dtype=int), attempts, rng)
            return
    elif rec == "clustered":
        width = _number(schedule["cluster_width"], "cluster_width")
        size = _integer(schedule["cluster_size"], "cluster_size")
        if width <= 0 or (unit == "samples" and int(width) != width):
            raise ValueError("cluster_width must be positive and integral in sample units")
        if not len(rows):
            return
        coordinate = rows if unit == "samples" else starts[rows]
        # Each band spans one width. Draw separated active bands to keep clusters distinct.
        band_ids = np.floor((coordinate - coordinate[0]) / width).astype(int)
        available = np.unique(band_ids)
        groups = []
        while len(available) and (count is None or len(groups) * size < count):
            band = rng.choice(available)
            groups.append(rows[band_ids == band])
            available = available[np.abs(available - band) > 1]
        if count is not None:
            for i in range(count):
                group = groups[i // size] if i // size < len(groups) else np.empty(0, dtype=int)
                yield Opportunity(table, group, attempts, rng)
            return
        rows = np.sort(np.concatenate([rng.choice(group, min(size, len(group)), replace=False)
                                       for group in groups]))

    if count is None:
        # Bernoulli draws are made once. Validation retries keep the selected onset.
        if diagnostics is not None:
            diagnostics.update(bernoulli_trials=len(rows), expected_starts=q * len(rows))
        for row in rows[rng.random(len(rows)) < q]:
            yield Opportunity(table, np.array([row]), attempts, rng)
    else:
        for _ in range(count):
            yield Opportunity(table, rows, attempts, rng)
