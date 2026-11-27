"""Render paired clean/corrupted measurements and their record-level differences."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from textwrap import wrap

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

from tsdq.data import elapsed_seconds


INK = "#172b35"
BEFORE = "#63737b"
AFTER = "#007a87"
CHANGE = "#bd4d16"
TAIL = "#734b92"
GRID = "#e2e8e9"
SCHEDULED = "#efe6c9"


def gallery_pages(cases):
    """Group complete cases into pages, keeping asynchronous channels together."""
    groups = defaultdict(list)
    for case, result in cases:
        groups[case.group].append((case, result))
    page_number = 0
    for group, items in groups.items():
        pages = []
        count = 0
        for pair in items:
            n_channels = pair[1].clean.sensor_id.nunique()
            if not pages or count + n_channels > 2:
                pages.append([])
                count = 0
            pages[-1].append(pair)
            count += n_channels
        slug = re.sub(r"[^a-z0-9]+", "-", group.lower()).strip("-")
        for i, page in enumerate(pages, start=1):
            page_number += 1
            suffix = f"-{i}" if len(pages) > 1 else ""
            yield f"{page_number:02d}-{slug}{suffix}.png", group, page


def _clock_bounds(clean, masks, config):
    """Get comparable axis limits from the actual changes plus nearby observations."""
    origin = clean.timestamp.min()
    limits = []
    for name, channel in clean.groupby("sensor_id", sort=False):
        mask = masks.loc[masks.sensor_id == name]
        positions = np.flatnonzero(channel.record_id.isin(mask.source_record_id))
        if len(positions):
            neighbors = channel.iloc[max(0, positions[0] - 6):positions[-1] + 7]
            limits.extend(elapsed_seconds(neighbors.timestamp, config, origin))
            limits.extend(elapsed_seconds(mask.recorded_timestamp, config, origin))
    if not limits:
        limits = elapsed_seconds(clean.timestamp, config, origin)
    lo, hi = min(limits), max(limits)
    span = hi - lo
    pad = .035 * span if span > 0 else 1.
    scale, unit = (3600., "h") if span >= 7200 else (60., "min") if span >= 300 else (1., "s")
    return origin, scale, unit, ((lo - pad) / scale, (hi + pad) / scale)


def _axes_style(ax):
    ax.grid(axis="y", color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#c5cfcf")
    ax.tick_params(length=3, color="#a1aeaf", labelcolor=INK, pad=4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))


def _marker(ax, times, values, *, color=CHANGE, symbol="o", hollow=False):
    if len(times):
        ax.scatter(times, values, s=33 if hollow else 24, marker=symbol, zorder=5,
                   facecolor="none" if hollow else color, edgecolor=color if hollow else "white",
                   linewidth=1.35 if hollow else .65)


def _shade_scheduled(top, bottom, channel, result, time):
    """Shade each scheduled support interval; changed readings are marked separately."""
    if channel.empty:
        return
    name = channel.sensor_id.iloc[0]
    clean_t = np.asarray(time(channel.timestamp), dtype=float)
    half = .5 * float(np.median(np.diff(clean_t))) if len(clean_t) > 1 else .5
    for ep in result.episodes:
        support = ep["channel_details"].get(name, {}).get("scheduled_support")
        if not support or ep["parent_episode_id"]:
            continue
        lo, hi = clean_t[support["start_index"]] - half, clean_t[support["end_index"]] + half
        for ax in (top, bottom):
            ax.axvspan(lo, hi, color=SCHEDULED, alpha=.55, linewidth=0, zorder=0)


def _format(value):
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def describe(result) -> str:
    """One-line summary of severity, extent, actual parameters and validation outcome."""
    roots = [ep for ep in result.episodes if not ep["parent_episode_id"]]
    first = roots[0]
    details = [d for ep in result.episodes for d in ep["channel_details"].values()]
    scheduled = sum(d["scheduled_support"]["end_index"] - d["scheduled_support"]["start_index"] + 1
                    for ep in roots for d in ep["channel_details"].values())
    checks = [c for ep in result.episodes for c in ep.get("generating_checks", [])]
    checks += [c for d in details for c in d.get("checks", [])]
    passed = sum(bool(c["passed"]) for c in checks)
    params = next(iter(first["channel_details"].values()))["parameters"]
    shown = [f"{k}={_format(v)}" for k, v in params.items()
             if not isinstance(v, (list, dict)) and v is not None][:6]
    root_ids = {ep["episode_id"] for ep in roots}
    changed = int(result.masks.episode_id.isin(root_ids).sum())
    follow_on = len(result.masks) - changed
    parts = [f"severity {first['severity']:.2f}",
             f"{len(roots)} episode{'s' if len(roots) != 1 else ''}",
             f"scheduled {scheduled} → changed {changed}"
             + (f" (+{follow_on} follow-on Bias)" if follow_on else ""),
             f"checks {passed}/{len(checks)} passed"]
    return " · ".join(parts) + ("\n" + ", ".join(shown) if shown else "")


def _draw_channel(top, bottom, channel, observed, masks, result, clock):
    origin, scale, unit, limits = clock
    cfg = result.manifest["config"]["adapter"]

    def time(series):
        return elapsed_seconds(series, cfg, origin) / scale

    def in_view(t):
        return (t >= limits[0]) & (t <= limits[1])

    _shade_scheduled(top, bottom, channel, result, time)
    # Crop each sensor after combining the true and recorded coordinates.
    clean_t = time(channel.timestamp)
    channel = channel.loc[in_view(clean_t)]
    clean_t = clean_t[in_view(clean_t)]
    obs_t = time(observed.timestamp)
    observed = observed.loc[in_view(obs_t)]
    measured = observed.sort_values("timestamp", kind="stable")
    missing = bool((result.masks.kind == "missing").any())
    timing = bool((result.masks.kind == "timestamp").any())
    if missing:
        # Insert plotting gaps at the known clean positions of deleted records.
        # The private clean reference is available only for this comparison.
        plotted_t = clean_t
        plotted_y = measured.set_index("record_id").value.reindex(channel.record_id)
    else:
        plotted_t, plotted_y = time(measured.timestamp), measured.value
    top.plot(plotted_t, plotted_y, color=AFTER, linewidth=1.65, marker=".", markersize=3.8, zorder=2)
    top.plot(clean_t, channel.value, color=BEFORE, linewidth=1.15,
             linestyle=(0, (3.3, 2.8)), marker=".", markersize=2.4, zorder=3)
    name = channel.sensor_id.iloc[0] if len(channel) else None
    meta = result.manifest["config"]["metadata"].get(name, {})
    units = meta.get("unit")
    label = "Temperature" if units in {"°C", "degC"} else "Value"
    top.set_ylabel(f"{label} ({units})" if units else label)
    for ax in (top, bottom):
        _axes_style(ax)
        ax.set_xlim(limits)
    top.tick_params(axis="x", labelbottom=False)
    top.margins(y=.16)
    bottom.set_xlabel(f"Elapsed time ({unit})")

    if missing:
        observed_mask = ~channel.record_id.isin(masks.source_record_id)
        bottom.scatter(clean_t[observed_mask], np.ones(observed_mask.sum()), color=AFTER, s=9, zorder=3)
        for dropped, group in masks.groupby("dropped"):
            _marker(bottom, time(group.true_timestamp), np.zeros(len(group)),
                    symbol="v" if dropped else "X", hollow=bool(dropped))
        bottom.set_yticks([0, 1], ["Missing", "Present"])
        bottom.set_ylim(-.46, 1.42)
        return

    copies = masks.record_id != masks.source_record_id
    unchanged = ~channel.record_id.isin(masks.loc[~copies, "source_record_id"])
    bottom.axhline(0, color="#8f9ea4", linewidth=.9, linestyle=(0, (2, 3)), zorder=1)
    bottom.scatter(clean_t[unchanged], np.zeros(unchanged.sum()), s=8, color="#85979e", zorder=2)
    if timing:
        # Subtract on the original time axis. Elapsed coordinates can be much
        # larger than a small jitter offset when the placement is late in a run.
        signed = []
        for true, written in zip(masks.true_timestamp, masks.recorded_timestamp):
            signed.append(float(elapsed_seconds([written], cfg, true)[0]))
        errors = np.asarray(signed)
        peak = np.max(np.abs(errors), initial=0)
        err_scale, err_unit = ((3600., "h") if peak >= 3600 else (.001, "ms") if 0 < peak < 1 else (1., "s"))
        bottom.set_ylabel(f"Clock error ({err_unit})")
        for is_copy in (False, True):
            keep = (copies == is_copy).to_numpy()
            data = masks.loc[keep]
            _marker(top, time(data.recorded_timestamp), data.corrupt_value,
                    symbol="D" if is_copy else "o", hollow=is_copy)
            _marker(bottom, time(data.true_timestamp), errors[keep] / err_scale,
                    symbol="D" if is_copy else "o", hollow=is_copy)
        if not peak:
            bottom.set_ylim(-.6, .6)
            bottom.set_yticks([0])
    else:
        bottom.set_ylabel(f"Value error ({units})" if units else "Value error")
        tails = {ep["episode_id"] for ep in result.episodes if ep["parent_episode_id"]}
        for is_tail, group in masks.groupby(masks.episode_id.isin(tails)):
            style = {"color": TAIL if is_tail else CHANGE, "symbol": "D" if is_tail else "o"}
            _marker(top, time(group.recorded_timestamp), group.corrupt_value, **style)
            _marker(bottom, time(group.true_timestamp), group.corrupt_value - group.clean_value, **style)
    bottom.margins(y=.24)


def _legend(page):
    handles = [Line2D([], [], color=BEFORE, linestyle="--", label="Before"),
               Line2D([], [], color=AFTER, marker=".", label="After"),
               Patch(facecolor=SCHEDULED, alpha=.55, label="Scheduled episode")]
    if any((r.masks.kind == "value").any() or
           ((r.masks.kind == "timestamp") & (r.masks.record_id == r.masks.source_record_id)).any()
           for _, r in page):
        handles.append(Line2D([], [], linestyle="none", marker="o", color=CHANGE, label="Changed reading"))
    if any(any(ep["parent_episode_id"] for ep in r.episodes) for _, r in page):
        handles.append(Line2D([], [], linestyle="none", marker="D", color=TAIL, label="Subsequent Bias"))
    if any((r.masks.record_id != r.masks.source_record_id).any() for _, r in page):
        handles.append(Line2D([], [], linestyle="none", marker="D", mfc="none", color=CHANGE, label="Added copy"))
    for dropped, marker, label in [(False, "X", "Explicit NaN"), (True, "v", "Removed record")]:
        if any(((r.masks.kind == "missing") & (r.masks.dropped == dropped)).any() for _, r in page):
            handles.append(Line2D([], [], linestyle="none", marker=marker, mfc="none" if dropped else CHANGE,
                                  color=CHANGE, label=label))
    return handles


def _page_figure(group, page, number, total):
    captions = [wrap(case.caption, width=105) for case, _ in page]
    infos = [[w for line in describe(r).split("\n") for w in wrap(line, width=128)] for _, r in page]
    counts = [r.clean.sensor_id.nunique() for _, r in page]
    col_left, col_right = .13, .965
    sensor_height = 2.84
    height = (1.06 + sum(.36 + .20 * len(lines) + .17 * len(info) + .05 + n * sensor_height
                         + (.24 * n if n > 1 else 0)
                         for lines, info, n in zip(captions, infos, counts)) + .16)
    fig = Figure(figsize=(10, height), facecolor="white")
    FigureCanvasAgg(fig)

    def text(x, y, value, **props):
        fig.text(x, 1 - y / height, value, va="top", **{"color": INK, **props})

    def axes(y, h):
        return fig.add_axes([col_left, 1 - (y + h) / height, col_right - col_left, h / height])

    text(col_left, .16, group, fontsize=22, weight="bold")
    text(col_right, .24, f"{number} / {total}", fontsize=10, ha="right")
    handles = _legend(page)
    fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="upper left",
               bbox_to_anchor=(col_left, 1 - .57 / height), borderaxespad=0,
               handlelength=2.1, columnspacing=1.7, fontsize=9.5)
    fig.add_artist(Line2D([col_left, col_right], [1 - .92 / height] * 2,
                          transform=fig.transFigure, color=GRID, linewidth=1))
    y = 1.03
    for (case, result), caption, info, n_channels in zip(page, captions, infos, counts):
        clean = result.clean
        copied = (result.masks.record_id != result.masks.source_record_id).all()
        affected = len(result.masks)
        text(col_left, y, case.title, fontsize=14.6, weight="bold")
        text(col_right, y + .035, f"{affected} {'added' if copied else 'affected'} "
             f"{'record' if affected == 1 else 'records'}", fontsize=9.1, ha="right")
        y += .29
        text(col_left, y, "\n".join(caption), fontsize=10, linespacing=1.25)
        y += .20 * len(caption) + .02
        text(col_left, y, "\n".join(info), fontsize=8.2, color=BEFORE, family="DejaVu Sans Mono",
             linespacing=1.2)
        y += .17 * len(info) + .10
        clock = _clock_bounds(clean, result.masks, result.manifest["config"]["adapter"])
        for name, channel in clean.groupby("sensor_id", sort=False):
            if n_channels > 1:
                text(col_left, y, str(name), fontsize=10.5, weight="bold")
                y += .24
            top = axes(y, 1.33)
            bottom = axes(y + 1.47, .70)
            _draw_channel(top, bottom, channel, result.corrupted[result.corrupted.sensor_id == name],
                          result.masks[result.masks.sensor_id == name], result, clock)
            y += sensor_height
    return fig


def render_gallery(cases, output_path) -> list[Path]:
    """Write a printable PDF and the same pages as PNG images in reading order."""
    destination = Path(output_path)
    destination.mkdir(parents=True, exist_ok=True)
    pages = list(gallery_pages(cases))
    paths = []
    style = {"font.family": "DejaVu Sans", "font.size": 9.5, "axes.labelcolor": INK,
             "axes.labelpad": 6, "pdf.fonttype": 42}
    metadata = {"Title": "Time-series defect gallery", "CreationDate": None, "ModDate": None}
    with mpl.rc_context(style), PdfPages(destination / "gallery.pdf", metadata=metadata) as pdf:
        for i, (filename, group, page) in enumerate(pages, start=1):
            fig = _page_figure(group, page, i, len(pages))
            path = destination / filename
            fig.savefig(path, dpi=150, format="png")
            pdf.savefig(fig)
            paths.append(path)
            fig.clear()
    return paths
