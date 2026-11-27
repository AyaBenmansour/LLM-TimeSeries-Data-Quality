"""Generate the illustrated comparison and its auditable before/after data."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gallery_cases import build_cases
from plot_gallery import gallery_pages, render_gallery


def compare_records(case, result) -> pd.DataFrame:
    """Independently diff clean and recorded acquisitions, then verify the labels."""
    if result.failures:
        raise ValueError(f"{case.key}: infeasible injection: {result.failures}")
    columns = ["sensor_id", "record_id", "timestamp", "value"]
    comparison = result.clean[columns].merge(
        result.corrupted[columns], on=["sensor_id", "record_id"], how="outer",
        suffixes=("_before", "_after"), indicator=True, validate="one_to_one",
    )
    both = comparison["_merge"] == "both"
    category = pd.Series("unchanged", index=comparison.index)
    before = comparison["value_before"]
    after = comparison["value_after"]
    same_value = before.eq(after) | (before.isna() & after.isna())
    category.loc[both & ~same_value] = "value_changed"
    category.loc[both & before.notna() & after.isna()] = "explicit_nan"
    category.loc[both & comparison.timestamp_before.ne(comparison.timestamp_after)] = "time_changed"
    category.loc[comparison["_merge"] == "left_only"] = "record_removed"
    category.loc[comparison["_merge"] == "right_only"] = "record_added"
    changed = set(comparison.loc[category.ne("unchanged"), "record_id"])
    labels = {rid: episode for episode in result.episodes for rid in episode["record_ids"]}
    if changed != set(labels) or changed != set(result.masks.record_id):
        raise ValueError(f"{case.key}: mismatched before/after differences, masks or episode labels")

    comparison.insert(0, "case", case.key)
    comparison["change"] = category
    for field in ("family", "subtype", "episode_id"):
        comparison[field] = comparison.record_id.map({rid: ep[field] for rid, ep in labels.items()})
    # An added record has no before row of its own. Record the ID it copied.
    sources = result.masks.set_index("record_id").source_record_id
    comparison["copied_from"] = comparison.record_id.map(sources).where(category == "record_added")
    comparison["plot_order"] = comparison.timestamp_before.combine_first(comparison.timestamp_after)
    return comparison.sort_values(["sensor_id", "plot_order"], kind="stable").drop(columns=["_merge", "plot_order"])


def write_index(cases, comparisons, images, output):
    lines = [
        "# Time-series defect gallery", "",
        "Open the [illustrated comparison](gallery.pdf) to see the same synthetic measurements "
        "before and after each injection. Each case starts with a fresh clean stream. "
        "Missing Values cases use scheduled gaps.", "",
        "The shaded band is the scheduled episode. Markers show only the readings whose final value, "
        "timestamp or presence differs from the clean reference. The grey line under each title lists "
        "the severity, scheduled versus changed counts, validation checks passed and the actual injected parameters.", "",
        "The [before-and-after CSV](before_after.csv) aligns the two streams by acquisition ID. "
        "Filter by `case` to inspect an example and by `change` to find its actual effects. "
        "Unchanged records have empty defect labels, including unchanged ramp endpoints and samples inside clipping limits.", "",
        "An `explicit_nan` retains its timestamp. A `record_removed` has no after row. "
        "A `time_changed` keeps its measured value. A `record_added` has its own ID and a `copied_from` source.", "",
    ]
    for (case, run), frame in zip(cases, comparisons, strict=True):
        if case.key == "drift_quadratic":
            support = run.episodes[0]["support_record_ids"]
            count = len(run.episodes[0]["record_ids"])
            lines.extend(["## Where the Drift label starts and ends", "",
                          f"The quadratic ramp and recovery span {len(support)} readings. "
                          f"The before/after comparison identifies {count} changed readings. "
                          "These four rows show the two boundaries.", "",
                          "| Time (s) | Before (°C) | After (°C) | Ground truth |",
                          "| ---: | ---: | ---: | --- |"])
            boundary = frame.set_index("record_id").loc[[support[0], support[1], support[-2], support[-1]]]
            for row in boundary.itertuples():
                label = "Clean" if row.change == "unchanged" else "Drift"
                lines.append(f"| {row.timestamp_before:g} | {row.value_before:.5f} | {row.value_after:.5f} | {label} |")
            break
    lines.extend(["", "## Figures", ""])
    case_images = {}
    for page, (path, (filename, group, entries)) in enumerate(zip(images, gallery_pages(cases), strict=True), 1):
        if path.name != filename:
            raise ValueError(f"Figure index mismatch for {filename}")
        names = ", ".join(case.title for case, _ in entries)
        lines.append(f"- [Page {page} · {group}]({filename}) — {names}")
        case_images.update({case.key: filename for case, _ in entries})
    lines.extend(["", "## Cases", "", "| Case | Family | Changed records | What to look for |",
                  "| --- | --- | ---: | --- |"])
    for (case, _), frame in zip(cases, comparisons):
        count = int(frame.change.ne("unchanged").sum())
        lines.append(f"| [{case.key}]({case_images[case.key]}) | {case.family} | {count} | {case.caption} |")
    lines.extend(["", "## Reproduce", "",
                  "From the repository root, run the following commands.", "",
                  "```sh", 'python -m pip install -e ".[gallery]"',
                  "python examples/make_defect_gallery.py --output examples/defect_gallery", "```", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/defect_gallery"))
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    cases = list(build_cases())
    comparisons = [compare_records(case, result) for case, result in cases]
    images = render_gallery(cases, output)
    pd.concat(comparisons, ignore_index=True).to_csv(output / "before_after.csv", index=False)
    write_index(cases, comparisons, images, output)
    print(f"{len(cases)} verified cases across {len(images)} pages. {output / 'gallery.pdf'}")


if __name__ == "__main__":
    main()
