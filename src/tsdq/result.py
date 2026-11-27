"""Separate evaluation observations from private calibration and ground truth."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .core import InjectionNotFeasible


def portable(value):
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [portable(v) for v in value]
    if is_dataclass(value):
        return portable(asdict(value))
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT or isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump_json(path: Path, obj, *, lines=False):
    with path.open("w", encoding="utf-8", newline="\n") as out:
        if lines:
            for row in obj:
                out.write(json.dumps(portable(row), allow_nan=False, ensure_ascii=False) + "\n")
        else:
            json.dump(portable(obj), out, indent=2, allow_nan=False, ensure_ascii=False)
            out.write("\n")


def write_table(folder: Path, name: str, table: pd.DataFrame, fmt: str):
    path = folder / f"{name}.{fmt}"
    if fmt == "jsonl":
        dump_json(path, table.to_dict("records"), lines=True)
    elif fmt == "csv":
        table.to_csv(path, index=False)
    elif fmt == "parquet":
        table.to_parquet(path, index=False)
    else:
        raise ValueError("output.format must be jsonl, csv or parquet")


@dataclass
class InjectionResult:
    _clean: pd.DataFrame
    corrupted: pd.DataFrame
    episodes: list[dict]
    masks: pd.DataFrame
    manifest: dict
    failures: list[InjectionNotFeasible]
    windows: list[dict]
    training: pd.DataFrame | None = None

    @property
    def clean(self) -> pd.DataFrame:
        return self._clean.copy(deep=True)

    def observations(self, window_id: str | None = None) -> pd.DataFrame:
        """Only explicitly authorized columns, with no generated clean-row IDs."""
        columns = self.manifest["config"]["output"]["observation_columns"]
        if window_id is None:
            keep = ~self.corrupted.record_id.isin(self.manifest["training_record_ids"])
            rows = self.corrupted.loc[keep]
        else:
            window = next((w for w in self.windows if w["window_id"] == window_id), None)
            if window is None:
                raise KeyError(window_id)
            rows = self.corrupted.set_index("record_id", drop=False).loc[window["observation_record_ids"]]
        return rows.loc[:, columns].copy().reset_index(drop=True)

    def write(self, destination: str | Path) -> Path:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
            raise FileExistsError(f"Choose a new or empty output directory: {dest}")
        temp = Path(tempfile.mkdtemp(prefix=".tsdq-", dir=dest.parent))
        try:
            private = temp / "ground_truth"
            private.mkdir()
            fmt = self.manifest["config"]["output"]["format"]
            write_table(temp, "observations", self.observations(), fmt)
            for name, table in (("clean", self.clean), ("corrupted", self.corrupted), ("masks", self.masks)):
                write_table(private, name, table, fmt)
            if self.training is not None:
                write_table(private, "training", self.training, fmt)
            dump_json(private / "episodes.jsonl", self.episodes, lines=True)
            dump_json(private / "windows.jsonl", self.windows, lines=True)
            dump_json(private / "manifest.json", self.manifest)
            if dest.exists():
                dest.rmdir()
            temp.rename(dest)
        except BaseException:
            shutil.rmtree(temp)
            raise
        return dest
