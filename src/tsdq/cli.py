import argparse
import json
from pathlib import Path
import re

import pandas as pd

from .engine import DefectInjector
from .config import ExperimentConfig


def read_table(path, adapter=None):
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        with Path(path).open(encoding="utf-8-sig") as source:
            return pd.DataFrame.from_records(json.loads(line) for line in source if line.strip())
    if suffix == ".csv":
        adapter = adapter or {}
        identity = adapter.get("id_col", "record_id")
        sensor = adapter.get("sensor_col", "sensor_id")
        columns = pd.read_csv(path, nrows=0).columns
        numeric_missing = ["", "NaN", "nan", "NA", "N/A", "null", "NULL", "<NA>"]
        ids = {identity, sensor, "source_record_id"}
        missing = {col: numeric_missing for col in columns if col not in ids}
        lexemes = {col: str for col in columns if col in ids}
        precision = {} if tuple(map(int, pd.__version__.split(".")[:2])) >= (3, 1) else {"float_precision": "round_trip"}
        frame = pd.read_csv(path, keep_default_na=False, na_values=missing, converters=lexemes, **precision)
        if identity in frame and all(re.fullmatch(r"0|-?[1-9][0-9]*", s) for s in frame[identity]):
            frame[identity] = [int(s) for s in frame[identity]]
        return frame
    raise ValueError("input extension must be .csv, .jsonl or .parquet")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inject configured episodes into a clean sensor table.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True, help="YAML or JSON overlay on the versioned preset")
    parser.add_argument("--output", required=True, help="new or empty run directory")
    parser.add_argument("--training", help="separate, eligible clean calibration data in the same schema")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    config = ExperimentConfig.load(args.config)
    adapter = config.settings["adapter"]
    result = DefectInjector(config, seed=args.seed).inject(
        read_table(args.input, adapter), training_df=read_table(args.training, adapter) if args.training else None)
    result.write(args.output)
    print(f"{len(result.episodes)} accepted phases, {len(result.failures)} infeasible opportunities. {args.output}")
    return 2 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
