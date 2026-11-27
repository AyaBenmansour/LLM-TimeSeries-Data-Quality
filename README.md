# LLM-Based Time-Series Data Quality Assessment

## Running the defect injection benchmark

The `tsdq` Python package injects nine families of defects into long or wide sensor tables. It fits healthy statistics per channel, generates episodes on the original time axes, and records every accepted change against a stable acquisition ID. The supplied regular and asynchronous datasets are small synthetic examples.

The illustrated gallery compares clean and corrupted measurements for all nine families and every subtype (36 cases, including all five Burst shapes, pattern-based missingness and timestamp defects on regular and irregular clocks). The shaded band marks the scheduled episode; markers show only readings whose final value, timestamp or presence actually changed. Each case prints its severity, scheduled-versus-changed counts, validation checks and actual parameters. [View the plots](examples/defect_gallery/gallery.pdf) · [Browse the cases and data](examples/defect_gallery/README.md)

```sh
python -m pip install -r requirements.txt                 # package + pytest + matplotlib
python -m pytest -q                                       # full test suite
python examples/make_defect_gallery.py                    # writes PNGs, PDF and before_after.csv to examples/defect_gallery/
python -m tsdq --input examples/regular.csv --config configs/example_regular.yaml --output results/regular-run
python -m tsdq --input examples/asynchronous.csv --config configs/example_async.json --output results/async-run
```

Use a new or empty output directory for each run. The regular example selects five evaluation windows with one clean, two single-defect, and two mixed windows. The asynchronous example removes actual records on two independent sensor clocks and applies signed timing errors. Both configurations override the same versioned preset. [Regular example](configs/example_regular.yaml) · [Asynchronous example](configs/example_async.json) · [Preset](configs/presets/pdf_v1.yaml)

```python
import pandas as pd
from tsdq import DefectInjector

clean = pd.read_csv("examples/regular.csv")
injector = DefectInjector("configs/example_regular.yaml", seed=47)
run = injector.inject(clean)
print(run.episodes[0]["channel_details"]["sensor"]["achieved"])
print(run.masks[["episode_id", "record_id", "clean_value", "final_value"]].head())
model_input = run.observations(run.windows[0]["window_id"])
```

### Using the injectors on a new dataset

Any long table with `timestamp`, `sensor_id` and `value` columns works (column names are configurable under `adapter`). Severity `s ∈ [0, 1]` maps every ranged parameter linearly, `θ(s) = θ_min + s(θ_max − θ_min)`. Override any `{min, max}` range from the [preset](configs/presets/pdf_v1.yaml) without touching the injectors; defaults are keyed by subtype (e.g. `global`, `contextual`, `spike`, `fixed_shift`). Unknown keys are rejected with an error.

```python
import numpy as np, pandas as pd
from tsdq import DefectInjector

t = np.arange(400)
clean = pd.concat([pd.DataFrame({"timestamp": t, "sensor_id": name, "value": 20 + np.sin(t / 15 + k)})
                   for k, name in enumerate(["temp_1", "temp_2"])], ignore_index=True)

config = {
    "reference": {"train_fraction": 0.3},                                   # healthy calibration split
    "defects": {"global": {"params": {"amplitude_multiplier": {"min": 2, "max": 5}}}},  # Burst K range
}
plan = [{
    "family": "burst", "subtype": "global", "severity": 0.6, "duration": 6,
    "params": {"shape": "half_sine", "end_mode": "recovery"},           # or end_mode: non_recovery
    "schedule": {"occurrence": {"mode": "bernoulli_start", "q": 0.01},  # or {mode: fixed_count, episode_count: K}
                 "channels": "all", "min_separation": 10, "max_attempts": 50},
}]
run = DefectInjector(config, seed=7).inject(clean, plan=plan)
print(len(run.episodes), run.failures)        # achieved episode count and any infeasible requests
run.write("results/my-run")                   # observations + ground_truth/{clean,corrupted,masks,episodes,manifest}
```

**Ground truth.** `run.masks` (and `ground_truth/masks.jsonl`) contains one row per record whose final value or timestamp differs from the clean reference, or that became missing (`dropped=True` rows keep their original `record_id` and `true_timestamp`). Unchanged points inside a scheduled interval — a ramp's zero-error start, readings below a saturation cap, a stuck value crossing the held level — are not labelled. Each episode keeps both `support_record_ids` (scheduled interval, for provenance) and `record_ids` (actually changed, for point-level scoring), plus per-channel `parameters`, `achieved` intensities and named `checks`. Rejected candidates are listed in `manifest["rejections"]` with a reason.

Use `plan=[]` to generate a clean control, or pass a list of episode requests to `inject` to replace the configured plan. By default, the first 30% of each channel is reserved for calibration. The split is configurable. Supply a separate clean acquisition with `training_df=` in Python or `--training` on the CLI to make the entire target stream eligible. The package keeps hidden truth under `ground_truth/` and exports the held-out observations separately. The CLI returns exit code 2 when a valid request exhausts its placement attempts, and saves the reasons with the run. [API and output guide](docs/injection.md) · [Sample output](examples/sample_run/ground_truth/manifest.json)

The source decisions, calibration assumptions and intentionally deferred features are documented alongside the relevant formulas. [Specification decisions](docs/spec_decisions.md) · [Value defects and Burst](docs/value_defects.md) · [Data, missingness and time defects](docs/data_time.md) · [Scheduler and evaluation windows](docs/scheduling.md)


## Overview

Data quality is a fundamental requirement for reliable time-series analytics. However, real-world sensor and telemetry data frequently suffer from missing values, noise, outliers, communication failures, and temporal or concept drift.

This research project investigates the capabilities of Large Language Models (LLMs) in detecting, localizing, quantifying, and interpreting data quality defects in time-series data.

The main objective is to develop a systematic and reproducible framework for benchmarking LLM-based approaches to time-series data quality assessment, with a potential extension toward targeted data quality mitigation.

## Research Objectives

### 1. Benchmark Curation and Controlled Defect Injection

- Review existing approaches to LLM-based time-series analytics and data quality assessment.
- Select and preprocess representative real-world time-series datasets.
- Develop controlled data quality defect injection pipelines.
- Characterize and quantify different types and severities of data quality defects.

### 2. LLM-Based Quality Detection and Rating

- Investigate the capabilities of open-source LLMs in identifying and localizing data quality defects.
- Evaluate their ability to estimate defect severity.
- Generate structured and interpretable data quality ratings.
- Assess diagnostic reliability and hallucination rates.
- Compare LLM-based approaches against traditional baseline methods.

### 3. Data Quality Mitigation (Optional Extension)

- Investigate targeted correction and imputation strategies.
- Explore the mitigation of selected data quality defects.
- Evaluate improvements in data quality after correction.

## Proposed Framework

The proposed methodology consists of the following stages:

1. Real-world time-series dataset collection and preprocessing.
2. Controlled defect injection and ground-truth generation.
3. LLM-based data quality assessment.
4. Defect detection, localization, and severity estimation.
5. Structured quality rating and diagnostic interpretation.
6. Performance evaluation and comparison against baseline methods.
7. Targeted quality mitigation (optional).

## Data Quality Dimensions

| Defect | Description |
|---|---|
| Missing Values | Missing observations within a time series. |
| Sensor Noise | Unwanted perturbations affecting recorded measurements. |
| Outliers | Abnormal observations deviating from expected patterns. |
| Communication Failures | Data disruptions caused by transmission or connectivity issues. |
| Temporal Drift | Changes in the temporal characteristics of the data. |
| Concept Drift | Changes in the underlying data-generating relationships. |

## Evaluation

The proposed framework will be evaluated across several dimensions:

- Defect detection accuracy.
- Anomaly localization performance.
- Defect severity estimation.
- Data quality rating accuracy.
- Hallucination rates.
- Semantic diagnostic reliability.
- Comparative performance against traditional methods.

## Research Roadmap

| Phase | Period | Focus |
|---|---|---|
| Phase 1 | October – November | Literature review, dataset curation, and baseline preparation. |
| Phase 2 | December – February | Defect injection, LLM evaluation, and framework development. |
| Phase 3 | March – April | Optional mitigation, comparative experiments, and framework finalization. |
| Phase 4 | Late April – Late May | Research manuscript, thesis writing, and reproducibility. |

The roadmap remains flexible and may evolve according to experimental findings and research progress.

## Expected Contributions

- A systematic benchmark for LLM-based time-series data quality assessment.
- Reproducible data quality defect generation pipelines.
- A structured framework for defect detection, localization, and quality rating.
- An empirical investigation of LLM capabilities and limitations in time-series data quality assessment.
- A potential targeted data quality mitigation approach.
- A documented open-source research artifact.



## Project Status

This repository accompanies an ongoing Master's thesis research project. The implementation, experimental results, datasets, and documentation will be progressively updated throughout the research.

## Keywords

Large Language Models · Time-Series Analytics · Data Quality Assessment · Anomaly Detection · Data-Centric AI · Time-Series Imputation · Edge Sensor Telemetry
