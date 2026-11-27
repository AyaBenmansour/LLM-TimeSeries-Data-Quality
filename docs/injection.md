# Injector API and experiment configuration

## Components

| Module | Responsibility |
| --- | --- |
| `data` | Canonical records, stable source IDs, datetime precision, physical time units and input validation. |
| `reference` | Fit global distributions and residuals on eligible clean training samples. Compute clean contextual neighborhoods for a proposed location. |
| `config` | Versioned YAML/JSON overlays, typed resolved requests, subtype aliases and independent severity maps. |
| `scheduler` | Sample joint opportunities from the selected channels and project them onto each sensor's original times. |
| `values`, `availability` | Generate per-channel candidates and optional follow-on phases. |
| `value_checks`, `core`, `composition` | Check the mathematical definitions, encoded time coordinates, eligibility, context and overlaps. Commit each accepted occurrence atomically across phases and channels. |
| `engine` | Own the clean reference, RNG streams, fit cache, retries, exact masks and provenance. |
| `windows`, `result`, `cli` | Slice the completed stream, prepare authorized observations and serialize a batch run. |

## Selecting an experiment

Configuration overlays are deep merged with the distributed preset. A scalar overrides a parameter directly. Ordered `{choices: [...]}` uses the nearest discrete severity level. `{min: a, max: b}` interpolates linearly, including decreasing bounds. Specify `mode: log` for positive log-space interpolation. Endpoints are included. A magnitude is expressed in the units of its own transform, and extent is measured separately. [Versioned preset](../configs/presets/pdf_v1.yaml)

Set `defects.<key>.params` for a run-wide default and `plan[].params` for an individual request. Keys are subtype names for Point Outlier, Burst, Flat Line and Timestamp. Other families use the family name. For example, `defects.spike.params.amplitude_multiplier.min` and `.max` control the two Spike endpoints. The same keys under `glitch` control only Glitches. `defects.contextual.reference.local_neighbors` selects a local window for contextual Bursts. Threshold multipliers, the healthy rate quantile and neighborhood lengths accept severity ranges. The full preset is an executable inventory of the defaults. [Value formulas and options](value_defects.md)

Plan entries require a `family` and, for families that define one, a `subtype`. An entry can override `severity`, `severity_magnitude`, `severity_duration`, `duration`, `duration_unit`, `params`, `reference`, `tolerance`, and `schedule`. Duration inherits the variant's unit. Set `duration_unit: samples` explicitly when changing a reference-window fraction into a record count. A positive sample duration rounds to the nearest integer with halves upward. `reference_fraction` multiplies the duration by the fixed `reference_window` in samples. `seconds` selects an elapsed-time interval. The scientific definition still constrains the result, for example, exactly one sample for a Spike and at least two meaningfully changed adjacent samples for a Burst.

The default run uses `plan: []`. Use the Python constructor with an overlay mapping, a YAML/JSON filename or an `ExperimentConfig`. Each `inject` call has fresh RNG and calibration state. A pipeline can inspect `result.failures` for typed `InjectionNotFeasible` outcomes and `manifest["rejections"]` for individual attempts, including rejection measurements and replay seeds. Invalid configuration raises `ValueError`. Finite retry budgets can leave some requested occurrences unfilled. Successful independent occurrences remain available.

## Calibration and composition

Use the per-channel chronological prefix, explicit `reference.train_record_ids`, or a separate `training_df` with the same adapter schema. Global bounds, gradients and scale never fit held-out observations. Local contextual comparisons deliberately use surrounding **clean** target measurements, excluding the proposed episode. Separate training acquisitions must be chosen independently by the researcher. Stable record IDs are unique within a table and may repeat in a separate acquisition. Fit subsets are identified in the private manifest. Univariate, wide and asynchronous input conventions are described in the adapter guide. [Data and time](data_time.md)

Episode starts and placement eligibility are evaluated in original coordinates. The default stage order is `[value, missing, timestamp]`. Set `composition.order` to another permutation and list permitted pairs such as `allow_overlap: [[value, missing]]` to model deliberate cross-stage interactions. Within-stage overlaps are rejected. A later missing episode may hide an earlier corrupted value. Both provenance rows survive and the final visibility is recorded. A previously deleted record cannot receive another transformation. Neighboring observations needed for an accepted Burst, Glitch or Stuck remain protected, including the Burst's recovery boundary where observed. A multi-channel proposal and its dependent phases succeed or fail together.

## Reading a run

| Path | Contents |
| --- | --- |
| `observations.{jsonl,csv,parquet}` | Authorized held-out columns only. The default projection is recorded timestamp, sensor ID and value. |
| `ground_truth/clean.*`, `corrupted.*` | Canonical clean and final full-stream records with stable IDs. Copies receive new IDs. Deleted rows remain traceable through the clean table. |
| `ground_truth/training.*` | External clean calibration table, when supplied. |
| `ground_truth/episodes.jsonl` | Absolute support and changed IDs, channels, aliases, subtype, separate magnitude/duration settings, resolved parameters, contextual identifiability, achieved effects and passed checks. |
| `ground_truth/masks.*` | One row per actual effect per episode. Includes true and recorded times, original and pre-stage values, the effect's value, final observation and deletion flags. A copy points back to its source ID. |
| `ground_truth/windows.jsonl` | Private membership and visibility maps, cut/edge flags, mixture class and eligible clean controls. |
| `ground_truth/manifest.json` | Seed, merged config, resolved requests, version and source digest, data fingerprints, attempt outcomes, missing extent and window count. |

For point-level localization or plotting, use `episodes[].record_ids` or the mask rows. Episode `onset` and `end` give the true acquisition times of the first and last actual effect. Each channel's `start_index`, `end_index` and `elapsed_seconds` use the same endpoints. An unchanged starting anchor, a fully recovered endpoint, and intervening clean samples receive no defect label. For example, if clipping changes samples 7, 8 and 12, those three IDs are the complete defect mask even though its endpoints are 7 and 12. A timestamp change is an effect on time, a dropped row is an effect on availability, and a duplicate labels the new record ID with a link to its original source.

The private episode fields `support_record_ids`, `scheduled_onset`, `scheduled_end`, and per-channel `scheduled_support` document where the generator ran. They allow an experiment to reproduce the full ramp, freeze or clipping interval. Overlap guards still reserve that interval. A window containing only unchanged scheduled measurements has no episode entry for that generator.

Use `run.observations(window_id)` for one window's authorized inputs in recorded-time order. The caller supplies authorized contextual metadata to its evaluator. Setting a context-visibility flag records an experimental assumption. The exporter writes only columns selected by `output.observation_columns`. Keep the private directory outside the evaluator's input path. [Window scoring rules](scheduling.md)

JSONL preserves JSON scalar identifiers and serializes missing observations as JSON null. Datetimes serialize as ISO timestamps with fractional seconds and UTC offsets. In CSV, IDs are read as text when any entry has a noncanonical integer spelling (including leading zeros). A column consisting entirely of canonical integers is read as integers. Use JSONL for experiments needing an exact distinction between the text `"7"` and integer `7`. Optional Parquet I/O uses `python -m pip install ".[parquet]"`.

Replay by passing `manifest["config"]` and the same input and training tables into `DefectInjector`. For a saved canonical clean table derived from a custom or wide schema, use the canonical long adapter column names while retaining the original physical time unit and timezone. Compare masks and episodes as well as the manifest's source digest. Seeded numerical draws are tied to the recorded library versions.
