# Scheduling and evaluation windows

The shared scheduler uses the same occurrence, recurrence, placement and spatial controls for every defect family. It works in the original observation order of each selected channel. The engine validates a joint placement and commits every channel together. This follows the [compact shared framework](../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/specs/FYP_Defect_Injection_Compact_Framework.md) and the [implementation brief, §§4–7](../Codex_FYP_Injection_Spec_Bundle/Codex_FYP_Injection_Spec_Bundle/IMPLEMENTATION_TASK.md).

## Scheduling

Each plan entry may override the top-level `schedule`. Index coordinates are zero-based and include the training prefix when calibration and injection share a table. Elapsed-second coordinates use the common origin of the input recording. Examples below assume the site selection includes eligible evaluation observations.

```yaml
family: missing_values
duration: 3
schedule:
  occurrence: {mode: fixed_count, episode_count: 4}
  recurrence: periodic
  period: 15
  recurrence_unit: samples
  onset_index: 40
  channels: [temperature]
  min_separation: 1
```

| Field | Meaning |
|---|---|
| `occurrence` | Choose `{mode: fixed_count, episode_count: K}` or `{mode: bernoulli_start, q: q}`. Both allow zero. |
| `recurrence` | `random`, `periodic` or `clustered`. Random fixed-count retries draw other eligible starts. Fixed period slots and selected Bernoulli onsets stay in place during transform retries. |
| `channels`, `channel_count` | Select a named pool or `all`. With no count an explicit pool targets every listed channel. With neither field the scheduler picks one channel. A count draws that many distinct channels from the pool. |
| `channel_selection`, `anchor_channel` | Choose `random` without replacement or `first` in input order. The group stays fixed across recurrences of one plan entry. The first selected channel supplies the scheduling axis unless an anchor channel is specified. Separate plan entries can select other groups. |
| `onset_index`, `onset_seconds` | Choose one coordinate for an exact proposal or for the first periodic slot. Seconds may fall between readings. A sample-periodic origin must occur on the anchor grid. |
| `allowed_indices`, `allowed_seconds` | Each accepts a half-open pair `[lower, upper)`. Index limits apply to each channel's own positions. The entire candidate support must consist of eligible records inside both limits. |
| `persistence` | `temporary` uses the requested duration. `unresolved` extends from onset through the last recorded observation. `whole_channel` covers the complete selected channel, requiring its full data to be eligible, typically via separate training data. |
| `period`, `recurrence_unit` | Distance between periodic slots in anchor observations or seconds. When onset is omitted, the first eligible anchor observation establishes phase. Ineligible fixed-count slots fail individually and preserve the later slots' planned times. |
| `cluster_size`, `cluster_width` | Each cluster takes up to the configured number of distinct starts within a band this wide in `recurrence_unit`. Populated bands are selected at random with an intervening band between clusters. The last fixed-count cluster can be shorter. These band boundaries are a configurable experimental placement convention. |
| `min_separation`, `max_attempts` | Minimum untouched observations between conflicting episodes on each channel, and maximum candidate draws per opportunity. A cluster of blocks uses a positive separation to keep blocks visibly distinct. |

A sample duration selects the specified number of consecutive records starting at the first reading on or after the proposed onset. Sample-based supports on multiple channels must intersect in time, taking the next clean reading as the exclusive boundary. A seconds duration selects all observations in the common half-open interval `[start, start + duration)` and requires coverage through its end on each channel. This can yield different sample counts per channel. Real-time periodic gaps use `duration_unit: seconds` in the plan and `recurrence_unit: seconds` in the schedule. The period multiplied by a chosen duty fraction gives the gap width in the same units.

For Bernoulli generation, eligible random opportunities are the valid anchor starts satisfying duration, scope and eligibility. Periodicity restricts trials to the grid. Clustered recurrence draws up to `cluster_size` opportunities without replacement per chosen band before applying independent Bernoulli trials. Conditional on a fixed set of N trials, the proposed count has binomial expectation Nq. Conflicts and semantic validation can reduce the accepted count. Skipped periodic trials produce a thinned grid. A fully regular grid uses fixed count or q=1.

For uniform sample-count episodes, `{mode: fixed_count, target_records: 12}` requests enough complete episodes to cover twelve records. `target_fraction` uses the total clean record count across the input channels, rounded to the nearest whole record with halves upward. The target must be divisible by duration times the selected channel count. Choose non-overlapping Missing Values episodes to build an exact missing count or fraction. A duration of one and at least one untouched record between episodes produces isolated gaps. A finite retry budget or a conflict can still leave a target unfulfilled. The achieved mask and failure records document this. Per-episode sample-count targets can be expressed directly with `duration`.

## Private window metadata

```yaml
windows:
  lengths: [32, 64]
  stride: 8
  scoring: families
  scoring_basis: ground_truth
  include_partial: false
  target_proportions: {clean: 0.2, single: 0.4, mixed: 0.4}
  count: 20
  natural_change_record_ids: []
```

Window length and stride count distinct clean timestamps in the evaluation timeline. Records sharing a timestamp stay together. Each window includes its first instant and excludes the next window boundary. At the recording end the last actual clean instant is inclusive. No extrapolated interval or grid is created. With `include_partial: true`, trailing shorter windows retain their actual lengths.

Each private window stores its absolute bounds, clean and observed member IDs, intersecting effects, changed observations visible at their recorded times, dropped records, and timing errors moved into or out of the interval. A moved-in timestamp can contribute observed evidence even if its acquisition lies outside the window. Dropped records supply ground truth with no observed record. Scientific identifiability also depends on the experiment's expected-reporting rule, metadata and reference visibility.

Window labels use exact changes. A drift's zero-error anchor, a recovered final value and an in-range sample between clipped peaks each remain eligible as clean context. A window containing only such measurements has no entry for that episode. `affected_record_ids` lists effects whose source falls in the window. The window episode's `support_record_ids` lists their source acquisitions, including the original acquisition for an added copy. `episode_coverage` is the fraction of all episode effects whose source falls in the window. `window_coverage` is the fraction of the window's clean acquisitions associated with those effects. The cut, edge and interior flags use the affected acquisitions. Context counts include unaffected observed measurements per targeted channel before and after the evidence. Full generator support is available separately in the private stream-level episode record.

Each window reports both `ground_truth_class` and `visible_class`. Choose `scoring_basis: ground_truth` to select the generating class for scoring or stratification. Choose `visible` to count episodes for which at least one targeted channel meets its recorded identifiability condition in this particular window. A Bias, Drift or Scaling episode needs a visible onset with preceding clean observations or an authorized independent reference. A dropped record can count under visible scoring when the experiment supplies the expected observation process or timing reference. A duplicate requires its source and extra copy to be observable together. Unseen, obscured or unidentifiable effects remain in the private episode labels. A visible-class clean window containing a known defect is excluded from genuine negative-control annotations.

`scoring: episodes` counts distinct accepted episodes that modified records in the interval or moved evidence into it. `families` counts their distinct top-level families. Counts zero, one and two or more determine the clean, single and mixed classes. Restricting counts to actual effects keeps a zero-offset ramp point alone from contributing a positive label. Unmodified source changes remain eligible clean controls. Supply researcher-reviewed `natural_change_record_ids` to mark clean windows containing known genuine changes.

Without sampling targets, the slicer returns all windows in length/start order. With a requested count, it samples windows without replacement. Proportions use largest-remainder integer quotas with ties settled in clean/single/mixed order. When count is omitted, the limiting class sets the sample size. An unavailable stratum or insufficient population for an explicit quota produces a diagnostic. Overlapping windows from the same stream are dependent observations and should remain in the same experiment split.

Model inputs consist of authorized observation columns. Window labels, clean coordinates, class quotas and record identities belong in private evaluation artifacts.
