import numpy as np
import pandas as pd
import pytest

from tsdq import DefectInjector, map_severity
from tsdq.config import Request, preset


def test_severity_mapping_endpoints_reversal_log_and_discrete():
    for s, descending, log_value, choice in [(0, .75, 1e-4, 1), (.5, .5, 1e-3, 3), (1, .25, 1e-2, 5)]:
        assert map_severity(s, [.75, .25]) == descending
        assert map_severity(s, {"min": 1e-4, "max": 1e-2, "mode": "log"}) == pytest.approx(log_value)
        assert map_severity(s, {"choices": [1, 3, 5]}) == choice
        assert map_severity(s, {"value": 9}) == 9


@pytest.mark.parametrize("s", [False, -0.01, 1.01, np.nan, np.inf, "0.5"])
def test_invalid_severity(s):
    with pytest.raises(ValueError):
        map_severity(s, [1, 2])


def test_invalid_endpoint_even_when_selected_duration_or_threshold_is_valid():
    for extra in ({"duration": {"min": -1, "max": 9}},
                  {"reference": {"rate_quantile": {"min": .8, "max": 1.01}}}):
        with pytest.raises(ValueError):
            Request.resolve(preset(), {"family": "bias", **extra})


def test_prior_separation_survives_later_plan_with_smaller_guard():
    clean = pd.DataFrame({"timestamp": range(90), "value": np.cos(np.arange(90))})
    plan = [{"family": "bias", "duration_unit": "samples", "duration": 1,
             "schedule": {"onset_index": onset, "min_separation": gap, "max_attempts": 1}}
            for onset, gap in ((40, 3), (43, 0), (44, 0))]
    result = DefectInjector().inject(clean, plan)
    assert result.masks.record_id.tolist() == [40, 44]
    assert [failure.reason for failure in result.failures] == ["overlap_or_separation"]


def test_bernoulli_diagnostics_count_trials_separately_from_accepted_mask():
    clean = pd.DataFrame({"timestamp": range(20), "value": range(20)})
    plan = [{"family": "missing_values", "duration": 2, "schedule": {
        "max_attempts": 1, "occurrence": {"mode": "bernoulli_start", "q": 1}}}]
    r = DefectInjector().inject(clean, plan)
    tally = r.manifest["schedule_results"][0]
    assert tally["bernoulli_trials"] == tally["expected_starts"] == 13
    assert 0 < tally["accepted_occurrences"] < tally["opportunities"]
    assert r.manifest["missing"]["record_count"] == 2*tally["accepted_occurrences"]
