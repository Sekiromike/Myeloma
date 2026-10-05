"""Scientific invariants of the MM forecast model.  Run: pytest Myeloma/tests"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lot_model as lm  # noqa: E402
import model_metrics as mm  # noqa: E402
from adoption import AdoptionEngine  # noqa: E402
from scientific_utils import WeibullParams  # noqa: E402


@pytest.fixture(scope='module')
def inputs():
    return lm.load_inputs()


@pytest.fixture(scope='module')
def base(inputs):
    return lm.scenario_from_params(inputs.params)


@pytest.fixture(scope='module')
def result(inputs, base):
    return lm.run_model(inputs, base)


def test_weibull_reproduces_median_and_landmark():
    w = WeibullParams.from_median(41.0, 1.3)
    assert w.survival_prob(41.0) == pytest.approx(0.5)
    w = WeibullParams.from_landmark(48, 0.843, 1.3)
    assert w.survival_prob(48) == pytest.approx(0.843)


def test_every_regimen_reproduces_its_published_efficacy(inputs):
    for r in inputs.regimens.values():
        if r.pfs_landmark:
            s = r.weibull.survival_prob(r.pfs_landmark['month'])
            assert s == pytest.approx(r.pfs_landmark['survival']), r.key
        else:
            assert r.weibull.survival_prob(r.pfs_median) == pytest.approx(0.5), r.key


def test_shares_sum_to_one(inputs):
    eng = AdoptionEngine(inputs.regimens)
    years = np.arange(1985, 2036, 0.25)
    for line, elig in [('1L', 'TE'), ('1L', 'TI'), ('2L', 'Both'), ('3L', 'Both'), ('4L+', 'Both')]:
        _, s = eng.share_matrix(years, line, elig)
        assert np.allclose(s.sum(axis=1), 1.0), (line, elig)


def test_no_share_before_uptake_or_after_withdrawal(inputs):
    eng = AdoptionEngine(inputs.regimens)
    years = np.arange(1985, 2036, 1 / 12)
    for line, elig in [('1L', 'TE'), ('1L', 'TI'), ('2L', 'Both'), ('3L', 'Both'), ('4L+', 'Both')]:
        keys, s = eng.share_matrix(years, line, elig)
        for i, k in enumerate(keys):
            r = inputs.regimens[k]
            if r.uptake_year > 1991:
                assert s[years < r.uptake_year, i].max(initial=0) == 0, k
    keys, s = eng.share_matrix(years, '4L+', 'Both')
    i = keys.index('4L+_Belamaf')
    assert s[years >= 2022.9, i].max() == 0


def test_cohort_conserves_patients(inputs, base):
    c = lm.run_cohort(inputs, base, 2016)
    accounted = sum(c[f'{k}_{l}'] for k in ['death', 'stop', 'still'] for l in mm.LINES)
    assert accounted + c['in_transit'] == pytest.approx(1000.0, rel=1e-6)
    assert c['in_transit'] < 10


def test_line_totals_equal_sum_of_regimens(inputs, result):
    m = result.monthly
    for line in mm.LINES:
        keys = [k for k, r in inputs.regimens.items() if r.line == line]
        assert np.allclose(m[keys].sum(axis=1), m[f'Total_{line}'])


def test_incidence_observed_years_untouched(inputs, result):
    inc = result.annual_incidence.set_index('Year')
    obs = inputs.uscs.groupby('Year')['Count'].sum()
    assert np.allclose(inc.loc[obs.index, 'Cases'], obs.values)
    assert (inc.loc[2023:, 'Cases'].diff().dropna() > 0).all()   # ageing population


def test_acs_anchor_hits_benchmark(inputs, base):
    sc = dict(base, anchor_to_acs=True)
    inc = lm.build_incidence(inputs, sc).set_index('Year')
    assert inc.loc[sc['acs_year'], 'Cases'] == pytest.approx(sc['acs_cases'])


def test_benchmarks_within_plausible_range(inputs, base, result):
    annual = mm.annualize(result.monthly).set_index('Year')
    ratio = annual.loc[2022, 'On_Therapy'] / mm.BENCHMARKS['seer_prevalence_2022']
    assert 0.5 < ratio < 0.85
    c = lm.run_cohort(inputs, base, 2016)
    assert 0.45 < c['start_2L'] / 1000 < 0.80
