"""
Derived metrics for the MM Disease Model Explorer.

All functions take model outputs (ModelResult.monthly / .starts) and the regimen
catalogue, so every number on the dashboard is traceable to the simulation.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

LINES = ['1L', '2L', '3L', '4L+']
LINE_NAMES = {'1L': '1st line', '2L': '2nd line', '3L': '3rd line', '4L+': '4th line+'}

# External benchmarks (verified October 2026)
BENCHMARKS = {
    'acs_2026_cases': 36000,         # Siegel RL et al. CA Cancer J Clin 2026
    'acs_2026_deaths': 10850,
    'seer_prevalence_2022': 192144,  # SEER Cancer Stat Facts: Myeloma
    'seer_5yr_rel_survival': 62.4,   # SEER 21, 2015–2021
    'seer_median_age_dx': 69,
    # Fonseca R et al. BMC Cancer 2020, doi:10.1186/s12885-020-07503-y (US claims; censored)
    'rwe_reach_2l_te': 0.79, 'rwe_reach_2l_nte': 0.43,
    'rwe_reach_3l_given_2l_te': 0.69, 'rwe_reach_3l_given_2l_nte': 0.55,
}


# ── Aggregation ──────────────────────────────────────────────────
def annualize(monthly: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year table: flows are summed, patient stocks are averaged."""
    df = monthly.copy()
    df['Year'] = df['Date'].dt.year
    flow_prefix = ('Incidence', 'Untreated', 'New_Starts_1L', 'Starts_', 'Progressions_',
                   'Deaths_', 'No_Next_Line_')
    flows = [c for c in df.columns if c.startswith(flow_prefix)]
    stocks = [c for c in df.columns if c not in flows + ['Date', 'Year']]
    out = pd.concat([df.groupby('Year')[flows].sum(), df.groupby('Year')[stocks].mean()], axis=1)
    out['On_Therapy'] = out[[f'Total_{l}' for l in LINES]].sum(axis=1)
    return out.reset_index()


def annual_starts(starts: pd.DataFrame) -> pd.DataFrame:
    df = starts.copy()
    df['Year'] = df['Date'].dt.year
    return df.drop(columns='Date').groupby('Year').sum().reset_index()


def line_occupancy_long(monthly: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for l in LINES:
        frames.append(pd.DataFrame({'Date': monthly['Date'], 'Line': LINE_NAMES[l],
                                    'Patients': monthly[f'Total_{l}']}))
    return pd.concat(frames, ignore_index=True)


# ── Regimen mix ──────────────────────────────────────────────────
def regimen_share(frame: pd.DataFrame, line: str, regimens: dict,
                  max_series: int = 7, min_peak: float = 0.03) -> pd.DataFrame:
    """Share of a line's patients by regimen; small/legacy regimens fold to 'Other'.

    frame: monthly stocks (patients on therapy) or starts (new patients).
    Returns Date + one column per regimen label (+ 'Other'), rows sum to 1.
    """
    keys = [k for k, r in regimens.items() if r.line == line and k in frame.columns]
    if not keys:
        return pd.DataFrame()
    m = frame[keys].to_numpy(dtype=float)
    tot = m.sum(axis=1, keepdims=True)
    sh = np.divide(m, tot, out=np.zeros_like(m), where=tot > 0)
    peaks = sh.max(axis=0)
    # Keep the regimens with the most recent relevance (share at end of window)
    order = np.argsort(-(sh[-1] + 0.25 * peaks))
    keep = [i for i in order if peaks[i] >= min_peak][:max_series]
    out = pd.DataFrame({'Date': frame['Date'].values})
    for i in sorted(keep, key=lambda i: regimens[keys[i]].approval_year):
        out[regimens[keys[i]].label] = sh[:, i]
    other = [i for i in range(len(keys)) if i not in keep]
    if other:
        out['Other'] = sh[:, other].sum(axis=1)
    return out


def class_exposure(monthly: pd.DataFrame, regimens: dict,
                   classes: List[str]) -> pd.DataFrame:
    """Patients on therapy receiving each drug class (classes overlap)."""
    out = pd.DataFrame({'Date': monthly['Date']})
    for c in classes:
        keys = [k for k, r in regimens.items() if c in r.classes and k in monthly.columns]
        out[c] = monthly[keys].sum(axis=1) if keys else 0.0
    out['On_Therapy'] = monthly[[f'Total_{l}' for l in LINES]].sum(axis=1)
    return out


def novel_share(monthly: pd.DataFrame, regimens: dict, since: float = 2020.0) -> pd.Series:
    """Share of on-therapy patients on regimens with US uptake from `since`."""
    keys = [k for k, r in regimens.items() if r.uptake_year >= since and k in monthly.columns]
    tot = monthly[[f'Total_{l}' for l in LINES]].sum(axis=1)
    return monthly[keys].sum(axis=1) / tot.replace(0, np.nan)


# ── Evidence table ───────────────────────────────────────────────
def regimen_evidence_table(regimens: dict, rwe: float = 1.0) -> pd.DataFrame:
    rows = []
    for r in regimens.values():
        rows.append({
            'Line': r.line,
            'Regimen': r.label,
            'Segment': {'TE': 'Transplant-eligible', 'TI': 'Transplant-ineligible'}.get(r.eligibility, 'All'),
            'Trial': r.trial,
            'Efficacy input': r.efficacy_basis,
            'Model median PFS (mo)': round(r.implied_median, 1),
            'Real-world median (mo)': round(r.implied_median * rwe, 1),
            'HR': r.hazard_ratio if r.hazard_ratio is not None else np.nan,
            'Comparator': r.comparator,
            'US approval': r.approval_date if r.approval_year > 1991 else 'Pre-1990s',
            'Classes': ', '.join(r.classes),
            'Citation': r.citation,
            'Note': r.note,
            '_order': (LINES.index(r.line), -r.approval_year),
        })
    df = pd.DataFrame(rows).sort_values('_order').drop(columns='_order')
    return df.reset_index(drop=True)


# ── Sensitivity (true re-simulation) ─────────────────────────────
TORNADO_PARAMS = [
    ('treated_fraction', 'Treated fraction', 'rel'),
    ('rwe_pfs_multiplier', 'Real-world PFS multiplier', 'rel'),
    ('p_2l', 'P(reach 2L)', 'rel'),
    ('p_3l', 'P(reach 3L | 2L)', 'rel'),
    ('p_4l', 'P(reach 4L+ | 3L)', 'rel'),
    ('frac_te', 'Transplant-eligible fraction', 'rel'),
    ('mort_multiplier', 'On-line mortality', 'rel'),
    ('rate_trend_pct', 'Incidence rate trend (±1 pp/yr)', 'abs'),
    ('new_launch_speed_multiplier', 'Uptake speed of 2024+ launches', 'rel'),
]
CAPPED = {'treated_fraction', 'p_2l', 'p_3l', 'p_4l', 'frac_te'}


def tornado(run: Callable[[dict], pd.DataFrame], sc: dict, year: int,
            metric: str = 'On_Therapy', delta: float = 0.20) -> pd.DataFrame:
    """One-way sensitivity: each lever ±delta, full model re-run each time.

    run(scenario) -> annualized DataFrame. Probabilities are capped at 1.
    """
    def value(s):
        a = run(s)
        return float(a.loc[a['Year'] == year, metric].iloc[0])

    base = value(sc)
    rows = []
    for key, label, mode in TORNADO_PARAMS:
        lo, hi = dict(sc), dict(sc)
        if mode == 'abs':
            lo[key], hi[key] = sc[key] - 1.0, sc[key] + 1.0
        else:
            lo[key], hi[key] = sc[key] * (1 - delta), sc[key] * (1 + delta)
            if key in CAPPED:
                hi[key] = min(hi[key], 1.0)
        v_lo, v_hi = value(lo), value(hi)
        rows.append({'Parameter': label, 'Low': v_lo, 'Base': base, 'High': v_hi,
                     'Low input': lo[key], 'High input': hi[key],
                     'Swing': abs(v_hi - v_lo)})
    return pd.DataFrame(rows).sort_values('Swing').reset_index(drop=True)


# ── Validation ───────────────────────────────────────────────────
def benchmark_table(annual: pd.DataFrame, incidence: pd.DataFrame,
                    cohort: Dict[str, float], frac_te: float) -> pd.DataFrame:
    def at(year, col, frame=annual):
        s = frame.loc[frame['Year'] == year, col]
        return float(s.iloc[0]) if len(s) else np.nan

    inc_2026 = at(2026, 'Cases', incidence)
    on_2022 = at(2022, 'On_Therapy')
    reach_2l = cohort['start_2L'] / cohort['start_1L']
    reach_3l = cohort['start_3L'] / max(cohort['start_2L'], 1e-9)
    rwe_2l = frac_te * BENCHMARKS['rwe_reach_2l_te'] + (1 - frac_te) * BENCHMARKS['rwe_reach_2l_nte']
    w_te = frac_te * BENCHMARKS['rwe_reach_2l_te']
    w_nte = (1 - frac_te) * BENCHMARKS['rwe_reach_2l_nte']
    rwe_3l = (w_te * BENCHMARKS['rwe_reach_3l_given_2l_te'] +
              w_nte * BENCHMARKS['rwe_reach_3l_given_2l_nte']) / (w_te + w_nte)
    return pd.DataFrame([
        {'Check': 'New cases, 2026',
         'Model': f"{inc_2026:,.0f}",
         'Benchmark': f"{BENCHMARKS['acs_2026_cases']:,} (ACS 2026 estimate)",
         'Gap': f"{inc_2026 / BENCHMARKS['acs_2026_cases'] - 1:+.0%}",
         'Read-out': 'ACS projects all ages with delay adjustment; USCS observed counts run lower.'},
        {'Check': 'Patients on therapy ÷ people living with MM, 2022',
         'Model': f"{on_2022:,.0f} ({on_2022 / BENCHMARKS['seer_prevalence_2022']:.0%})",
         'Benchmark': f"{BENCHMARKS['seer_prevalence_2022']:,} prevalent (SEER)",
         'Gap': '—',
         'Read-out': 'Continuous-therapy era: most prevalent patients are on treatment or maintenance.'},
        {'Check': '1L → 2L reach (2016 cohort, 10 y)',
         'Model': f"{reach_2l:.0%}",
         'Benchmark': f"{rwe_2l:.0%} (Fonseca 2020, TE/TI-weighted)",
         'Gap': f"{reach_2l - rwe_2l:+.0%}",
         'Read-out': 'Claims data are follow-up-censored, so true lifetime reach is somewhat higher.'},
        {'Check': '2L → 3L reach (2016 cohort, 10 y)',
         'Model': f"{reach_3l:.0%}",
         'Benchmark': f"{rwe_3l:.0%} (Fonseca 2020, TE/TI-weighted)",
         'Gap': f"{reach_3l - rwe_3l:+.0%}",
         'Read-out': 'Same caveat as above.'},
    ])


# ── Narrative ────────────────────────────────────────────────────
def executive_summary(annual: pd.DataFrame, incidence: pd.DataFrame, regimens: dict,
                      monthly: pd.DataFrame, now: int, horizon: int) -> str:
    a = annual.set_index('Year')
    inc = incidence.set_index('Year')['Cases']
    on_now, on_h = a.loc[now, 'On_Therapy'], a.loc[horizon, 'On_Therapy']
    cagr = (on_h / on_now) ** (1 / max(horizon - now, 1)) - 1
    sh = class_exposure(monthly, regimens, ['Anti-CD38', 'BCMA bispecific', 'BCMA CAR-T'])
    sh['Year'] = sh['Date'].dt.year
    y = sh.groupby('Year').mean(numeric_only=True)
    cd38_now = y.loc[now, 'Anti-CD38'] / y.loc[now, 'On_Therapy']
    tcr_h = (y.loc[horizon, 'BCMA bispecific'] + y.loc[horizon, 'BCMA CAR-T']) / y.loc[horizon, 'On_Therapy']
    return (
        f"About <b>{inc.loc[now]:,.0f}</b> Americans are projected to be diagnosed with myeloma in {now}, "
        f"rising to <b>{inc.loc[horizon]:,.0f}</b> by {horizon} as the population ages. "
        f"The model puts <b>{on_now:,.0f}</b> patients on active therapy in {now}, growing "
        f"<b>{cagr:+.1%}</b> a year to <b>{on_h:,.0f}</b> in {horizon}: quadruplet induction and "
        f"continuous maintenance keep patients on first line longer. "
        f"Anti-CD38 regimens reach <b>{cd38_now:.0%}</b> of treated patients today; BCMA-directed "
        f"T-cell therapies (CAR-T and bispecifics) reach <b>{tcr_h:.0%}</b> by {horizon} as "
        f"teclistamab–daratumumab moves to first relapse."
    )


def pitfalls(sc: dict) -> List[dict]:
    return [
        {'severity': 'high', 'title': 'Adoption curves are calibrated, not fitted',
         'detail': 'Regimen shares come from logistic diffusion with analyst-set peak, speed and '
                   'displacement parameters, tuned to approximate published US treatment patterns. '
                   'They are not fitted to claims or EMR data (IQVIA, Komodo, Flatiron). Calibrate '
                   'against observed patient-level share before using for revenue forecasts.'},
        {'severity': 'high', 'title': 'Extrapolated PFS for immature quadruplets and bispecific combos',
         'detail': 'Where the median is not reached (PERSEUS, IMROZ, CEPHEUS, CARTITUDE-4, MajesTEC-3) '
                   'the Weibull curve is solved from one landmark PFS rate with an assumed shape. '
                   'Implied medians (e.g. D-VRd with ASCT ≈ 141 months before the real-world adjustment) '
                   'are sensitive to that shape. Treat long-horizon first-line stock as uncertain.'},
        {'severity': 'medium', 'title': 'Trial PFS ≠ real-world time to next treatment',
         'detail': f"A single efficacy–effectiveness multiplier ({sc['rwe_pfs_multiplier']:.2f}) "
                   'converts trial PFS to real-world duration. Real-world cohorts are older and frailer, '
                   'and some stop therapy before progression, so the true gap varies by regimen.'},
        {'severity': 'medium', 'title': 'Registry counts include smoldering myeloma',
         'detail': 'USCS myeloma (ICD-O-3 9732) includes some smoldering cases that are not treated at '
                   f"diagnosis. The treated fraction ({sc['treated_fraction']:.0%}) absorbs this and frail "
                   'untreated patients. Later progression of smoldering disease to treatment is not modelled.'},
        {'severity': 'medium', 'title': 'Prior-therapy exposure does not drive regimen choice',
         'detail': '2L+ shares are pooled across prior regimens. In practice, anti-CD38- or lenalidomide-'
                   'refractory patients are steered away from re-challenge. Exposure-conditional '
                   'sequencing would shift 2L share from DRd/DPd toward PI- and BCMA-based options.'},
        {'severity': 'low', 'title': 'Homogeneous population',
         'detail': 'There is no stratification by race or ethnicity, cytogenetic risk or frailty. '
                   'Myeloma incidence is about 2× higher in Black Americans, and outcomes differ by risk group.'},
        {'severity': 'low', 'title': 'Competing risks with constant on-line mortality',
         'detail': 'Progression (Weibull) and death (constant hazard per line) are cause-specific hazards '
                   'combined exactly within each month. Trial PFS already includes some deaths, so on-line '
                   'death is slightly double-counted, mainly in later lines.'},
    ]
