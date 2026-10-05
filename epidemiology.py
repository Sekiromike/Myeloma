"""
Incidence engine: observed US registry counts plus forward projection.

Observed:  CDC WONDER — U.S. Cancer Statistics (USCS) 1999–2022, myeloma,
           by 5-year age group and sex (NPCR + SEER, ~100% population coverage).
Projected: age-/sex-specific rates (pooled over recent non-COVID years) applied
           to U.S. Census Bureau 2023 National Population Projections (middle
           series), with an optional annual change in underlying rates.

Projection methods
    demographic  rates x projected population  (captures population ageing)
    trend        log-linear fit to observed counts 2010–2022 (ex. 2020)
    flat         last observed year carried forward (legacy behaviour)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

AGE_ORDER = ['25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
             '60-64', '65-69', '70-74', '75-79', '80-84', '85+']
COVID_YEARS = (2020,)
DEFAULT_BASE_YEARS = (2017, 2018, 2019, 2021, 2022)


# ── Loaders ──────────────────────────────────────────────────────
def load_uscs(path: Path) -> pd.DataFrame:
    """Return Year, Sex, Age_Group, Count, Population from the USCS extract."""
    df = pd.read_csv(path)
    if 'Age Groups Code' not in df.columns:
        raise ValueError(f"Unexpected USCS layout in {path}")
    df = df[pd.to_numeric(df['Year'], errors='coerce').notna()].copy()
    out = pd.DataFrame({
        'Year': df['Year'].astype(int),
        'Sex': df['Sex'].astype(str).str.strip(),
        'Age_Group': df['Age Groups Code'].astype(str).str.strip(),
        'Count': pd.to_numeric(df['Count'], errors='coerce'),
        'Population': pd.to_numeric(df['Population'], errors='coerce'),
    }).dropna()
    return out


def load_census(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ── Observed summaries ───────────────────────────────────────────
def observed_annual(uscs: pd.DataFrame) -> pd.DataFrame:
    a = uscs.groupby('Year')['Count'].sum().reset_index(name='Cases')
    a['Type'] = 'Observed'
    return a


def age_specific_rates(uscs: pd.DataFrame,
                       base_years: Iterable[int] = DEFAULT_BASE_YEARS) -> pd.DataFrame:
    """Pooled rate per person-year by sex x age group (sum counts / sum population)."""
    sub = uscs[uscs['Year'].isin(list(base_years))]
    r = sub.groupby(['Sex', 'Age_Group'])[['Count', 'Population']].sum()
    r['Rate'] = r['Count'] / r['Population']
    return r[['Rate']].reset_index()


def age_standardised_rate(uscs: pd.DataFrame, census: pd.DataFrame,
                          std_year: int = 2022) -> pd.Series:
    """Directly age/sex-standardised rate per 100k (std = Census adult pop)."""
    std = census[census['Year'] == std_year].set_index(['Sex', 'Age_Group'])['Population']
    out = {}
    for y, g in uscs.groupby('Year'):
        rate = (g.set_index(['Sex', 'Age_Group'])['Count'] /
                g.set_index(['Sex', 'Age_Group'])['Population']).reindex(std.index).fillna(0)
        out[y] = float((rate * std).sum() / std.sum() * 1e5)
    return pd.Series(out, name='ASR_per_100k')


# ── Projection ───────────────────────────────────────────────────
def project_incidence(uscs: pd.DataFrame, census: pd.DataFrame,
                      end_year: int = 2035, method: str = 'demographic',
                      rate_trend_pct: float = 0.0,
                      base_years: Iterable[int] = DEFAULT_BASE_YEARS,
                      anchor: Optional[tuple] = None) -> pd.DataFrame:
    """Annual incidence (Year, Cases, Type) from first observed year to end_year.

    anchor: optional (year, cases) external benchmark, e.g. (2026, 36000) from
            ACS Cancer Statistics 2026. When given, the whole series is scaled
            by a single registry-completeness factor so the projection hits it.
    """
    obs = observed_annual(uscs)
    last = int(obs['Year'].max())
    years = np.arange(last + 1, end_year + 1)

    if method == 'demographic':
        rates = age_specific_rates(uscs, base_years)
        pop = census[census['Year'].isin(years)].merge(rates, on=['Sex', 'Age_Group'], how='left')
        pop['Cases'] = pop['Population'] * pop['Rate'].fillna(0)
        proj = pop.groupby('Year')['Cases'].sum().reindex(years).values
        # Census and registry denominators differ slightly in the overlap year;
        # rescale Census populations onto the registry denominator.
        census_last = census[census['Year'] == last].merge(rates, on=['Sex', 'Age_Group'])
        uscs_last = uscs[uscs['Year'] == last].merge(rates, on=['Sex', 'Age_Group'])
        census_pred = float((census_last['Population'] * census_last['Rate']).sum())
        uscs_pred = float((uscs_last['Population'] * uscs_last['Rate']).sum())
        if census_pred > 0 and uscs_pred > 0:
            proj = proj * (uscs_pred / census_pred)
    elif method == 'trend':
        fit = obs[(obs['Year'] >= 2010) & ~obs['Year'].isin(COVID_YEARS)]
        b, a = np.polyfit(fit['Year'], np.log(fit['Cases']), 1)
        proj = np.exp(a + b * years)
    elif method == 'flat':
        proj = np.repeat(float(obs.loc[obs['Year'] == last, 'Cases'].iloc[0]), len(years))
    else:
        raise ValueError(f"Unknown method {method}")

    if method != 'trend' and rate_trend_pct:
        proj = proj * (1 + rate_trend_pct / 100.0) ** (years - last)

    out = pd.concat([obs, pd.DataFrame({'Year': years, 'Cases': proj, 'Type': 'Projected'})],
                    ignore_index=True)
    out['Completeness_Factor'] = 1.0
    if anchor is not None:
        a_year, a_cases = anchor
        ref = out.loc[out['Year'] == a_year, 'Cases']
        if len(ref) and ref.iloc[0] > 0:
            f = a_cases / float(ref.iloc[0])
            out['Cases'] *= f
            out['Completeness_Factor'] = f
    return out


def projected_age_distribution(uscs: pd.DataFrame, census: pd.DataFrame,
                               years: Iterable[int],
                               base_years: Iterable[int] = DEFAULT_BASE_YEARS) -> pd.DataFrame:
    """Expected cases by age group for selected projection years."""
    rates = age_specific_rates(uscs, base_years)
    pop = census[census['Year'].isin(list(years))].merge(rates, on=['Sex', 'Age_Group'])
    pop['Cases'] = pop['Population'] * pop['Rate']
    return pop.groupby(['Year', 'Age_Group'])['Cases'].sum().reset_index()


def monthly_incidence(annual: pd.DataFrame, start_year: int, end_year: int,
                      backcast_decline_pct: float = 2.0) -> pd.DataFrame:
    """Spread annual cases evenly across months; backcast years before data.

    Years before the first observed year are a model burn-in only: cases are
    back-extrapolated at -backcast_decline_pct per year so that 1999 starts
    with a realistic prevalent pool on each line.
    """
    s = annual.set_index('Year')['Cases']
    first = int(s.index.min())
    rows = []
    for y in range(start_year, end_year + 1):
        c = float(s.loc[y]) if y in s.index else \
            float(s.loc[first]) * (1 - backcast_decline_pct / 100.0) ** (first - y)
        for m in range(1, 13):
            rows.append((pd.Timestamp(year=y, month=m, day=1), c / 12.0))
    return pd.DataFrame(rows, columns=['Date', 'Cases'])
