"""
Line-of-Therapy (LoT) patient-flow model for US multiple myeloma.

Structure (monthly, discrete time)
    Incidence (USCS observed + Census-based projection)
      → × treated fraction, after dx-to-treatment delay → 1L starts (TE / TI)
      → regimen assignment by calendar-time market share (AdoptionEngine)
      → on-line survival per regimen: cause-specific competing risks
            progression  Weibull(scale × RWE multiplier, shape)   [from trial PFS]
            death        constant line-specific hazard
      → progression × P(next line) → 2L → 3L → 4L+ (4L+ re-entry = 5L, 6L, ...)

Because each regimen cohort's exit kernel depends only on time since start,
stocks and flows are convolutions of starts with survival kernels; the whole
1984–2035 run takes well under a second, so the dashboard re-simulates live.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

try:
    from scientific_utils import load_regimens, Regimen
    from adoption import AdoptionEngine
    import epidemiology as epi
except ImportError:  # imported as package module
    from .scientific_utils import load_regimens, Regimen
    from .adoption import AdoptionEngine
    from . import epidemiology as epi

logger = logging.getLogger(__name__)

LINES = ['1L', '2L', '3L', '4L+']
BASE_DIR = Path(__file__).resolve().parent


# ── Scenario ─────────────────────────────────────────────────────
def scenario_from_params(params: dict) -> dict:
    """Flatten params.yaml into the scenario dictionary used by the engine."""
    h, inc = params['horizon'], params['incidence']
    up, att = params['uptake'], params['attrition']
    eff, mort = params['effectiveness'], params['mortality']
    return {
        'start_year': int(h['start_year']),
        'burn_in_years': int(h['burn_in_years']),
        'end_year': int(h['end_year']),
        'incidence_method': inc['method'],
        'rate_trend_pct': float(inc['rate_trend_pct']),
        'base_years': tuple(inc['base_years']),
        'anchor_to_acs': bool(inc['anchor_to_acs']),
        'acs_year': int(inc['acs_benchmark']['year']),
        'acs_cases': float(inc['acs_benchmark']['cases']),
        'treated_fraction': float(up['treated_fraction']),
        'delay_months': int(up['dx_to_1l_delay_months']),
        'frac_te': float(up['fraction_transplant_eligible']),
        'p_2l': float(att['p_reach_2l']),
        'p_3l': float(att['p_reach_3l_given_2l']),
        'p_4l': float(att['p_reach_4l_given_3l']),
        'p_later': float(att['p_next_given_4l']),
        'lag_months': int(att['progression_to_next_line_months']),
        'rwe_pfs_multiplier': float(eff['rwe_pfs_multiplier']),
        'new_launch_speed_multiplier': float(eff['new_launch_speed_multiplier']),
        'mort_1l': float(mort['monthly_death_hazard_1l']),
        'mort_2l': float(mort['monthly_death_hazard_2l']),
        'mort_3l': float(mort['monthly_death_hazard_3l']),
        'mort_4l': float(mort['monthly_death_hazard_4l_plus']),
        'mort_multiplier': float(mort['multiplier']),
    }


# ── Inputs ───────────────────────────────────────────────────────
@dataclass
class ModelInputs:
    uscs: pd.DataFrame
    census: pd.DataFrame
    params: dict
    regimens_yaml: dict
    events_yaml: dict
    regimens: Dict[str, Regimen] = field(init=False)

    def __post_init__(self):
        self.regimens = load_regimens(self.regimens_yaml)


def _read_yaml(p: Path) -> dict:
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_inputs(base_dir: Path = BASE_DIR) -> ModelInputs:
    uscs_path = base_dir / 'outputs' / 'uscs_myeloma_incidence_clean.csv'
    if not uscs_path.exists():
        uscs_path = base_dir / 'United States and Puerto Rico Cancer Statistics, 1999-2022 Incidence.csv'
    return ModelInputs(
        uscs=epi.load_uscs(uscs_path),
        census=epi.load_census(base_dir / 'data' / 'census_np2023_population_age_sex.csv'),
        params=_read_yaml(base_dir / 'params.yaml'),
        regimens_yaml=_read_yaml(base_dir / 'regimens.yaml'),
        events_yaml=_read_yaml(base_dir / 'events.yaml'),
    )


def build_incidence(inputs: ModelInputs, sc: dict) -> pd.DataFrame:
    anchor = (sc['acs_year'], sc['acs_cases']) if sc['anchor_to_acs'] else None
    return epi.project_incidence(inputs.uscs, inputs.census, end_year=sc['end_year'],
                                 method=sc['incidence_method'],
                                 rate_trend_pct=sc['rate_trend_pct'],
                                 base_years=sc['base_years'], anchor=anchor)


# ── Engine ───────────────────────────────────────────────────────
@dataclass
class ModelResult:
    monthly: pd.DataFrame          # stocks by regimen + line totals + flows
    starts: pd.DataFrame           # new patient starts by regimen (monthly)
    annual_incidence: pd.DataFrame
    scenario: dict


def _kernels(reg: Regimen, mu: float, rwe: float, n: int):
    """Monthly exit kernels for one regimen with competing risks.

    ks[a] = P(still on line at end of month a)         a = 0..n-1
    kp[a] = P(progress during month a), kd[a] = P(die on line during month a)
    """
    w = reg.weibull.scaled(rwe)
    a = np.arange(n + 1, dtype=float)
    hp = w.cumulative_hazard(a)
    surv = np.exp(-hp - mu * a)
    d_s = surv[:-1] - surv[1:]
    d_hp = np.diff(hp)
    frac_p = np.divide(d_hp, d_hp + mu, out=np.zeros_like(d_hp), where=(d_hp + mu) > 0)
    return surv[1:], d_s * frac_p, d_s * (1 - frac_p)


def _conv(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.convolve(x, k)[:len(x)]


def _shift(x: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return x.copy()
    return np.concatenate([np.zeros(lag), x[:-lag]])


def _simulate(inputs: ModelInputs, sc: dict, years: np.ndarray, starts_1l: np.ndarray):
    """Propagate 1L starts through all lines. Returns (stocks, starts, flows)."""
    T = len(years)
    engine = AdoptionEngine(inputs.regimens, inputs.events_yaml.get('events', []),
                            new_launch_speed_multiplier=sc['new_launch_speed_multiplier'])
    mu = {'1L': sc['mort_1l'], '2L': sc['mort_2l'], '3L': sc['mort_3l'], '4L+': sc['mort_4l']}
    mu = {k: v * sc['mort_multiplier'] for k, v in mu.items()}
    rwe = sc['rwe_pfs_multiplier']
    lag = max(int(sc['lag_months']), 0)

    stocks: Dict[str, np.ndarray] = {}
    starts: Dict[str, np.ndarray] = {}
    flows: Dict[str, np.ndarray] = {}

    def run_line(line: str, seg_starts: Dict[str, np.ndarray]) -> np.ndarray:
        """seg_starts: eligibility segment -> total starts vector."""
        prog, death, total = np.zeros(T), np.zeros(T), np.zeros(T)
        for elig, s_total in seg_starts.items():
            keys, shares = engine.share_matrix(years, line, elig)
            for i, key in enumerate(keys):
                s = s_total * shares[:, i]
                ks, kp, kd = _kernels(inputs.regimens[key], mu[line], rwe, T)
                st = _conv(s, ks)
                starts[key] = starts.get(key, 0) + s
                stocks[key] = stocks.get(key, 0) + st
                prog += _conv(s, kp)
                death += _conv(s, kd)
                total += st
        flows[f'Starts_{line}'] = sum(seg_starts.values())
        flows[f'Progressions_{line}'] = prog
        flows[f'Deaths_{line}'] = death
        stocks[f'Total_{line}'] = total
        return prog

    prog1 = run_line('1L', {'TE': starts_1l * sc['frac_te'],
                            'TI': starts_1l * (1 - sc['frac_te'])})
    prog2 = run_line('2L', {'Both': _shift(prog1, lag) * sc['p_2l']})
    prog3 = run_line('3L', {'Both': _shift(prog2, lag) * sc['p_3l']})

    # 4L+ is recursive (progression on 4L+ can re-enter 4L+ as 5L, 6L, ...)
    keys4, shares4 = engine.share_matrix(years, '4L+', 'Both')
    kern4 = [_kernels(inputs.regimens[k], mu['4L+'], rwe, T) for k in keys4]
    KP = np.stack([k[1] for k in kern4], axis=1)          # [T, R]
    entry4 = _shift(prog3, lag) * sc['p_4l']
    s4 = np.zeros((T, len(keys4)))
    prog4 = np.zeros(T)
    reentry = np.zeros(T)
    lag4 = max(lag, 1)
    for t in range(T):
        reentry[t] = prog4[t - lag4] * sc['p_later'] if t >= lag4 else 0.0
        s4[t] = (entry4[t] + reentry[t]) * shares4[t]
        # progressions in month t from all 4L+ cohorts started at months 0..t
        prog4[t] = np.einsum('ar,ar->', s4[t::-1], KP[:t + 1])
    death4, total4 = np.zeros(T), np.zeros(T)
    for i, key in enumerate(keys4):
        ks, _, kd = kern4[i]
        st = _conv(s4[:, i], ks)
        starts[key] = s4[:, i]
        stocks[key] = st
        total4 += st
        death4 += _conv(s4[:, i], kd)
    flows['Starts_4L+'] = entry4 + reentry
    flows['Starts_4L+_first'] = entry4
    flows['Progressions_4L+'] = prog4
    flows['Deaths_4L+'] = death4
    stocks['Total_4L+'] = total4

    for line, p in [('1L', sc['p_2l']), ('2L', sc['p_3l']),
                    ('3L', sc['p_4l']), ('4L+', sc['p_later'])]:
        flows[f'No_Next_Line_{line}'] = flows[f'Progressions_{line}'] * (1 - p)
    return stocks, starts, flows


def run_model(inputs: ModelInputs, sc: dict) -> ModelResult:
    annual = build_incidence(inputs, sc)
    first_obs = int(annual['Year'].min())
    sim_start = first_obs - sc['burn_in_years']
    monthly_inc = epi.monthly_incidence(annual, sim_start, sc['end_year'])
    dates = monthly_inc['Date']
    years = (dates.dt.year + (dates.dt.month - 1) / 12.0).to_numpy()

    dx = monthly_inc['Cases'].to_numpy()
    starts_1l = _shift(dx, sc['delay_months']) * sc['treated_fraction']
    stocks, starts, flows = _simulate(inputs, sc, years, starts_1l)

    out = pd.DataFrame({'Date': dates, 'Incidence': dx,
                        'Untreated': _shift(dx, sc['delay_months']) * (1 - sc['treated_fraction']),
                        'New_Starts_1L': starts_1l})
    reg_keys = [k for k in stocks if not k.startswith('Total_')]
    out = pd.concat([out,
                     pd.DataFrame({k: stocks[k] for k in reg_keys}),
                     pd.DataFrame({f'Total_{l}': stocks[f'Total_{l}'] for l in LINES}),
                     pd.DataFrame(flows)], axis=1)
    st_df = pd.concat([dates.rename('Date'), pd.DataFrame(starts)], axis=1)

    keep = out['Date'].dt.year >= sc['start_year']
    return ModelResult(monthly=out.loc[keep].reset_index(drop=True),
                       starts=st_df.loc[keep].reset_index(drop=True),
                       annual_incidence=annual, scenario=dict(sc))


def run_cohort(inputs: ModelInputs, sc: dict, start_year: int,
               n_patients: float = 1000.0, horizon_years: int = 10) -> Dict[str, float]:
    """Lifetime-style journey of n patients starting 1L in January of start_year.

    Returns cumulative counts over the horizon: starts per line, deaths on each
    line, progressions without further therapy, and patients still on each line.
    """
    T = horizon_years * 12
    years = start_year + np.arange(T) / 12.0
    impulse = np.zeros(T)
    impulse[0] = n_patients
    stocks, _, flows = _simulate(inputs, sc, years, impulse)
    out = {}
    for line in LINES:
        out[f'start_{line}'] = float(flows[f'Starts_{line}'].sum()) if line != '4L+' \
            else float(flows['Starts_4L+_first'].sum())
        out[f'death_{line}'] = float(flows[f'Deaths_{line}'].sum())
        out[f'stop_{line}'] = float(flows[f'No_Next_Line_{line}'].sum())
        out[f'still_{line}'] = float(stocks[f'Total_{line}'][-1])
    # Progressed in the final lag month(s) but not yet started on the next line
    out['in_transit'] = max(0.0, n_patients - sum(
        out[f'death_{l}'] + out[f'stop_{l}'] + out[f'still_{l}'] for l in LINES))
    return out


# ── CLI ──────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    inputs = load_inputs()
    sc = scenario_from_params(inputs.params)
    res = run_model(inputs, sc)
    out_dir = BASE_DIR / 'outputs'
    out_dir.mkdir(exist_ok=True)
    res.monthly.to_csv(out_dir / 'mm_detailed_simulation.csv', index=False)
    res.starts.to_csv(out_dir / 'mm_regimen_starts.csv', index=False)
    res.annual_incidence.to_csv(out_dir / 'mm_incidence_projection.csv', index=False)
    logger.info("Simulation complete: %d months, %d regimen series → %s",
                len(res.monthly), res.starts.shape[1] - 1, out_dir)


if __name__ == '__main__':
    main()
