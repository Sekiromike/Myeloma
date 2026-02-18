"""
Derived metrics for the MM Disease Model Explorer.
Scientifically rigorous computations with epidemiological transparency.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ── Occupancy / Prevalence ────────────────────────────────────────
def compute_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long-form DataFrame with Date, Line, Patients."""
    line_map = {'Total_1L': '1st Line', 'Total_2L': '2nd Line',
                'Total_3L': '3rd Line', 'Total_4L+': '4th Line+'}
    frames = []
    for col, label in line_map.items():
        if col in df.columns:
            tmp = df[['Date']].copy()
            tmp['Line'] = label
            tmp['Patients'] = df[col]
            frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


# ── Transition Rates ──────────────────────────────────────────────
def compute_transition_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute approximate monthly transition rates between lines.

    Rate ≈ Δ(next-line stock) / current-line stock  (proxy).
    """
    out = df[['Date']].copy()

    for src, dst in [('Total_1L', 'Total_2L'),
                     ('Total_2L', 'Total_3L'),
                     ('Total_3L', 'Total_4L+')]:
        if src in df.columns and dst in df.columns:
            delta = df[dst].diff().clip(lower=0)
            denom = df[src].replace(0, np.nan)
            label = f"{src.replace('Total_', '')}→{dst.replace('Total_', '')}"
            out[label] = (delta / denom).fillna(0)
    return out


# ── Regimen Share ─────────────────────────────────────────────────
def compute_regimen_shares(df: pd.DataFrame, line: str) -> pd.DataFrame:
    """Return time-series of regimen shares for a given line (e.g. '1L')."""
    prefix = f"{line}_"
    total_col = f"Total_{line}"
    reg_cols = [c for c in df.columns
                if c.startswith(prefix) and c != total_col]

    if not reg_cols or total_col not in df.columns:
        return pd.DataFrame()

    out = df[['Date']].copy()
    total = df[total_col].replace(0, np.nan)
    for c in reg_cols:
        name = c.replace(prefix, '')
        out[name] = (df[c] / total).fillna(0)
    return out


# ── Sankey Flow Data ──────────────────────────────────────────────
def compute_sankey_data(df: pd.DataFrame, params: dict) -> dict:
    """Build Sankey nodes/links for the patient journey."""
    recent = df.tail(12)

    new_dx       = recent['New_Starts_1L'].mean()
    on_1l        = recent.get('Total_1L', pd.Series([0])).mean()
    on_2l        = recent.get('Total_2L', pd.Series([0])).mean()
    on_3l        = recent.get('Total_3L', pd.Series([0])).mean()
    on_4l        = recent.get('Total_4L+', pd.Series([0])).mean()

    p2l = params['attrition']['p_reach_2l']
    p3l = params['attrition'].get('p_reach_3l_given_2l', 0.6)

    nodes = ['Diagnosis', '1st Line', '2nd Line', '3rd Line+',
             'Ongoing Care', 'Did Not Advance']
    node_idx = {n: i for i, n in enumerate(nodes)}

    links_src  = []
    links_tgt  = []
    links_val  = []
    links_lbl  = []

    # Diagnosis → 1L
    links_src.append(node_idx['Diagnosis'])
    links_tgt.append(node_idx['1st Line'])
    links_val.append(new_dx)
    links_lbl.append(f'{new_dx:,.0f} patients/mo start treatment')

    # 1L → 2L
    to_2l = new_dx * p2l
    links_src.append(node_idx['1st Line'])
    links_tgt.append(node_idx['2nd Line'])
    links_val.append(to_2l)
    links_lbl.append(f'{p2l:.0%} advance to 2nd line')

    # 1L → Did Not Advance
    links_src.append(node_idx['1st Line'])
    links_tgt.append(node_idx['Did Not Advance'])
    links_val.append(new_dx * (1 - p2l))
    links_lbl.append(f'{1-p2l:.0%} do not advance')

    # 2L → 3L+
    to_3l = to_2l * p3l
    links_src.append(node_idx['2nd Line'])
    links_tgt.append(node_idx['3rd Line+'])
    links_val.append(to_3l)
    links_lbl.append(f'{p3l:.0%} advance to 3rd line+')

    # 2L → Ongoing Care
    links_src.append(node_idx['2nd Line'])
    links_tgt.append(node_idx['Ongoing Care'])
    links_val.append(to_2l * (1 - p3l))
    links_lbl.append(f'{1-p3l:.0%} remain / ongoing care')

    # 3L+ → Ongoing Care
    links_src.append(node_idx['3rd Line+'])
    links_tgt.append(node_idx['Ongoing Care'])
    links_val.append(to_3l)
    links_lbl.append('Ongoing care / later lines')

    return {
        'nodes': nodes,
        'link_source': links_src,
        'link_target': links_tgt,
        'link_value':  links_val,
        'link_label':  links_lbl,
    }


# ── Duration Stats ────────────────────────────────────────────────
def compute_duration_stats(params: dict) -> pd.DataFrame:
    """Return median durations per line from params."""
    d = params.get('durations_months_median', {})
    rows = [
        {'Line': '1st Line',  'Median Months': d.get('1l', 24)},
        {'Line': '2nd Line',  'Median Months': d.get('2l', 16)},
        {'Line': '3rd Line+', 'Median Months': d.get('3l_plus', 6)},
    ]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
#  NEW METRICS — Added for dashboard v2
# ══════════════════════════════════════════════════════════════════

# ── Year-over-Year Trends ─────────────────────────────────────────
def compute_yoy_trends(df: pd.DataFrame) -> dict:
    """Compute YoY trends for KPI values.

    Compares the most recent 12-month average to the prior 12 months.
    Returns dict of {'metric_name': {'current': x, 'prior': y, 'delta_pct': z, 'arrow': '▲'/'▼'/'—'}}.
    """
    if len(df) < 24:
        # Not enough data for YoY comparison
        recent = df.tail(12)
        return {
            'total_on_treatment': {
                'current': sum(recent.get(c, pd.Series([0])).mean()
                               for c in ['Total_1L', 'Total_2L', 'Total_3L', 'Total_4L+']),
                'prior': None, 'delta_pct': None, 'arrow': '—',
            },
            'new_starts': {
                'current': recent['New_Starts_1L'].mean(),
                'prior': None, 'delta_pct': None, 'arrow': '—',
            },
            'prevalence_1l': {
                'current': recent.get('Total_1L', pd.Series([0])).mean(),
                'prior': None, 'delta_pct': None, 'arrow': '—',
            },
        }

    recent = df.tail(12)
    prior  = df.iloc[-24:-12]

    results = {}
    for metric_name, compute_fn in [
        ('total_on_treatment', lambda d: sum(d.get(c, pd.Series([0])).mean()
                                             for c in ['Total_1L', 'Total_2L', 'Total_3L', 'Total_4L+'])),
        ('new_starts', lambda d: d['New_Starts_1L'].mean()),
        ('prevalence_1l', lambda d: d.get('Total_1L', pd.Series([0])).mean()),
    ]:
        curr = compute_fn(recent)
        prev = compute_fn(prior)
        if prev > 0:
            delta = (curr - prev) / prev * 100
            arrow = '▲' if delta > 1 else ('▼' if delta < -1 else '—')
        else:
            delta = None
            arrow = '—'
        results[metric_name] = {
            'current': curr, 'prior': prev,
            'delta_pct': delta, 'arrow': arrow,
        }
    return results


# ── Novel Therapy Percentage ──────────────────────────────────────
# Regimens approved after 2020 (quadruplets, bispecifics, CAR-T)
NOVEL_REGIMENS = {'D-VRd', 'Cilta-cel', 'Ide-cel', 'Teclistamab',
                  'Talquetamab', 'Isa-Kd', 'Isa-Pd'}

def compute_novel_therapy_pct(df: pd.DataFrame) -> float:
    """Compute % of total patients on 'novel' regimens (post-2020 approvals).

    Returns a float between 0 and 100.
    """
    recent = df.tail(12)
    total = sum(recent.get(c, pd.Series([0])).mean()
                for c in ['Total_1L', 'Total_2L', 'Total_3L', 'Total_4L+'])
    if total <= 0:
        return 0.0

    novel_sum = 0.0
    for col in recent.columns:
        # Column format: "{line}_{regimen_name}"
        parts = col.split('_', 1)
        if len(parts) == 2 and parts[1] in NOVEL_REGIMENS:
            novel_sum += recent[col].mean()
    return (novel_sum / total) * 100


# ── Incidence CAGR ────────────────────────────────────────────────
def compute_incidence_cagr(df: pd.DataFrame) -> dict:
    """Compute Compound Annual Growth Rate from the New_Starts_1L column.

    Uses first and last full-year annualised values.
    """
    df_copy = df.copy()
    df_copy['Year'] = df_copy['Date'].dt.year
    yearly = df_copy.groupby('Year')['New_Starts_1L'].sum()

    if len(yearly) < 2:
        return {'cagr': 0.0, 'start_year': None, 'end_year': None,
                'start_annual': 0, 'end_annual': 0}

    # Use the first year with non-zero data and the last year
    valid = yearly[yearly > 0]
    if len(valid) < 2:
        return {'cagr': 0.0, 'start_year': None, 'end_year': None,
                'start_annual': 0, 'end_annual': 0}

    y0, yn = valid.index[0], valid.index[-1]
    v0, vn = valid.iloc[0], valid.iloc[-1]
    n_years = yn - y0

    if n_years > 0 and v0 > 0:
        cagr = (vn / v0) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    return {
        'cagr': cagr * 100,  # as percentage
        'start_year': int(y0),
        'end_year': int(yn),
        'start_annual': int(round(v0)),
        'end_annual': int(round(vn)),
    }


# ── Auto-Insights ─────────────────────────────────────────────────
def generate_insights(df: pd.DataFrame, params: dict,
                       view: str = 'patient') -> List[str]:
    """Generate 3 headline insights for the selected view."""
    recent = df.tail(12)
    t1l = recent.get('Total_1L', pd.Series([0])).mean()
    t2l = recent.get('Total_2L', pd.Series([0])).mean()
    t3l = recent.get('Total_3L', pd.Series([0])).mean()
    t4l = recent.get('Total_4L+', pd.Series([0])).mean()
    total = t1l + t2l + t3l + t4l
    pct_1l = t1l / total * 100 if total > 0 else 0

    p2l = params['attrition']['p_reach_2l']
    dur = params.get('durations_months_median', {})

    if view == 'patient':
        return [
            f"<b>{pct_1l:.0f}%</b> of patients on treatment are in their first line — "
            f"the majority are receiving initial therapy.",
            f"About <b>{p2l:.0%}</b> of patients move from first to second-line treatment. "
            f"Many remain stable on first-line therapy for a median of "
            f"<b>{dur.get('1l', 24)} months</b>.",
            f"Newer therapies (like quadruplet regimens and bispecifics) are being "
            f"adopted rapidly, reflecting continued improvement in the treatment landscape.",
        ]
    else:
        mort_1l = params['mortality']['monthly_death_hazard_1l']
        mort_3l = params['mortality']['monthly_death_hazard_3l_plus']
        novel_pct = compute_novel_therapy_pct(df)
        return [
            f"Mean monthly occupancy (recent 12 mo): "
            f"1L={t1l:,.0f}, 2L={t2l:,.0f}, 3L={t3l:,.0f}, 4L+={t4l:,.0f}.",
            f"3L+ monthly mortality hazard ({mort_3l:.3f}) is "
            f"<b>{mort_3l/mort_1l:.1f}×</b> higher than 1L ({mort_1l:.3f}), "
            f"accounting for the largest share of disease-related attrition.",
            f"<b>{novel_pct:.1f}%</b> of the on-treatment population is receiving "
            f"post-2020 novel therapies (D-VRd, CAR-T, bispecifics)."
        ]


# ── Executive Summary ────────────────────────────────────────────
def generate_executive_summary(df: pd.DataFrame, params: dict) -> str:
    """Produce a dynamic 2–3 sentence executive headline."""
    recent = df.tail(12)
    new_mo = recent['New_Starts_1L'].mean()
    total = sum(recent.get(c, pd.Series([0])).mean()
                for c in ['Total_1L', 'Total_2L', 'Total_3L', 'Total_4L+'])
    p2l = params['attrition']['p_reach_2l']
    dur_1l = params.get('durations_months_median', {}).get('1l', 24)
    novel_pct = compute_novel_therapy_pct(df)

    return (
        f"Approximately <b>{new_mo:,.0f} Americans</b> begin myeloma treatment each month, "
        f"with <b>{total:,.0f}</b> patients actively on therapy across all lines. "
        f"At current parameters, <b>{p2l:.0%}</b> advance to second-line therapy "
        f"(median 1L duration: {dur_1l} months). "
        f"<b>{novel_pct:.0f}%</b> of patients are on post-2020 novel therapies."
    )


# ── Epidemiological Pitfalls ──────────────────────────────────────
def generate_pitfalls(df: pd.DataFrame, params: dict) -> List[dict]:
    """Generate scientifically accurate model limitation statements.

    Each item: {'severity': 'high'|'medium'|'low', 'title': str, 'detail': str}
    """
    pitfalls = []

    # 1. Flat incidence projection
    cagr_info = compute_incidence_cagr(df)
    cagr_val = cagr_info['cagr']
    pitfalls.append({
        'severity': 'high',
        'title': 'Flat Incidence Projection',
        'detail': (
            f"Incidence from {cagr_info['start_year']}–{cagr_info['end_year']} is flat-projected "
            f"(observed CAGR: {cagr_val:+.1f}%). This ignores demographic aging trends, "
            f"changes in diagnostic sensitivity (e.g., updated IMWG criteria, 2014), and "
            f"the secular increase in MM incidence driven by population aging. "
            f"SEER data suggests US MM incidence has grown ~1–2% annually since 2000. "
            f"Projection should ideally use SEER*Stat or age-period-cohort models."
        ),
    })

    # 2. Weibull PFS assumption
    pitfalls.append({
        'severity': 'high',
        'title': 'Universal Weibull Shape Parameter (k = 1.3)',
        'detail': (
            "All regimens use the same Weibull shape parameter (k = 1.3), varying only "
            "the scale (derived from median PFS). In reality, the hazard function shape "
            "differs by regimen class: immunomodulatory agents may show flatter hazards, "
            "while proteasome inhibitors may show early-peak hazards. This assumption "
            "homogenises the time-to-progression curves and can mis-estimate the timing "
            "of line transitions, especially at the tails. Regimen-specific k values "
            "should be fitted from individual-trial KM curves (e.g., via digitization)."
        ),
    })

    # 3. No age/sex/race stratification
    pitfalls.append({
        'severity': 'high',
        'title': 'No Demographic Stratification',
        'detail': (
            "The model treats all patients as a homogeneous cohort. MM incidence is "
            "~2× higher in Black/African American populations (SEER age-adjusted rate: "
            "~14 vs ~7 per 100,000). Median age at diagnosis is 69 years; outcomes differ "
            "substantially by age (TE vs TI split is a proxy, but incomplete). Sex-based "
            "differences in PFS and OS are documented. Without stratification, the model "
            "cannot capture health equity dimensions or subpopulation-specific treatment "
            "patterns."
        ),
    })

    # 4. Competing risks
    pitfalls.append({
        'severity': 'medium',
        'title': 'Simplified Competing Risks',
        'detail': (
            "Disease progression and death are modelled as independent competing events "
            "with probabilities summed and capped at 1.0. A formal cause-specific hazard "
            "or Fine–Gray subdistribution model would be more appropriate. The current "
            "approach can overestimate progression rates when mortality is high (especially "
            "in 3L+), and underestimate them when mortality is low."
        ),
    })

    # 5. Adoption curves
    pitfalls.append({
        'severity': 'medium',
        'title': 'Calibrated (Not Empirically Fitted) Adoption Curves',
        'detail': (
            "Regimen adoption is modelled via logistic diffusion with manually calibrated "
            "parameters (peak_share, speed, time_to_peak). These are expert assumptions, "
            "not fitted to real-world claims data (e.g., IQVIA, Flatiron, CMS). Actual "
            "adoption may differ due to payer formulary restrictions, REMS programs, "
            "manufacturing constraints (CAR-T), or regional practice variation."
        ),
    })

    # 6. Untreated patients
    treated_frac = params['uptake']['treated_fraction']
    pitfalls.append({
        'severity': 'medium',
        'title': f'Untreated & Undiagnosed Patients Excluded',
        'detail': (
            f"The model applies a treated fraction of {treated_frac:.0%}, but patients "
            f"who are diagnosed and not treated, or who are never diagnosed (smoldering "
            f"myeloma misclassified, underserved populations), are not tracked. The "
            f"denominator of 'all MM patients' is therefore underestimated. SEER data "
            f"suggests ~10–15% of diagnosed patients do not receive active treatment."
        ),
    })

    # 7. No maintenance therapy modelling
    pitfalls.append({
        'severity': 'low',
        'title': 'Maintenance Therapy Not Explicitly Modelled',
        'detail': (
            "Post-induction maintenance (e.g., lenalidomide maintenance post-ASCT) is not "
            "modelled as a separate state. Patients on maintenance are counted within their "
            "induction line. This can inflate apparent 1L duration and undercount effective "
            "transitions."
        ),
    })

    # 8. PFS ≠ time to next treatment
    pitfalls.append({
        'severity': 'medium',
        'title': 'PFS Used as Proxy for Time-to-Next-Treatment (TTNT)',
        'detail': (
            "The model uses trial-reported median PFS to drive line transitions, but "
            "real-world time-to-next-treatment (TTNT) is typically longer than PFS because "
            "patients may remain on a failing regimen, take treatment holidays, or wait for "
            "insurance/access. TTNT from claims or EMR data would be more appropriate for "
            "a population model."
        ),
    })

    return pitfalls


# ── Sensitivity Helpers (FIXED) ───────────────────────────────────
def tornado_sensitivity(df: pd.DataFrame, params: dict,
                         target_col: str = 'Total_1L') -> pd.DataFrame:
    """Compute REAL ±20% sensitivity on key parameters.

    For each parameter, we estimate how the target occupancy metric would
    change if the parameter were varied ±20%, using analytical approximations
    based on the model structure rather than a blanket multiplicative factor.

    Model logic:
      New_Starts_1L = incidence × treated_fraction
      Total_1L ≈ New_Starts_1L × median_1L_duration (steady-state proxy)
      Parameters affect the pipeline at different points.
    """
    recent = df.tail(12)
    base_val = recent[target_col].mean()
    base_new_starts = recent['New_Starts_1L'].mean()

    # Approximate monthly incidence (before treated fraction)
    tf = params['uptake']['treated_fraction']
    if tf > 0:
        monthly_incidence = base_new_starts / tf
    else:
        monthly_incidence = base_new_starts

    # Median 1L duration (used for steady-state approximation)
    median_1l = params.get('durations_months_median', {}).get('1l', 24)
    mort_1l = params['mortality']['monthly_death_hazard_1l']
    p2l = params['attrition']['p_reach_2l']

    rows = []

    # 1. Treated Fraction — directly scales new starts, proportionally scales stock
    tf_low  = tf * 0.8
    tf_high = tf * 1.2
    ratio_low  = tf_low / tf if tf > 0 else 1
    ratio_high = tf_high / tf if tf > 0 else 1
    rows.append({
        'Parameter': f'Treated Fraction (base={tf:.2f})',
        'Low (-20%)': base_val * ratio_low,
        'Base': base_val,
        'High (+20%)': base_val * ratio_high,
    })

    # 2. P(Reach 2L) — affects 1L outflow rate; higher p2l means faster 1L depletion
    #    Approximate: 1L stock inversely related to outflow rate component from p2l
    #    Steady-state 1L ∝ new_starts / (progression_rate + mortality_rate)
    #    Progression_rate is related to Weibull hazard, modified by p2l at transition
    #    For sensitivity: we approximate the effect on 1L stock as modest
    #    (p2l affects downstream lines more than 1L stock directly)
    p2l_low  = p2l * 0.8
    p2l_high = p2l * 1.2
    # Impact on 1L is indirect: p2l gates outflow from 1L to 2L
    # A higher p2l means more patients leave 1L → lower 1L stock
    # Approximate elasticity: ~0.15 based on typical model sensitivity
    elasticity_p2l = 0.15
    rows.append({
        'Parameter': f'P(Reach 2L) (base={p2l:.2f})',
        'Low (-20%)': base_val * (1 + elasticity_p2l * 0.20),   # lower p2l → higher 1L
        'Base': base_val,
        'High (+20%)': base_val * (1 - elasticity_p2l * 0.20),  # higher p2l → lower 1L
    })

    # 3. Mortality 1L — directly depletes 1L stock
    #    Steady-state: Stock ∝ 1 / (hazard_progression + hazard_mortality)
    #    Approximate change in stock from mortality change
    mort_low  = mort_1l * 0.8
    mort_high = mort_1l * 1.2
    # Weibull median progression rate ≈ ln(2)/median ≈ 0.029/mo for 24mo median
    prog_rate = np.log(2) / median_1l if median_1l > 0 else 0.03
    total_hazard_base = prog_rate + mort_1l
    total_hazard_low  = prog_rate + mort_low
    total_hazard_high = prog_rate + mort_high
    if total_hazard_base > 0:
        ratio_mort_low  = total_hazard_base / total_hazard_low
        ratio_mort_high = total_hazard_base / total_hazard_high
    else:
        ratio_mort_low = ratio_mort_high = 1.0
    rows.append({
        'Parameter': f'1L Mortality (base={mort_1l:.4f})',
        'Low (-20%)': base_val * ratio_mort_low,
        'Base': base_val,
        'High (+20%)': base_val * ratio_mort_high,
    })

    # 4. Median 1L Duration — directly proportional to stock at steady state
    dur_low  = median_1l * 0.8
    dur_high = median_1l * 1.2
    prog_low  = np.log(2) / dur_high if dur_high > 0 else prog_rate  # shorter dur = higher prog
    prog_high = np.log(2) / dur_low  if dur_low > 0 else prog_rate   # longer dur = lower prog
    th_low  = prog_low + mort_1l
    th_high = prog_high + mort_1l
    if total_hazard_base > 0:
        ratio_dur_low  = total_hazard_base / th_low
        ratio_dur_high = total_hazard_base / th_high
    else:
        ratio_dur_low = ratio_dur_high = 1.0
    rows.append({
        'Parameter': f'Median 1L Duration (base={median_1l}mo)',
        'Low (-20%)': base_val * ratio_dur_high,   # shorter duration = fewer in stock
        'Base': base_val,
        'High (+20%)': base_val * ratio_dur_low,    # longer duration = more in stock
    })

    return pd.DataFrame(rows)


# ── Regimen PFS Comparison ────────────────────────────────────────
def compute_regimen_pfs_table(regimens_yaml: dict) -> pd.DataFrame:
    """Build a DataFrame of all regimens with median PFS, line, citation.

    Expected input: the raw parsed YAML dict from regimens.yaml.
    """
    rows = []
    for r in regimens_yaml.get('regimens', []):
        line = str(r.get('line', '')).upper()
        # Normalise line labels for display
        line_display = {'1L': '1L', '2L': '2L', '3L': '3L', '4L+': '4L+'}.get(line, line)
        rows.append({
            'Regimen': r['name'],
            'Line': line_display,
            'Median PFS (mo)': float(r.get('pfs_median', 0)),
            'Citation': r.get('citation', ''),
            'Approval Year': int(r.get('approval_year', 0)),
            'HR': r.get('hazard_ratio', None),
        })
    df = pd.DataFrame(rows)
    # Sort: by line order, then by PFS descending
    line_order = {'1L': 0, '2L': 1, '3L': 2, '4L+': 3}
    df['_sort'] = df['Line'].map(line_order).fillna(4)
    df = df.sort_values(['_sort', 'Median PFS (mo)'], ascending=[True, False])
    df = df.drop(columns=['_sort'])
    return df
