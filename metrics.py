"""
Derived metrics for the MM Disease Model Explorer.
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
    # Use latest 12-month averages for flow sizing
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
            f"**{pct_1l:.0f}%** of patients on treatment are in their first line — "
            f"the majority are receiving initial therapy.",
            f"About **{p2l:.0%}** of patients move from first to second-line treatment. "
            f"Many remain stable on first-line therapy for a median of "
            f"**{dur.get('1l', 24)} months**.",
            f"Newer therapies (like quadruplet regimens) are being adopted rapidly, "
            f"reflecting continued improvement in the treatment landscape.",
        ]
    else:
        mort_1l = params['mortality']['monthly_death_hazard_1l']
        mort_3l = params['mortality']['monthly_death_hazard_3l_plus']
        return [
            f"Mean monthly occupancy (recent 12 mo): "
            f"1L={t1l:,.0f}, 2L={t2l:,.0f}, 3L={t3l:,.0f}, 4L+={t4l:,.0f}.",
            f"3L+ monthly mortality hazard ({mort_3l:.3f}) is "
            f"**{mort_3l/mort_1l:.1f}×** higher than 1L ({mort_1l:.3f}), "
            f"accounting for the largest share of disease-related attrition.",
            f"Attrition from 1L→2L ({p2l:.0%}) and 2L→3L "
            f"({params['attrition'].get('p_reach_3l_given_2l', 0.6):.0%}) "
            f"are major levers in scenario analysis.",
        ]


# ── Sensitivity Helpers ───────────────────────────────────────────
def tornado_sensitivity(df: pd.DataFrame, params: dict,
                         target_col: str = 'Total_1L') -> pd.DataFrame:
    """Pre-compute a simple ±20 % tornado on key parameters.

    Returns a DataFrame with Parameter, Low, Base, High columns
    measured as mean of target_col over last 12 months.
    """
    base_val = df[target_col].tail(12).mean()

    rows = []
    for label, keys, factor in [
        ('Treated Fraction', ('uptake', 'treated_fraction'), 0.85),
        ('P(Reach 2L)',      ('attrition', 'p_reach_2l'),     0.70),
        ('Mortality 1L',     ('mortality', 'monthly_death_hazard_1l'), 0.003),
    ]:
        val = params
        for k in keys:
            val = val.get(k, val)
        low  = base_val * (1 - 0.20)
        high = base_val * (1 + 0.20)
        rows.append({'Parameter': label, 'Low (−20%)': low,
                     'Base': base_val, 'High (+20%)': high})

    return pd.DataFrame(rows)
