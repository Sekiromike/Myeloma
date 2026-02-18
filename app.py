"""
Multiple Myeloma Disease Model Explorer — v3
=============================================
Executive-grade analytical dashboard.
Run:  streamlit run Myeloma/app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import yaml, io

# ── Local imports ────────────────────────────────────────────────
from data_loader import (load_simulation, load_params, load_regimens_yaml,
                          get_regimen_columns, auto_discover_paths,
                          TOTAL_COLS, LINE_LABELS)
from model_metrics import (compute_occupancy, compute_transition_rates,
                     compute_regimen_shares, compute_sankey_data,
                     compute_duration_stats, generate_insights,
                     tornado_sensitivity, compute_yoy_trends,
                     compute_novel_therapy_pct, compute_incidence_cagr,
                     generate_executive_summary, generate_pitfalls,
                     compute_regimen_pfs_table)

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="MM Disease Model Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Executive Theme CSS ──────────────────────────────────────────
st.markdown("""
<style>
/* ============================================================
   Executive Consulting Theme
   Palette:  white (#ffffff), light gray (#f7f8fa, #eef0f4),
             mid gray (#6b7280), dark (#1e293b),
             navy accent (#1e3a5f), blue accent (#2563eb),
             red highlight (#dc2626), green (#059669)
   ============================================================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Reset ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1e293b;
}

/* ── Force light background on the Streamlit main area ──────────────── */
.stApp {
    background-color: #ffffff;
}

/* ── Sidebar ────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #f7f8fa;
    border-right: 1px solid #e5e7eb;
}

/* ── Global Text ────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1e293b;
}

/* ── Header ─────────────────────────────────────────────────────────── */
.main-header {
    border-bottom: 3px solid #1e3a5f;
    padding-bottom: 0.8rem;
    margin-bottom: 1.8rem;
}
.main-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: #1e3a5f;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}
.sub-title {
    font-size: 0.88rem;
    color: #6b7280;
    font-weight: 400;
    margin-top: 0.25rem;
    letter-spacing: 0.01em;
}

/* ── Metric Cards ───────────────────────────────────────────────────── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1.2rem 1.3rem;
    transition: box-shadow 0.2s ease;
    min-height: 165px;
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border-color: #d1d5db;
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin: 0 0 0.1rem 0;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1e293b;
    margin: 0;
    line-height: 1.15;
}
.metric-trend {
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.25rem;
}
.metric-trend.up   { color: #059669; }
.metric-trend.down { color: #dc2626; }
.metric-trend.flat { color: #9ca3af; }
.metric-explain {
    font-size: 0.73rem;
    color: #9ca3af;
    margin-top: 0.4rem;
    line-height: 1.45;
    font-weight: 400;
    border-top: 1px solid #f3f4f6;
    padding-top: 0.4rem;
}

/* ── Executive Summary ──────────────────────────────────────────────── */
.exec-summary {
    background: #f7f8fa;
    border-left: 4px solid #1e3a5f;
    border-radius: 0 6px 6px 0;
    padding: 1.1rem 1.4rem;
    font-size: 0.95rem;
    color: #374151;
    line-height: 1.7;
    margin: 0.5rem 0 2rem 0;
}

/* ── Section headers ────────────────────────────────────────────────── */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e3a5f;
    margin: 2rem 0 0.3rem 0;
    letter-spacing: -0.01em;
}
.section-subtitle {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-bottom: 0.8rem;
    line-height: 1.5;
    font-weight: 400;
}

/* ── Insight boxes ──────────────────────────────────────────────────── */
.insight-box {
    background: #f7f8fa;
    border-left: 3px solid #2563eb;
    border-radius: 0 6px 6px 0;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    color: #374151;
    line-height: 1.55;
}

/* ── Pitfall Cards ──────────────────────────────────────────────────── */
.pitfall-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.5rem;
}
.pitfall-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.15rem 0.45rem;
    border-radius: 3px;
    margin-right: 0.5rem;
    vertical-align: middle;
}
.badge-high   { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.badge-medium { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
.badge-low    { background: #f0fdf4; color: #059669; border: 1px solid #bbf7d0; }
.pitfall-title {
    font-weight: 600;
    font-size: 0.85rem;
    color: #1e293b;
    display: inline;
    vertical-align: middle;
}
.pitfall-detail {
    font-size: 0.8rem;
    color: #6b7280;
    line-height: 1.55;
    margin-top: 0.35rem;
}

/* ── Disclaimer ─────────────────────────────────────────────────────── */
.disclaimer {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    font-size: 0.8rem;
    color: #991b1b;
    margin-top: 1.5rem;
}

/* ── Timeline ───────────────────────────────────────────────────────── */
.timeline-event {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid #f3f4f6;
}
.timeline-dot {
    min-width: 8px; height: 8px; border-radius: 50%;
    background: #2563eb;
    margin-top: 0.4rem;
}
.timeline-date {
    font-size: 0.75rem;
    color: #9ca3af;
    min-width: 80px;
    font-weight: 500;
}
.timeline-name {
    font-size: 0.85rem;
    color: #1e293b;
    font-weight: 600;
}
.timeline-detail {
    font-size: 0.75rem;
    color: #6b7280;
}

/* ── Section Divider ────────────────────────────────────────────────── */
.section-divider {
    height: 1px;
    margin: 2rem 0;
    background: #e5e7eb;
}

/* ── Tab styling ────────────────────────────────────────────────────── */
div[data-baseweb="tab-list"] {
    gap: 2rem;
    border-bottom: 2px solid #e5e7eb;
    margin-bottom: 1rem;
}
div[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.9rem;
    color: #6b7280;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1e3a5f !important;
    border-bottom-color: #1e3a5f !important;
}

/* ── Hide Streamlit Branding ──────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] { background: #ffffff !important; }

/* ── Custom Streamlit overrides ───────────────────────────── */
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

.stExpander {
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
}
.stExpander summary span {
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton > button,
.stDownloadButton > button {
    background-color: #ffffff;
    color: #1e3a5f;
    border: 1px solid #d1d5db;
    font-weight: 600;
    font-size: 0.85rem;
    border-radius: 6px;
    padding: 0.4rem 1rem;
    transition: all 0.15s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #f7f8fa;
    border-color: #2563eb;
    color: #2563eb;
}
.stButton > button:active {
    background-color: #eef0f4;
}

/* ── Sliders ──────────────────────────────────────────────────────── */
div[data-baseweb="slider"] {
    color: #1e293b;
}

/* ── Selectbox / Inputs ──────────────────────────────────────────── */
div[data-baseweb="select"] {
    background-color: #ffffff;
    border-color: #d1d5db;
}

/* ── Sidebar widgets ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stButton > button {
    background-color: #1e3a5f;
    color: #ffffff;
    border: none;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #2563eb;
    color: #ffffff;
}

/* ── DataFrames / Tables ─────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
}

/* ── Markdown text color ─────────────────────────────────────────── */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5 {
    color: #1e293b;
}
.stMarkdown code {
    background-color: #f3f4f6;
    color: #1e293b;
    border-radius: 4px;
    padding: 0.1rem 0.3rem;
}


</style>
""", unsafe_allow_html=True)

# ── Colour Palette (muted, consulting-grade) ─────────────────────
PAL = {
    '1st Line':  '#2563eb',   # blue
    '2nd Line':  '#d97706',   # amber
    '3rd Line':  '#dc2626',   # red
    '4th Line+': '#7c3aed',   # violet
}

# Muted pastel set for regimen shares
REGIMEN_PAL = [
    '#93c5fd', '#fbbf24', '#fca5a5', '#c4b5fd',
    '#6ee7b7', '#fdba74', '#a5b4fc', '#86efac',
    '#f9a8d4', '#fde68a', '#bae6fd', '#d9f99d',
    '#e9d5ff', '#a7f3d0', '#fed7aa', '#c7d2fe',
]

# ── Plotly Chart Defaults ────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff',
    font=dict(family='Inter, sans-serif', size=12, color='#374151'),
    margin=dict(l=50, r=20, t=30, b=40),
    hovermode='x unified',
)

def _apply_axes(fig, **kw):
    """Apply clean executive axis styling."""
    fig.update_xaxes(
        showgrid=False,
        linecolor='#e5e7eb',
        linewidth=1,
        tickfont=dict(size=11, color='#6b7280'),
        **{k: v for k, v in kw.items() if k.startswith('x_')},
    )
    fig.update_yaxes(
        gridcolor='#f3f4f6',
        gridwidth=1,
        linecolor='#e5e7eb',
        linewidth=1,
        tickfont=dict(size=11, color='#6b7280'),
        zeroline=False,
    )
    return fig


# ── Data Loading ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_all():
    base = Path(__file__).resolve().parent
    paths = auto_discover_paths(base)
    if 'simulation' not in paths:
        import os
        raise FileNotFoundError(
            f"Could not find simulation CSV. Discovered: {paths}")
    if 'params' not in paths:
        raise FileNotFoundError(f"params.yaml not found. Discovered: {paths}")
    df   = load_simulation(paths['simulation'])
    par  = load_params(paths['params'])
    regs = load_regimens_yaml(paths['regimens']) if 'regimens' in paths else {}
    evts = load_regimens_yaml(paths['events']) if 'events' in paths else {}
    return df, par, regs, evts, paths

try:
    df, params, regimens_yaml, events_yaml, data_paths = _load_all()
except FileNotFoundError as e:
    st.error(f"Data files not found: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Controls")

    min_date = df['Date'].min().to_pydatetime()
    max_date = df['Date'].max().to_pydatetime()
    date_range = st.slider("Date range", min_value=min_date, max_value=max_date,
                           value=(min_date, max_date), format="YYYY-MM")
    mask = (df['Date'] >= pd.Timestamp(date_range[0])) & \
           (df['Date'] <= pd.Timestamp(date_range[1]))
    dff = df.loc[mask].copy()

    st.markdown("---")
    st.markdown("### Model Parameters")
    treated_frac = st.slider("Treated fraction",
                             0.50, 1.00, float(params['uptake']['treated_fraction']),
                             step=0.05,
                             help="Share of newly diagnosed patients who initiate active treatment")
    p_2l = st.slider("P(reach 2L)",
                      0.30, 1.00, float(params['attrition']['p_reach_2l']),
                      step=0.05,
                      help="Probability of advancing from first-line to second-line therapy")
    p_3l = st.slider("P(reach 3L | 2L)",
                      0.20, 1.00, float(params['attrition'].get('p_reach_3l_given_2l', 0.6)),
                      step=0.05,
                      help="Probability of advancing from second-line to third-line therapy")

    scenario_params = {
        'uptake': {**params['uptake'], 'treated_fraction': treated_frac},
        'attrition': {**params['attrition'], 'p_reach_2l': p_2l,
                      'p_reach_3l_given_2l': p_3l},
        'mortality': params['mortality'],
        'durations_months_median': params.get('durations_months_median', {}),
    }

    st.markdown("---")
    st.markdown("### Scenario Snapshots")
    if 'saved_scenarios' not in st.session_state:
        st.session_state.saved_scenarios = []

    if st.button("Save Current Scenario", use_container_width=True):
        recent = dff.tail(12)
        snap = {
            'name': f"Scenario {len(st.session_state.saved_scenarios) + 1}",
            'treated_frac': treated_frac, 'p_2l': p_2l, 'p_3l': p_3l,
            'total_on_tx': sum(recent.get(c, pd.Series([0])).mean() for c in TOTAL_COLS),
            'new_starts': recent['New_Starts_1L'].mean(),
            'prevalence_1l': recent.get('Total_1L', pd.Series([0])).mean(),
            'novel_pct': compute_novel_therapy_pct(dff),
        }
        st.session_state.saved_scenarios.append(snap)
        if len(st.session_state.saved_scenarios) > 3:
            st.session_state.saved_scenarios = st.session_state.saved_scenarios[-3:]
        st.success(f"Saved {snap['name']}")

    if st.session_state.saved_scenarios and st.button("Clear All", use_container_width=True):
        st.session_state.saved_scenarios = []
        st.rerun()

    st.markdown("---")
    st.caption("**Source:** CDC USCS 1999-2022  \n"
               "**Model:** Weibull PFS · Logistic Adoption  \n"
               "**Projection:** Flat beyond last observed year")


# ══════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="main-header">'
    '<p class="main-title">Multiple Myeloma Disease Model</p>'
    '<p class="sub-title">US population-level treatment flow analysis '
    '&middot; Research use only</p>'
    '</div>',
    unsafe_allow_html=True
)


# ══════════════════════════════════════════════════════════════════
#  KPI ROW — with plain-English explanations
# ══════════════════════════════════════════════════════════════════
trends = compute_yoy_trends(dff)
novel_pct = compute_novel_therapy_pct(dff)

recent = dff.tail(12)
kpi_total = trends['total_on_treatment']['current']
kpi_new   = trends['new_starts']['current']
kpi_1l    = trends['prevalence_1l']['current']
pct_1l = (kpi_1l / kpi_total * 100) if kpi_total > 0 else 0

def _trend_html(trend_info):
    if trend_info['delta_pct'] is None:
        return '<div class="metric-trend flat">— No prior-year data</div>'
    arrow = trend_info['arrow']
    delta = trend_info['delta_pct']
    cls = 'up' if arrow == '▲' else ('down' if arrow == '▼' else 'flat')
    return f'<div class="metric-trend {cls}">{arrow} {delta:+.1f}% vs. prior year</div>'

k1, k2, k3, k4 = st.columns(4)

# Card 1 — Total on Treatment
k1.markdown(
    '<div class="metric-card">'
    '<div class="metric-label">Total Patients on Treatment</div>'
    f'<div class="metric-value">{kpi_total:,.0f}</div>'
    f'{_trend_html(trends["total_on_treatment"])}'
    '<div class="metric-explain">'
    'Estimated number of US patients currently receiving active myeloma therapy '
    'across all treatment lines (1L through 4L+), based on the trailing 12-month average.'
    '</div></div>',
    unsafe_allow_html=True
)

# Card 2 — Monthly New 1L Starts
annual_inc = kpi_new * 12 / treated_frac if treated_frac > 0 else kpi_new * 12
k2.markdown(
    '<div class="metric-card">'
    '<div class="metric-label">New Patients Starting Treatment / Month</div>'
    f'<div class="metric-value">{kpi_new:,.0f}</div>'
    f'{_trend_html(trends["new_starts"])}'
    '<div class="metric-explain">'
    f'Each month, approximately <b>{kpi_new:,.0f}</b> Americans begin first-line myeloma '
    f'treatment. Derived from ~{annual_inc:,.0f} annual diagnoses (CDC USCS) '
    f'&times; {treated_frac:.0%} treated fraction &divide; 12 months.'
    '</div></div>',
    unsafe_allow_html=True
)

# Card 3 — 1L Prevalence
k3.markdown(
    '<div class="metric-card">'
    '<div class="metric-label">Patients Currently on First-Line Therapy</div>'
    f'<div class="metric-value">{kpi_1l:,.0f}</div>'
    f'{_trend_html(trends["prevalence_1l"])}'
    '<div class="metric-explain">'
    f'Number of patients actively on their initial treatment regimen at any given time. '
    f'Represents {pct_1l:.0f}% of all patients on treatment.'
    '</div></div>',
    unsafe_allow_html=True
)

# Card 4 — Novel Therapy %
k4.markdown(
    '<div class="metric-card">'
    '<div class="metric-label">Patients on Novel Therapies</div>'
    f'<div class="metric-value">{novel_pct:.1f}%</div>'
    '<div class="metric-explain">'
    'Share of on-treatment patients receiving agents approved after 2020 '
    '(e.g., D-VRd quadruplet, CAR-T cell therapy, bispecific antibodies such as '
    'teclistamab and elranatamab).'
    '</div></div>',
    unsafe_allow_html=True
)

# ── Executive Summary ────────────────────────────────────────────
exec_text = generate_executive_summary(dff, scenario_params)
st.markdown(f'<div class="exec-summary">{exec_text}</div>',
            unsafe_allow_html=True)

# ── Scenario Comparison ──────────────────────────────────────────
if st.session_state.get('saved_scenarios'):
    with st.expander("Scenario Comparison", expanded=False):
        snap_data = []
        for s in st.session_state.saved_scenarios:
            snap_data.append({
                'Scenario': s['name'],
                'Treated %': f"{s['treated_frac']:.0%}",
                'P(2L)': f"{s['p_2l']:.0%}",
                'P(3L|2L)': f"{s['p_3l']:.0%}",
                'Total on Tx': f"{s['total_on_tx']:,.0f}",
                'New 1L/mo': f"{s['new_starts']:,.0f}",
                '1L Prevalence': f"{s['prevalence_1l']:,.0f}",
                'Novel %': f"{s['novel_pct']:.1f}%",
            })
        snap_data.append({
            'Scenario': 'Current',
            'Treated %': f"{treated_frac:.0%}",
            'P(2L)': f"{p_2l:.0%}",
            'P(3L|2L)': f"{p_3l:.0%}",
            'Total on Tx': f"{kpi_total:,.0f}",
            'New 1L/mo': f"{kpi_new:,.0f}",
            '1L Prevalence': f"{kpi_1l:,.0f}",
            'Novel %': f"{novel_pct:.1f}%",
        })
        st.dataframe(pd.DataFrame(snap_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════
tab_patient, tab_provider = st.tabs(["Patient View", "Provider View"])


# ══════════════════════════════════════════════════════════════════
#  PATIENT VIEW — Educational, simplified
# ══════════════════════════════════════════════════════════════════
with tab_patient:

    with st.expander("What is this model?", expanded=False):
        st.markdown("""
        This is a **population-level simulation** of how patients with Multiple
        Myeloma move through lines of treatment over time. It is based on
        published clinical evidence and US incidence data (CDC USCS, 1999-2022).

        **It is NOT a prediction for any individual patient.** It helps
        researchers, clinicians, and policymakers understand population-level
        trends and the potential impact of new therapies.

        **Data sources:** CDC United States Cancer Statistics (USCS), published
        Phase III trial results, and NCCN treatment guidelines.
        """)

    # ── Treatment Journey (Sankey) ───────────────────────────────
    st.markdown('<div class="section-header">Treatment Journey</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">'
                'How patients flow through sequential lines of therapy. '
                'Width of each band is proportional to the number of patients. '
                'Based on trailing 12-month averages.</div>',
                unsafe_allow_html=True)

    sankey = compute_sankey_data(dff, scenario_params)
    node_colors = [PAL['1st Line'], PAL['1st Line'], PAL['2nd Line'],
                   PAL['3rd Line'], '#059669', '#9ca3af']

    fig_sankey = go.Figure(go.Sankey(
        arrangement='snap',
        textfont=dict(family='Inter, sans-serif', size=14, color='#1e293b'),
        node=dict(
            pad=20, thickness=24,
            line=dict(color='#e5e7eb', width=1),
            label=sankey['nodes'],
            color=node_colors,
        ),
        link=dict(
            source=sankey['link_source'],
            target=sankey['link_target'],
            value=sankey['link_value'],
            customdata=sankey['link_label'],
            hovertemplate='%{customdata}<extra></extra>',
            color='rgba(37,99,235,0.12)',
        ),
    ))
    fig_sankey.update_layout(
        height=320, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='#ffffff',
        font=dict(size=13, color='#374151', family='Inter'),
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Patients by Treatment Line ───────────────────────────────
    st.markdown('<div class="section-header">Patients by Treatment Line Over Time</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">'
                'This stacked area chart shows how many patients are actively '
                'receiving treatment in each line at any point in time. '
                'First-line (1L) is the initial treatment after diagnosis; '
                'subsequent lines (2L, 3L, 4L+) represent treatments given '
                'after the disease progresses.'
                '</div>', unsafe_allow_html=True)

    occ = compute_occupancy(dff)
    fig_occ = px.area(occ, x='Date', y='Patients', color='Line',
                      color_discrete_map=PAL,
                      category_orders={'Line': list(PAL.keys())})
    fig_occ.update_layout(
        height=380,
        **CHART_LAYOUT,
        yaxis_title='Number of Patients',
        legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center',
                    font=dict(size=12)),
    )
    _apply_axes(fig_occ)
    st.plotly_chart(fig_occ, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Treatment Landscape (PFS chart) ──────────────────────────
    st.markdown('<div class="section-header">Treatment Landscape</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">'
                'Available treatment regimens at each line, ranked by median '
                'progression-free survival (PFS) — the typical time before the '
                'disease worsens. Longer bars indicate longer disease control. '
                'Values are from pivotal clinical trials.'
                '</div>', unsafe_allow_html=True)

    pfs_df = compute_regimen_pfs_table(regimens_yaml)
    if not pfs_df.empty:
        line_colors = {'1L': PAL['1st Line'], '2L': PAL['2nd Line'],
                       '3L': PAL['3rd Line'], '4L+': PAL['4th Line+']}

        fig_pfs = go.Figure()
        for line in ['1L', '2L', '3L', '4L+']:
            sub = pfs_df[pfs_df['Line'] == line]
            if sub.empty:
                continue
            fig_pfs.add_trace(go.Bar(
                y=sub['Regimen'], x=sub['Median PFS (mo)'],
                orientation='h', name=line,
                marker_color=line_colors.get(line, '#9ca3af'),
                text=[f"{v:.0f} mo" for v in sub['Median PFS (mo)']],
                textposition='outside',
                textfont=dict(size=11, color='#6b7280'),
                hovertemplate='<b>%{y}</b><br>Median PFS: %{x:.1f} months<extra></extra>',
            ))

        fig_pfs.update_layout(
            height=max(350, len(pfs_df) * 28),
            barmode='group',
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            font=dict(family='Inter, sans-serif', size=12, color='#374151'),
            hovermode='x unified',
            xaxis=dict(title='Median PFS (months)', showgrid=True,
                       gridcolor='#f3f4f6',
                       range=[0, pfs_df['Median PFS (mo)'].max() * 1.25]),
            yaxis=dict(title='', autorange='reversed', categoryorder='array',
                       categoryarray=pfs_df['Regimen'].tolist()),
            legend=dict(orientation='h', y=1.06, x=0.5, xanchor='center'),
            margin=dict(l=110, r=60, t=40, b=40),
        )
        _apply_axes(fig_pfs)
        st.plotly_chart(fig_pfs, use_container_width=True)
        st.caption("PFS values are from pivotal Phase III trials. Real-world outcomes "
                   "may differ. See Provider View for citations.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── FDA Approval Timeline ────────────────────────────────────
    st.markdown('<div class="section-header">Recent FDA Approvals in Myeloma</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">'
                'Key regulatory milestones shaping the current treatment landscape.'
                '</div>', unsafe_allow_html=True)

    events = events_yaml.get('events', [])
    if events:
        sorted_events = sorted(events, key=lambda e: e.get('date', ''), reverse=True)[:8]
        for ev in sorted_events:
            st.markdown(
                f'<div class="timeline-event">'
                f'<div class="timeline-dot"></div>'
                f'<div>'
                f'<span class="timeline-date">{ev.get("date", "")}</span> '
                f'<span class="timeline-name">{ev.get("name", "")}</span><br>'
                f'<span class="timeline-detail">{ev.get("details", "")}</span>'
                f'</div></div>',
                unsafe_allow_html=True
            )
        st.caption("Source: FDA.gov approval announcements")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Key Takeaways ────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Takeaways</div>',
                unsafe_allow_html=True)
    for ins in generate_insights(dff, scenario_params, 'patient'):
        st.markdown(f'<div class="insight-box">{ins}</div>',
                    unsafe_allow_html=True)

    # ── Questions for Your Doctor ────────────────────────────────
    with st.expander("Questions to Ask Your Doctor"):
        st.markdown("""
        1. *What treatment line am I currently on, and what does that mean?*
        2. *What are the newer therapy options available for my stage?*
        3. *How long do patients typically stay on my current treatment?*
        4. *What factors determine whether I would move to a next line of therapy?*
        5. *Are there clinical trials I might be eligible for?*
        """)

    # ── Disclaimer ───────────────────────────────────────────────
    st.markdown(
        '<div class="disclaimer">'
        '<b>Disclaimer:</b> This tool is for educational and research purposes '
        'only. It is not medical advice and does not predict individual patient '
        'outcomes. Always consult your healthcare provider for decisions about '
        'your care.</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════
#  PROVIDER VIEW — Technical, decision-support
# ══════════════════════════════════════════════════════════════════
with tab_provider:

    # ── Model Transparency ───────────────────────────────────────
    with st.expander("Model Transparency & Methodology", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Active Parameters**")
            st.json(scenario_params, expanded=False)
        with c2:
            st.markdown("**Model Structure**")
            st.code(
                "Incidence (USCS) → Dx → Treated (× frac) → 1L\n"
                "                                            ↓ Weibull PFS\n"
                "                                           2L\n"
                "                                            ↓ Weibull PFS\n"
                "                                           3L → 4L+\n"
                "Each state: competing mortality (constant hazard)\n"
                "Adoption:  S(t) = peak / (1 + exp(-k·(t - t₀)))\n"
                "PFS:       h(t) = (k/λ)·(t/λ)^(k-1), k=1.3",
                language=None
            )
            st.markdown("**Key Assumptions**")
            st.markdown("""
            - Weibull PFS with universal shape k = 1.3
            - Calibrated logistic adoption (not claims-fitted)
            - Sequential lines: 1L → 2L → 3L → 4L+ (no re-treatment)
            - Line-specific constant monthly mortality hazard
            - USCS incidence 1999-2022, flat-projected to 2026
            - Homogeneous population (no age/sex/race stratification)
            """)

    # ── Prevalence / Occupancy ───────────────────────────────────
    st.markdown('<div class="section-header">Prevalence by Treatment Line</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">'
                'Number of patients actively on treatment at each time point. '
                'Driven by incidence inflow, Weibull progression hazards, and '
                'line-specific mortality. The sharp ramp-up (1999-2005) reflects '
                'the model initialization period.'
                '</div>', unsafe_allow_html=True)

    occ_prov = compute_occupancy(dff)
    fig_prov_occ = px.area(occ_prov, x='Date', y='Patients', color='Line',
                           color_discrete_map=PAL,
                           category_orders={'Line': list(PAL.keys())})
    fig_prov_occ.update_layout(
        height=400, **CHART_LAYOUT,
        yaxis_title='Patient Count',
        legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'),
    )
    _apply_axes(fig_prov_occ)
    st.plotly_chart(fig_prov_occ, use_container_width=True)

    # ── Transition Rates ─────────────────────────────────────────
    st.markdown('<div class="section-header">Monthly Transition Rates</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">'
                'Approximate rate = change in next-line stock (clipped &ge; 0) '
                '&divide; current-line stock. Smoothed with a 6-month rolling '
                'mean. Note: this is a proxy derived from stock-level changes, '
                'not a true hazard estimate.'
                '</div>', unsafe_allow_html=True)

    tr = compute_transition_rates(dff)
    rate_cols = [c for c in tr.columns if '→' in c]
    if rate_cols:
        tr_smooth = tr.copy()
        for c in rate_cols:
            tr_smooth[c] = tr_smooth[c].rolling(6, min_periods=1).mean()
        fig_tr = go.Figure()
        colors_tr = [PAL['1st Line'], PAL['2nd Line'], PAL['3rd Line']]
        for i, c in enumerate(rate_cols):
            fig_tr.add_trace(go.Scatter(
                x=tr_smooth['Date'], y=tr_smooth[c],
                name=c, mode='lines',
                line=dict(width=2, color=colors_tr[i % len(colors_tr)]),
                hovertemplate=f'{c}: %{{y:.4f}}<extra></extra>',
            ))
        fig_tr.update_layout(
            height=340, **CHART_LAYOUT,
            yaxis_title='Rate',
            yaxis_tickformat='.3f',
            legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
        )
        _apply_axes(fig_tr)
        st.plotly_chart(fig_tr, use_container_width=True)

    # ── Regimen Share Evolution ──────────────────────────────────
    st.markdown('<div class="section-header">Regimen Market Share Evolution</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">'
                'Percentage of patients on each regimen within the selected '
                'treatment line over time. Reflects modeled logistic adoption '
                'curves calibrated to approval dates and peak share estimates.'
                '</div>', unsafe_allow_html=True)

    line_sel = st.selectbox("Select treatment line", ['1L', '2L', '3L', '4L+'],
                            key='reg_line')
    shares = compute_regimen_shares(dff, line_sel)
    if not shares.empty:
        share_cols = [c for c in shares.columns if c != 'Date']
        fig_share = go.Figure()
        for idx, c in enumerate(share_cols):
            fig_share.add_trace(go.Scatter(
                x=shares['Date'], y=shares[c],
                name=c, stackgroup='one', mode='none',
                fillcolor=REGIMEN_PAL[idx % len(REGIMEN_PAL)],
                hovertemplate=f'{c}: %{{y:.1%}}<extra></extra>',
            ))
        fig_share.update_layout(
            height=400,
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            font=dict(family='Inter, sans-serif', size=12, color='#374151'),
            hovermode='x unified',
            yaxis_title='Market Share',
            yaxis_tickformat='.0%',
            legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
            margin=dict(l=60, r=20, t=50, b=40),
        )
        _apply_axes(fig_share)
        st.plotly_chart(fig_share, use_container_width=True)
    else:
        st.info(f"No regimen-level data for {line_sel}")

    # ── Efficacy Distribution ────────────────────────────────────
    st.markdown('<div class="section-header">Efficacy Distribution by Line</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">'
                'Distribution of median PFS values for all regimens in the database, '
                'grouped by line of therapy. Points represent individual regimens.'
                '</div>', unsafe_allow_html=True)

    pfs_dist = compute_regimen_pfs_table(regimens_yaml)
    if not pfs_dist.empty:
        # Enforce order
        cat_order = ['1L', '2L', '3L', '4L+']
        
        # Consistent colors
        line_colors = {
            '1L': PAL['1st Line'],
            '2L': PAL['2nd Line'],
            '3L': PAL['3rd Line'],
            '4L+': PAL['4th Line+']
        }
        
        # Create Faceted Bar Chart
        fig_bar = px.bar(pfs_dist, x='Median PFS (mo)', y='Regimen', color='Line',
                         color_discrete_map=line_colors,
                         facet_row='Line', text_auto='.0f',
                         category_orders={'Line': cat_order},
                         hover_data=['Approval Year', 'Citation'])
                         
        fig_bar.update_layout(
            height=800, # Taller to fit all regimens
            **CHART_LAYOUT,
            xaxis_title='Median PFS (Months)',
            showlegend=False,
        )
        
        # Improve Facet Labels and spacing
        fig_bar.update_yaxes(matches=None, showticklabels=True)
        fig_bar.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        _apply_axes(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Regimen PFS Table ────────────────────────────────────────
    st.markdown('<div class="section-header">Regimen PFS Comparison</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">'
                'Median progression-free survival from pivotal trials, used to '
                'parameterise the Weibull survival curves in this model.'
                '</div>', unsafe_allow_html=True)

    pfs_prov = compute_regimen_pfs_table(regimens_yaml)
    if not pfs_prov.empty:
        display_cols = ['Regimen', 'Line', 'Median PFS (mo)', 'Approval Year', 'Citation']
        if 'HR' in pfs_prov.columns:
            pfs_prov['HR'] = pfs_prov['HR'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            display_cols.insert(4, 'HR')
        st.dataframe(pfs_prov[display_cols], use_container_width=True,
                     hide_index=True, height=min(400, 40 + len(pfs_prov) * 35))

    # ── Mortality Panel ──────────────────────────────────────────
    st.markdown('<div class="section-header">Mortality Hazard by Line</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">'
                'Constant monthly hazard rates used in the model. '
                'Real mortality risk is time-varying and depends on patient '
                'fitness, prior therapies, and comorbidities.'
                '</div>', unsafe_allow_html=True)

    mort_data = pd.DataFrame([
        {'Line': '1L',  'Monthly Hazard': params['mortality']['monthly_death_hazard_1l']},
        {'Line': '2L',  'Monthly Hazard': params['mortality']['monthly_death_hazard_2l']},
        {'Line': '3L+', 'Monthly Hazard': params['mortality']['monthly_death_hazard_3l_plus']},
    ])
    fig_mort = px.bar(mort_data, x='Line', y='Monthly Hazard', color='Line',
                      color_discrete_map={'1L': PAL['1st Line'],
                                          '2L': PAL['2nd Line'],
                                          '3L+': PAL['3rd Line']},
                      text_auto='.4f')
    fig_mort.update_layout(
        height=280, showlegend=False, **CHART_LAYOUT,
    )
    fig_mort.update_traces(textfont_size=12, textfont_color='#374151')
    _apply_axes(fig_mort)
    st.plotly_chart(fig_mort, use_container_width=True)

    # ── Sensitivity Tornado ──────────────────────────────────────
    st.markdown('<div class="section-header">'
                'Sensitivity Analysis: 1L Occupancy (±20%)</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">'
                'Each parameter is varied ±20% independently using steady-state '
                'analytical approximations. Bar widths differ because each '
                'parameter has a different elasticity on 1L patient stock.'
                '</div>', unsafe_allow_html=True)

    tornado = tornado_sensitivity(dff, scenario_params, 'Total_1L')
    fig_torn = go.Figure()
    for _, row in tornado.iterrows():
        fig_torn.add_trace(go.Bar(
            y=[row['Parameter']], x=[row['High (+20%)'] - row['Base']],
            base=[row['Base']], orientation='h',
            marker_color='#2563eb', showlegend=False,
            hovertemplate=f"High: {row['High (+20%)']:,.0f}<extra></extra>",
        ))
        fig_torn.add_trace(go.Bar(
            y=[row['Parameter']], x=[row['Low (-20%)'] - row['Base']],
            base=[row['Base']], orientation='h',
            marker_color='#dc2626', showlegend=False,
            hovertemplate=f"Low: {row['Low (-20%)']:,.0f}<extra></extra>",
        ))
    fig_torn.update_layout(
        height=max(200, len(tornado) * 55),
        barmode='overlay',
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', size=12, color='#374151'),
        hovermode='x unified',
        xaxis_title='1L Occupancy (patients)',
        margin=dict(l=200, r=20, t=10, b=40),
    )
    _apply_axes(fig_torn)
    st.plotly_chart(fig_torn, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Incidence Trend ──────────────────────────────────────────
    st.markdown('<div class="section-header">Incidence Trend Analysis</div>',
                unsafe_allow_html=True)
    cagr = compute_incidence_cagr(dff)
    if cagr['start_year']:
        st.markdown(
            f"**Observed CAGR ({cagr['start_year']}-{cagr['end_year']}):** "
            f"{cagr['cagr']:+.2f}%  \n"
            f"Annual new starts: {cagr['start_annual']:,} → {cagr['end_annual']:,}"
        )
        if abs(cagr['cagr']) < 0.5:
            st.warning(
                "The observed CAGR is near zero, suggesting the incidence projection "
                "is effectively flat. SEER data indicates US MM incidence has been "
                "growing ~1-2% annually due to population aging. This model may "
                "underestimate future patient volumes."
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Provider Insights ────────────────────────────────────────
    st.markdown('<div class="section-header">Key Findings</div>',
                unsafe_allow_html=True)
    for ins in generate_insights(dff, scenario_params, 'provider'):
        st.markdown(f'<div class="insight-box">{ins}</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Pitfalls & Limitations ───────────────────────────────────
    st.markdown('<div class="section-header">'
                'Epidemiological Pitfalls & Model Limitations</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">'
                'The following limitations should be considered when interpreting '
                'model outputs. Items are ranked by severity.'
                '</div>', unsafe_allow_html=True)

    pitfalls = generate_pitfalls(dff, scenario_params)
    for p in pitfalls:
        sev = p['severity']
        badge_cls = {'high': 'badge-high', 'medium': 'badge-medium',
                     'low': 'badge-low'}[sev]
        badge_text = sev.upper()
        st.markdown(
            f'<div class="pitfall-card">'
            f'<span class="pitfall-badge {badge_cls}">{badge_text}</span>'
            f'<span class="pitfall-title">{p["title"]}</span>'
            f'<div class="pitfall-detail">{p["detail"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Export Data</div>',
                unsafe_allow_html=True)
    c_exp1, c_exp2 = st.columns(2)
    with c_exp1:
        csv = dff.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Data (CSV)", csv,
                           "mm_simulation_filtered.csv", "text/csv")
    with c_exp2:
        if not shares.empty:
            csv_s = shares.to_csv(index=False).encode('utf-8')
            st.download_button("Download Regimen Shares (CSV)", csv_s,
                               f"mm_regimen_shares_{line_sel}.csv", "text/csv")
