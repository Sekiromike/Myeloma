"""
Multiple Myeloma Forecast Studio — v4
=====================================
US epidemiology + line-of-therapy patient-flow forecast, re-simulated live.
Run:  streamlit run Myeloma/app.py
"""
from __future__ import annotations

import json
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import epidemiology as epi
import lot_model as lm
import model_metrics as mm

NOW_YEAR = 2026            # "today" for KPIs (data vintage: October 2026)
TODAY = pd.Timestamp('2026-10-01')
LAST_OBS_YEAR = 2022       # last year of observed USCS incidence

st.set_page_config(
    page_title="Multiple Myeloma Forecast Studio · US Epidemiology & Treatment Forecast",
    page_icon="📈", layout="wide", initial_sidebar_state="expanded",
    menu_items={'About': "Multiple Myeloma Forecast Studio: interactive US myeloma epidemiology and "
                         "line-of-therapy forecast to 2035. Methods and references: "
                         "https://sekiromike.github.io/multiple-myeloma.html"})

# ══════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════
INK, INK_2, INK_3 = '#0f1b2d', '#4b5565', '#8a909b'
GRID, AXIS, SURFACE = '#e8e8e3', '#c3c2b7', '#ffffff'
BRAND = '#12355b'
# Validated categorical palette (fixed slot order; see dataviz reference palette)
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
OTHER = '#b9b7ae'
LINE_COLORS = {'1st line': SLOTS[0], '2nd line': SLOTS[1], '3rd line': SLOTS[2], '4th line+': SLOTS[3]}
LINE_CODE_COLORS = dict(zip(mm.LINES, SLOTS[:4]))

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --ink:#0f1b2d; --ink-2:#4b5565; --ink-3:#8a909b; --line:#e6e7ea; --plane:#f5f6f8;
  --brand:#12355b; --accent:#2a78d6; --good:#0b7a3e; --bad:#b42318;
}
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif; color:var(--ink); }
.stApp { background:var(--plane); }
.block-container { padding-top:1.4rem; padding-bottom:3rem; max-width:1480px; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }

/* Sidebar */
section[data-testid="stSidebar"] { background:#0f1b2d; }
section[data-testid="stSidebar"] * { color:#e6e9ef; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] { background:#ffffff; border-color:#ffffff; }
section[data-testid="stSidebar"] [data-testid="stExpander"] { border:1px solid rgba(255,255,255,.12); border-radius:10px; background:rgba(255,255,255,.03); }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary { background:transparent !important; color:#e6e9ef !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background:rgba(255,255,255,.05) !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] details { background:transparent !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p { font-weight:600; font-size:.86rem; }
section[data-testid="stSidebar"] input { background:#1a2840 !important; color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; }
section[data-testid="stSidebar"] [data-baseweb="input"] { background:#1a2840 !important; border-color:rgba(255,255,255,.18) !important; }
[data-testid="stToolbar"] { display:none; }
section[data-testid="stSidebar"] .stButton > button { background:transparent; border:1px solid rgba(255,255,255,.28); color:#fff; border-radius:8px; font-weight:600; }
section[data-testid="stSidebar"] .stButton > button:hover { border-color:#fff; background:rgba(255,255,255,.08); }
section[data-testid="stSidebar"] small, section[data-testid="stSidebar"] .stCaption { color:#9aa4b2 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div { background:#1a2840; border-color:rgba(255,255,255,.18); }
.sb-brand { font-weight:800; font-size:1.02rem; letter-spacing:-.01em; margin:.2rem 0 .1rem; }
.sb-sub { font-size:.74rem; color:#9aa4b2 !important; margin-bottom:1rem; }
.sb-h { font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:#9aa4b2 !important; font-weight:700; margin:1.1rem 0 .4rem; }

/* Hero */
.hero { background:linear-gradient(135deg,#0f1b2d 0%,#12355b 60%,#1d4f86 100%); border-radius:16px; padding:1.5rem 1.8rem 1.3rem; color:#fff; margin-bottom:1.1rem; }
.hero h1 { font-size:1.65rem; font-weight:800; letter-spacing:-.025em; margin:0; color:#fff; line-height:1.2; }
.hero p { margin:.35rem 0 .9rem; color:#c9d4e3; font-size:.92rem; }
.chip { display:inline-block; font-size:.7rem; font-weight:600; padding:.22rem .6rem; border-radius:999px; margin:0 .35rem .3rem 0; background:rgba(255,255,255,.1); color:#e6ecf5; border:1px solid rgba(255,255,255,.16); }
.chip.warn { background:rgba(237,161,0,.18); border-color:rgba(237,161,0,.5); }

/* KPI tiles */
.kpi { min-height:178px; background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem .9rem; height:100%; box-shadow:0 1px 2px rgba(16,24,40,.04); }
.kpi-label { font-size:.68rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--ink-3); }
.kpi-value { font-size:1.85rem; font-weight:800; letter-spacing:-.03em; color:var(--ink); line-height:1.15; margin-top:.25rem; }
.kpi-delta { font-size:.78rem; font-weight:600; margin-top:.2rem; color:var(--ink-2); }
.kpi-delta .up { color:var(--good); } .kpi-delta .down { color:var(--bad); }
.kpi-foot { font-size:.72rem; color:var(--ink-3); margin-top:.55rem; padding-top:.5rem; border-top:1px solid #f0f1f3; line-height:1.45; }

/* Cards & sections */
.card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.25rem; box-shadow:0 1px 2px rgba(16,24,40,.04); }
.sec-title { font-size:1.02rem; font-weight:700; letter-spacing:-.01em; color:var(--ink); margin:.2rem 0 .1rem; }
.sec-sub { font-size:.8rem; color:var(--ink-3); margin-bottom:.6rem; line-height:1.5; }
.summary { background:#fff; border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:12px; padding:1rem 1.25rem; font-size:.93rem; line-height:1.7; color:#27303f; margin:.9rem 0 .4rem; }
.callout { background:#f0f5fc; border:1px solid #d6e4f7; border-radius:10px; padding:.75rem .95rem; font-size:.82rem; color:#24324a; line-height:1.55; }
.stat-row { display:flex; gap:.6rem; flex-wrap:wrap; margin-bottom:.6rem; }
.stat { flex:1; min-width:130px; background:#f7f8fa; border:1px solid var(--line); border-radius:10px; padding:.6rem .8rem; }
.stat b { display:block; font-size:1.25rem; font-weight:800; letter-spacing:-.02em; }
.stat span { font-size:.72rem; color:var(--ink-3); }

/* Pitfalls / timeline */
.pit { background:#fff; border:1px solid var(--line); border-radius:10px; padding:.8rem 1rem; margin-bottom:.5rem; }
.badge { display:inline-block; font-size:.62rem; font-weight:800; letter-spacing:.07em; padding:.12rem .45rem; border-radius:4px; margin-right:.45rem; vertical-align:middle; }
.b-high { background:#fef3f2; color:#b42318; border:1px solid #fecdca; }
.b-medium { background:#fffaeb; color:#b54708; border:1px solid #fedf89; }
.b-low { background:#ecfdf3; color:#067647; border:1px solid #abefc6; }
.pit-t { font-weight:600; font-size:.86rem; vertical-align:middle; }
.pit-d { font-size:.8rem; color:var(--ink-2); margin-top:.35rem; line-height:1.55; }
.tl { display:flex; gap:.85rem; padding:.6rem 0; border-bottom:1px solid #eef0f2; }
.tl-date { min-width:88px; font-size:.75rem; color:var(--ink-3); font-weight:600; padding-top:.1rem; }
.tl-name { font-size:.86rem; font-weight:650; color:var(--ink); }
.tl-det { font-size:.78rem; color:var(--ink-2); line-height:1.45; }
.tag { display:inline-block; font-size:.62rem; font-weight:700; padding:.08rem .4rem; border-radius:4px; margin-left:.35rem; background:#eef2f7; color:#3b4a61; vertical-align:middle; }
.tag.pending { background:#fff4e0; color:#8a5300; } .tag.withdrawal { background:#fef3f2; color:#b42318; }
.disclaimer { background:#fff; border:1px solid var(--line); border-radius:10px; padding:.8rem 1rem; font-size:.78rem; color:var(--ink-2); margin-top:1rem; }

/* Tabs */
.stTabs [role="tablist"] { gap:.25rem; background:#fff; border:1px solid var(--line); border-radius:12px; padding:.3rem; }
.stTabs [data-testid="stTab"] { padding:.45rem 1rem !important; border-radius:8px; font-weight:600; color:var(--ink-2); }
.stTabs [data-testid="stTab"]:hover { background:#f1f3f6; }
.stTabs [data-testid="stTab"][aria-selected="true"] { background:#0f1b2d !important; color:#fff !important; }
.stTabs [data-testid="stTab"][aria-selected="true"] p { color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stTabs [role="tabpanel"] { padding-top:1.1rem; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.stDownloadButton > button { border-radius:8px; font-weight:600; border:1px solid #cfd4dc; }
</style>
""", unsafe_allow_html=True)


def chart(fig: go.Figure, height: int = 380, legend: bool = True, hover: str = 'x unified',
          margin=None, legend_right: bool = False, stacked: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='Inter, system-ui, sans-serif', size=12, color=INK_2),
        margin=margin or dict(l=8, r=12, t=36 if legend else 12, b=8),
        hovermode=hover, showlegend=legend,
        legend=(dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.01, traceorder='reversed' if stacked else 'normal',
                     font=dict(size=11.5, color=INK_2), bgcolor='rgba(0,0,0,0)') if legend_right else
                dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0, traceorder='normal',
                     font=dict(size=11.5, color=INK_2), bgcolor='rgba(0,0,0,0)')),
        hoverlabel=dict(bgcolor='#ffffff', bordercolor='#d0d3d9',
                        font=dict(family='Inter, sans-serif', size=12, color=INK)),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(size=11, color=INK_3),
                     ticks='outside', tickcolor=AXIS, ticklen=4)
    fig.update_yaxes(gridcolor=GRID, linecolor='rgba(0,0,0,0)', zeroline=False,
                     tickfont=dict(size=11, color=INK_3))
    return fig


def mark_projection(fig: go.Figure, x0, x1, label='Projected incidence'):
    """Shade the forecast window and mark today (shapes, not vline, for date axes)."""
    fig.add_shape(type='rect', xref='x', yref='paper', x0=x0, x1=x1, y0=0, y1=1,
                  fillcolor='rgba(18,53,91,0.045)', line_width=0, layer='below')
    fig.add_annotation(x=x0, y=1, xref='x', yref='paper', text=f'  {label} →', showarrow=False,
                       xanchor='left', yanchor='top', font=dict(size=10.5, color=INK_3))
    if pd.Timestamp(x0) <= TODAY <= pd.Timestamp(x1):
        fig.add_shape(type='line', xref='x', yref='paper', x0=TODAY, x1=TODAY, y0=0, y1=1,
                      line=dict(color=INK_3, width=1))
        fig.add_annotation(x=TODAY, y=0, xref='x', yref='paper', text=' Today', showarrow=False,
                           xanchor='left', yanchor='bottom', font=dict(size=10.5, color=INK_3))


def section(title: str, sub: str = ''):
    st.markdown(f'<div class="sec-title">{title}</div>' +
                (f'<div class="sec-sub">{sub}</div>' if sub else ''), unsafe_allow_html=True)


def fmt(n: float) -> str:
    return f"{n:,.0f}"


# ══════════════════════════════════════════════════════════════════
#  DATA & MODEL (cached)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_inputs() -> lm.ModelInputs:
    return lm.load_inputs()


try:
    INPUTS = get_inputs()
except FileNotFoundError as e:
    st.error(f"Model input files not found: {e}")
    st.stop()

BASE = lm.scenario_from_params(INPUTS.params)
REGS = INPUTS.regimens
CLASS_ORDER = INPUTS.regimens_yaml.get('class_order', [])


@st.cache_data(show_spinner=False, max_entries=64)
def simulate(sc_json: str):
    res = lm.run_model(INPUTS, json.loads(sc_json))
    return res.monthly, res.starts, res.annual_incidence, mm.annualize(res.monthly)


@st.cache_data(show_spinner=False, max_entries=64)
def cohort(sc_json: str, year: int):
    return lm.run_cohort(INPUTS, json.loads(sc_json), year)


@st.cache_data(show_spinner=False, max_entries=16)
def sensitivity(sc_json: str, year: int):
    run = lambda s: mm.annualize(lm.run_model(INPUTS, s).monthly)
    return mm.tornado(run, json.loads(sc_json), year)


def to_json(sc: dict) -> str:
    return json.dumps(sc, sort_keys=True, default=list)


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR — scenario levers (every change re-runs the model)
# ══════════════════════════════════════════════════════════════════
LEVERS = {  # key: (label, min, max, step, help, format)
    'treated_fraction': ('Treated fraction', 0.60, 1.00, 0.01,
                         'Share of registry-diagnosed patients who start 1L therapy. Registry counts '
                         'include smoldering myeloma and frail patients who are never treated.', '%.2f'),
    'frac_te': ('Transplant-eligible share', 0.15, 0.60, 0.01,
                'Share of 1L starts on a transplant pathway (US ASCT use is ~30–40% of NDMM).', '%.2f'),
    'p_2l': ('P(start 2L | progression)', 0.40, 1.00, 0.01, 'Probability that 1L progression leads to 2L therapy.', '%.2f'),
    'p_3l': ('P(start 3L | progression)', 0.40, 1.00, 0.01, 'Probability that 2L progression leads to 3L.', '%.2f'),
    'p_4l': ('P(start 4L+ | progression)', 0.30, 1.00, 0.01, 'Probability that 3L progression leads to 4L+.', '%.2f'),
    'p_later': ('P(re-treat within 4L+)', 0.20, 0.90, 0.01, 'Progression on 4L+ leading to a further line (5L, 6L…).', '%.2f'),
    'rwe_pfs_multiplier': ('Real-world PFS multiplier', 0.50, 1.20, 0.01,
                           'Real-world duration as a fraction of trial PFS (efficacy–effectiveness gap).', '%.2f'),
    'mort_multiplier': ('On-line mortality multiplier', 0.50, 2.00, 0.05, 'Scales the pre-progression death hazard on every line.', '%.2f'),
    'new_launch_speed_multiplier': ('Uptake speed, 2024+ launches', 0.50, 2.00, 0.05,
                                    'Speeds up or slows down diffusion of D-VRd, Isa-VRd, Tec-Dara, BVd and other recent launches.', '%.2f'),
    'rate_trend_pct': ('Underlying rate trend (%/yr)', -2.0, 2.0, 0.1,
                       'Annual change in age/sex-specific incidence rates beyond population ageing.', '%.1f'),
}
METHODS = {'Demographic (Census ageing)': 'demographic', 'Log-linear trend': 'trend',
           'Flat (last year carried)': 'flat'}

for k in LEVERS:
    st.session_state.setdefault(k, BASE[k])
st.session_state.setdefault('incidence_method_label', 'Demographic (Census ageing)')
st.session_state.setdefault('anchor_to_acs', BASE['anchor_to_acs'])


def reset_levers():
    for k in LEVERS:
        st.session_state[k] = BASE[k]
    st.session_state['incidence_method_label'] = 'Demographic (Census ageing)'
    st.session_state['anchor_to_acs'] = BASE['anchor_to_acs']


def lever(k):
    label, lo, hi, step, hlp, f = LEVERS[k]
    return st.slider(label, lo, hi, step=step, key=k, help=hlp, format=f)


with st.sidebar:
    st.markdown('<div class="sb-brand">Myeloma Forecast Studio</div>'
                '<div class="sb-sub">Scenario levers · re-simulates live</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-h">Forecast window</div>', unsafe_allow_html=True)
    horizon = st.select_slider('Horizon year', options=list(range(2028, BASE['end_year'] + 1)),
                               value=2035, help='Year used for forward-looking KPIs and sensitivity.')
    view_start = st.select_slider('Charts start in', options=list(range(1999, 2026)), value=2010)

    st.markdown('<div class="sb-h">Scenario</div>', unsafe_allow_html=True)
    with st.expander('Epidemiology', expanded=False):
        st.radio('Incidence projection', list(METHODS), key='incidence_method_label')
        lever('rate_trend_pct')
        st.toggle('Anchor to ACS 2026 estimate (36,000)', key='anchor_to_acs',
                  help='Scales the whole incidence series by one completeness factor so 2026 matches '
                       'the American Cancer Society estimate. Off = registry-observed basis.')
    with st.expander('Treatment pathway', expanded=True):
        for k in ['treated_fraction', 'frac_te', 'p_2l', 'p_3l', 'p_4l', 'p_later']:
            lever(k)
    with st.expander('Effectiveness & uptake', expanded=False):
        for k in ['rwe_pfs_multiplier', 'mort_multiplier', 'new_launch_speed_multiplier']:
            lever(k)
    st.button('Reset to base case', on_click=reset_levers, width='stretch')

    sc = dict(BASE)
    sc.update({k: st.session_state[k] for k in LEVERS})
    sc['incidence_method'] = METHODS[st.session_state['incidence_method_label']]
    sc['anchor_to_acs'] = bool(st.session_state['anchor_to_acs'])
    is_base = all(abs(sc[k] - BASE[k]) < 1e-9 for k in LEVERS) and \
        sc['incidence_method'] == BASE['incidence_method'] and sc['anchor_to_acs'] == BASE['anchor_to_acs']

    st.markdown('<div class="sb-h">Saved scenarios</div>', unsafe_allow_html=True)
    st.session_state.setdefault('saved', [])
    save_name = st.text_input('Name', value=f"Scenario {len(st.session_state.saved) + 1}",
                              label_visibility='collapsed')
    c1, c2 = st.columns(2)
    save_clicked = c1.button('Save', width='stretch')
    if c2.button('Clear', width='stretch', disabled=not st.session_state.saved):
        st.session_state.saved = []
    st.caption('Sources: CDC USCS 1999–2022 · US Census NP2023 · ACS 2026 · '
               'SEER · FDA approvals to Jul 2026 · pivotal-trial PFS')

SC_JSON = to_json(sc)
with st.spinner('Simulating…'):
    monthly, starts, incidence, annual = simulate(SC_JSON)
A = annual.set_index('Year')
INC = incidence.set_index('Year')
view = monthly[monthly['Date'].dt.year >= view_start]

if save_clicked:
    st.session_state.saved = (st.session_state.saved + [{
        'Scenario': save_name, 'json': SC_JSON,
        f'Dx {NOW_YEAR}': INC.loc[NOW_YEAR, 'Cases'], f'On therapy {NOW_YEAR}': A.loc[NOW_YEAR, 'On_Therapy'],
        f'On therapy {horizon}': A.loc[horizon, 'On_Therapy'],
        f'1L starts {horizon}': A.loc[horizon, 'New_Starts_1L'],
        f'4L+ patients {horizon}': A.loc[horizon, 'Total_4L+'],
    }])[-6:]

# ══════════════════════════════════════════════════════════════════
#  HERO + KPIs
# ══════════════════════════════════════════════════════════════════
chips = ['US · all adults', 'USCS incidence 1999–2022', 'Census NP2023 projections',
         'FDA approvals through Jul 2026', 'Evidence as of Oct 2026']
chips_html = ''.join(f'<span class="chip">{c}</span>' for c in chips)
chips_html += ('<span class="chip">Base case</span>' if is_base else
               '<span class="chip warn">Custom scenario</span>')
st.markdown(f"""
<div class="hero">
  <h1>Multiple Myeloma · US Epidemiology &amp; Treatment Forecast</h1>
  <p>Population model of incidence, line-of-therapy patient flow and regimen mix from 1999 to {BASE['end_year']}, built on registry data and pivotal-trial evidence.</p>
  {chips_html}
</div>""", unsafe_allow_html=True)


def kpi(col, label, value, delta_html, foot):
    col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-value">{value}</div><div class="kpi-delta">{delta_html}</div>'
                 f'<div class="kpi-foot">{foot}</div></div>', unsafe_allow_html=True)


def delta(cur, ref, suffix):
    d = cur / ref - 1 if ref else 0
    cls = 'up' if d > 0.005 else ('down' if d < -0.005 else '')
    arrow = '▲' if d > 0.005 else ('▼' if d < -0.005 else '■')
    return f'<span class="{cls}">{arrow} {d:+.1%}</span> {suffix}'


cls_m = mm.class_exposure(monthly, REGS, CLASS_ORDER)
cls_m['Year'] = cls_m['Date'].dt.year
CLS = cls_m.groupby('Year').mean(numeric_only=True)
tcr_now = CLS.loc[NOW_YEAR, 'BCMA bispecific'] + CLS.loc[NOW_YEAR, 'BCMA CAR-T'] + CLS.loc[NOW_YEAR, 'GPRC5D bispecific']
tcr_h = CLS.loc[horizon, 'BCMA bispecific'] + CLS.loc[horizon, 'BCMA CAR-T'] + CLS.loc[horizon, 'GPRC5D bispecific']
cd38_now = CLS.loc[NOW_YEAR, 'Anti-CD38'] / CLS.loc[NOW_YEAR, 'On_Therapy']

k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, f'New diagnoses · {NOW_YEAR}', fmt(INC.loc[NOW_YEAR, 'Cases']),
    delta(INC.loc[NOW_YEAR, 'Cases'], INC.loc[LAST_OBS_YEAR, 'Cases'], f'vs {LAST_OBS_YEAR} observed'),
    f"→ {fmt(INC.loc[horizon, 'Cases'])} in {horizon}. ACS 2026 estimate: 36,000.")
kpi(k2, f'Patients on therapy · {NOW_YEAR}', fmt(A.loc[NOW_YEAR, 'On_Therapy']),
    delta(A.loc[horizon, 'On_Therapy'], A.loc[NOW_YEAR, 'On_Therapy'], f'by {horizon}'),
    f"→ {fmt(A.loc[horizon, 'On_Therapy'])} in {horizon}. Includes maintenance; annual mean.")
kpi(k3, f'New 1L starts · {NOW_YEAR}', fmt(A.loc[NOW_YEAR, 'New_Starts_1L']),
    delta(A.loc[horizon, 'New_Starts_1L'], A.loc[NOW_YEAR, 'New_Starts_1L'], f'by {horizon}'),
    f"{sc['treated_fraction']:.0%} of diagnosed start 1L; {sc['frac_te']:.0%} on a transplant pathway.")
kpi(k4, f'Anti-CD38 share · {NOW_YEAR}', f"{cd38_now:.0%}",
    f"<span>{fmt(CLS.loc[NOW_YEAR, 'Anti-CD38'])} patients</span>",
    'Daratumumab- or isatuximab-containing, across all lines.')
kpi(k5, f'T-cell redirectors · {horizon}', fmt(tcr_h),
    delta(tcr_h, tcr_now, f'vs {NOW_YEAR}'),
    'Patients on CAR-T (in follow-up) or BCMA/GPRC5D bispecifics.')

st.markdown(f'<div class="summary">{mm.executive_summary(annual, incidence, REGS, monthly, NOW_YEAR, horizon)}</div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════
tabs = st.tabs(['Overview', 'Patient flow', 'Regimen forecast', 'Epidemiology',
                'Evidence & approvals', 'Validation & sensitivity', 'Methods'])
proj_start = pd.Timestamp(f'{LAST_OBS_YEAR + 1}-01-01')
proj_end = monthly['Date'].max()

# ── OVERVIEW ─────────────────────────────────────────────────────
with tabs[0]:
    c1, c2 = st.columns([1.65, 1])
    with c1:
        section('Patients on active therapy, by line',
                'Monthly patients receiving treatment (including maintenance). Shading marks the '
                'period after the last observed registry year, where incidence is projected.')
        occ = mm.line_occupancy_long(view)
        fig = go.Figure()
        for name, color in LINE_COLORS.items():
            d = occ[occ['Line'] == name]
            fig.add_trace(go.Scatter(x=d['Date'], y=d['Patients'], name=name, stackgroup='one',
                                     mode='lines', line=dict(width=0.6, color='#ffffff'),
                                     fillcolor=color, hovertemplate='%{y:,.0f}'))
        mark_projection(fig, proj_start, proj_end)
        fig.update_yaxes(title=None, tickformat=',.0f')
        st.plotly_chart(chart(fig, 420), width='stretch')
    with c2:
        section('Snapshot', 'Annual means for patients; annual totals for flows.')
        snap_years = sorted({LAST_OBS_YEAR, NOW_YEAR, 2030, horizon})
        rows = {
            'New diagnoses': INC.loc[snap_years, 'Cases'],
            'New 1L starts': A.loc[snap_years, 'New_Starts_1L'],
            'On 1st line': A.loc[snap_years, 'Total_1L'],
            'On 2nd line': A.loc[snap_years, 'Total_2L'],
            'On 3rd line': A.loc[snap_years, 'Total_3L'],
            'On 4th line+': A.loc[snap_years, 'Total_4L+'],
            'Total on therapy': A.loc[snap_years, 'On_Therapy'],
        }
        snap = pd.DataFrame(rows).T
        snap.columns = [str(y) for y in snap_years]
        st.dataframe(snap.style.format('{:,.0f}'), width='stretch', height=38 + 35 * len(snap))
        share_1l = A.loc[horizon, 'Total_1L'] / A.loc[horizon, 'On_Therapy']
        st.markdown(f'<div class="callout"><b>Why first line keeps growing.</b> Quadruplet induction '
                    f'(D-VRd, Isa-VRd) and continuous maintenance stretch time on 1L. By {horizon}, '
                    f'<b>{share_1l:.0%}</b> of treated patients are still on their first regimen, '
                    f'so patients reach later lines older and fewer in number.</div>',
                    unsafe_allow_html=True)

    section('Annual patient starts by line',
            'New patients initiating each line per year; 4L+ includes re-treatment at 5L and beyond.')
    fig = go.Figure()
    av = annual[annual['Year'] >= view_start]
    for l, col in [('1L', 'New_Starts_1L'), ('2L', 'Starts_2L'), ('3L', 'Starts_3L'), ('4L+', 'Starts_4L+')]:
        fig.add_trace(go.Scatter(x=av['Year'], y=av[col], name=mm.LINE_NAMES[l], mode='lines',
                                 line=dict(width=2, color=LINE_CODE_COLORS[l]),
                                 hovertemplate='%{y:,.0f}'))
    fig.add_shape(type='rect', xref='x', yref='paper', x0=LAST_OBS_YEAR + 0.5, x1=BASE['end_year'] + 0.5,
                  y0=0, y1=1, fillcolor='rgba(18,53,91,0.045)', line_width=0, layer='below')
    fig.update_yaxes(tickformat=',.0f', rangemode='tozero')
    st.plotly_chart(chart(fig, 320), width='stretch')

# ── PATIENT FLOW ─────────────────────────────────────────────────
with tabs[1]:
    section('Ten-year journey of 1,000 patients starting first-line therapy',
            'A cohort beginning 1L in January of the chosen year, followed for 10 years with that '
            'era’s regimen mix at each line. Every patient is accounted for.')
    cy = st.segmented_control('Cohort start year', [2012, 2016, 2021, 2026], default=2026,
                              key='cohort_year') or 2026
    c = cohort(SC_JSON, int(cy))
    stats = [(c['start_2L'] / 10, 'reach 2nd line'), (c['start_3L'] / 10, 'reach 3rd line'),
             (c['start_4L+'] / 10, 'reach 4th line+'), (c['still_1L'] / 10, 'still on 1st line at 10 y')]
    st.markdown('<div class="stat-row">' + ''.join(
        f'<div class="stat"><b>{v:.0f}%</b><span>{t}</span></div>' for v, t in stats) + '</div>',
        unsafe_allow_html=True)

    labels = ['Start 1L', '2nd line', '3rd line', '4th line+',
              'Died on therapy', 'Progressed, no further therapy', 'Still on therapy at 10 y']
    src, tgt, val = [], [], []
    for i, l in enumerate(mm.LINES):
        if i < 3:
            src.append(i); tgt.append(i + 1); val.append(c[f'start_{mm.LINES[i + 1]}'])
        src += [i, i, i]; tgt += [4, 5, 6]
        val += [c[f'death_{l}'], c[f'stop_{l}'], c[f'still_{l}']]
    node_colors = SLOTS[:4] + ['#7b8494', '#b9b7ae', '#12355b']
    link_colors = [f'rgba({int(node_colors[s][1:3], 16)},{int(node_colors[s][3:5], 16)},'
                   f'{int(node_colors[s][5:7], 16)},0.28)' for s in src]
    fig = go.Figure(go.Sankey(
        arrangement='snap', valueformat=',.0f', valuesuffix=' patients',
        node=dict(label=labels, color=node_colors, pad=22, thickness=18,
                  line=dict(color='#ffffff', width=1),
                  hovertemplate='%{label}: %{value:,.0f}<extra></extra>'),
        link=dict(source=src, target=tgt, value=val, color=link_colors,
                  hovertemplate='%{source.label} → %{target.label}: %{value:,.0f}<extra></extra>'),
    ))
    fig.update_traces(textfont=dict(family='Inter, sans-serif', size=12.5, color=INK))
    st.plotly_chart(chart(fig, 430, legend=False, hover='closest', margin=dict(l=8, r=8, t=8, b=8)),
                    width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        section('Where patients exit therapy', 'Of the 1,000 starting patients over 10 years.')
        ex = pd.DataFrame({
            'Line': [mm.LINE_NAMES[l] for l in mm.LINES],
            'Died on therapy': [c[f'death_{l}'] for l in mm.LINES],
            'No further therapy': [c[f'stop_{l}'] for l in mm.LINES],
            'Still on therapy': [c[f'still_{l}'] for l in mm.LINES],
        })
        fig = go.Figure()
        for i, colname in enumerate(['Died on therapy', 'No further therapy', 'Still on therapy']):
            fig.add_trace(go.Bar(y=ex['Line'], x=ex[colname], name=colname, orientation='h',
                                 marker=dict(color=['#7b8494', '#b9b7ae', '#12355b'][i],
                                             line=dict(color='#fff', width=2)),
                                 hovertemplate='%{x:,.0f}'))
        fig.update_layout(barmode='stack', bargap=0.35)
        fig.update_yaxes(autorange='reversed', gridcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showgrid=True, gridcolor=GRID)
        st.plotly_chart(chart(fig, 280, hover='y unified'), width='stretch')
    with c2:
        section('Line transition rates over time',
                'Patients starting the next line per year ÷ mean patients on the current line.')
        fig = go.Figure()
        for (src_l, dst_col, name), color in zip(
                [('1L', 'Starts_2L', '1L → 2L'), ('2L', 'Starts_3L', '2L → 3L'),
                 ('3L', 'Starts_4L+_first', '3L → 4L+')], SLOTS[:3]):
            rate = av[dst_col] / av[f'Total_{src_l}']
            fig.add_trace(go.Scatter(x=av['Year'], y=rate, name=name, mode='lines',
                                     line=dict(width=2, color=color), hovertemplate='%{y:.1%}'))
        fig.update_yaxes(tickformat='.0%', rangemode='tozero')
        st.plotly_chart(chart(fig, 280), width='stretch')

# ── REGIMEN FORECAST ─────────────────────────────────────────────
with tabs[2]:
    cA, cB = st.columns([1, 1])
    with cA:
        line_sel = st.segmented_control('Line of therapy', mm.LINES, default='1L', key='reg_line') or '1L'
    with cB:
        basis = st.segmented_control('Measure', ['Patients on therapy', 'New patient starts'],
                                     default='Patients on therapy', key='reg_basis') or 'Patients on therapy'
    frame = monthly if basis == 'Patients on therapy' else starts
    frame = frame[frame['Date'].dt.year >= view_start].reset_index(drop=True)
    if basis == 'New patient starts':   # smooth monthly starts to quarterly
        frame = frame.set_index('Date').resample('QS').sum().reset_index()
    shares = mm.regimen_share(frame, line_sel, REGS)

    section(f'{mm.LINE_NAMES[line_sel].capitalize()} regimen mix · {basis.lower()}',
            'Modelled share of the line. Logistic adoption from FDA approval or guideline listing, '
            'with displacement by successors. Regimens that never exceed 3% fold into “Other”.')
    if shares.empty:
        st.info('No regimen data for this line.')
    else:
        series = [c for c in shares.columns if c != 'Date']
        fig = go.Figure()
        n_named = 0
        for s in series:
            color = OTHER if s == 'Other' else SLOTS[n_named % len(SLOTS)]
            n_named += s != 'Other'
            fig.add_trace(go.Scatter(x=shares['Date'], y=shares[s], name=s, stackgroup='one',
                                     mode='lines', line=dict(width=0.6, color='#ffffff'),
                                     fillcolor=color, hovertemplate='%{y:.0%}'))
        mark_projection(fig, proj_start, proj_end)
        fig.update_yaxes(tickformat='.0%', range=[0, 1])
        st.plotly_chart(chart(fig, 430, legend_right=True, margin=dict(l=8, r=12, t=16, b=8)),
                        width='stretch')

    yrs = [y for y in [2022, NOW_YEAR, 2028, 2030, horizon] if y <= BASE['end_year']]
    yrs = sorted(set(yrs))
    reg_keys = [k for k, r in REGS.items() if r.line == line_sel]
    src_frame = monthly if basis == 'Patients on therapy' else starts
    tmp = src_frame[['Date'] + reg_keys].copy()
    tmp['Year'] = tmp['Date'].dt.year
    agg = tmp.groupby('Year')[reg_keys].mean() if basis == 'Patients on therapy' \
        else tmp.groupby('Year')[reg_keys].sum()
    tbl = agg.loc[yrs].T
    tbl.index = [REGS[k].label for k in tbl.index]
    tbl = tbl[tbl.max(axis=1) >= 50].sort_values(horizon, ascending=False)
    tbl.columns = [str(y) for y in yrs]
    c1, c2 = st.columns([1.15, 1])
    with c1:
        section(f'{basis} by regimen', 'Annual mean patients on therapy, or annual total starts. '
                                        'Regimens with fewer than 50 patients in every year are hidden.')
        st.dataframe(tbl.style.format('{:,.0f}'), width='stretch',
                     height=min(420, 38 + 35 * len(tbl)))
    with c2:
        section('Drug-class exposure', 'Share of patients on therapy receiving each class. Classes overlap: '
                'a quadruplet counts in each of its classes.')
        cls_v = cls_m[cls_m['Year'] >= view_start]
        fig = go.Figure()
        for i, cname in enumerate(['Anti-CD38', 'PI', 'IMiD', 'BCMA bispecific', 'BCMA CAR-T',
                                   'GPRC5D bispecific', 'BCMA ADC', 'XPO1']):
            fig.add_trace(go.Scatter(x=cls_v['Date'], y=cls_v[cname] / cls_v['On_Therapy'], name=cname,
                                     mode='lines', line=dict(width=2, color=SLOTS[i]),
                                     customdata=cls_v[cname], hovertemplate='%{y:.1%} (%{customdata:,.0f})'))
        mark_projection(fig, proj_start, proj_end, label='Projected')
        fig.update_yaxes(tickformat='.0%', rangemode='tozero')
        st.plotly_chart(chart(fig, 400, legend_right=True, stacked=False, margin=dict(l=8, r=12, t=16, b=8)),
                        width='stretch')

    d1, d2, d3 = st.columns(3)
    d1.download_button('Monthly simulation (CSV)', monthly.to_csv(index=False).encode(),
                       'mm_monthly_simulation.csv', 'text/csv', width='stretch')
    d2.download_button('Annual forecast (CSV)', annual.to_csv(index=False).encode(),
                       'mm_annual_forecast.csv', 'text/csv', width='stretch')
    d3.download_button('Regimen starts by year (CSV)',
                       mm.annual_starts(starts).to_csv(index=False).encode(),
                       'mm_regimen_starts_annual.csv', 'text/csv', width='stretch')

# ── EPIDEMIOLOGY ─────────────────────────────────────────────────
with tabs[3]:
    c1, c2 = st.columns([1.6, 1])
    with c1:
        section('New myeloma diagnoses per year',
                f'Observed: CDC USCS ({incidence["Year"].min()}–{LAST_OBS_YEAR}). Projected: '
                f'{st.session_state["incidence_method_label"].lower()}'
                + (', scaled to the ACS 2026 estimate' if sc['anchor_to_acs'] else '') + '.')
        inc_v = incidence[incidence['Year'] >= min(view_start, 2005)]
        obs = inc_v[inc_v['Type'] == 'Observed']
        prj = inc_v[inc_v['Type'] == 'Projected']
        fig = go.Figure()
        fig.add_trace(go.Bar(x=obs['Year'], y=obs['Cases'], name='Observed (USCS)',
                             marker=dict(color=SLOTS[0], line=dict(color='#fff', width=1.5)),
                             hovertemplate='%{y:,.0f}'))
        fig.add_trace(go.Bar(x=prj['Year'], y=prj['Cases'], name='Projected',
                             marker=dict(color='#9ec5f4', line=dict(color='#fff', width=1.5)),
                             hovertemplate='%{y:,.0f}'))
        fig.add_trace(go.Scatter(x=[2026], y=[mm.BENCHMARKS['acs_2026_cases']], name='ACS 2026 estimate',
                                 mode='markers', marker=dict(symbol='diamond', size=11, color=SLOTS[1],
                                                             line=dict(color='#fff', width=2)),
                                 hovertemplate='%{y:,.0f}'))
        fig.update_layout(bargap=0.18)
        fig.update_yaxes(tickformat=',.0f', rangemode='tozero')
        st.plotly_chart(chart(fig, 380), width='stretch')
    with c2:
        asr = epi.age_standardised_rate(INPUTS.uscs, INPUTS.census)
        cf = float(incidence['Completeness_Factor'].iloc[0])
        growth = (INC.loc[horizon, 'Cases'] / INC.loc[LAST_OBS_YEAR, 'Cases']) ** (1 / (horizon - LAST_OBS_YEAR)) - 1
        section('Key epidemiology')
        st.markdown(f"""
<div class="stat-row">
 <div class="stat"><b>{INC.loc[LAST_OBS_YEAR, 'Cases']:,.0f}</b><span>registry cases, {LAST_OBS_YEAR}</span></div>
 <div class="stat"><b>{growth:+.1%}</b><span>case growth per year to {horizon}</span></div>
</div>
<div class="stat-row">
 <div class="stat"><b>{asr.loc[LAST_OBS_YEAR]:.1f}</b><span>per 100k adults 25+, standardised ({LAST_OBS_YEAR})</span></div>
 <div class="stat"><b>{mm.BENCHMARKS['seer_median_age_dx']}</b><span>median age at diagnosis (SEER)</span></div>
</div>
<div class="stat-row">
 <div class="stat"><b>{mm.BENCHMARKS['seer_prevalence_2022']:,}</b><span>living with myeloma, 2022 (SEER)</span></div>
 <div class="stat"><b>{mm.BENCHMARKS['seer_5yr_rel_survival']}%</b><span>5-year relative survival (SEER 2015–21)</span></div>
</div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="callout">Growth is driven by <b>population ageing</b>: the 65+ population '
                    f'grows much faster than the total, and myeloma risk rises steeply with age. '
                    f'2020 is excluded from base rates (COVID-era under-diagnosis). '
                    f'Completeness factor applied: <b>{cf:.2f}×</b>.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section('Age-specific incidence', 'Cases per 100,000 by 5-year age group, pooled base years (2017–19, 2021–22).')
        r = epi.age_specific_rates(INPUTS.uscs, BASE['base_years'])
        fig = go.Figure()
        for i, sex in enumerate(['Male', 'Female']):
            d = r[r['Sex'] == sex].set_index('Age_Group').reindex(epi.AGE_ORDER)
            fig.add_trace(go.Scatter(x=epi.AGE_ORDER, y=d['Rate'] * 1e5, name=sex, mode='lines+markers',
                                     line=dict(width=2, color=SLOTS[i]), marker=dict(size=8, line=dict(color='#fff', width=2)),
                                     hovertemplate='%{y:.1f} per 100k'))
        fig.update_yaxes(rangemode='tozero')
        st.plotly_chart(chart(fig, 330), width='stretch')
    with c2:
        section('Projected cases by age group', f'{LAST_OBS_YEAR + 1} vs {horizon}, demographic method.')
        dist = epi.projected_age_distribution(INPUTS.uscs, INPUTS.census, [LAST_OBS_YEAR + 1, horizon], BASE['base_years'])
        fig = go.Figure()
        for i, y in enumerate([LAST_OBS_YEAR + 1, horizon]):
            d = dist[dist['Year'] == y].set_index('Age_Group').reindex(epi.AGE_ORDER[4:])
            fig.add_trace(go.Bar(x=d.index, y=d['Cases'], name=str(y),
                                 marker=dict(color=['#9ec5f4', SLOTS[0]][i], line=dict(color='#fff', width=1.5)),
                                 hovertemplate='%{y:,.0f}'))
        fig.update_layout(barmode='group', bargap=0.25, bargroupgap=0.05)
        fig.update_yaxes(tickformat=',.0f')
        st.plotly_chart(chart(fig, 330), width='stretch')

# ── EVIDENCE & APPROVALS ─────────────────────────────────────────
with tabs[4]:
    ev = mm.regimen_evidence_table(REGS, sc['rwe_pfs_multiplier'])
    ev_mod = ev[ev['US approval'] != 'Pre-1990s']
    section('Efficacy inputs by regimen',
            'Model median PFS from pivotal trials. Hollow markers are medians extrapolated from a landmark '
            'PFS rate (median not reached); the grey tick is the real-world adjusted median used in the forecast.')
    line_pick = st.segmented_control('Line', mm.LINES, default='1L', key='ev_line') or '1L'
    d = ev_mod[ev_mod['Line'] == line_pick].sort_values('Model median PFS (mo)')
    is_lm = d['Efficacy input'].str.contains('NR')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['Real-world median (mo)'], y=d['Regimen'], mode='markers', name='Real-world adjusted',
                             marker=dict(symbol='line-ns', size=14, line=dict(width=2, color=INK_3)),
                             hovertemplate='Real-world: %{x:.0f} mo<extra></extra>'))
    fig.add_trace(go.Scatter(x=d['Model median PFS (mo)'], y=d['Regimen'], mode='markers', name='Trial median (reported)',
                             marker=dict(size=11, color=np.where(is_lm, '#ffffff', LINE_CODE_COLORS[line_pick]),
                                         line=dict(width=2, color=LINE_CODE_COLORS[line_pick])),
                             customdata=np.stack([d['Trial'], d['Efficacy input'], d['Citation']], axis=1),
                             hovertemplate='<b>%{y}</b><br>%{customdata[0]}: %{customdata[1]}<br>'
                                           'Model median %{x:.1f} mo<br><span style="color:#8a909b">%{customdata[2]}</span><extra></extra>'))
    fig.update_xaxes(title='Median PFS (months)', showgrid=True, gridcolor=GRID, rangemode='tozero')
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(chart(fig, max(260, 34 * len(d) + 80), hover='closest',
                          margin=dict(l=8, r=12, t=36, b=40)), width='stretch')

    section('Evidence register', 'Every efficacy input with its source. Search or sort any column.')
    st.dataframe(ev.drop(columns=['Classes']), width='stretch', hide_index=True, height=420,
                 column_config={
                     'HR': st.column_config.NumberColumn('HR', format='%.2f'),
                     'Model median PFS (mo)': st.column_config.NumberColumn(format='%.1f'),
                     'Real-world median (mo)': st.column_config.NumberColumn(format='%.1f'),
                     'Citation': st.column_config.TextColumn(width='large'),
                     'Note': st.column_config.TextColumn(width='large'),
                 })

    section('FDA regulatory timeline', 'Approvals, label expansions and withdrawals since 2015, plus pending decisions.')
    events = sorted(INPUTS.events_yaml.get('events', []), key=lambda e: e['date'], reverse=True)
    lines_f = st.pills('Filter by line', ['1L', '2L', '3L', '4L+', 'SMM'], selection_mode='multi',
                       default=['1L', '2L', '3L', '4L+', 'SMM'], key='ev_filter') or []
    html = ''
    for e in events:
        if e.get('line') not in lines_f:
            continue
        t = e.get('type', '')
        tcls = 'pending' if t == 'Pending' else ('withdrawal' if t == 'Withdrawal' else '')
        html += (f'<div class="tl"><div class="tl-date">{escape(e["date"])}</div><div>'
                 f'<span class="tl-name">{escape(e["name"])}</span>'
                 f'<span class="tag">{escape(e.get("line", ""))}</span><span class="tag {tcls}">{escape(t)}</span>'
                 f'<div class="tl-det">{escape(e.get("details", ""))}</div></div></div>')
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)

# ── VALIDATION & SENSITIVITY ─────────────────────────────────────
with tabs[5]:
    section('External benchmarks', 'Model outputs compared with independent US sources.')
    bt = mm.benchmark_table(annual, incidence, cohort(SC_JSON, 2016), sc['frac_te'])
    st.dataframe(bt, width='stretch', hide_index=True,
                 column_config={'Read-out': st.column_config.TextColumn(width='large')})

    section(f'What drives patients on therapy in {horizon}?',
            'One-way sensitivity: each lever is moved ±20% (rate trend ±1 pp/yr) and the full model is '
            're-run. Bars show the resulting range around the current scenario.')
    with st.spinner('Running sensitivity simulations…'):
        tor = sensitivity(SC_JSON, horizon)
    base_v = float(tor['Base'].iloc[0])
    fig = go.Figure()
    fig.add_trace(go.Bar(y=tor['Parameter'], x=tor['Low'] - base_v, base=base_v, orientation='h',
                         name='Lever −20%', marker=dict(color=SLOTS[0], line=dict(color='#fff', width=2)),
                         customdata=np.stack([tor['Low'], tor['Low input']], axis=1),
                         hovertemplate='Input %{customdata[1]:.3g} → %{customdata[0]:,.0f}<extra>−20%</extra>'))
    fig.add_trace(go.Bar(y=tor['Parameter'], x=tor['High'] - base_v, base=base_v, orientation='h',
                         name='Lever +20%', marker=dict(color=SLOTS[1], line=dict(color='#fff', width=2)),
                         customdata=np.stack([tor['High'], tor['High input']], axis=1),
                         hovertemplate='Input %{customdata[1]:.3g} → %{customdata[0]:,.0f}<extra>+20%</extra>'))
    fig.add_shape(type='line', x0=base_v, x1=base_v, y0=-0.5, y1=len(tor) - 0.5, line=dict(color=INK, width=1))
    fig.update_layout(barmode='overlay', bargap=0.35)
    fig.update_xaxes(tickformat=',.0f', showgrid=True, gridcolor=GRID, title=f'Patients on therapy, {horizon}')
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(chart(fig, 60 + 42 * len(tor), hover='closest', margin=dict(l=8, r=12, t=36, b=40)),
                    width='stretch')

    section('Scenario comparison', 'Save scenarios from the sidebar to compare them side by side.')
    if st.session_state.saved:
        comp = pd.DataFrame([{k: v for k, v in s.items() if k != 'json'} for s in st.session_state.saved])
        cur = {'Scenario': 'Current', f'Dx {NOW_YEAR}': INC.loc[NOW_YEAR, 'Cases'],
               f'On therapy {NOW_YEAR}': A.loc[NOW_YEAR, 'On_Therapy'],
               f'On therapy {horizon}': A.loc[horizon, 'On_Therapy'],
               f'1L starts {horizon}': A.loc[horizon, 'New_Starts_1L'],
               f'4L+ patients {horizon}': A.loc[horizon, 'Total_4L+']}
        comp = pd.concat([comp, pd.DataFrame([cur])], ignore_index=True)
        num = [c for c in comp.columns if c != 'Scenario']
        st.dataframe(comp.style.format({c: '{:,.0f}' for c in num}), width='stretch', hide_index=True)
    else:
        st.markdown('<div class="callout">No saved scenarios yet. Adjust levers in the sidebar, then '
                    'click <b>Save</b>.</div>', unsafe_allow_html=True)

# ── METHODS ──────────────────────────────────────────────────────
with tabs[6]:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        section('Model structure')
        st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.3;
  node [shape=box, style="rounded,filled", fontname="Inter", fontsize=11, color="#d0d5dd", fillcolor="#ffffff", fontcolor="#0f1b2d"];
  edge [color="#8a909b", fontname="Inter", fontsize=9, fontcolor="#4b5565"];
  inc [label="Incidence\\nUSCS 1999–2022\\n+ Census projection", fillcolor="#eaf2fc"];
  tx  [label="Treated\\n(× treated fraction)"];
  te  [label="1L transplant-\\neligible"]; ti [label="1L transplant-\\nineligible"];
  l2 [label="2L"]; l3 [label="3L"]; l4 [label="4L+"];
  inc -> tx -> te; tx -> ti;
  te -> l2 [label="Weibull PFS × p(2L)"]; ti -> l2;
  l2 -> l3 [label="× p(3L)"]; l3 -> l4 [label="× p(4L+)"]; l4 -> l4 [label="re-treat"];
}""", width='stretch')
        st.caption('Every line also exits to death on therapy or to no further treatment.')
        st.markdown("""
**Incidence.** Age- and sex-specific rates from CDC USCS (pooled 2017–2019 and 2021–2022)
are applied to U.S. Census Bureau 2023 National Population Projections (middle series).
Optional underlying rate trend; optional single-factor anchoring to the ACS 2026 estimate.

**Regimen assignment.** Within each segment (1L transplant-eligible, 1L transplant-ineligible,
2L, 3L, 4L+), regimen share follows logistic diffusion from FDA approval or guideline listing,
with logistic displacement by successors, normalised to 100%.

**Time on line.** For each regimen, a Weibull progression hazard is fitted to the
median PFS from the pivotal trial, or to a landmark PFS rate where the median has not been
reached. The real-world multiplier rescales it, and a line-specific constant death hazard
competes with it. Within-month competing risks are exact:
`P(exit) = 1 − exp(−(ΔΛ_prog + μ))`, split pro rata.

**Computation.** Stocks and flows are convolutions of monthly starts with each regimen's
survival and exit kernels. 4L+ is solved recursively to allow re-treatment. A 15-year
back-cast burn-in seeds prevalent patients at 1999.
""")
    with c2:
        section('Limitations', 'Ranked by impact on forecast use.')
        for p in mm.pitfalls(sc):
            st.markdown(f'<div class="pit"><span class="badge b-{p["severity"]}">{p["severity"].upper()}</span>'
                        f'<span class="pit-t">{escape(p["title"])}</span>'
                        f'<div class="pit-d">{escape(p["detail"])}</div></div>', unsafe_allow_html=True)

    with st.expander('Active scenario parameters'):
        st.json(sc, expanded=False)
    with st.expander('Patient guide: what this model can and cannot tell you'):
        st.markdown("""
This is a **population-level** model of how people with multiple myeloma in the US move through
treatment over time. It does **not** predict what will happen to any individual.

*Questions you might ask your care team:*
1. Which line of therapy am I on, and what is the goal of this treatment?
2. Am I a candidate for a stem-cell transplant, a quadruplet regimen, or a clinical trial?
3. How long do people usually stay on this treatment, and what happens if it stops working?
4. Are CAR-T or bispecific antibody therapies options for me now or later?
5. What side effects should I watch for, and who do I call about them?
""")
    st.markdown('<div class="disclaimer"><b>Research use only.</b> This tool supports epidemiology '
                'and forecasting work. It is not medical advice. Adoption parameters are analyst '
                'calibrations and should be fitted to real-world data before commercial use.</div>',
                unsafe_allow_html=True)
