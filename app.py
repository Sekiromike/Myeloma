"""
Multiple Myeloma Disease Model Explorer
========================================
A scientific & interactive Streamlit dashboard.
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
from metrics import (compute_occupancy, compute_transition_rates,
                     compute_regimen_shares, compute_sankey_data,
                     compute_duration_stats, generate_insights,
                     tornado_sensitivity)

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="MM Disease Model Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header */
.main-title {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.sub-title { font-size: 1rem; color: #888; margin-top: -0.4rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    border: 1px solid #333; border-radius: 12px;
    padding: 1.2rem 1.4rem; text-align: center;
}
.metric-card h3 { color: #aaa; font-size: 0.78rem; font-weight: 500;
                   text-transform: uppercase; letter-spacing: 0.05em; margin:0; }
.metric-card p  { color: #fff; font-size: 1.9rem; font-weight: 700; margin: .3rem 0 0 0; }

/* Insight pills */
.insight-box {
    background: #1a1a2e; border-left: 4px solid #667eea;
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: .7rem;
    font-size: 0.92rem; color: #ccc; line-height: 1.5;
}

/* Disclaimer */
.disclaimer {
    background: #2a1a1a; border: 1px solid #553333; border-radius: 8px;
    padding: 1rem; font-size: 0.82rem; color: #cc8888; margin-top: 1rem;
}

/* Tab styling */
div[data-baseweb="tab-list"] { gap: 0.5rem; }
div[data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;} footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# ── Colour Palette ───────────────────────────────────────────────
PAL = {
    '1st Line':  '#667eea',
    '2nd Line':  '#f59e0b',
    '3rd Line':  '#ef4444',
    '4th Line+': '#8b5cf6',
}
REGIMEN_PAL = px.colors.qualitative.Pastel + px.colors.qualitative.Set2

# ── Data Loading ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_all():
    base = Path(__file__).resolve().parent
    paths = auto_discover_paths(base)

    # Diagnostics for cloud debugging
    if 'simulation' not in paths:
        import os
        files_here = os.listdir(base)
        outputs_dir = base / 'outputs'
        outputs_files = os.listdir(outputs_dir) if outputs_dir.exists() else []
        raise FileNotFoundError(
            f"Could not find simulation CSV.\n"
            f"  base dir = {base}\n"
            f"  files in base = {files_here}\n"
            f"  outputs/ contents = {outputs_files}\n"
            f"  discovered paths = {paths}"
        )

    if 'params' not in paths:
        raise FileNotFoundError(f"params.yaml not found. Discovered: {paths}")

    df   = load_simulation(paths['simulation'])
    par  = load_params(paths['params'])
    regs = load_regimens_yaml(paths['regimens']) if 'regimens' in paths else {}
    return df, par, regs, paths

try:
    df, params, regimens_yaml, data_paths = _load_all()
except FileNotFoundError as e:
    st.error(f"🚨 Data files not found. Details:\n\n```\n{e}\n```")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    # Time range
    min_date = df['Date'].min().to_pydatetime()
    max_date = df['Date'].max().to_pydatetime()
    date_range = st.slider("Date range", min_value=min_date, max_value=max_date,
                           value=(min_date, max_date), format="YYYY-MM")
    mask = (df['Date'] >= pd.Timestamp(date_range[0])) & \
           (df['Date'] <= pd.Timestamp(date_range[1]))
    dff = df.loc[mask].copy()

    st.markdown("---")
    st.markdown("### 📊 Parameters")
    treated_frac = st.slider("Treated fraction",
                             0.50, 1.00, float(params['uptake']['treated_fraction']),
                             step=0.05, help="Share of diagnosed patients initiating treatment")
    p_2l = st.slider("P(reach 2L)",
                      0.30, 1.00, float(params['attrition']['p_reach_2l']),
                      step=0.05, help="Probability of advancing from 1L to 2L")
    p_3l = st.slider("P(reach 3L | 2L)",
                      0.20, 1.00, float(params['attrition'].get('p_reach_3l_given_2l', 0.6)),
                      step=0.05)

    # Build scenario params dict
    scenario_params = {
        'uptake': {**params['uptake'], 'treated_fraction': treated_frac},
        'attrition': {**params['attrition'], 'p_reach_2l': p_2l,
                      'p_reach_3l_given_2l': p_3l},
        'mortality': params['mortality'],
        'durations_months_median': params.get('durations_months_median', {}),
    }

    st.markdown("---")
    st.caption("Data: USCS Incidence + Microsimulation  \n"
               "Model: Weibull PFS, Logistic Adoption")

# ── Header ───────────────────────────────────────────────────────
st.markdown('<p class="main-title">Multiple Myeloma Disease Model Explorer</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">Interactive visualization of patient flow, '
            'treatment lines, and regimen adoption</p>', unsafe_allow_html=True)

# ── KPI Row ──────────────────────────────────────────────────────
recent = dff.tail(12)
kpi_total = sum(recent.get(c, pd.Series([0])).mean() for c in TOTAL_COLS)
kpi_new   = recent['New_Starts_1L'].mean()
kpi_1l    = recent.get('Total_1L', pd.Series([0])).mean()

k1, k2, k3, k4 = st.columns(4)
for col, title, val, fmt in [
    (k1, 'Total on Treatment', kpi_total, '{:,.0f}'),
    (k2, 'Monthly New 1L Starts', kpi_new, '{:,.0f}'),
    (k3, '1L Prevalence', kpi_1l, '{:,.0f}'),
    (k4, 'Lines Modeled', 4, '{}'),
]:
    col.markdown(f'<div class="metric-card"><h3>{title}</h3>'
                 f'<p>{fmt.format(val)}</p></div>', unsafe_allow_html=True)

st.markdown("")

# ═══════════════════ TABS ═══════════════════════════════════════
tab_patient, tab_provider = st.tabs(["🧑‍🤝‍🧑  Patient View", "🏥  Provider View"])

# ─────────────────────────────────────────────────────────────────
#  PATIENT VIEW
# ─────────────────────────────────────────────────────────────────
with tab_patient:

    # ── What is this model? ──────────────────────────────────────
    with st.expander("ℹ️  What is this model?", expanded=False):
        st.markdown("""
        This is a **population-level simulation** of how patients with
        Multiple Myeloma move through lines of treatment over time.
        It is based on published clinical evidence and US incidence data.

        **It is NOT a prediction for any individual patient.**
        It helps researchers, clinicians, and policymakers understand
        population-level trends and the impact of new therapies.
        """)

    # ── Journey Sankey ───────────────────────────────────────────
    st.markdown("#### 🗺️ The Treatment Journey")
    st.caption("Average monthly patient flow (most recent 12 months)")

    sankey = compute_sankey_data(dff, scenario_params)
    node_colors = ['#667eea', '#667eea', '#f59e0b', '#ef4444',
                   '#10b981', '#6b7280']

    fig_sankey = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            pad=20, thickness=28, line=dict(color='#333', width=1),
            label=sankey['nodes'],
            color=node_colors,
        ),
        link=dict(
            source=sankey['link_source'],
            target=sankey['link_target'],
            value=sankey['link_value'],
            customdata=sankey['link_label'],
            hovertemplate='%{customdata}<extra></extra>',
            color='rgba(102,126,234,0.25)',
        ),
    ))
    fig_sankey.update_layout(
        height=340, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13, color='#ccc'),
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

    # ── People in Each Treatment Line ────────────────────────────
    st.markdown("#### 📈 People in Each Treatment Line Over Time")

    occ = compute_occupancy(dff)
    fig_occ = px.area(occ, x='Date', y='Patients', color='Line',
                      color_discrete_map=PAL,
                      category_orders={'Line': list(PAL.keys())})
    fig_occ.update_layout(
        height=370,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(title='Number of Patients', gridcolor='#222'),
        legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'),
        margin=dict(l=50, r=20, t=30, b=40),
        hovermode='x unified',
    )
    st.plotly_chart(fig_occ, use_container_width=True)

    # ── Typical Time in Each Line ────────────────────────────────
    st.markdown("#### ⏱️ Typical Time Spent in Each Treatment Line")
    dur = compute_duration_stats(scenario_params)

    fig_dur = px.bar(dur, x='Median Months', y='Line', orientation='h',
                     color='Line', color_discrete_map={
                         '1st Line': PAL['1st Line'],
                         '2nd Line': PAL['2nd Line'],
                         '3rd Line+': PAL['3rd Line'],
                     },
                     text='Median Months')
    fig_dur.update_traces(texttemplate='%{text} mo', textposition='outside')
    fig_dur.update_layout(
        height=220, showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Months', showgrid=True, gridcolor='#222'),
        yaxis=dict(title=''),
        margin=dict(l=80, r=40, t=10, b=40),
    )
    st.plotly_chart(fig_dur, use_container_width=True)

    # ── Insights ─────────────────────────────────────────────────
    st.markdown("#### 💡 Key Takeaways")
    for ins in generate_insights(dff, scenario_params, 'patient'):
        st.markdown(f'<div class="insight-box">{ins}</div>',
                    unsafe_allow_html=True)

    # ── Questions to Ask Your Doctor ─────────────────────────────
    with st.expander("🩺 Questions to Ask Your Doctor"):
        st.markdown("""
        1. *What treatment line am I currently on, and what does that mean?*
        2. *What are the newer therapy options available for my stage?*
        3. *How long do patients typically stay on my current treatment?*
        4. *What factors determine whether I would move to a next line of therapy?*
        5. *Are there clinical trials I might be eligible for?*
        """)

    # ── Disclaimer ───────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This tool is for <em>educational and research
    purposes only</em>. It is <strong>not medical advice</strong> and does not
    predict individual patient outcomes. Always consult your healthcare provider
    for decisions about your care.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  PROVIDER VIEW
# ─────────────────────────────────────────────────────────────────
with tab_provider:

    # ── Model Transparency ───────────────────────────────────────
    with st.expander("📋 Model Transparency & Parameters", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Active Parameters**")
            st.json(scenario_params, expanded=False)
        with c2:
            st.markdown("**Assumptions**")
            st.markdown("""
            - **Hazard model:** Weibull PFS (shape k = 1.3)
            - **Adoption:** Calibrated logistic diffusion
            - **Lines:** 1L → 2L → 3L → 4L+ (sequential)
            - **Mortality:** Line-specific constant monthly hazard
            - **Incidence:** USCS 1999-2022, projected to 2026
            """)

    # ── Stacked Occupancy ────────────────────────────────────────
    st.markdown("#### Prevalence / Occupancy by Line")
    st.caption("Definition: Number of patients actively on treatment in each line "
               "at each time point")

    occ_prov = compute_occupancy(dff)
    fig_prov_occ = px.area(occ_prov, x='Date', y='Patients', color='Line',
                           color_discrete_map=PAL,
                           category_orders={'Line': list(PAL.keys())})
    fig_prov_occ.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(title='Patient Count', gridcolor='#222'),
        legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'),
        margin=dict(l=60, r=20, t=30, b=40),
        hovermode='x unified',
    )
    st.plotly_chart(fig_prov_occ, use_container_width=True)

    # ── Transition Rates ─────────────────────────────────────────
    st.markdown("#### Transition Rates (Monthly)")
    st.caption("Definition: Approximate rate = Δ(next-line stock, clipped ≥ 0) "
               "÷ current-line stock.  Smoothed with 6-month rolling mean.")

    tr = compute_transition_rates(dff)
    rate_cols = [c for c in tr.columns if '→' in c]
    if rate_cols:
        tr_smooth = tr.copy()
        for c in rate_cols:
            tr_smooth[c] = tr_smooth[c].rolling(6, min_periods=1).mean()
        fig_tr = go.Figure()
        colors_tr = ['#667eea', '#f59e0b', '#ef4444']
        for i, c in enumerate(rate_cols):
            fig_tr.add_trace(go.Scatter(
                x=tr_smooth['Date'], y=tr_smooth[c],
                name=c, mode='lines',
                line=dict(width=2.5, color=colors_tr[i % len(colors_tr)]),
                hovertemplate=f'{c}: %{{y:.4f}}<extra></extra>',
            ))
        fig_tr.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(title='Rate', gridcolor='#222', tickformat='.3f'),
            xaxis=dict(showgrid=False),
            legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
            margin=dict(l=60, r=20, t=30, b=40),
            hovermode='x unified',
        )
        st.plotly_chart(fig_tr, use_container_width=True)

    # ── Regimen Share Evolution ──────────────────────────────────
    st.markdown("#### Regimen Share Evolution")
    line_sel = st.selectbox("Select line", ['1L', '2L', '3L', '4L+'],
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
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(title='Market Share', tickformat='.0%', gridcolor='#222'),
            xaxis=dict(showgrid=False),
            legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
            margin=dict(l=60, r=20, t=50, b=40),
            hovermode='x unified',
        )
        st.plotly_chart(fig_share, use_container_width=True)
    else:
        st.info(f"No regimen-level data for {line_sel}")

    # ── Mortality Panel ──────────────────────────────────────────
    st.markdown("#### Mortality Hazard by Line")
    st.caption("Constant monthly hazard rates from `params.yaml`")
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
        height=280, showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='#222'),
        margin=dict(l=60, r=20, t=10, b=40),
    )
    st.plotly_chart(fig_mort, use_container_width=True)

    # ── Sensitivity Tornado ──────────────────────────────────────
    st.markdown("#### Sensitivity Analysis (±20% on 1L Occupancy)")
    st.caption("Illustrative linear sensitivity. For full model re-runs, "
               "adjust sidebar parameters.")
    tornado = tornado_sensitivity(dff, scenario_params, 'Total_1L')
    fig_torn = go.Figure()
    for _, row in tornado.iterrows():
        fig_torn.add_trace(go.Bar(
            y=[row['Parameter']], x=[row['High (+20%)'] - row['Base']],
            base=[row['Base']], orientation='h', name='High',
            marker_color='#667eea', showlegend=False,
            hovertemplate=f"High: {row['High (+20%)']:,.0f}<extra></extra>",
        ))
        fig_torn.add_trace(go.Bar(
            y=[row['Parameter']], x=[row['Low (−20%)'] - row['Base']],
            base=[row['Base']], orientation='h', name='Low',
            marker_color='#ef4444', showlegend=False,
            hovertemplate=f"Low: {row['Low (−20%)']:,.0f}<extra></extra>",
        ))
    fig_torn.update_layout(
        height=220, barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='1L Occupancy (patients)', gridcolor='#222'),
        yaxis=dict(title=''),
        margin=dict(l=140, r=20, t=10, b=40),
    )
    st.plotly_chart(fig_torn, use_container_width=True)

    # ── Provider Insights ────────────────────────────────────────
    st.markdown("#### 🔍 Key Findings")
    for ins in generate_insights(dff, scenario_params, 'provider'):
        st.markdown(f'<div class="insight-box">{ins}</div>',
                    unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📥 Export")
    c_exp1, c_exp2 = st.columns(2)
    with c_exp1:
        csv = dff.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Filtered Data (CSV)", csv,
                           "mm_simulation_filtered.csv", "text/csv")
    with c_exp2:
        if not shares.empty:
            csv_s = shares.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Regimen Shares (CSV)", csv_s,
                               f"mm_regimen_shares_{line_sel}.csv", "text/csv")
