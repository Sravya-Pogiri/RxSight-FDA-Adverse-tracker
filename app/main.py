import sys
import os
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.openfda_api import fetch_drug_events
from scripts.models import (
    get_reaction_trends,
    train_trend_regression,
    cluster_reactions,
    get_demographics,
    get_severity_breakdown,
)

st.set_page_config(
    page_title="FAERS · Drug Insight Engine",
    layout="wide",
    page_icon="⬡",
    initial_sidebar_state="expanded",
)

# all the colors in one place so i don't have to hunt through the file
# when i want to change something
PALETTE = {
    "bg": "#080C14",
    "surface": "#0E1420",
    "surface2": "#141B2D",
    "surface3": "#1A2238",
    "border": "#1E2A42",
    "border2": "#263450",
    "teal": "#00D4AA",
    "teal_dim": "#007A62",
    "teal_glow": "rgba(0,212,170,0.08)",
    "coral": "#FF6B6B",
    "coral_dim": "#8B2C2C",
    "amber": "#F5A623",
    "blue": "#4A90E2",
    "purple": "#9B6DFF",
    "text_pri": "#E8EDF5",
    "text_sec": "#7A8AA8",
    "text_muted": "#4A5568",
}

# reusing this dict across every chart so they all look consistent
# beats copy-pasting layout kwargs into every single figure
CHART_THEME = dict(
    plot_bgcolor=PALETTE["surface"],
    paper_bgcolor=PALETTE["surface"],
    font=dict(family="DM Mono, monospace", color=PALETTE["text_sec"], size=11),
    xaxis=dict(
        gridcolor=PALETTE["border"],
        linecolor=PALETTE["border"],
        tickcolor=PALETTE["border2"],
        tickfont=dict(size=10, color=PALETTE["text_muted"]),
        title_font=dict(size=11, color=PALETTE["text_sec"]),
    ),
    yaxis=dict(
        gridcolor=PALETTE["border"],
        linecolor=PALETTE["border"],
        tickcolor=PALETTE["border2"],
        tickfont=dict(size=10, color=PALETTE["text_muted"]),
        title_font=dict(size=11, color=PALETTE["text_sec"]),
    ),
    legend=dict(
        bgcolor="rgba(14,20,32,0.9)",
        bordercolor=PALETTE["border2"],
        borderwidth=1,
        font=dict(size=11, color=PALETTE["text_sec"]),
    ),
    margin=dict(l=8, r=8, t=16, b=8),
)

# streamlit doesn't give you a lot of styling hooks out of the box
# so we inject a big css block at the top — not ideal but it works
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    -webkit-font-smoothing: antialiased;
}}
.stApp {{
    background-color: {PALETTE['bg']};
    color: {PALETTE['text_pri']};
}}
.block-container {{ padding-top: 0 !important; max-width: 100% !important; }}
[data-testid="stSidebar"] {{
    background-color: {PALETTE['surface']} !important;
    border-right: 1px solid {PALETTE['border']} !important;
}}
[data-testid="stSidebar"] .stMarkdown h2 {{
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: {PALETTE['text_muted']} !important;
    margin: 20px 0 10px !important;
}}
[data-testid="stSidebar"] .stTextInput input {{
    background: {PALETTE['surface2']} !important;
    border: 1px solid {PALETTE['border2']} !important;
    color: {PALETTE['text_pri']} !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.05em !important;
}}
[data-testid="stSidebar"] .stTextInput input:focus {{
    border-color: {PALETTE['teal']} !important;
    box-shadow: 0 0 0 3px {PALETTE['teal_glow']} !important;
}}
[data-testid="stSidebar"] .stSlider [data-testid="stSliderThumb"] {{
    background: {PALETTE['teal']} !important;
}}
[data-testid="stSidebar"] .stCheckbox label {{
    color: {PALETTE['text_sec']} !important;
    font-size: 13px !important;
}}
.page-header {{
    background: linear-gradient(180deg, {PALETTE['surface2']} 0%, {PALETTE['bg']} 100%);
    border-bottom: 1px solid {PALETTE['border']};
    padding: 28px 40px 20px;
    margin: 0 -1rem 32px;
    position: relative;
    overflow: hidden;
}}
.page-header::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,212,170,0.06) 0%, transparent 70%);
    pointer-events: none;
}}
.header-eyebrow {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {PALETTE['teal']};
    margin-bottom: 6px;
    font-family: 'DM Mono', monospace;
}}
.header-title {{
    font-size: 30px;
    font-weight: 300;
    color: {PALETTE['text_pri']};
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin: 0;
}}
.header-title strong {{
    font-weight: 600;
    background: linear-gradient(90deg, {PALETTE['teal']} 0%, {PALETTE['blue']} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.header-sub {{
    font-size: 13px;
    color: {PALETTE['text_muted']};
    margin-top: 8px;
    font-weight: 300;
    letter-spacing: 0.02em;
}}
.drug-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: {PALETTE['surface2']};
    border: 1px solid {PALETTE['teal_dim']};
    border-radius: 6px;
    padding: 6px 14px 6px 10px;
    margin-bottom: 24px;
}}
.drug-badge-dot {{
    width: 7px; height: 7px;
    border-radius: 50%;
    background: {PALETTE['teal']};
    box-shadow: 0 0 6px {PALETTE['teal']};
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.5; transform: scale(0.85); }}
}}
.drug-badge-text {{
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    color: {PALETTE['teal']};
    letter-spacing: 0.08em;
}}
.drug-badge-label {{
    font-size: 10px;
    color: {PALETTE['text_muted']};
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}
.metric-row {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
.m-card {{
    flex: 1;
    min-width: 130px;
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}}
.m-card:hover {{ border-color: {PALETTE['border2']}; transform: translateY(-2px); }}
.m-card-accent {{
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: 10px 0 0 10px;
}}
.m-card-label {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {PALETTE['text_muted']};
    margin-bottom: 10px;
}}
.m-card-value {{
    font-family: 'DM Mono', monospace;
    font-size: 26px;
    font-weight: 500;
    color: {PALETTE['text_pri']};
    line-height: 1;
}}
.m-card-sub {{
    font-size: 11px;
    color: {PALETTE['text_muted']};
    margin-top: 5px;
    font-weight: 300;
}}
.section-label {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {PALETTE['text_muted']};
    margin: 28px 0 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {PALETTE['border']};
}}
.chart-card {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}}
.chart-card-title {{
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['text_sec']};
    margin-bottom: 16px;
}}
.stTabs [data-baseweb="tab-list"] {{
    background: {PALETTE['surface']} !important;
    border-bottom: 1px solid {PALETTE['border']} !important;
    gap: 4px;
    padding: 0 4px;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    color: {PALETTE['text_muted']} !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 18px !important;
    border: none !important;
    background: transparent !important;
    transition: color 0.15s !important;
}}
.stTabs [aria-selected="true"] {{
    color: {PALETTE['teal']} !important;
    background: {PALETTE['teal_glow']} !important;
    border-bottom: 2px solid {PALETTE['teal']} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding: 24px 0 0 !important;
}}
.cluster-card {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
}}
.cluster-card-head {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['teal']};
    margin-bottom: 10px;
    font-family: 'DM Mono', monospace;
}}
.rxn-pill {{
    display: inline-block;
    background: {PALETTE['surface2']};
    border: 1px solid {PALETTE['border2']};
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    color: {PALETTE['text_sec']};
    margin: 3px 3px 0 0;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.02em;
}}
.forecast-card {{
    background: {PALETTE['surface2']};
    border: 1px solid {PALETTE['border2']};
    border-top: 2px solid {PALETTE['teal']};
    border-radius: 0 0 10px 10px;
    padding: 18px 20px;
    text-align: center;
}}
.forecast-date {{
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {PALETTE['text_muted']};
    font-family: 'DM Mono', monospace;
    margin-bottom: 8px;
}}
.forecast-val {{
    font-size: 34px;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
    color: {PALETTE['text_pri']};
    line-height: 1;
}}
.forecast-unit {{
    font-size: 11px;
    color: {PALETTE['text_muted']};
    margin-top: 4px;
    font-weight: 300;
}}
.eval-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 24px;
}}
.eval-cell {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}}
.eval-cell-label {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {PALETTE['text_muted']};
    margin-bottom: 8px;
}}
.eval-cell-val {{
    font-family: 'DM Mono', monospace;
    font-size: 22px;
    font-weight: 500;
    color: {PALETTE['teal']};
}}
.eval-cell-unit {{
    font-size: 10px;
    color: {PALETTE['text_muted']};
    margin-top: 3px;
}}
.sb-drug-card {{
    background: {PALETTE['surface2']};
    border: 1px solid {PALETTE['border2']};
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 12px;
}}
.sb-drug-card-name {{
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    font-weight: 500;
    color: {PALETTE['teal']};
    letter-spacing: 0.06em;
}}
.sb-drug-card-sub {{
    font-size: 11px;
    color: {PALETTE['text_muted']};
    margin-top: 4px;
}}
.empty-state {{
    background: {PALETTE['surface']};
    border: 1px dashed {PALETTE['border2']};
    border-radius: 10px;
    padding: 36px 24px;
    text-align: center;
    color: {PALETTE['text_muted']};
    font-size: 13px;
}}
.empty-state-icon {{ font-size: 28px; margin-bottom: 10px; }}
.stAlert {{ border-radius: 8px !important; border: 1px solid {PALETTE['border2']} !important; }}
.stSpinner > div {{ border-top-color: {PALETTE['teal']} !important; }}
</style>
""", unsafe_allow_html=True)


# sidebar lives here — kept it simple, just a search box and a couple toggles
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 4px;">
        <div style="font-size:10px; font-weight:600; letter-spacing:0.18em; text-transform:uppercase; color:#4A5568; font-family:'DM Mono',monospace;">
            FDA · FAERS
        </div>
        <div style="font-size:18px; font-weight:600; color:#E8EDF5; margin-top:4px; letter-spacing:-0.3px;">
            Drug Insight Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Query")
    drug_query = st.text_input(
        "",
        placeholder="e.g. IBUPROFEN",
        label_visibility="collapsed",
        help="Enter the brand name as it appears in FAERS (all-caps preferred).",
    )

    st.markdown("## Data Source")
    use_live_api = st.checkbox("Live OpenFDA API", value=True)
    api_limit = st.slider("Record limit", 10, 200, 50, step=10)

    # show a little confirmation card so the user knows what they searched
    if drug_query:
        drug_upper = drug_query.strip().upper()
        st.markdown("## Active Query")
        st.markdown(f"""
        <div class="sb-drug-card">
            <div class="sb-drug-card-name">{drug_upper}</div>
            <div class="sb-drug-card-sub">FAERS adverse event search</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#4A5568; line-height:1.6;">
        Data sourced from FDA FAERS quarterly bulk data
        and the OpenFDA REST API. For research use only.
    </div>
    """, unsafe_allow_html=True)


# main page header
st.markdown("""
<div class="page-header">
    <div class="header-eyebrow">⬡ &nbsp; FDA Adverse Event Reporting System</div>
    <h1 class="header-title">Rx<strong>Sight</strong></h1>
    <p class="header-sub">
        Adverse event search · reaction clustering · trend forecasting · patient demographics
    </p>
</div>
""", unsafe_allow_html=True)

# try to ping postgres — if it's not running we still want the app to load,
# just with a warning banner and sample data instead of crashing
db_connected = False
try:
    from scripts.db_config import get_postgres_engine
    from sqlalchemy import text
    engine = get_postgres_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    db_connected = True
except (ImportError, OSError):
    pass

if not db_connected:
    st.warning("⚠️ Database not connected. Severity and demographics charts are showing sample placeholder data. Make sure Docker is running and the ETL pipeline has been executed.")

# nothing to show until someone actually types something
if not drug_query:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">⬡</div>
        Enter a drug name in the sidebar to begin analysis
    </div>
    """, unsafe_allow_html=True)
    st.stop()

drug_upper = drug_query.strip().upper()

# little animated dot badge so it's obvious what drug is being analyzed
st.markdown(f"""
<div class="drug-badge">
    <span class="drug-badge-dot"></span>
    <span>
        <span class="drug-badge-label">Active Query &nbsp;·&nbsp;</span>
        <span class="drug-badge-text">{drug_upper}</span>
    </span>
</div>
""", unsafe_allow_html=True)

live_events = []
reactions_flat = []

# hit the API and flatten all the reaction terms into a single list
# so they're easy to count later
if use_live_api:
    with st.spinner("Fetching records from OpenFDA …"):
        live_events = fetch_drug_events(drug_upper, limit=api_limit)
    if live_events:
        for event in live_events:
            for rxn in event['patient']['reaction']:
                term = rxn['reactionmeddrapt'].strip()
                if term:
                    reactions_flat.append(term)

n_events = len(live_events)
n_unique = len(set(reactions_flat))
n_reactions = len(reactions_flat)

# top-level summary cards
st.markdown(f"""
<div class="metric-row">
    <div class="m-card">
        <div class="m-card-accent" style="background:{PALETTE['teal']};"></div>
        <div class="m-card-label">Reports Fetched</div>
        <div class="m-card-value">{n_events:,}</div>
        <div class="m-card-sub">live FAERS records</div>
    </div>
    <div class="m-card">
        <div class="m-card-accent" style="background:{PALETTE['blue']};"></div>
        <div class="m-card-label">Total Reactions</div>
        <div class="m-card-value">{n_reactions:,}</div>
        <div class="m-card-sub">MedDRA terms reported</div>
    </div>
    <div class="m-card">
        <div class="m-card-accent" style="background:{PALETTE['purple']};"></div>
        <div class="m-card-label">Unique Reactions</div>
        <div class="m-card-value">{n_unique:,}</div>
        <div class="m-card-sub">distinct adverse events</div>
    </div>
    <div class="m-card">
        <div class="m-card-accent" style="background:{PALETTE['coral']};"></div>
        <div class="m-card-label">API Limit</div>
        <div class="m-card-value">{api_limit}</div>
        <div class="m-card-sub">records requested</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "  Reactions & Profile  ",
    "  Trend Forecast  ",
    "  Reaction Clusters  ",
])

# -- TAB 1: CORE DATA & DEMOGRAPHICS --
with tab1:
    col_left, col_right = st.columns([1.7, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-label">Top Reported Reactions</div>', unsafe_allow_html=True)

        if reactions_flat:
            # counter does the heavy lifting here, then we just take the top 15
            rxn_counts = Counter(reactions_flat)
            top_rxns = pd.DataFrame(rxn_counts.most_common(15), columns=['Reaction', 'Count'])

            fig_bar = go.Figure(go.Bar(
                y=top_rxns['Reaction'],
                x=top_rxns['Count'],
                orientation='h',
                marker=dict(
                    color=top_rxns['Count'],
                    colorscale=[
                        [0, PALETTE["surface3"]],
                        [0.4, PALETTE["teal_dim"]],
                        [1, PALETTE["teal"]],
                    ],
                    line=dict(width=0),
                ),
                text=top_rxns['Count'],
                textposition='outside',
                textfont=dict(size=10, color=PALETTE["text_muted"], family="DM Mono, monospace"),
                hovertemplate='<b>%{y}</b><br>%{x} reports<extra></extra>',
            ))
            fig_bar.update_layout(**CHART_THEME)
            fig_bar.update_layout(
                height=400,
                yaxis=dict(autorange='reversed', tickfont=dict(size=11, color=PALETTE["text_sec"])),
                xaxis=dict(title="Reports"),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<div class="empty-state"><div class="empty-state-icon">⬡</div>Enable the OpenFDA API to see reaction data</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-label">Outcome Severity</div>', unsafe_allow_html=True)

        sev_df = get_severity_breakdown(drug_upper)
        # manual color map because the default plotly colors look terrible for medical data
        SEVERITY_COLORS = {
            'Death':                 '#FF4B4B',
            'Life-threatening':      '#FF6B35',
            'Hospitalization':       '#F5A623',
            'Disability':            '#00D4AA',
            'Congenital anomaly':    '#4A90E2',
            'Required intervention': '#9B6DFF',
            'Other':                 '#4A5568',
            'Standard':              '#3D4E6A',
        }
        colors = [SEVERITY_COLORS.get(s, '#4A5568') for s in sev_df['severity']]

        fig_pie = go.Figure(go.Pie(
            labels=sev_df['severity'],
            values=sev_df['count'],
            hole=0.6,
            marker=dict(colors=colors, line=dict(color=PALETTE["surface"], width=2)),
            textinfo='percent',
            textfont=dict(size=10, family="DM Mono, monospace"),
            hovertemplate='<b>%{label}</b><br>%{value} cases (%{percent})<extra></extra>',
        ))
        # total case count in the donut hole
        fig_pie.add_annotation(
            text=f"<b>{sev_df['count'].sum():,}</b><br><span style='font-size:8px'>cases</span>",
            x=0.5, y=0.5,
            font=dict(size=14, color=PALETTE["text_pri"], family="DM Mono, monospace"),
            showarrow=False,
        )
        fig_pie.update_layout(**CHART_THEME)
        fig_pie.update_layout(
            height=300,
            showlegend=True,
            legend=dict(orientation="v", font=dict(size=10, color=PALETTE["text_sec"])),
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-label">Patient Demographics</div>', unsafe_allow_html=True)
    age_df, sex_df = get_demographics(drug_upper)
    d1, d2 = st.columns(2, gap="large")

    with d1:
        fig_age = go.Figure(go.Histogram(
            x=age_df['age'],
            nbinsx=20,
            marker=dict(
                color=PALETTE["blue"],
                opacity=0.85,
                line=dict(color=PALETTE["surface"], width=0.5),
            ),
            hovertemplate='Age %{x}<br>%{y} reports<extra></extra>',
        ))
        fig_age.update_layout(**CHART_THEME)
        fig_age.update_layout(
            height=240,
            xaxis=dict(title="Patient Age"),
            yaxis=dict(title="Reports"),
            bargap=0.08,
        )
        st.markdown('<div style="font-size:11px; color:#4A5568; margin-bottom:8px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;">Age Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})

    with d2:
        # map the raw codes to readable labels before rendering
        sex_label_map = {'M': 'Male', 'F': 'Female', 'UNK': 'Unknown'}
        sex_df['label'] = sex_df['sex'].map(sex_label_map)

        fig_sex = go.Figure(go.Pie(
            labels=sex_df['label'],
            values=sex_df['count'],
            hole=0.55,
            marker=dict(
                colors=[PALETTE["blue"], PALETTE["coral"], PALETTE["text_muted"]],
                line=dict(color=PALETTE["surface"], width=2),
            ),
            textinfo='label+percent',
            textfont=dict(size=10, family="DM Mono, monospace"),
            hovertemplate='<b>%{label}</b><br>%{value} reports (%{percent})<extra></extra>',
        ))
        fig_sex.update_layout(**CHART_THEME)
        fig_sex.update_layout(height=240, showlegend=False)
        st.markdown('<div style="font-size:11px; color:#4A5568; margin-bottom:8px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;">Sex Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_sex, use_container_width=True, config={"displayModeBar": False})


# -- TAB 2: TIME-SERIES FORECASTING --
with tab2:
    st.markdown('<div class="section-label">Quarterly Reporting Trend</div>', unsafe_allow_html=True)

    trend_df = get_reaction_trends(drug_upper)
    model, stats, overlay_df = train_trend_regression(trend_df)

    if model is None:
        st.markdown('<div class="empty-state"><div class="empty-state-icon">⬡</div>Not enough data points for regression (need ≥ 4 quarters)</div>', unsafe_allow_html=True)
    else:
        lr_stats = stats.get('lr', {})
        dt_stats = stats.get('dt', {})

        fig_trend = go.Figure()
        # split the overlay df into the slices we need for coloring train/test/forecast differently
        actual_data = overlay_df[overlay_df['actual'].notna()].copy()
        train_data  = overlay_df[overlay_df['split'] == 'train']
        test_data   = overlay_df[overlay_df['split'] == 'test']
        fcast_data  = overlay_df[overlay_df['split'] == 'forecast']

        # actual reported values with a subtle fill to make the trend readable
        fig_trend.add_trace(go.Scatter(
            x=actual_data['quarter_end_date'],
            y=actual_data['actual'],
            name='Actual Reports',
            mode='lines+markers',
            fill='tozeroy',
            fillcolor='rgba(74,144,226,0.06)',
            line=dict(color=PALETTE["blue"], width=2),
            marker=dict(size=6, color=PALETTE["blue"],
                        line=dict(width=1.5, color=PALETTE["surface"])),
            hovertemplate='Q %{x|%b %Y}<br><b>%{y:,} reports</b><extra>Actual</extra>',
        ))
        fig_trend.add_trace(go.Scatter(
            x=train_data['quarter_end_date'],
            y=train_data['lr_fitted'],
            name='Linear Regression (Train)',
            mode='lines',
            line=dict(color=PALETTE["text_muted"], width=1.5, dash='dot'),
            hovertemplate='Q %{x|%b %Y}<br><b>%{y:,.0f}</b><extra>Linear Regression</extra>',
        ))
        if not test_data.empty:
            fig_trend.add_trace(go.Scatter(
                x=test_data['quarter_end_date'],
                y=test_data['lr_fitted'],
                name='Linear Regression (Test)',
                mode='lines+markers',
                line=dict(color=PALETTE["text_muted"], width=1.5, dash='dash'),
                marker=dict(symbol='x', size=9, color=PALETTE["text_muted"]),
                hovertemplate='Q %{x|%b %Y}<br><b>%{y:,.0f}</b><extra>Linear Regression Test</extra>',
            ))
        fig_trend.add_trace(go.Scatter(
            x=train_data['quarter_end_date'],
            y=train_data['dt_fitted'],
            name='Decision Tree (Train)',
            mode='lines',
            line=dict(color=PALETTE["teal"], width=1.5, dash='dot'),
            hovertemplate='Q %{x|%b %Y}<br><b>%{y:,.0f}</b><extra>Decision Tree</extra>',
        ))
        if not test_data.empty:
            fig_trend.add_trace(go.Scatter(
                x=test_data['quarter_end_date'],
                y=test_data['dt_fitted'],
                name='Decision Tree (Test)',
                mode='lines+markers',
                line=dict(color=PALETTE["teal"], width=1.5, dash='dash'),
                marker=dict(symbol='diamond', size=9, color=PALETTE["teal"],
                            line=dict(width=1.5, color=PALETTE["surface"])),
                hovertemplate='Q %{x|%b %Y}<br><b>%{y:,.0f}</b><extra>Decision Tree Test</extra>',
            ))
        if not fcast_data.empty:
            # forecast points get coral so they visually stand apart from historical data
            fig_trend.add_trace(go.Scatter(
                x=fcast_data['quarter_end_date'],
                y=fcast_data['lr_fitted'],
                name='Forecast (Linear Regression)',
                mode='lines+markers',
                line=dict(color=PALETTE["coral"], width=2, dash='longdash'),
                marker=dict(symbol='diamond', size=10, color=PALETTE["coral"],
                            line=dict(width=1.5, color=PALETTE["surface"])),
                hovertemplate='Q %{x|%b %Y}<br><b>%{y:,.0f}</b><extra>Forecast</extra>',
            ))

        fig_trend.update_layout(**CHART_THEME)
        fig_trend.update_layout(
            height=420,
            xaxis=dict(title="Quarter"),
            yaxis=dict(title="Reports"),
            hovermode='x unified',
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:24px;">
            <div style="background:{PALETTE['surface']}; border:1px solid {PALETTE['border']};
                        border-top:2px solid {PALETTE['text_muted']}; border-radius:10px; padding:18px;">
                <div style="font-size:11px; font-weight:600; letter-spacing:0.1em;
                            text-transform:uppercase; color:{PALETTE['text_muted']}; margin-bottom:14px;">
                    Linear Regression
                </div>
                <div class="eval-grid" style="grid-template-columns: repeat(2,1fr);">
                    <div class="eval-cell">
                        <div class="eval-cell-label">R² Train</div>
                        <div class="eval-cell-val" style="color:{PALETTE['text_muted']};">{lr_stats.get('r2_train','—')}</div>
                    </div>
                    <div class="eval-cell">
                        <div class="eval-cell-label">R² Test</div>
                        <div class="eval-cell-val" style="color:{PALETTE['text_muted']};">{lr_stats.get('r2_test','—')}</div>
                    </div>
                    <div class="eval-cell">
                        <div class="eval-cell-label">RMSE Test</div>
                        <div class="eval-cell-val" style="color:{PALETTE['text_muted']};">{lr_stats.get('rmse_test','—')}</div>
                        <div class="eval-cell-unit">reports</div>
                    </div>
                    <div class="eval-cell">
                        <div class="eval-cell-label">Slope</div>
                        <div class="eval-cell-val" style="color:{PALETTE['text_muted']};">{lr_stats.get('slope','—')}</div>
                        <div class="eval-cell-unit">reports/day</div>
                    </div>
                </div>
            </div>
            <div style="background:{PALETTE['surface']}; border:1px solid {PALETTE['border']};
                        border-top:2px solid {PALETTE['teal']}; border-radius:10px; padding:18px;">
                <div style="font-size:11px; font-weight:600; letter-spacing:0.1em;
                            text-transform:uppercase; color:{PALETTE['teal']}; margin-bottom:14px;">
                    Decision Tree Improved Model
                </div>
                <div class="eval-grid" style="grid-template-columns: repeat(2,1fr);">
                    <div class="eval-cell">
                        <div class="eval-cell-label">R² Train</div>
                        <div class="eval-cell-val">{dt_stats.get('r2_train','—')}</div>
                    </div>
                    <div class="eval-cell">
                        <div class="eval-cell-label">R² Test</div>
                        <div class="eval-cell-val">{dt_stats.get('r2_test','—')}</div>
                    </div>
                    <div class="eval-cell">
                        <div class="eval-cell-label">RMSE Test</div>
                        <div class="eval-cell-val" style="color:{PALETTE['amber']};">{dt_stats.get('rmse_test','—')}</div>
                        <div class="eval-cell-unit">reports</div>
                    </div>
                    <div class="eval-cell">
                        <div class="eval-cell-label">Max Depth</div>
                        <div class="eval-cell-val" style="color:{PALETTE['purple']};">4</div>
                        <div class="eval-cell-unit">tree depth</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # worth explaining this in the UI — tripped me up at first too
        st.markdown(f"""
        <div style="background:{PALETTE['surface2']}; border:1px solid {PALETTE['border2']};
                    border-radius:8px; padding:14px 18px; font-size:12px;
                    color:{PALETTE['text_muted']}; margin-bottom:24px;">
            ⚠️ The 2-quarter forecast uses <b style="color:{PALETTE['text_sec']}">linear regression</b>
            because decision trees cannot extrapolate beyond the range of training data —
            they will always predict the last seen value rather than a new one.
            The decision tree is shown for its superior fit on historical data.
        </div>
        """, unsafe_allow_html=True)

        if not fcast_data.empty:
            st.markdown('<div class="section-label">2-Quarter Forecast</div>', unsafe_allow_html=True)

            forecast_dates = fcast_data['quarter_end_date'].dt.strftime('%b %Y').tolist()
            lr_vals = fcast_data['lr_fitted'].tolist()
            # dt can't forecast so we just repeat the last training value as a flat line
            dt_last = overlay_df[overlay_df['split'] == 'train']['dt_fitted'].iloc[-1]

            st.markdown(f"""
            <div style="font-size:11px; font-weight:600; letter-spacing:0.1em;
                        text-transform:uppercase; color:{PALETTE['text_muted']}; margin-bottom:10px;">
                Linear Regression
            </div>
            """, unsafe_allow_html=True)

            f1, f2 = st.columns(2, gap="large")
            with f1:
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-date">Next Quarter · {forecast_dates[0]}</div>
                    <div class="forecast-val">{round(lr_vals[0]):,}</div>
                    <div class="forecast-unit">projected reports</div>
                </div>""", unsafe_allow_html=True)
            with f2:
                st.markdown(f"""
                <div class="forecast-card" style="border-top-color:{PALETTE['blue']};">
                    <div class="forecast-date">Q+2 · {forecast_dates[1]}</div>
                    <div class="forecast-val">{round(lr_vals[1]):,}</div>
                    <div class="forecast-unit">projected reports</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="font-size:11px; font-weight:600; letter-spacing:0.1em;
                        text-transform:uppercase; color:{PALETTE['teal']};
                        margin-top:20px; margin-bottom:10px;">
                Decision Tree Improved Model
            </div>
            """, unsafe_allow_html=True)

            d1, d2 = st.columns(2, gap="large")
            with d1:
                st.markdown(f"""
                <div class="forecast-card" style="border-top-color:{PALETTE['teal']};">
                    <div class="forecast-date">Next Quarter · {forecast_dates[0]}</div>
                    <div class="forecast-val">{round(dt_last):,}</div>
                    <div class="forecast-unit">projected reports (flat — last training value)</div>
                </div>""", unsafe_allow_html=True)
            with d2:
                st.markdown(f"""
                <div class="forecast-card" style="border-top-color:{PALETTE['teal']};">
                    <div class="forecast-date">Q+2 · {forecast_dates[1]}</div>
                    <div class="forecast-val">{round(dt_last):,}</div>
                    <div class="forecast-unit">projected reports (flat — last training value)</div>
                </div>""", unsafe_allow_html=True)


# -- TAB 3: NLP & CLUSTERING --
with tab3:
    st.markdown('<div class="section-label">Adverse Reaction Clusters · PCA 2-D Projection</div>', unsafe_allow_html=True)
    st.markdown(f'<span style="font-size:12px; color:{PALETTE["text_muted"]};">K-Means clusters reaction terms via TF-IDF similarity. PCA reduces the high-dimensional vector space to 2-D. Each point is one reaction term; colour encodes assigned body system.</span>', unsafe_allow_html=True)

    if not reactions_flat:
        st.markdown('<div class="empty-state"><div class="empty-state-icon">⬡</div>Enable the OpenFDA API in the sidebar to see clustering</div>', unsafe_allow_html=True)
    else:
        # cap at 300 unique terms — beyond that kmeans gets slow and the chart gets unreadable
        unique_rxns = list(set(reactions_flat[:300]))
        if len(unique_rxns) < 5:
            st.markdown('<div class="empty-state">Not enough unique reactions to cluster (need ≥ 5)</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Clustering reactions …"):
                from scripts.etl_processor import map_body_system
                cluster_df = cluster_reactions(unique_rxns, n_clusters=5)

            cluster_df['body_system'] = cluster_df['reaction'].apply(map_body_system)

            # color by body system rather than cluster number — more medically meaningful
            BODY_COLORS = {
                'Cardiovascular':   '#FF6B6B',
                'Gastrointestinal': '#F5A623',
                'Neurological':     '#00D4AA',
                'Respiratory':      '#4A90E2',
                'Dermatological':   '#FFD700',
                'Musculoskeletal':  '#7ED321',
                'Renal':            '#9B6DFF',
                'Endocrine':        '#FF8C69',
                'Immunological':    '#87CEEB',
                'Haematological':   '#FF69B4',
                'Other':            '#4A5568',
            }

            fig_scatter = px.scatter(
                cluster_df,
                x='x', y='y',
                color='body_system',
                symbol='cluster',
                hover_data={'reaction': True, 'cluster': True, 'x': False, 'y': False},
                color_discrete_map=BODY_COLORS,
                labels={
                    'x': 'PCA Component 1',
                    'y': 'PCA Component 2',
                    'body_system': 'Body System',
                    'cluster': 'K-Means Cluster',
                },
            )
            fig_scatter.update_traces(
                marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color=PALETTE["surface"])),
                selector=dict(mode='markers'),
            )
            fig_scatter.update_layout(**CHART_THEME)
            fig_scatter.update_layout(
                height=480,
                xaxis=dict(title="PCA Component 1", zeroline=False),
                yaxis=dict(title="PCA Component 2", zeroline=False),
                legend=dict(
                    title=dict(text="Body System", font=dict(size=10, color=PALETTE["text_muted"])),
                ),
            )
            st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="section-label">Cluster Members</div>', unsafe_allow_html=True)
            cluster_cols = st.columns(min(5, cluster_df['cluster'].nunique()))
            for i, (cluster_name, group) in enumerate(cluster_df.groupby('cluster')):
                with cluster_cols[i % len(cluster_cols)]:
                    top_members = group.sort_values('reaction')['reaction'].head(12).tolist()
                    pills = "".join(f'<span class="rxn-pill">{r}</span>' for r in top_members)
                    st.markdown(f"""
                    <div class="cluster-card">
                        <div class="cluster-card-head">{cluster_name}</div>
                        {pills}
                    </div>""", unsafe_allow_html=True)