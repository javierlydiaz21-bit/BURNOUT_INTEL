import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from datetime import datetime
import subprocess

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Burnout Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
#  DESIGN TOKENS — Minimal / Corporate Cold
# ══════════════════════════════════════════════════════════
C = {
    'bg':        '#F7F8FA',
    'surface':   '#FFFFFF',
    'border':    '#E2E6ED',
    'border2':   '#C8CDD6',
    'text':      '#111318',
    'subtext':   '#6B7280',
    'muted':     '#9CA3AF',
    'accent':    '#1A56DB',
    'accent_lt': '#EEF2FF',
    'low':       '#059669',
    'moderate':  '#D97706',
    'high':      '#EA580C',
    'severe':    '#DC2626',
    'low_lt':    '#ECFDF5',
    'mod_lt':    '#FFFBEB',
    'high_lt':   '#FFF7ED',
    'sev_lt':    '#FEF2F2',
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: {C['text']};
}}
.stApp {{
    background: {C['bg']};
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {C['surface']};
    border-right: 1px solid {C['border']};
}}
section[data-testid="stSidebar"] > div {{
    padding: 2rem 1.5rem;
}}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding: 2rem 2.5rem 4rem 2.5rem;
    max-width: 1400px;
}}

/* ── Typography system ── */
.brand-wordmark {{
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: {C['accent']};
    margin: 0 0 0.25rem 0;
}}
.page-title {{
    font-size: 1.75rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: {C['text']};
    margin: 0 0 0.35rem 0;
    line-height: 1.2;
}}
.page-subtitle {{
    font-size: 0.875rem;
    color: {C['subtext']};
    font-weight: 400;
    margin: 0 0 2rem 0;
}}

/* ── Cards ── */
.card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
}}
.card-sm {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}}

/* ── Section labels ── */
.section-label {{
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {C['muted']};
    margin: 0 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid {C['border']};
}}
.sidebar-section-label {{
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {C['muted']};
    margin: 1.25rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {C['border']};
}}

/* ── KPI chips ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-bottom: 1rem;
}}
.kpi-chip {{
    background: {C['bg']};
    border: 1px solid {C['border']};
    border-radius: 7px;
    padding: 0.65rem 0.75rem;
}}
.kpi-chip-label {{
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {C['muted']};
    margin: 0 0 0.2rem 0;
}}
.kpi-chip-value {{
    font-size: 1.35rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: {C['text']};
    margin: 0;
    font-family: 'DM Mono', monospace;
}}

/* ── Model badge ── */
.model-tag {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: {C['accent_lt']};
    color: {C['accent']};
    border: 1px solid #C7D7F8;
    border-radius: 5px;
    padding: 0.2rem 0.55rem;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}}
.model-dot {{
    width: 6px; height: 6px;
    background: {C['accent']};
    border-radius: 50%;
    display: inline-block;
    animation: blink 2s infinite;
}}
@keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}

/* ── Prediction result box ── */
.result-box {{
    border-radius: 10px;
    padding: 2rem 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    border: 1px solid;
}}
.result-level {{
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin: 0 0 0.5rem 0;
}}
.result-label {{
    font-size: 2.75rem;
    font-weight: 600;
    letter-spacing: -0.04em;
    margin: 0 0 0.75rem 0;
    line-height: 1;
}}
.result-badge {{
    display: inline-block;
    border-radius: 4px;
    padding: 0.3rem 0.75rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    font-family: 'DM Mono', monospace;
}}
.waiting-box {{
    border: 1px dashed {C['border2']};
    border-radius: 10px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    color: {C['muted']};
}}
.waiting-icon {{
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    display: block;
}}

/* ── Prob bar ── */
.prob-row {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
    font-size: 0.78rem;
}}
.prob-label {{
    width: 72px;
    color: {C['subtext']};
    font-size: 0.72rem;
    font-weight: 500;
    flex-shrink: 0;
}}
.prob-bar-track {{
    flex: 1;
    height: 5px;
    background: {C['bg']};
    border-radius: 99px;
    overflow: hidden;
}}
.prob-bar-fill {{
    height: 100%;
    border-radius: 99px;
    transition: width 0.5s ease;
}}
.prob-pct {{
    width: 38px;
    text-align: right;
    color: {C['text']};
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    flex-shrink: 0;
}}

/* ── Data table ── */
.stDataFrame {{
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
}}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {C['border']};
    gap: 0;
    padding: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {C['subtext']};
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    padding: 0.6rem 1.25rem;
    border: none;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}}
.stTabs [aria-selected="true"] {{
    color: {C['text']} !important;
    border-bottom: 2px solid {C['accent']} !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 1.5rem;
}}

/* ── Buttons ── */
.stButton > button[kind="primary"] {{
    background: {C['accent']};
    border: none;
    border-radius: 7px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.55rem 1.25rem;
    color: white;
}}
.stButton > button[kind="secondary"] {{
    background: {C['surface']};
    border: 1px solid {C['border2']};
    border-radius: 7px;
    font-size: 0.8rem;
    font-weight: 500;
    color: {C['text']};
}}
.stButton > button:hover {{
    opacity: 0.88;
    transform: translateY(-1px);
    transition: all 0.15s ease;
}}

/* ── Number inputs ── */
.stNumberInput input {{
    border-radius: 7px !important;
    border-color: {C['border']} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}}
.stNumberInput label {{
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: {C['subtext']} !important;
}}

/* ── Input labels ── */
.stRadio label, .stSelectbox label {{
    font-size: 0.75rem !important;
    color: {C['subtext']} !important;
}}

/* ── Divider ── */
hr {{ border-color: {C['border']}; margin: 1rem 0; }}

/* ── Alerts ── */
.stAlert {{
    border-radius: 7px !important;
    font-size: 0.82rem !important;
}}

/* ══════════════════════════════════════════════════
   OVERRIDES GLOBALES - forzar tema claro
   ══════════════════════════════════════════════════ */

/* Texto base */
p, span, div, label, h1, h2, h3, h4, h5, h6 {{ color: {C['text']}; }}

/* Metricas nativas */
[data-testid="stMetric"] {{
    background: {C['bg']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
}}
[data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] p {{
    color: {C['muted']} !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}}
[data-testid="stMetricValue"] > div, [data-testid="stMetricValue"] p {{
    color: {C['text']} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.3rem !important;
}}
[data-testid="stMetricDelta"] {{ display: none; }}

/* Inputs */
.stNumberInput input, .stTextInput input {{
    background: {C['surface']} !important;
    color: {C['text']} !important;
    border-color: {C['border']} !important;
}}
.stNumberInput label p, .stTextInput label p {{
    color: {C['subtext']} !important;
}}

/* Radio */
.stRadio label, .stRadio [data-testid="stMarkdownContainer"] p {{
    color: {C['text']} !important;
    font-size: 0.8rem !important;
}}

/* File uploader */
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {{
    color: {C['subtext']} !important;
}}

/* Captions */
[data-testid="stCaptionContainer"] p {{ color: {C['muted']} !important; font-size: 0.72rem !important; }}

/* Sidebar: forzar texto oscuro en todos los nodos */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {{ color: {C['text']} !important; }}

/* Dataframe */
.stDataFrame th {{ background: {C['bg']} !important; color: {C['text']} !important; font-size: 0.72rem !important; }}
.stDataFrame td {{ color: {C['text']} !important; font-size: 0.78rem !important; }}

/* Tabs */
button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {{ color: inherit !important; }}

/* ── ELIMINAR BLUR DE ACTUALIZACIÓN ── */
[data-testid="stAppViewBlockContainer"], 
[data-testid="stAppViewContainer"],
.st-emotion-cache-16idsys, 
.st-emotion-cache-z5fcl4 {{
    opacity: 1 !important;
    filter: none !important;
}}

/* ── Sidebar logo area ── */
.sidebar-logo {{
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid {C['border']};
}}
.sidebar-logo-mark {{
    width: 32px; height: 32px;
    background: {C['text']};
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1rem;
    flex-shrink: 0;
}}
.sidebar-logo-text {{
    font-size: 0.78rem;
    font-weight: 600;
    color: {C['text']};
    letter-spacing: -0.01em;
    line-height: 1.3;
}}
.sidebar-logo-sub {{
    font-size: 0.65rem;
    color: {C['muted']};
    font-weight: 400;
}}

/* ── Matrix section divider ── */
.matrix-header {{
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {C['muted']};
    margin: 1.25rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.matrix-header::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {C['border']};
}}

/* ── Stat row ── */
.stat-inline {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.4rem 0;
    border-bottom: 1px solid {C['border']};
    font-size: 0.78rem;
}}
.stat-inline:last-child {{ border-bottom: none; }}
.stat-inline-label {{ color: {C['subtext']}; }}
.stat-inline-value {{
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    color: {C['text']};
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  AUTO-TRAINING
# ══════════════════════════════════════════════════════════
def check_and_retrain():
    dataset_path = 'mental_health_burnout_tech_2026.csv'
    model_path = 'burnout_nn_model.pkl'
    needs_training = False
    if os.path.exists(dataset_path):
        if not os.path.exists(model_path):
            needs_training = True
        else:
            data_time = os.path.getmtime(dataset_path)
            model_time = os.path.getmtime(model_path)
            if data_time > (model_time + 2):
                needs_training = True
    if needs_training:
        st.warning("Nuevos datos detectados — reiniciando motores de inferencia.")
        with st.spinner("Actualizando modelos..."):
            try:
                subprocess.run([sys.executable, "train_burnout_nn.py"], check=True)
                subprocess.run([sys.executable, "train_burnout_logistic.py"], check=True)
                st.success("Modelos actualizados.")
                st.cache_resource.clear()
                return True
            except Exception as e:
                st.error(f"Error en auto-entrenamiento: {e}")
    return False

if check_and_retrain():
    st.rerun()


# ══════════════════════════════════════════════════════════
#  ASSETS
# ══════════════════════════════════════════════════════════
def get_resource_fingerprint():
    files = ['burnout_logistic_model.pkl', 'burnout_nn_model.pkl', 'burnout_logistic_metrics.json', 'burnout_nn_metrics.json']
    fingerprint = []
    for f in files:
        if os.path.exists(f):
            stats = os.stat(f)
            fingerprint.append((f, stats.st_mtime, stats.st_size))
    return tuple(fingerprint)

@st.cache_resource(show_spinner="Cargando modelos...")
def load_assets(fingerprint):
    l_m = joblib.load('burnout_logistic_model.pkl')
    n_m = joblib.load('burnout_nn_model.pkl')
    s   = joblib.load('burnout_scaler.pkl')
    with open('burnout_logistic_metrics.json') as f: m_l = json.load(f)
    with open('burnout_nn_metrics.json')       as f: m_n = json.load(f)
    return l_m, n_m, s, m_l, m_n, datetime.now().strftime("%H:%M")

l_model, n_model, scaler, m_log, m_nn, load_time = load_assets(get_resource_fingerprint())

LABEL_MAP   = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Severe'}
REVERSE_MAP = {'Low': 0, 'Moderate': 1, 'High': 2, 'Severe': 3}
FEATURE_COLS = [
    'work_hours_per_week', 'meetings_per_day', 'sleep_hours_per_night',
    'stress_score', 'work_life_balance_score', 'job_satisfaction_score',
    'manager_support_score', 'vacation_days_taken'
]

LEVEL_COLORS = {
    'Low':      (C['low'],      C['low_lt']),
    'Moderate': (C['moderate'], C['mod_lt']),
    'High':     (C['high'],     C['high_lt']),
    'Severe':   (C['severe'],   C['sev_lt']),
}
LABEL_COLORS_BAR = {
    'Low': C['low'], 'Moderate': C['moderate'],
    'High': C['high'], 'Severe': C['severe']
}


# ══════════════════════════════════════════════════════════
#  CONFUSION MATRIX — minimal style
# ══════════════════════════════════════════════════════════
def plot_matrix(cm, labels, title="", subtitle=""):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    # Custom colormap: white → accent blue
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('bip', ['#F7F8FA', '#1A56DB'], N=256)

    cm_arr = np.array(cm, dtype=float)
    cm_norm = cm_arr / (cm_arr.max() + 1e-9)

    for i in range(n):
        for j in range(n):
            intensity = cm_norm[i, j]
            bg_color = cmap(intensity)
            rect = mpatches.FancyBboxPatch(
                (j - 0.45, i - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                linewidth=0,
                facecolor=bg_color,
                zorder=1
            )
            ax.add_patch(rect)
            txt_color = 'white' if intensity > 0.5 else '#111318'
            val = int(cm_arr[i, j])
            ax.text(j, i, str(val),
                    ha='center', va='center',
                    color=txt_color,
                    fontsize=11, fontweight='600',
                    fontfamily='monospace', zorder=2)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=7.5, color='#6B7280',
                        fontfamily='DM Sans', fontweight='500')
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7.5, color='#6B7280',
                        fontfamily='DM Sans', fontweight='500')
    ax.xaxis.set_label_position('bottom')
    ax.set_xlabel("Predicción", fontsize=7, color='#9CA3AF',
                   labelpad=8, fontfamily='DM Sans')
    ax.set_ylabel("Real", fontsize=7, color='#9CA3AF',
                   labelpad=8, fontfamily='DM Sans')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    if title:
        ax.set_title(title, fontsize=8.5, color='#111318',
                     fontweight='600', pad=12, fontfamily='DM Sans',
                     loc='left')

    fig.tight_layout(pad=0.8)
    return fig


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-logo">
        <div class="sidebar-logo-mark">◈</div>
        <div>
            <div class="sidebar-logo-text">Burnout Intel</div>
            <div class="sidebar-logo-sub">Sincronizado {load_time}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='sidebar-section-label'>Motor de inferencia</p>", unsafe_allow_html=True)
    model_choice = st.radio(
        "", ["Regresión Logística", "Red Neuronal"],
        index=1, label_visibility="collapsed"
    )
    act_mod = l_model if model_choice == "Regresión Logística" else n_model
    act_met = m_log   if model_choice == "Regresión Logística" else m_nn

    badge = "LOG · REG" if model_choice == "Regresión Logística" else "MLP · NEURAL"
    st.markdown(f"""<div class="model-tag"><span class="model-dot"></span>{badge}</div>""",
                unsafe_allow_html=True)

    # KPI grid
    acc  = act_met['accuracy']  * 100
    f1   = act_met['f1_score']  * 100
    prec = act_met.get('precision', 0) * 100
    rec  = act_met.get('recall', 0) * 100

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-chip">
            <p class="kpi-chip-label">Accuracy</p>
            <p class="kpi-chip-value">{acc:.1f}<span style="font-size:0.75rem;color:#6B7280">%</span></p>
        </div>
        <div class="kpi-chip">
            <p class="kpi-chip-label">F1 Score</p>
            <p class="kpi-chip-value">{f1:.1f}<span style="font-size:0.75rem;color:#6B7280">%</span></p>
        </div>
        <div class="kpi-chip">
            <p class="kpi-chip-label">Precision</p>
            <p class="kpi-chip-value">{prec:.1f}<span style="font-size:0.75rem;color:#6B7280">%</span></p>
        </div>
        <div class="kpi-chip">
            <p class="kpi-chip-label">Recall</p>
            <p class="kpi-chip-value">{rec:.1f}<span style="font-size:0.75rem;color:#6B7280">%</span></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Global Confusion Matrix — always visible ──
    st.markdown("<p class='matrix-header'>Matriz de Confusión</p>", unsafe_allow_html=True)
    classes = [LABEL_MAP[c] for c in act_mod.classes_]
    fig_side = plot_matrix(
        np.array(act_met['confusion_matrix']),
        classes,
        title="Rendimiento en Test"
    )
    st.pyplot(fig_side, use_container_width=True)
    plt.close(fig_side)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺  Recargar modelos", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<p class="brand-wordmark">Plataforma de Análisis de Burnout</p>
<h1 class="page-title">Monitor de Bienestar Organizacional</h1>
<p class="page-subtitle">Predicción de riesgo de agotamiento profesional mediante modelos de aprendizaje automático</p>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════
for key in FEATURE_COLS:
    if key not in st.session_state:
        st.session_state[key] = 0.0

def randomize_data():
    st.session_state.work_hours_per_week     = round(random.uniform(20, 80), 1)
    st.session_state.meetings_per_day        = round(random.uniform(0, 15), 1)
    st.session_state.sleep_hours_per_night   = round(random.uniform(4, 10), 1)
    st.session_state.stress_score            = round(random.uniform(1, 10), 1)
    st.session_state.work_life_balance_score = round(random.uniform(1, 10), 1)
    st.session_state.job_satisfaction_score  = round(random.uniform(1, 10), 1)
    st.session_state.manager_support_score   = round(random.uniform(1, 10), 1)
    st.session_state.vacation_days_taken     = round(random.uniform(0, 30), 1)


# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["Análisis Individual", "Procesamiento por Lotes"])


# ─────────────────────────────────────────────────────────
#  TAB 1 — INDIVIDUAL
# ─────────────────────────────────────────────────────────
with tab1:
    col_form, col_gap, col_result = st.columns([2, 0.12, 1.4])

    with col_form:
        st.markdown("<p class='section-label'>Factores del empleado</p>", unsafe_allow_html=True)

        btn_rand = st.button("Escenario aleatorio", type="secondary")
        if btn_rand:
            randomize_data()

        c1, c2 = st.columns(2)
        with c1:
            wh     = st.number_input("Horas de trabajo / semana", 0.0, 120.0, key='work_hours_per_week')
            mpd    = st.number_input("Reuniones / día",            0.0, 30.0,  key='meetings_per_day')
            slp    = st.number_input("Horas de sueño / noche",    0.0, 24.0,  key='sleep_hours_per_night')
            str_sc = st.number_input("Nivel de estrés (1–10)",    1.0, 10.0,  key='stress_score')
        with c2:
            wlb    = st.number_input("Balance vida / trabajo (1–10)", 1.0, 10.0, key='work_life_balance_score')
            js     = st.number_input("Satisfacción laboral (1–10)",   1.0, 10.0, key='job_satisfaction_score')
            ms     = st.number_input("Apoyo del manager (1–10)",      1.0, 10.0, key='manager_support_score')
            vac    = st.number_input("Días de vacaciones tomados",    0.0, 100.0, key='vacation_days_taken')

        btn_run = st.button("Ejecutar diagnóstico", type="primary", use_container_width=True)

    with col_result:
        st.markdown("<p class='section-label'>Resultado de inferencia</p>", unsafe_allow_html=True)

        if btn_run:
            df_inf    = pd.DataFrame([[wh, mpd, slp, str_sc, wlb, js, ms, vac]], columns=FEATURE_COLS)
            inf_s     = scaler.transform(df_inf)
            pred      = act_mod.predict(inf_s)[0]
            prob      = act_mod.predict_proba(inf_s)[0]
            label     = LABEL_MAP[pred]
            ink, bg   = LEVEL_COLORS[label]
            conf_pct  = prob[pred] * 100

            st.toast(f"Diagnóstico completado — {label}")

            st.markdown(f"""
            <div class="result-box" style="background:{bg}; border-color:{ink}22;">
                <p class="result-level" style="color:{ink};">Nivel de burnout</p>
                <p class="result-label" style="color:{ink};">{label}</p>
                <span class="result-badge" style="background:{ink}18; color:{ink}; border:1px solid {ink}44;">
                    Confianza {conf_pct:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p class='section-label' style='margin-bottom:0.75rem;'>Distribución de probabilidad</p>", unsafe_allow_html=True)

            for idx, lbl in LABEL_MAP.items():
                p_val = prob[idx] * 100 if idx < len(prob) else 0
                bar_color = LABEL_COLORS_BAR[lbl]
                st.markdown(f"""
                <div class="prob-row">
                    <span class="prob-label">{lbl}</span>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill" style="width:{p_val:.1f}%; background:{bar_color};"></div>
                    </div>
                    <span class="prob-pct">{p_val:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            # Feature summary
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p class='section-label' style='margin-bottom:0.6rem;'>Resumen de inputs</p>", unsafe_allow_html=True)
            labels_display = [
                ("Horas / semana", wh), ("Reuniones / día", mpd),
                ("Horas de sueño", slp), ("Estrés", str_sc),
                ("Balance V/T", wlb), ("Satisfacción", js),
                ("Apoyo manager", ms), ("Vacaciones", vac),
            ]
            rows_html = "".join([
                f"""<div class="stat-inline">
                        <span class="stat-inline-label">{lbl}</span>
                        <span class="stat-inline-value">{val:.1f}</span>
                    </div>"""
                for lbl, val in labels_display
            ])
            st.markdown(f'<div class="card-sm" style="padding:0.75rem 1rem;">{rows_html}</div>', unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="waiting-box">
                <span class="waiting-icon">◈</span>
                <p style="font-size:0.85rem; font-weight:500; margin:0 0 0.25rem 0;">Sin datos</p>
                <p style="font-size:0.75rem; margin:0; color:{C['muted']};">Complete el formulario o use un escenario aleatorio</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  TAB 2 — BATCH
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("<p class='section-label'>Carga de datos organizacionales</p>", unsafe_allow_html=True)

    st.info("Sube un archivo .csv departamental. Se combinará con el dataset maestro y desencadenará el re-entrenamiento automático de los modelos.")
    up_file = st.file_uploader("Archivo .csv", type=['csv'])

    if up_file:
        df_batch = pd.read_csv(up_file)

        col_preview, col_action = st.columns([3, 1])
        with col_preview:
            st.markdown(f"<p style='font-size:0.75rem; color:{C['subtext']}; margin-bottom:0.5rem;'>"
                        f"{len(df_batch):,} registros · {len(df_batch.columns)} columnas</p>",
                        unsafe_allow_html=True)
            st.dataframe(df_batch.head(5), use_container_width=True, height=180)

        with col_action:
            st.markdown("<br>", unsafe_allow_html=True)
            btn_batch = st.button("Procesar lote", type="primary", use_container_width=True)

        if btn_batch:
            X_b = pd.DataFrame()
            for c in FEATURE_COLS:
                found = next((f for f in df_batch.columns if c in f.lower() or f.lower() in c), None)
                X_b[c] = df_batch[found] if found else 0.0
            X_b.fillna(X_b.mean(), inplace=True)
            preds = act_mod.predict(scaler.transform(X_b))
            df_batch['PREDICCIÓN'] = [LABEL_MAP[p] for p in preds]
            st.toast("Procesamiento completado")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Distribution summary ──
            pred_counts = pd.Series([LABEL_MAP[p] for p in preds]).value_counts()
            total = len(preds)

            st.markdown("<p class='section-label'>Distribución del lote</p>", unsafe_allow_html=True)
            dist_cols = st.columns(4)
            for idx, (lbl, cnt) in enumerate(
                sorted(pred_counts.items(), key=lambda x: list(LABEL_MAP.values()).index(x[0]))
            ):
                ink, bg = LEVEL_COLORS[lbl]
                pct = cnt / total * 100
                with dist_cols[idx % 4]:
                    st.markdown(f"""
                    <div class="card-sm" style="border-left: 3px solid {ink}; border-radius:7px;">
                        <p class="kpi-chip-label" style="color:{ink};">{lbl}</p>
                        <p class="kpi-chip-value">{cnt:,}</p>
                        <p style="font-size:0.7rem; color:{C['muted']}; margin:0;">{pct:.1f}% del total</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Results table ──
            st.markdown("<p class='section-label'>Resultados de clasificación</p>", unsafe_allow_html=True)
            st.dataframe(
                df_batch[['PREDICCIÓN'] + [c for c in df_batch.columns if c != 'PREDICCIÓN']].head(50),
                use_container_width=True
            )

            # ── Batch Confusion Matrix ──
            l_col = next(
                (c for c in df_batch.columns if 'burnout' in c.lower() and 'predic' not in c.lower()),
                None
            )
            if l_col:
                from sklearn.metrics import confusion_matrix as sk_cm, accuracy_score, f1_score

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<p class='section-label'>Análisis de precisión del lote</p>", unsafe_allow_html=True)

                y_t       = df_batch[l_col].map(REVERSE_MAP)
                valid_idx = y_t.dropna().index

                if len(valid_idx) > 0:
                    y_true_v = y_t.loc[valid_idx].astype(int)
                    y_pred_v = preds[valid_idx]

                    cm_batch = sk_cm(y_true_v, y_pred_v, labels=act_mod.classes_)
                    batch_acc = accuracy_score(y_true_v, y_pred_v) * 100
                    batch_f1  = f1_score(y_true_v, y_pred_v, average='weighted') * 100

                    m_col1, m_col2 = st.columns([1.1, 1])

                    with m_col1:
                        fig_b = plot_matrix(
                            cm_batch,
                            [LABEL_MAP[c] for c in act_mod.classes_],
                            title="Matriz de Confusión — Lote"
                        )
                        st.pyplot(fig_b, use_container_width=True)
                        plt.close(fig_b)

                    with m_col2:
                        st.markdown("<p class='section-label' style='margin-top:0.5rem;'>Métricas del lote</p>",
                                    unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="kpi-grid" style="margin-top:0.5rem;">
                            <div class="kpi-chip">
                                <p class="kpi-chip-label">Accuracy</p>
                                <p class="kpi-chip-value">{batch_acc:.1f}<span style="font-size:0.75rem;color:#6B7280">%</span></p>
                            </div>
                            <div class="kpi-chip">
                                <p class="kpi-chip-label">F1 Score</p>
                                <p class="kpi-chip-value">{batch_f1:.1f}<span style="font-size:0.75rem;color:#6B7280">%</span></p>
                            </div>
                            <div class="kpi-chip">
                                <p class="kpi-chip-label">Muestras</p>
                                <p class="kpi-chip-value">{len(valid_idx):,}</p>
                            </div>
                            <div class="kpi-chip">
                                <p class="kpi-chip-label">Clases</p>
                                <p class="kpi-chip-value">{len(act_mod.classes_)}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Per-class diagonal (correct predictions)
                        st.markdown("<p class='section-label' style='margin-top:1.25rem;'>Aciertos por clase</p>",
                                    unsafe_allow_html=True)
                        for i, cls_label in enumerate([LABEL_MAP[c] for c in act_mod.classes_]):
                            correct = cm_batch[i, i]
                            total_cls = cm_batch[i].sum()
                            pct_cls = (correct / total_cls * 100) if total_cls > 0 else 0
                            ink = LABEL_COLORS_BAR[cls_label]
                            st.markdown(f"""
                            <div class="prob-row">
                                <span class="prob-label">{cls_label}</span>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill" style="width:{pct_cls:.1f}%; background:{ink};"></div>
                                </div>
                                <span class="prob-pct">{pct_cls:.0f}%</span>
                            </div>
                            """, unsafe_allow_html=True)

            # ── Dataset merge ──
            try:
                dataset_path = 'mental_health_burnout_tech_2026.csv'
                df_original  = pd.read_csv(dataset_path)
                df_combined  = pd.concat(
                    [df_original, df_batch.drop(columns=['PREDICCIÓN'], errors='ignore')],
                    ignore_index=True
                )
                df_combined.to_csv(dataset_path, index=False)
                st.success("Dataset combinado con éxito.")
                st.warning("Archivo maestro modificado — recargando para iniciar re-entrenamiento.")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error al combinar datasets: {e}")
