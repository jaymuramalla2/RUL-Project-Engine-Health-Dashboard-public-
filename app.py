import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import pickle
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="RUL Predict · Engine Health",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# MATPLOTLIB FONT SETUP
# System-safe fallbacks — no local install needed
# If you have DM Sans .ttf files in your project folder,
# uncomment the lines below for better chart fonts:
# for f in ['DMSans-Regular.ttf','DMSans-Medium.ttf','DMMono-Regular.ttf']:
#     try: fm.fontManager.addfont(f)
#     except: pass
# ==========================================
CHART_FONT   = 'DejaVu Sans'    # safe fallback, always available
MONO_FONT    = 'DejaVu Sans Mono'  # safe fallback

plt.rcParams.update({
    'font.family':       CHART_FONT,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'figure.dpi':        130,
})

# ==========================================
# CSS + GOOGLE FONTS (browser only, not matplotlib)
# ==========================================
st.markdown("""<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap" rel="stylesheet">""", unsafe_allow_html=True)

# Force sidebar open always
st.markdown("""
<script>
(function() {
    function showSidebar() {
        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        var collapseBtn = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]');
        var collapsedCtrl = window.parent.document.querySelector('[data-testid="collapsedControl"]');
        if (collapseBtn) collapseBtn.style.display = 'none';
        if (collapsedCtrl) collapsedCtrl.style.display = 'none';
    }
    setTimeout(showSidebar, 100);
    setTimeout(showSidebar, 500);
    setTimeout(showSidebar, 1000);
})();
</script>
""", unsafe_allow_html=True)

st.markdown("""<style>
html,body,[data-testid="stAppViewContainer"]{background:#F4F2EE !important;font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"]>.main{background:#F4F2EE !important;}
[data-testid="stSidebar"]{background:#111827 !important;border-right:none !important;min-width:260px !important;max-width:260px !important;}
[data-testid="stSidebar"] *{color:#9CA3AF !important;}
[data-testid="stFileUploader"]{background:#1F2937 !important;border:1.5px dashed #374151 !important;border-radius:12px !important;}
[data-testid="stFileUploader"] *{color:#6B7280 !important;font-size:12px !important;}
[data-testid="stFileUploader"] button{background:#1E3A5F !important;color:#93C5FD !important;border:1px solid #2563EB !important;border-radius:8px !important;font-size:11px !important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 2.5rem 3rem !important;max-width:1400px !important;}
section[data-testid="stSidebar"]>div{padding-top:1.5rem !important;}
/* hide ALL collapse/expand buttons */
[data-testid="stSidebarCollapseButton"]{display:none !important;}
[data-testid="collapsedControl"]{display:none !important;}
button[kind="header"]{display:none !important;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#EDEBE6;}
::-webkit-scrollbar-thumb{background:#C5BFB5;border-radius:3px;}
</style>""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS
# ==========================================
FEATURES = [
    's_2','s_3','s_4','s_7','s_8','s_9','s_11',
    's_12','s_13','s_14','s_15','s_17','s_20','s_21','life_ratio'
]
WINDOW = 30

# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    xgb = XGBRegressor()
    xgb.load_model("model_xgb.json")
    with open("model_gbr.pkl","rb") as f:
        gbr = pickle.load(f)
    gru = load_model("model_gru.keras")
    return xgb, gbr, gru

# ==========================================
# HELPERS
# ==========================================
def get_stage(r):
    if r <= 0.3:   return "Early"
    elif r <= 0.7: return "Mid"
    else:          return "Late"

def risk_category(rul):
    if rul >= 130:
        return "SAFE",     "Continue maintenance schedule as normal",   "#065F46","#ECFDF5","#10B981"
    elif rul >= 90:
        return "MONITOR",  "Standard schedule — inspect thoroughly",    "#78350F","#FFFBEB","#F59E0B"
    elif rul >= 50:
        return "CAUTION",  "Increase inspection frequency immediately", "#92400E","#FFF7ED","#F97316"
    else:
        return "CRITICAL", "Immediate maintenance — do not operate",    "#7F1D1D","#FEF2F2","#EF4444"

def stage_ensemble(row):
    if pd.isna(row['gru_pred']):
        return row['gbr_pred']
    if row['stage'] == 'Early':
        return 0.1*row['gru_pred'] + 0.2*row['xgb_pred'] + 0.7*row['gbr_pred']
    elif row['stage'] == 'Mid':
        return 0.25*row['gru_pred'] + 0.25*row['xgb_pred'] + 0.5*row['gbr_pred']
    else:
        return 0.4*row['gru_pred'] + 0.3*row['xgb_pred'] + 0.3*row['gbr_pred']

def create_sequences(X, window=30):
    return np.array([X[i:i+window] for i in range(len(X)-window+1)])

# ==========================================
# GAUGE
# ==========================================
def draw_gauge(health_pct, risk_label, accent_color):
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='none')
    ax.set_facecolor('none')
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.7, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')

    zones = [
        (0.00, 0.25, '#EF4444', '#FEE2E2'),
        (0.25, 0.50, '#F97316', '#FFF7ED'),
        (0.50, 0.75, '#F59E0B', '#FFFBEB'),
        (0.75, 1.00, '#10B981', '#DCFCE7'),
    ]
    for z0, z1, stroke, fill in zones:
        t0 = np.pi*(1-z0)
        t1 = np.pi*(1-z1)
        theta = np.linspace(t0, t1, 80)
        r_out, r_in = 1.02, 0.66
        xo = r_out*np.cos(theta); yo = r_out*np.sin(theta)
        xi = r_in *np.cos(theta[::-1]); yi = r_in*np.sin(theta[::-1])
        ax.fill(np.concatenate([xo,xi]), np.concatenate([yo,yi]),
                color=fill, zorder=2, linewidth=0)
        ax.plot(r_out*np.cos(theta), r_out*np.sin(theta), color=stroke, lw=2.5, zorder=3)
        ax.plot(r_in *np.cos(theta), r_in *np.sin(theta), color=stroke, lw=1.2, zorder=3, alpha=0.4)

    for frac in [0.25, 0.50, 0.75]:
        angle = np.pi*(1-frac)
        ax.plot([0.64*np.cos(angle),1.04*np.cos(angle)],
                [0.64*np.sin(angle),1.04*np.sin(angle)],
                color='#F4F2EE', lw=3.5, zorder=4)

    for i, frac in enumerate(np.linspace(0,1,21)):
        angle = np.pi*(1-frac)
        is_major = (i%5==0)
        r1 = 1.13 if is_major else 1.07
        ax.plot([1.03*np.cos(angle),r1*np.cos(angle)],
                [1.03*np.sin(angle),r1*np.sin(angle)],
                color='#9CA3AF', lw=1.8 if is_major else 0.7, zorder=4)

    for frac, lbl in [(0,'0%'),(0.25,'25%'),(0.5,'50%'),(0.75,'75%'),(1.0,'100%')]:
        angle = np.pi*(1-frac)
        ax.text(1.26*np.cos(angle), 1.26*np.sin(angle), lbl,
                ha='center', va='center', fontsize=7,
                color='#6B7280', fontfamily=CHART_FONT, fontweight='bold')

    # Needle
    needle_angle = np.pi*(1 - health_pct/100)
    nl = 0.84
    nx, ny = nl*np.cos(needle_angle), nl*np.sin(needle_angle)
    ax.plot([0,nx*0.97],[0,ny*0.97], color='#D1D5DB', lw=5, solid_capstyle='round', zorder=5)
    ax.plot([0,nx],[0,ny], color='#111827', lw=3, solid_capstyle='round', zorder=6)
    ax.plot([nx*0.55,nx],[ny*0.55,ny], color='#374151', lw=1.8, solid_capstyle='round', zorder=7)
    tail = needle_angle+np.pi
    ax.plot([0,0.13*np.cos(tail)],[0,0.13*np.sin(tail)],
            color='#374151', lw=3, solid_capstyle='round', zorder=6)
    for r, c in [(0.085,'#111827'),(0.058,'#374151'),(0.032,'#F4F2EE')]:
        ax.add_patch(plt.Circle((0,0), r, color=c, zorder=8))

    ax.text(0, -0.26, f"{health_pct:.1f}%",
            ha='center', va='center', fontsize=23, fontweight='bold',
            color='#111827', fontfamily=CHART_FONT, zorder=9)
    ax.text(0, -0.41, "ENGINE HEALTH",
            ha='center', va='center', fontsize=7.5,
            color='#9CA3AF', fontfamily=CHART_FONT, zorder=9)

    badge_styles = {
        'SAFE':    ('#ECFDF5','#065F46'),
        'MONITOR': ('#FFFBEB','#78350F'),
        'CAUTION': ('#FFF7ED','#92400E'),
        'CRITICAL':('#FEF2F2','#7F1D1D'),
    }
    bb, bf = badge_styles.get(risk_label, ('#F3F4F6','#374151'))
    badge = mpatches.FancyBboxPatch((-0.32,-0.58), 0.64, 0.105,
                                    boxstyle="round,pad=0.02",
                                    facecolor=bb, edgecolor=accent_color,
                                    linewidth=1.3, zorder=8)
    ax.add_patch(badge)
    ax.text(0, -0.525, risk_label,
            ha='center', va='center', fontsize=9.5,
            color=bf, fontfamily=MONO_FONT, fontweight='bold', zorder=9)
    fig.tight_layout(pad=0)
    return fig

# ==========================================
# RUL TREND
# ==========================================
def draw_rul_trend(df):
    fig, ax = plt.subplots(figsize=(11, 4), facecolor='#FFFFFF')
    ax.set_facecolor('#FAFAF8')
    cycles = df['time_in_cycles']
    ens    = df['ensemble_pred']
    valid  = [c for c in ['xgb_pred','gbr_pred','gru_pred'] if c in df.columns]
    mn = df[valid].min(axis=1)
    mx = df[valid].max(axis=1)

    ax.fill_between(cycles, mn, mx, alpha=0.10, color='#3B82F6', zorder=1)
    for col, color in [('xgb_pred','#10B981'),('gbr_pred','#F59E0B'),('gru_pred','#8B5CF6')]:
        if col in df.columns:
            ax.plot(cycles, df[col], color=color, lw=1.2, alpha=0.38, linestyle='--', zorder=2)
    ax.plot(cycles, ens, color='#2563EB', lw=2.8, zorder=3)
    ax.scatter([cycles.iloc[-1]], [ens.iloc[-1]], color='#2563EB', s=72, zorder=5,
               edgecolors='white', linewidths=2)

    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#E5E7EB')
    ax.tick_params(colors='#9CA3AF', labelsize=9)
    ax.set_xlabel("Cycle", fontsize=10, color='#6B7280', labelpad=8)
    ax.set_ylabel("Predicted RUL", fontsize=10, color='#6B7280', labelpad=8)
    ax.grid(axis='y', color='#F3F4F6', linewidth=1.0, zorder=0)
    ax.grid(axis='x', color='#F9FAFB', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        mpatches.Patch(color='#2563EB', label='Ensemble'),
        mpatches.Patch(color='#10B981', alpha=0.7, label='XGBoost'),
        mpatches.Patch(color='#F59E0B', alpha=0.7, label='GBR'),
        mpatches.Patch(color='#8B5CF6', alpha=0.7, label='GRU'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=True, framealpha=0.95,
              facecolor='white', edgecolor='#E5E7EB', fontsize=9)
    fig.tight_layout(pad=1.2)
    return fig

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
def draw_importance(xgb_model):
    importance = xgb_model.feature_importances_
    imp_df = pd.DataFrame({'feature':FEATURES,'importance':importance})
    imp_df = imp_df.sort_values('importance', ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#FFFFFF')
    ax.set_facecolor('#FAFAF8')
    colors = plt.cm.Blues(np.linspace(0.3, 0.85, len(imp_df)))
    bars = ax.barh(imp_df['feature'], imp_df['importance'],
                   color=colors, height=0.62, zorder=2, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, imp_df['importance']):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8.5, color='#6B7280',
                fontfamily=MONO_FONT)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.tick_params(colors='#6B7280', labelsize=9.5, left=False)
    ax.set_xlabel("Importance Score", fontsize=10, color='#6B7280', labelpad=8)
    ax.xaxis.grid(True, color='#F3F4F6', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily(MONO_FONT)
        lbl.set_fontsize(9)
    fig.tight_layout(pad=1.2)
    return fig

# ==========================================
# STAGE DISTRIBUTION
# ==========================================
def draw_stage_dist(df):
    stages = ['Early','Mid','Late']
    sc_map = {'Early':'#10B981','Mid':'#F59E0B','Late':'#EF4444'}
    counts = [df['stage'].value_counts().get(s,0) for s in stages]
    colors = [sc_map[s] for s in stages]
    fig, ax = plt.subplots(figsize=(4, 3.2), facecolor='#FFFFFF')
    ax.set_facecolor('#FAFAF8')
    for i,(s,c,col) in enumerate(zip(stages,counts,colors)):
        ax.bar(s, c, color=col, width=0.52, alpha=0.82, zorder=2,
               edgecolor='white', linewidth=1.5)
        offset = max(counts)*0.025 if max(counts)>0 else 1
        ax.text(i, c+offset, str(c), ha='center', va='bottom', fontsize=11,
                color='#374151', fontfamily=MONO_FONT, fontweight='bold')
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.tick_params(colors='#6B7280', labelsize=10, left=False)
    ax.set_ylabel("Cycles", fontsize=9.5, color='#6B7280', labelpad=6)
    ax.yaxis.grid(True, color='#F3F4F6', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily(CHART_FONT)
        lbl.set_fontweight('bold')
        lbl.set_color('#374151')
    fig.tight_layout(pad=1.0)
    return fig

# ==========================================
# CARD HELPER — avoids HTML string concat bugs
# ==========================================
def card_open(title_label):
    return f"""<div style="background:#FFFFFF;border-radius:16px;border:1px solid #E5E7EB;
        padding:1.2rem 1.4rem 0.6rem;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="font-family:'DM Mono',monospace;font-size:9px;color:#9CA3AF;
                    letter-spacing:0.12em;margin-bottom:0.6rem;">{title_label}</div>"""

def card_close():
    return "</div>"

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0.8rem 1rem;">
        <div style="font-family:'DM Mono',monospace;font-size:9px;color:#4B5563;letter-spacing:0.14em;margin-bottom:8px;">SYSTEM</div>
        <div style="font-family:'Playfair Display',serif;font-size:26px;color:#F9FAFB;line-height:1.15;font-weight:700;">RUL<br>Predict</div>
        <div style="font-family:'DM Mono',monospace;font-size:8.5px;color:#374151;margin-top:6px;letter-spacing:0.1em;">ENGINE HEALTH MONITOR v2.0</div>
    </div>
    <div style="border-top:1px solid #1F2937;margin:0 0.8rem 1.2rem;"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:9px;color:#4B5563;letter-spacing:0.12em;margin-bottom:8px;padding:0 0.8rem;">DATA SOURCE</div>',
                unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")

    st.markdown("""
    <div style="border-top:1px solid #1F2937;margin:1.2rem 0.8rem;"></div>
    <div style="font-family:'DM Mono',monospace;font-size:9px;color:#374151;letter-spacing:0.1em;margin-bottom:10px;padding:0 0.8rem;">MODELS ACTIVE</div>
    """, unsafe_allow_html=True)

    for m, c in [("GRU","#8B5CF6"),("XGBoost","#10B981"),("GBR","#F59E0B")]:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin:0 0.8rem 6px;padding:7px 10px;
                    background:#1F2937;border-radius:8px;border:1px solid #374151;">
            <div style="width:7px;height:7px;border-radius:50%;background:{c};flex-shrink:0;"></div>
            <span style="font-family:'DM Mono',monospace;font-size:11px;color:#9CA3AF;">{m}</span>
            <span style="margin-left:auto;font-family:'DM Mono',monospace;font-size:8.5px;color:#10B981;">● READY</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="border-top:1px solid #1F2937;margin:1.2rem 0.8rem 0.8rem;"></div>
    <div style="padding:0 0.8rem;">
        <div style="font-family:'DM Mono',monospace;font-size:8.5px;color:#374151;letter-spacing:0.1em;margin-bottom:4px;">DATASET</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:11px;color:#6B7280;line-height:1.7;">NASA C-MAPSS<br>FD001 · FD002 · FD003</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# MAIN
# ==========================================
def main():
    # Header
    st.markdown("""
    <div style="display:flex;align-items:baseline;gap:18px;margin-bottom:4px;margin-top:0.5rem;">
        <span style="font-family:'Playfair Display',serif;font-size:34px;font-weight:700;color:#111827;line-height:1;">
            Engine Health Dashboard
        </span>
        <span style="font-family:'DM Mono',monospace;font-size:10px;color:#9CA3AF;letter-spacing:0.1em;">
            TURBOFAN · RUL PREDICTION
        </span>
    </div>
    <div style="font-family:'DM Sans',sans-serif;font-size:13.5px;color:#9CA3AF;margin-bottom:1.8rem;font-style:italic;">
        Weighted ensemble · GRU + XGBoost + GBR · NASA C-MAPSS dataset
    </div>
    """, unsafe_allow_html=True)

    # ── EMPTY STATE ──
    if file is None:
        col_l, col_c, col_r = st.columns([1,2,1])
        with col_c:
            st.markdown("""
            <div style="background:#FFFFFF;border:1.5px dashed #D1D5DB;border-radius:20px;
                        padding:3.5rem 2rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                <div style="font-size:44px;margin-bottom:1rem;opacity:0.3;">⚙</div>
                <div style="font-family:'Playfair Display',serif;font-size:24px;color:#1F2937;
                            margin-bottom:0.6rem;font-weight:700;">No Engine Data Loaded</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:14px;color:#9CA3AF;
                            line-height:1.7;max-width:320px;margin:0 auto 1.5rem;">
                    Upload a CSV file from the sidebar to begin real-time RUL prediction and health analysis.
                </div>
                <div style="display:inline-block;background:#F9FAFB;border:1px solid #E5E7EB;
                            border-radius:10px;padding:10px 20px;">
                    <span style="font-family:'DM Mono',monospace;font-size:10px;color:#9CA3AF;letter-spacing:0.06em;">
                        EXPECTED: time_in_cycles · s_2 through s_21
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        i1, i2, i3 = st.columns(3)
        for col, icon, title, desc in [
            (i1, "🧠", "Weighted Ensemble",
             "Combines GRU, XGBoost and GBR using inverse-MAE weights for optimal accuracy"),
            (i2, "📊", "Stage-Aware Prediction",
             "Dynamically adjusts model weights based on Early / Mid / Late degradation stage"),
            (i3, "🔍", "Explainable AI",
             "XGBoost feature importance reveals which sensors drive each RUL prediction"),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:16px;
                            padding:1.4rem 1.2rem;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                    <div style="font-size:24px;margin-bottom:10px;">{icon}</div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:14px;font-weight:600;
                                color:#1F2937;margin-bottom:6px;">{title}</div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:12.5px;color:#9CA3AF;line-height:1.6;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        return

    # ── LOAD & PREDICT ──
    try:
        xgb, gbr, gru = load_models()
        df = pd.read_csv(file)

        df['life_ratio'] = df['time_in_cycles'] / 200.0
        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        init = xgb.predict(df[FEATURES])
        df['life_ratio'] = df['time_in_cycles'] / (df['time_in_cycles'] + init)
        df['stage']   = df['life_ratio'].apply(get_stage)
        df['xgb_pred'] = xgb.predict(df[FEATURES])
        df['gbr_pred'] = gbr.predict(df[FEATURES])
        df['gru_pred'] = np.nan

        if len(df) >= WINDOW:
            scaler   = MinMaxScaler()
            X_scaled = scaler.fit_transform(df[FEATURES])
            seq      = create_sequences(X_scaled)
            gru_out  = gru.predict(seq, verbose=0).flatten()
            df.iloc[WINDOW-1:, df.columns.get_loc('gru_pred')] = gru_out

        df['ensemble_pred'] = df.apply(stage_ensemble, axis=1)

        rul    = float(df['ensemble_pred'].iloc[-1])
        stage  = df['stage'].iloc[-1]
        ratio  = float(df['life_ratio'].iloc[-1])
        cycle  = float(df['time_in_cycles'].iloc[-1])
        total  = cycle + rul
        health = min(max((rul/total)*100, 0), 100) if total > 0 else 0
        risk, desc, txt_col, bg_col, accent = risk_category(rul)

        sc_map = {'Early':'#10B981','Mid':'#F59E0B','Late':'#EF4444'}
        sb_map = {'Early':'#ECFDF5','Mid':'#FFFBEB','Late':'#FEF2F2'}
        sc = sc_map.get(stage,'#6B7280')
        sb = sb_map.get(stage,'#F9FAFB')

        # ── KPI CARDS ──
        base_card = ("background:#FFFFFF;border-radius:14px;border:1px solid #E5E7EB;"
                     "padding:1.1rem 1.2rem;box-shadow:0 1px 4px rgba(0,0,0,0.04);")
        k1,k2,k3,k4,k5 = st.columns(5)
        for col, lbl, val, clr, sub in [
            (k1,"CURRENT CYCLE",  str(int(cycle)),  "#111827","cycles elapsed"),
            (k2,"RUL ESTIMATE",   str(int(rul)),    "#2563EB","cycles remaining"),
            (k3,"TOTAL ESTIMATE", str(int(total)),  "#111827","projected lifetime"),
            (k4,"LIFE RATIO",     f"{ratio:.2f}",   "#111827","0.0 → 1.0 scale"),
        ]:
            with col:
                st.markdown(f"""
                <div style="{base_card}">
                    <div style="font-family:'DM Mono',monospace;font-size:9px;color:#9CA3AF;
                                letter-spacing:0.12em;margin-bottom:8px;">{lbl}</div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:30px;
                                font-weight:600;color:{clr};line-height:1;">{val}</div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:11px;
                                color:#9CA3AF;margin-top:5px;">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        with k5:
            st.markdown(f"""
            <div style="background:{bg_col};border-radius:14px;border:1.5px solid {accent}44;
                        padding:1.1rem 1.2rem;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
                <div style="font-family:'DM Mono',monospace;font-size:9px;color:{txt_col}88;
                            letter-spacing:0.12em;margin-bottom:8px;">STATUS</div>
                <div style="font-family:'DM Mono',monospace;font-size:24px;font-weight:500;
                            color:{txt_col};line-height:1;">{risk}</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:11px;
                            color:{txt_col}99;margin-top:5px;">operational status</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Alert banner
        icons = {'SAFE':'✓','MONITOR':'◉','CAUTION':'⚠','CRITICAL':'✕'}
        st.markdown(f"""
        <div style="background:{bg_col};border-left:4px solid {accent};
                    border-radius:0 12px 12px 0;padding:0.8rem 1.3rem;
                    margin-bottom:1.2rem;display:flex;align-items:center;gap:14px;">
            <span style="font-size:16px;color:{accent};font-weight:700;">{icons.get(risk,'·')}</span>
            <span style="font-family:'DM Mono',monospace;font-size:11px;font-weight:500;
                         color:{txt_col};letter-spacing:0.06em;">{risk}</span>
            <span style="font-family:'DM Sans',sans-serif;font-size:13px;
                         color:{txt_col}CC;margin-left:6px;">— {desc}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── GAUGE + LIFECYCLE + STAGE ──
        g_col, lc_col, sd_col = st.columns([2.4, 1.5, 1.3])

        with g_col:
            st.markdown(card_open("HEALTH GAUGE"), unsafe_allow_html=True)
            st.pyplot(draw_gauge(health, risk, accent), use_container_width=True)
            st.markdown(card_close(), unsafe_allow_html=True)

        with lc_col:
            pct = (cycle/total*100) if total > 0 else 0
            st.markdown(f"""
            <div style="background:#FFFFFF;border-radius:16px;border:1px solid #E5E7EB;
                        padding:1.2rem 1.4rem;box-shadow:0 1px 4px rgba(0,0,0,0.04);height:100%;">
                <div style="font-family:'DM Mono',monospace;font-size:9px;color:#9CA3AF;
                            letter-spacing:0.12em;margin-bottom:1rem;">LIFECYCLE</div>
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-family:'DM Sans',sans-serif;font-size:11px;color:#6B7280;">Progress</span>
                    <span style="font-family:'DM Mono',monospace;font-size:11px;color:#111827;">{pct:.1f}%</span>
                </div>
                <div style="background:#F3F4F6;border-radius:100px;height:8px;overflow:hidden;margin-bottom:1.2rem;">
                    <div style="width:{pct:.1f}%;height:100%;
                                background:linear-gradient(90deg,#2563EB,#60A5FA);
                                border-radius:100px;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:9px 0;border-bottom:1px solid #F3F4F6;">
                    <span style="font-family:'DM Sans',sans-serif;font-size:12px;color:#9CA3AF;">Total Lifetime</span>
                    <span style="font-family:'DM Mono',monospace;font-size:16px;font-weight:500;color:#111827;">{int(total)}</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:9px 0;border-bottom:1px solid #F3F4F6;">
                    <span style="font-family:'DM Sans',sans-serif;font-size:12px;color:#9CA3AF;">Completed</span>
                    <span style="font-family:'DM Mono',monospace;font-size:16px;font-weight:500;color:#6B7280;">{int(cycle)}</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:9px 0;border-bottom:1px solid #F3F4F6;">
                    <span style="font-family:'DM Sans',sans-serif;font-size:12px;color:#9CA3AF;">Remaining</span>
                    <span style="font-family:'DM Mono',monospace;font-size:16px;font-weight:500;color:#2563EB;">{int(rul)}</span>
                </div>
                <div style="margin-top:1rem;display:flex;align-items:center;gap:8px;
                            background:{sb};border-radius:10px;padding:9px 12px;">
                    <div style="width:9px;height:9px;border-radius:50%;background:{sc};"></div>
                    <span style="font-family:'DM Sans',sans-serif;font-size:12px;
                                 font-weight:500;color:{sc};">{stage} Stage</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with sd_col:
            st.markdown(card_open("STAGE DISTRIBUTION"), unsafe_allow_html=True)
            st.pyplot(draw_stage_dist(df), use_container_width=True)
            st.markdown(card_close(), unsafe_allow_html=True)

        st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

        # ── RUL TREND ──
        st.markdown(card_open("RUL PREDICTION TREND — ALL MODELS"), unsafe_allow_html=True)
        st.pyplot(draw_rul_trend(df), use_container_width=True)
        st.markdown(card_close(), unsafe_allow_html=True)

        st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

        # ── FEATURE IMPORTANCE + ENSEMBLE WEIGHTS ──
        fi_col, ew_col = st.columns([2.2, 1])

        with fi_col:
            st.markdown(card_open("FEATURE IMPORTANCE · XGBoost"), unsafe_allow_html=True)
            st.pyplot(draw_importance(xgb), use_container_width=True)
            st.markdown(card_close(), unsafe_allow_html=True)

        with ew_col:
            wt_map = {
                'Early': [('GRU','#8B5CF6',10),('XGBoost','#10B981',20),('GBR','#F59E0B',70)],
                'Mid':   [('GRU','#8B5CF6',25),('XGBoost','#10B981',25),('GBR','#F59E0B',50)],
                'Late':  [('GRU','#8B5CF6',40),('XGBoost','#10B981',30),('GBR','#F59E0B',30)],
            }
            wts = wt_map.get(stage, wt_map['Mid'])

            with st.expander(f"Ensemble Weights · {stage}", expanded=False):
                for m, c, w in wts:
                    st.markdown(f"""
                    <div style="margin-bottom:14px;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                            <span style="font-family:'DM Mono',monospace;font-size:11px;color:#374151;">{m}</span>
                            <span style="font-family:'DM Mono',monospace;font-size:11px;color:{c};font-weight:500;">{w}%</span>
                        </div>
                        <div style="background:#F3F4F6;border-radius:100px;height:9px;overflow:hidden;">
                            <div style="width:{w}%;height:100%;background:{c};border-radius:100px;opacity:0.85;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:1rem;background:#F9FAFB;border-radius:10px;
                            border:1px solid #E5E7EB;padding:10px 12px;">
                    <div style="font-family:'DM Mono',monospace;font-size:8.5px;color:#9CA3AF;
                                letter-spacing:0.08em;margin-bottom:5px;">FORMULA</div>
                    <div style="font-family:'DM Mono',monospace;font-size:10px;
                                color:#374151;line-height:1.8;">
                        RUL = &Sigma; w&#x1D62; &times; RUL&#x1D62;<br>
                        w&#x1D62; &prop; 1 / MAE&#x1D62;
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid #E5E7EB;text-align:center;">
            <span style="font-family:'DM Mono',monospace;font-size:9px;color:#D1D5DB;letter-spacing:0.12em;">
                RUL PREDICT &nbsp;·&nbsp; NASA C-MAPSS &nbsp;·&nbsp; GRU + XGBoost + GBR WEIGHTED ENSEMBLE
            </span>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:12px;
                    padding:1.2rem 1.5rem;margin-top:1rem;">
            <div style="font-family:'DM Mono',monospace;font-size:11px;color:#7F1D1D;
                        margin-bottom:6px;letter-spacing:0.08em;">ERROR</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:#991B1B;">{str(e)}</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()