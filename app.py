import matplotlib
matplotlib.use('Agg')  # Forces a headless backend to prevent GUI DLL clashes on Windows
import shap
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
import config
from data_loader import load_pond_data, load_retrospective_data, load_model
from streamlit_autorefresh import st_autorefresh
from ml_models import compute_shap_values
from data_loader import load_pond_data, load_retrospective_data, load_model, load_inspection_log, save_inspection
from chat_module import init_semantic_index

init_semantic_index()

st.set_page_config(
    page_title="AshPond System",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0a0f !important;
    color: #e2e8f0 !important;
}
.stApp { background-color: #0a0a0f !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0f1a !important;
    border-right: 1px solid rgba(0,240,255,0.08) !important;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ── Sidebar divider ── */
[data-testid="stSidebar"] hr {
    border-color: rgba(0,240,255,0.1) !important;
}

/* ── Sidebar button ── */
[data-testid="stSidebar"] [data-testid="stButton"] button {
    background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 50px !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
    width: 100% !important;
    transition: opacity 0.2s ease !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    opacity: 0.82 !important;
}

/* ── Page headers ── */
h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(120deg, #ffffff 30%, #a855f7 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
h2, h3 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
    letter-spacing: -0.01em !important;
}
[data-testid="stCaption"] p {
    color: rgba(255,255,255,0.35) !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.04em !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,240,255,0.1) !important;
    margin: 1.4rem 0 !important;
}

/* ── Main area selectbox ── */
[data-testid="stSelectbox"] label p {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: rgba(255,255,255,0.4) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #13172a !important;
    border: 1px solid rgba(0,240,255,0.15) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #13172a !important;
    border: 1px solid rgba(0,240,255,0.1) !important;
    border-radius: 10px !important;
    padding: 16px 18px !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: rgba(255,255,255,0.4) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"] > div {
    font-size: 0.7rem !important;
    color: rgba(0,240,255,0.65) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ── Slider ── */
[data-testid="stSlider"] label p {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: rgba(255,255,255,0.4) !important;
}

/* ── Plotly chart card ── */
[data-testid="stPlotlyChart"] {
    background: #0d0f1a !important;
    border: 1px solid rgba(0,240,255,0.08) !important;
    border-radius: 12px !important;
    padding: 2px !important;
}

/* ── Folium map ── */
iframe {
    border-radius: 12px !important;
    border: 1px solid rgba(0,240,255,0.1) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #13172a !important;
    border: 1px solid rgba(0,240,255,0.08) !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea { color: #e2e8f0 !important; }
[data-testid="stChatInput"] {
    background: #13172a !important;
    border: 1px solid rgba(168,85,247,0.3) !important;
    border-radius: 50px !important;
}

/* ── Warnings / alerts ── */
[data-testid="stAlert"] {
    background: rgba(124,58,237,0.08) !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 10px !important;
    color: #c4b5fd !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: #13172a !important;
    border: 1px solid rgba(0,240,255,0.1) !important;
    border-radius: 10px !important;
}

/* ── Animated critical pulse ── */
@keyframes pulse-critical {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
    50%      { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
}

/* ── Feature section header card ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    background: linear-gradient(135deg, rgba(124,58,237,0.1) 0%, rgba(6,182,212,0.05) 100%);
    border: 1px solid rgba(0,240,255,0.12);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 14px;
}
.section-header .icon-box {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.section-header .label {
    font-size: 0.95rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.01em;
}
.section-header .sublabel {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.35);
    margin-top: 1px;
    letter-spacing: 0.03em;
}

/* ── Page hero banner ── */
.hero-banner {
    background: linear-gradient(135deg,
        rgba(124,58,237,0.14) 0%,
        rgba(6,182,212,0.06) 60%,
        rgba(0,0,0,0) 100%);
    border: 1px solid rgba(0,240,255,0.12);
    border-radius: 16px;
    padding: 30px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(168,85,247,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner h1 {
    margin: 0 0 6px 0 !important;
    font-size: 1.85rem !important;
}
.hero-banner .sub {
    color: rgba(255,255,255,0.38) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.04em;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Risk badge pills ── */
.badge {
    display: inline-block;
    border-radius: 50px;
    padding: 3px 14px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-critical {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.45);
    color: #fca5a5;
    animation: pulse-critical 1.8s ease-in-out infinite;
}
.badge-elevated {
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.4);
    color: #fcd34d;
}
.badge-stable {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.35);
    color: #86efac;
}
</style>
""", unsafe_allow_html=True)

st_autorefresh(interval=30 * 60 * 1000, key="data_refresh")

# ── CACHED DATA ───────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_pond_data().copy()

@st.cache_data
def get_retro():
    return load_retrospective_data().copy()

@st.cache_resource
def get_ml_model():
    return load_model()

#app.py

@st.cache_data(show_spinner="Computing SHAP explanations...")
def get_cached_shap(ndwi, ndvi, slope, rainfall, proximity):
    model, scaler = get_ml_model()
    dummy_row = {
        'ndwi_score': ndwi,
        'ndvi_score': ndvi,
        'slope_degrees': slope,
        'rainfall_forecast_mm': rainfall,
        'breach_proximity_score': proximity
    }
    return compute_shap_values(dummy_row, model, scaler)

def get_inspection_log():
    return load_inspection_log()

try:
    df       = get_data()
    retro_df = get_retro()
    model, scaler = get_ml_model()
    inspection_log = get_inspection_log()
except Exception as e:
    st.error(f"Fatal Pipeline Data Load Error: {e}")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    rain_source = df['rainfall_source'].iloc[0] if 'rainfall_source' in df.columns else 'unknown'
    rain_label  = "🟢 Live" if rain_source == "live" else "🟡 Cached"
    n_critical  = len(df[df.risk_category == "CRITICAL"])
    n_elevated  = len(df[df.risk_category == "ELEVATED"])
    n_stable    = len(df[df.risk_category == "STABLE"])
    anomalies   = int(df['profile_anomaly'].sum()) if 'profile_anomaly' in df.columns else 0

    st.markdown(f"""
    <div style="padding:10px 0 18px 0;">
        <div style="font-size:1.15rem;font-weight:800;letter-spacing:-0.01em;
                    background:linear-gradient(120deg,#ffffff 30%,#a855f7 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">
            ⚠️ AshPond System
        </div>
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.3);margin-top:3px;
                    font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;">
            AI-powered Breach Early Warning
        </div>
    </div>

    <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:16px;">
        <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.45);
                    border-radius:10px;padding:11px 14px;
                    animation:pulse-critical 1.8s ease-in-out infinite;">
            <div style="font-size:0.65rem;color:#fca5a5;letter-spacing:0.1em;
                        font-weight:700;text-transform:uppercase;">Critical</div>
            <div style="font-size:2.1rem;font-weight:800;color:#fca5a5;line-height:1.1;
                        font-family:'JetBrains Mono',monospace;">{n_critical}</div>
        </div>
        <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.4);
                    border-radius:10px;padding:11px 14px;">
            <div style="font-size:0.65rem;color:#fcd34d;letter-spacing:0.1em;
                        font-weight:700;text-transform:uppercase;">Elevated</div>
            <div style="font-size:2.1rem;font-weight:800;color:#fcd34d;line-height:1.1;
                        font-family:'JetBrains Mono',monospace;">{n_elevated}</div>
        </div>
        <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.35);
                    border-radius:10px;padding:11px 14px;">
            <div style="font-size:0.65rem;color:#86efac;letter-spacing:0.1em;
                        font-weight:700;text-transform:uppercase;">Stable</div>
            <div style="font-size:2.1rem;font-weight:800;color:#86efac;line-height:1.1;
                        font-family:'JetBrains Mono',monospace;">{n_stable}</div>
        </div>
        <div style="background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.35);
                    border-radius:10px;padding:11px 14px;">
            <div style="font-size:0.65rem;color:#fb923c;letter-spacing:0.1em;
                        font-weight:700;text-transform:uppercase;">Anomalous</div>
            <div style="font-size:2.1rem;font-weight:800;color:#fb923c;line-height:1.1;
                        font-family:'JetBrains Mono',monospace;">{anomalies}</div>
        </div>
    </div>

    <div style="background:rgba(13,15,26,0.8);border:1px solid rgba(0,240,255,0.08);
                border-radius:10px;padding:11px 14px;margin-bottom:10px;
                font-size:0.76rem;color:rgba(255,255,255,0.45);line-height:2;">
        <div>📍 Singrauli, MP — 5 ponds</div>
        <div>📍 Korba, CG — 5 ponds</div>
    </div>

    <div style="background:rgba(13,15,26,0.8);border:1px solid rgba(0,240,255,0.08);
                border-radius:10px;padding:11px 14px;margin-bottom:16px;
                font-size:0.72rem;color:rgba(255,255,255,0.3);
                font-family:'JetBrains Mono',monospace;line-height:1.9;">
        <div>🛰 Sync: {getattr(config, 'LAST_FETCH', 'Unknown')}</div>
        <div>🌧 Rainfall: {rain_label}</div>
        <div>📡 Sentinel-2 Optical + SAR</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.08em;color:rgba(255,255,255,0.4);
                margin-bottom:8px;">
        📋 Field Inspection Log
    </div>
    """, unsafe_allow_html=True)

    with st.form("inspection_form", clear_on_submit=True):
        pond_options = df[['pond_id','pond_name']].values.tolist()
        pond_labels  = [f"{p[1]}" for p in pond_options]
        selected_idx = st.selectbox("Pond", range(len(pond_labels)),
                                    format_func=lambda i: pond_labels[i],
                                    key="insp_pond")
        inspector    = st.text_input("Inspector Name", placeholder="e.g. R. Sharma")
        notes        = st.text_area("Observations", placeholder="Seepage visible at north face...", height=80)
        photo        = st.file_uploader("Photo (optional)", type=["jpg","jpeg","png"])
        submitted    = st.form_submit_button("Submit Inspection", use_container_width=True)

        if submitted and inspector.strip():
            pid       = pond_options[selected_idx][0]
            pname     = pond_options[selected_idx][1]
            photo_fn  = ""
            if photo is not None:
                import os
                photo_fn = f"{pid}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{photo.name}"
                photo_path = os.path.join("data", "photos", photo_fn)
                os.makedirs(os.path.dirname(photo_path), exist_ok=True)
                with open(photo_path, "wb") as f:
                    f.write(photo.getbuffer())
            save_inspection(pid, pname, inspector.strip(), notes.strip(), photo_fn)
            st.cache_data.clear()
            st.success("✅ Inspection logged.")
            st.rerun()

    st.divider()

    if st.button("🔄 Refresh Pipeline", use_container_width=True):
        import sys, subprocess
        with st.spinner("Running pipeline..."):
            subprocess.run([sys.executable, "fetch_rainfall.py"])
            subprocess.run([sys.executable, "compute_dii.py"])
            subprocess.run([sys.executable, "ml_models.py"])
        st.cache_data.clear()
        st.rerun()

# ── HERO BANNER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>Coal Ash Pond<br>Breach Early Warning</h1>
    <p class="sub">DII &amp; Breach Probability · SHAP Explainability · Anomaly Detection · Change Point Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── FEATURE 1 — MAP ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon-box">🗺️</div>
    <div>
        <div class="label">Risk Mapping Matrix</div>
        <div class="sublabel">Live geospatial risk overlay — Singrauli &amp; Korba</div>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    from map_module import build_risk_map
    map_obj = build_risk_map(df, inspection_log)
    st_folium(map_obj, width="100%", height=480, returned_objects=[], key="risk_map")
except Exception as e:
    st.warning(f"Map Rendering Offline: {e}")
    st.dataframe(df[['pond_name', 'dii_score', 'breach_probability', 'risk_category']])

st.divider()

# ── FEATURE 2 — DII DASHBOARD ─────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon-box">📊</div>
    <div>
        <div class="label">DII Dashboard &amp; SHAP Explainability</div>
        <div class="sublabel">Dyke Instability Index · AI-driven breach probability</div>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    from charts_module import build_dii_bar_chart, build_shap_waterfall, build_factor_breakdown_chart
    col1, col2 = st.columns([1.8, 1])

    with col1:
        st.plotly_chart(build_dii_bar_chart(df), use_container_width=True)

    with col2:
        pond_names = df.sort_values('dii_score', ascending=False)['pond_name'].tolist()
        selected   = st.selectbox("Select Pond for AI Inspection", pond_names, key="pond_selector")
        row        = df[df['pond_name'] == selected].iloc[0]

        badge_map = {
            "CRITICAL": '<span class="badge badge-critical">Critical</span>',
            "ELEVATED": '<span class="badge badge-elevated">Elevated</span>',
            "STABLE":   '<span class="badge badge-stable">Stable</span>',
        }
        st.markdown(badge_map.get(row['risk_category'], ''), unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        risk_icon = {"CRITICAL": "🔴", "ELEVATED": "🟠", "STABLE": "🟢"}
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                f"{risk_icon.get(row['risk_category'], '⚪')} DII Score",
                f"{row['dii_score']:.2f} {row.get('dii_trend', '→')}",
                delta=f"±{(row.get('dii_upper', row['dii_score']) - row.get('dii_lower', row['dii_score']))/2:.2f} uncertainty"
            )
        with col_b:
            breach_pct = int(row.get('breach_probability', 0) * 100)
            st.metric(
                "P(Breach / 30d)",
                f"{breach_pct}%",
                delta="ANOMALOUS PROFILE" if row.get('profile_anomaly', False) else "Normal baseline"
            )

        extra_rain = st.slider("Simulate Rainfall Shift (mm)", 0, 200, 0, step=10, key="rain_slider")
        if extra_rain > 0:
            sim_dii = min(row['dii_score'] + (extra_rain / 500), 1.0)
            sim_cat = "CRITICAL" if sim_dii > 0.70 else "ELEVATED" if sim_dii > 0.40 else "STABLE"
            st.metric("Simulated DII", f"{sim_dii:.2f}", delta=f"→ {sim_cat} (+{extra_rain}mm)")

        #app.py

        if model is not None:
            # We pass the slider's extra_rain added to the original rainfall value
            shap_dict = get_cached_shap(
                row['ndwi_score'],
                row['ndvi_score'],
                row['slope_degrees'],
                row['rainfall_forecast_mm'] + extra_rain,
                row['breach_proximity_score']
            )
            st.plotly_chart(build_shap_waterfall(shap_dict, row['pond_name']), use_container_width=True)
        else:
            st.plotly_chart(build_factor_breakdown_chart(row), use_container_width=True)

except Exception as e:
    st.warning(f"Analytics Dashboard Engine Error: {e}")

st.divider()

# ── FEATURE 3 — RETROSPECTIVE ─────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon-box">📈</div>
    <div>
        <div class="label">Retrospective Timeline &amp; Change Point Detection</div>
        <div class="sublabel">Sasan UMPP 2020 — simulated DII vs actual breach event</div>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    from charts_module import build_retrospective_chart
    st.plotly_chart(build_retrospective_chart(retro_df), use_container_width=True)
except Exception as e:
    st.warning(f"Historical Simulation Engine Error: {e}")

st.divider()

# ── FEATURE 4 — CHAT ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon-box">💬</div>
    <div>
        <div class="label">Intelligence Assistant</div>
        <div class="sublabel">Tool-augmented · natural language risk queries</div>
    </div>
</div>
<p style="color:rgba(255,255,255,0.28);font-size:0.74rem;margin:-6px 0 14px 0;
          font-family:'JetBrains Mono',monospace;letter-spacing:0.03em;">
    Try: "Which pond is most dangerous?" · "Simulate 120mm rain at Sasan" · "Explain the SHAP factors"
</p>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Query current ash pond risk metrics..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Executing analytical verification across framework tables..."):
            try:
                from chat_module import ask_assistant
                response = ask_assistant(prompt, df)
            except Exception as e:
                response = f"Chat Client Connection Timeout: {str(e)[:60]}"
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

