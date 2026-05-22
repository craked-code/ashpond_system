import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
import config
from data_loader import load_pond_data, load_retrospective_data, load_model
from streamlit_autorefresh import st_autorefresh

from chat_module import init_semantic_index
init_semantic_index()

#st_autorefresh(interval=30 * 60 * 1000, key="data_refresh")

st.set_page_config(
    page_title="AshPond System",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e0e0e0;
}

[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a2f3e;
}

[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

[data-testid="stMetric"] {
    background: #1c2333;
    border-radius: 8px;
    padding: 12px 16px;
    border: 1px solid #2a2f3e;
}

[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}

div[data-testid="stChatMessage"] {
    background: #1c2333;
    border-radius: 8px;
    border: 1px solid #2a2f3e;
    margin-bottom: 8px;
}

.critical-badge {
    display: inline-block;
    background: #7f1d1d;
    color: #fca5a5;
    border: 1px solid #ef4444;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    animation: pulse-critical 1.8s ease-in-out infinite;
}

.elevated-badge {
    display: inline-block;
    background: #78350f;
    color: #fcd34d;
    border: 1px solid #f59e0b;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}

.stable-badge {
    display: inline-block;
    background: #14532d;
    color: #86efac;
    border: 1px solid #22c55e;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}

@keyframes pulse-critical {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50%       { box-shadow: 0 0 0 6px rgba(239,68,68,0); }
}

[data-testid="stHorizontalBlock"] > div {
    gap: 1rem;
}

h1, h2, h3 {
    color: #e2e8f0 !important;
}

hr {
    border-color: #2a2f3e !important;
}
</style>
""", unsafe_allow_html=True)

st_autorefresh(interval=30 * 60 * 1000, key="data_refresh")

@st.cache_data
def get_data():
    return load_pond_data().copy()

@st.cache_data
def get_retro():
    return load_retrospective_data().copy()

@st.cache_resource
def get_ml_model():
    return load_model()

# Global Data Loads
try:
    df = get_data()
    retro_df = get_retro()
    model, scaler = get_ml_model()
except Exception as e:
    st.error(f"Fatal Pipeline Data Load Error: {e}")
    st.stop()

# SIDEBAR MONITORING METRICS
with st.sidebar:
    rain_source = df['rainfall_source'].iloc[0] if 'rainfall_source' in df.columns else 'unknown'
    rain_label  = "🟢 Live" if rain_source == "live" else "🟡 Cached"
    n_critical = len(df[df.risk_category == "CRITICAL"])
    n_elevated = len(df[df.risk_category == "ELEVATED"])
    n_stable   = len(df[df.risk_category == "STABLE"])
    anomalies  = int(df['profile_anomaly'].sum()) if 'profile_anomaly' in df.columns else 0

    st.markdown(f"""
    <div style="padding: 8px 0 16px 0;">
        <div style="font-size:1.25rem;font-weight:700;color:#e2e8f0;letter-spacing:0.03em;">
            ⚠️ AshPond System
        </div>
        <div style="font-size:0.75rem;color:#64748b;margin-top:2px;">
            AI-powered Breach Early Warning
        </div>
    </div>

    <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:16px;">
        <div style="background:#7f1d1d;border:1px solid #ef4444;border-radius:8px;padding:10px 14px;
                    animation: pulse-critical 1.8s ease-in-out infinite;">
            <div style="font-size:0.7rem;color:#fca5a5;letter-spacing:0.08em;font-weight:600;">CRITICAL</div>
            <div style="font-size:2rem;font-weight:700;color:#fca5a5;line-height:1.1;">{n_critical}</div>
        </div>
        <div style="background:#78350f;border:1px solid #f59e0b;border-radius:8px;padding:10px 14px;">
            <div style="font-size:0.7rem;color:#fcd34d;letter-spacing:0.08em;font-weight:600;">ELEVATED</div>
            <div style="font-size:2rem;font-weight:700;color:#fcd34d;line-height:1.1;">{n_elevated}</div>
        </div>
        <div style="background:#14532d;border:1px solid #22c55e;border-radius:8px;padding:10px 14px;">
            <div style="font-size:0.7rem;color:#86efac;letter-spacing:0.08em;font-weight:600;">STABLE</div>
            <div style="font-size:2rem;font-weight:700;color:#86efac;line-height:1.1;">{n_stable}</div>
        </div>
        <div style="background:#1c2333;border:1px solid #f97316;border-radius:8px;padding:10px 14px;">
            <div style="font-size:0.7rem;color:#fb923c;letter-spacing:0.08em;font-weight:600;">ANOMALOUS</div>
            <div style="font-size:2rem;font-weight:700;color:#fb923c;line-height:1.1;">{anomalies}</div>
        </div>
    </div>

    <div style="background:#1c2333;border:1px solid #2a2f3e;border-radius:8px;padding:10px 14px;
                margin-bottom:12px;font-size:0.78rem;color:#94a3b8;line-height:1.8;">
        <div>📍 Singrauli, MP — 5 ponds</div>
        <div>📍 Korba, CG — 5 ponds</div>
    </div>

    <div style="background:#1c2333;border:1px solid #2a2f3e;border-radius:8px;padding:10px 14px;
                font-size:0.75rem;color:#64748b;line-height:1.8;">
        <div>🛰️ Sync: {getattr(config, 'LAST_FETCH', 'Unknown')}</div>
        <div>🌧️ Rainfall: {rain_label}</div>
        <div>📡 Sentinel-2 Optical + Sentinel-1 SAR</div>
    </div>
    """, unsafe_allow_html=True)

st.title("⚠️ Coal Ash Pond Breach Early Warning System")
st.caption("DII & Breach Probability · SHAP Explainability · Anomaly Detection · Change Point Analysis")
st.divider()

if st.sidebar.button("🔄 Refresh Now"):
    import sys, subprocess
    with st.spinner("Running pipeline..."):
        subprocess.run([sys.executable, "fetch_rainfall.py"])
        subprocess.run([sys.executable, "compute_dii.py"])
        subprocess.run([sys.executable, "ml_models.py"])
    st.cache_data.clear()
    st.rerun()

# FEATURE 1: FOLIUM MAP (Person 1 Component)
st.subheader("🗺️ Feature 1 — Risk Mapping Matrix")
try:
    from map_module import build_risk_map
    map_obj = build_risk_map(df)
    st_folium(map_obj, width="100%", height=480, returned_objects=[], key="risk_map")
except Exception as e:
    st.warning(f"Map Rendering Offline: {e}")
    st.dataframe(df[['pond_name', 'dii_score', 'breach_probability', 'risk_category']])

st.divider()

# FEATURE 2: DII DASHBOARD + EXPLAINABLE AI (Person 2 Component)
st.subheader("📊 Feature 2 — DII Dashboard & SHAP Explainability")
try:
    from charts_module import build_dii_bar_chart, build_shap_waterfall, build_factor_breakdown_chart
    col1, col2 = st.columns([1.8, 1])

    with col1:
        st.plotly_chart(build_dii_bar_chart(df), use_container_width=True)
    with col2:
        pond_names = df.sort_values('dii_score', ascending=False)['pond_name'].tolist()
        selected = st.selectbox("Select Pond for Component-Level AI Inspection", pond_names, key="pond_selector")
        row = df[df['pond_name'] == selected].iloc[0]
        
        risk_icon = {"CRITICAL": "🔴", "ELEVATED": "🟠", "STABLE": "🟢"}
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                f"{risk_icon.get(row['risk_category'], '⚪')} DII Score",
                f"{row['dii_score']:.2f} {row.get('dii_trend', '→')}",
                delta=f"±{(row.get('dii_upper', row['dii_score']) - row.get('dii_lower', row['dii_score']))/2:.2f} Uncertainty"
            )
        with col_b:
            breach_pct = int(row.get('breach_probability', 0) * 100)
            st.metric("P(Breach Within 30 Days)", f"{breach_pct}%",
                      delta="ANOMALOUS FACTOR PROFILE" if row.get('profile_anomaly', False) else "Normal Baseline")

        # Interactive Simulation Sliders
        extra_rain = st.slider("Simulate Local Monsoonal Rainfall Shift (mm)", 0, 200, 0, step=10, key="rain_slider")
        if extra_rain > 0:
            sim_dii = min(row['dii_score'] + (extra_rain / 500), 1.0)
            sim_cat = "CRITICAL" if sim_dii > 0.70 else "ELEVATED" if sim_dii > 0.40 else "STABLE"
            st.metric("Simulated DII Response Target", f"{sim_dii:.2f}", delta=f"Transitions to {sim_cat} (+{extra_rain}mm)")

        if model is not None:
            from ml_models import compute_shap_values
            shap_dict = compute_shap_values(row, model, scaler)
            st.plotly_chart(build_shap_waterfall(shap_dict, row['pond_name']), use_container_width=True)
        else:
            st.plotly_chart(build_factor_breakdown_chart(row), use_container_width=True)
except Exception as e:
    st.warning(f"Analytics Dashboard Engine Error: {e}")

st.divider()

# FEATURE 3: RETROSPECTIVE CPD VALIDATION (Person 2 Component)
st.subheader("📈 Feature 3 — Retrospective Timeline Analysis & Change Point Detection")
try:
    from charts_module import build_retrospective_chart
    st.plotly_chart(build_retrospective_chart(retro_df), use_container_width=True)
except Exception as e:
    st.warning(f"Historical Simulation Engine Error: {e}")

st.divider()

# FEATURE 4: LIVE AI RISK ASSISTANT CHAT TERMINAL (Your Module)
st.subheader("💬 Feature 4 — Intelligence Assistant (Tool-Augmented System)")
st.caption("Interface verified for natural language operations like: 'Which pond is most dangerous?' or 'Simulate 120mm rain at Sasan'.")

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