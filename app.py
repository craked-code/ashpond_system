import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
import config
from data_loader import load_pond_data, load_retrospective_data, load_model
from streamlit_autorefresh import st_autorefresh

from ml_models import compute_shap_values

from chat_module import init_semantic_index
init_semantic_index()

#st_autorefresh(interval=30 * 60 * 1000, key="data_refresh")

st.set_page_config(
    page_title="AshPond System",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_autorefresh(interval=30 * 60 * 1000, key="data_refresh")

@st.cache_data
def get_data():
    return load_pond_data()

@st.cache_data
def get_retro():
    return load_retrospective_data()

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
    st.title("AshPond System")
    st.caption("AI-powered Breach Early Warning")
    st.divider()
    st.metric("🔴 CRITICAL", len(df[df.risk_category == "CRITICAL"]))
    st.metric("🟠 ELEVATED", len(df[df.risk_category == "ELEVATED"]))
    st.metric("🟢 STABLE",   len(df[df.risk_category == "STABLE"]))
    anomalies = df['profile_anomaly'].sum() if 'profile_anomaly' in df.columns else 0
    st.metric("🔶 ANOMALOUS", int(anomalies))
    st.divider()
    st.caption("Singrauli, MP — 5 ponds")
    st.caption("Korba, CG — 5 ponds")
    st.divider()
    st.caption(f"Satellite sync: {getattr(config, 'LAST_FETCH', 'Unknown')}")
    st.caption("Sensors: Sentinel-2 Optical + Sentinel-1 SAR")

st.title("⚠️ Coal Ash Pond Breach Early Warning System")
st.caption("DII & Breach Probability · SHAP Explainability · Anomaly Detection · Change Point Analysis")
st.divider()
if st.sidebar.button("🔄 Refresh Now"):
    import subprocess
    subprocess.run(["python", "fetch_rainfall.py"])
    subprocess.run(["python", "compute_dii.py"])
    subprocess.run(["python", "ml_models.py"])
    st.cache_data.clear()
    st.rerun()

# FEATURE 1: FOLIUM MAP (Person 1 Component)
st.subheader("🗺️ Feature 1 — Risk Mapping Matrix")
try:
    from map_module import build_risk_map
    map_obj = build_risk_map(df)
    st_folium(map_obj, width="100%", height=480, returned_objects=[])
except Exception as e:
    st.warning(f"Map Rendering Offline: {e}")
    st.dataframe(df[['pond_name', 'dii_score', 'breach_probability', 'risk_category']])

st.divider()

# FEATURE 2: DII DASHBOARD + EXPLAINABLE AI (Person 2 Component)
st.subheader("📊 Feature 2 — DII Dashboard & SHAP Explainability")
try:
    from charts_module import build_dii_bar_chart, build_shap_waterfall, build_factor_breakdown_chart
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.plotly_chart(build_dii_bar_chart(df), use_container_width=True)
    with col2:
        pond_names = df.sort_values('dii_score', ascending=False)['pond_name'].tolist()
        selected = st.selectbox("Select Pond for Component-Level AI Inspection", pond_names)
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
        extra_rain = st.slider("Simulate Local Monsoonal Rainfall Shift (mm)", 0, 200, 0, step=10)
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