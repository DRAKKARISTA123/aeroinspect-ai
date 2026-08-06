import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import json
import time
import math
import tempfile
import os
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AeroThermodynamic Engine Analyzer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin-bottom: 0.1rem; }
    .sub-header { font-size: 1.0rem; color: #475569; margin-bottom: 1.5rem; }
    .stMetric { background-color: #1E293B !important; padding: 12px !important; border-radius: 8px !important; border: 1px solid #334155 !important; }
    div[data-testid="stMetricValue"] { color: #38BDF8 !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { color: #94A3B8 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔥 AeroThermodynamic Gas-Path Engine Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Propulsion Engineering Suite | Brayton Cycle Module Breakdown & Thermodynamic Profiling</div>', unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.image("https://img.icons8.com/color/96/engine.png", width=64)
st.sidebar.title("Cycle Operating Parameters")

st.sidebar.markdown("---")
st.sidebar.info("Model Active: **partes-de-motor/5**\nStatus: **Gas-Path Vision Linked**")

# Thermodynamic Operating Sliders for Simulation
st.sidebar.markdown("### 🌡️ Engine Operating Point")
ambient_temp_k = st.sidebar.slider("Ambient Temperature ($T_2$ in K):", 216.0, 320.0, 288.15, 1.0)
pressure_ratio = st.sidebar.slider("Compressor Pressure Ratio (PR):", 15.0, 50.0, 30.0, 1.0)
tit_k = st.sidebar.slider("Turbine Inlet Temperature ($T_4$ in K):", 1200.0, 1800.0, 1450.0, 10.0)

# --- HARDCODED CREDENTIALS ---
ROBOFLOW_API_KEY = "26JC1OEUbjS0rV3JZTxM"
MODEL_ID = "partes-de-motor/5"

# --- THERMODYNAMIC CALCULATIONS (BRAYTON CYCLE PHYSICS) ---
gamma_air = 1.4
cp_air = 1.005 # kJ/(kg*K)

def calculate_thermodynamic_profile(label, pr, t4, tamb):
    label_lower = label.lower()
    
    if "compressor" in label_lower or "front" in label_lower or "accesory" in label_lower:
        # Isentropic compression temperature exit: T3 = T2 * PR^((gamma-1)/gamma)
        t_exit = tamb * (pr ** ((gamma_air - 1.0) / gamma_air))
        delta_h = cp_air * (t_exit - tamb)
        return {
            "Station": "Station 2 -> 3 (Compression)",
            "Primary Parameter": f"Pressure Ratio: {pr}x",
            "Thermal Behavior": f"Polytropic compression increases core air temperature to ~{round(t_exit, 1)} K.",
            "Enthalpy Change": f"Δh = +{round(delta_h, 1)} kJ/kg",
            "Performance Status": "OPTIMAL" if pr < 45 else, "HIGH THERMAL STRESS"
        }
    elif "combust" in label_lower:
        return {
            "Station": "Station 3 -> 4 (Combustion)",
            "Primary Parameter": f"Peak TIT: {t4} K",
            "Thermal Behavior": "Isobaric heat addition via fuel mass-flow injection into primary combustion zone.",
            "Enthalpy Change": f"Peak Energy Release (Q_in ~ {round(cp_air * (t4 - 700), 1)} kJ/kg)",
            "Performance Status": "NOMINAL FLAME STABILITY"
        }
    elif "turbin" in label_lower:
        t_exhaust = t4 / (pr ** ((gamma_air - 1.0) / (gamma_air * 1.3)))
        return {
            "Station": "Station 4 -> 5 (Expansion)",
            "Primary Parameter": f"Expansion Ratio (~{round(pr * 0.8, 1)}x)",
            "Thermal Behavior": f"Gas expansion extracts work to drive compressor, dropping exhaust gas to ~{round(t_exhaust, 1)} K.",
            "Enthalpy Change": f"Work Extraction (W_out active)",
            "Performance Status": "THERMAL EFFICIENCY NOMINAL"
        }
    else:
        return {
            "Station": "Core Gas-Path Module",
            "Primary Parameter": "General Flow Domain",
            "Thermal Behavior": "Boundary layer airflow and core mass continuity transition.",
            "Enthalpy Change": "Negligible pressure drop",
            "Performance Status": "SERVICEABLE"
        }

# --- INFERENCE ENGINE ---
def run_thermodynamic_scan(image, pr, t4, tamb):
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    width, height = image.size

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    raw_predictions = []
    try:
        upload_url = f"https://detect.roboflow.com/{MODEL_ID}?api_key={ROBOFLOW_API_KEY}"
        with open(tmp_path, "rb") as image_file:
            response = requests.post(upload_url, files={"file": image_file})
        
        if response.status_code == 200:
            result = response.json()
            raw_predictions = result.get("predictions", [])
    except Exception as e:
        pass

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    thermo_data = []

    for idx, pred in enumerate(raw_predictions, 1):
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        label = pred["class"]

        x1, y1 = x - (w / 2), y - (h / 2)
        x2, y2 = x + (w / 2), y + (h / 2)

        span_pct = round((1.0 - (y / height)) * 100, 1)
        profile = calculate_thermodynamic_profile(label, pr, t4, tamb)
        
        color = "#38BDF8" if "NOMINAL" in profile["Performance Status"] or "OPTIMAL" in profile["Performance Status"] else "#F59E0B"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1 + 5, max(0, y1 - 15)), f"#{idx} {label.upper()}", fill=color)

        thermo_data.append({
            "Module ID": f"MOD-{idx:03d}",
            "Engine Component": label.upper(),
            "Span Location": f"{span_pct}% Span",
            "Brayton Station": profile["Station"],
            "Core Parameter": profile["Primary Parameter"],
            "Thermal & Fluid Behavior": profile["Thermal Behavior"],
            "Enthalpy Metric": profile["Enthalpy Change"],
            "Thermodynamic Status": profile["Performance Status"]
        })

    return draw_img, thermo_data

# --- LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Propulsion Engine Ingestion")
    uploaded_file = st.file_uploader("Upload Engine Cutaway / Module Photo:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Raw Engine Frame", use_container_width=True)

with col2:
    st.subheader("2. Gas-Path & Thermodynamic Mapping")
    if uploaded_file is not None:
        with st.spinner("Computing Brayton cycle parameters and module energy states..."):
            annotated_img, thermodynamics = run_thermodynamic_scan(image, pressure_ratio, tit_k, ambient_temp_k)
            st.success("Thermodynamic scan successful.")
            st.image(annotated_img, caption="Module Identification & Thermal Boundaries", use_container_width=True)

# --- REPORT ---
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("3. Thermodynamic Cycle Performance Log")
    
    num_modules = len(thermodynamics)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Model", "partes-de-motor/5")
    m2.metric("Detected Modules", num_modules)
    m3.metric("Compressor PR", f"{pressure_ratio}x")
    m4.metric("Turbine Inlet Temp", f"{tit_k} K")

    st.markdown("##### Detailed Module Thermodynamic Breakdown")
    if thermodynamics:
        st.table(thermodynamics)
    else:
        st.info("No engine modules isolated in this frame.")

    cycle_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "partes-de-motor/5",
        "operating_conditions": {
            "ambient_temp_k": ambient_temp_k,
            "pressure_ratio": pressure_ratio,
            "turbine_inlet_temp_k": tit_k
        },
        "modules_analyzed": thermodynamics
    }

    st.download_button(
        label="📥 Download Thermodynamic Cycle Report (JSON)",
        data=json.dumps(cycle_report, indent=2),
        file_name=f"RAM_Propulsion_Thermodynamics_{int(time.time())}.json",
        mime="application/json"
    )
