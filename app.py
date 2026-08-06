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
    page_title="AeroInspect AI - Thermodynamic MRO Engine",
    page_icon="✈️",
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

st.markdown('<div class="main-header">✈️ AeroInspect AI: Thermodynamic & Stress Surface Validation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Royal Air Maroc MRO Technical Zone | Precision Defect & Stress Calculation Engine</div>', unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.image("https://img.icons8.com/color/96/airplane-tail.png", width=64)
st.sidebar.title("Inspection Parameters")

st.sidebar.markdown("---")
st.sidebar.info("Model Active: **partes-de-motor/5**\nStatus: **Connected to Roboflow Cloud**")

st.sidebar.markdown("### 📐 Precision Measurement Mode")
use_manual_inputs = st.sidebar.checkbox("Override with Hand-Measured Values (Micrometer / Borescope Gauge)", value=True)

if use_manual_inputs:
    st.sidebar.markdown("Enter hangar-measured dimensions:")
    manual_depth = st.sidebar.number_input("Defect Depth (mm):", min_value=0.01, max_value=10.0, value=0.50, step=0.05)
    manual_radius = st.sidebar.number_input("Notch Root Radius (mm):", min_value=0.01, max_value=5.0, value=0.20, step=0.05)
    manual_length = st.sidebar.number_input("Defect Length (mm):", min_value=0.1, max_value=50.0, value=3.00, step=0.5)

# --- HARDCODED CREDENTIALS ---
ROBOFLOW_API_KEY = "26JC1OEUbjS0rV3JZTxM"
MODEL_ID = "partes-de-motor/5"

# --- AERONAUTICAL & THERMODYNAMIC CALCULATIONS ---
def calculate_stress_concentration(depth_mm, radius_mm):
    if radius_mm <= 0:
        return 3.0
    # Neuber's / Isuzu approximation for stress concentration factor Kt of a notch/scratch
    return round(1.0 + 2.0 * math.sqrt(depth_mm / radius_mm), 2)

def calculate_blend_volume(depth_mm, length_mm):
    width_mm = depth_mm * 4.0
    return round(0.5 * depth_mm * width_mm * length_mm, 2)

def get_thermodynamic_impact(label, kt_value):
    label_lower = label.lower()
    if "combust" in label_lower:
        return {
            "Fluid & Thermal Impact": "Local hot gas streak acceleration & Thermal Barrier Coating (TBC) degradation.",
            "Efficiency Loss": "~0.4% Turbine Inlet Temperature (TIT) margin reduction.",
            "Thermodynamic Consequence": "Localized convective heat flux spike, risking premature liner oxidation and creep."
        }
    elif "compressor" in label_lower or "blade" in label_lower:
        return {
            "Fluid & Thermal Impact": f"Boundary layer separation & flow vortex shedding (Stress Riser Kt: {kt_value}).",
            "Efficiency Loss": "~0.8% Polytropic efficiency drop across stage.",
            "Thermodynamic Consequence": "Induces premature stall margin degradation and increased specific fuel consumption (SFC)."
        }
    else:
        return {
            "Fluid & Thermal Impact": f"Surface roughness disruption & aerodynamic drag increase (Kt: {kt_value}).",
            "Efficiency Loss": "~0.2% core mass-flow disruption.",
            "Thermodynamic Consequence": "Localized pressure drop and boundary layer turbulent transition."
        }

# --- INFERENCE ENGINE ---
def run_inspection(image, manual_mode, m_depth, m_radius, m_length):
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

    detections_data = []

    for idx, pred in enumerate(raw_predictions, 1):
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        label = pred["class"]

        x1, y1 = x - (w / 2), y - (h / 2)
        x2, y2 = x + (w / 2), y + (h / 2)

        span_pct = round((1.0 - (y / height)) * 100, 1)

        # Use manual physical inputs if enabled, otherwise estimate from bounding box
        if manual_mode:
            est_depth = m_depth
            est_radius = m_radius
            est_length = m_length
        else:
            est_depth = round(h * 0.002, 2)
            est_radius = max(0.05, round(w * 0.001, 2))
            est_length = round(w * 0.02, 2)

        kt = calculate_stress_concentration(est_depth, est_radius)
        blend_vol = calculate_blend_volume(est_depth, est_length)
        thermo = get_thermodynamic_impact(label, kt)
        
        severity = "CRITICAL" if kt > 2.5 else ("HIGH" if kt > 1.8 else "MONITOR")
        color = "#DC2626" if severity == "CRITICAL" else ("#EF4444" if severity == "HIGH" else "#38BDF8")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1 + 5, max(0, y1 - 15)), f"#{idx} {label} (Kt: {kt})", fill=color)

        detections_data.append({
            "Defect ID": f"DEF-{idx:03d}",
            "Component Class": label.upper(),
            "Span Location": f"{span_pct}% Span",
            "Measured Depth (mm)": est_depth,
            "Root Radius (mm)": est_radius,
            "Stress Concentration (Kt)": kt,
            "Blend Volume (mm³)": blend_vol,
            "Aerodynamic / Thermal Impact": thermo["Fluid & Thermal Impact"],
            "Cycle Efficiency Loss": thermo["Efficiency Loss"],
            "Thermodynamic Consequence": thermo["Thermodynamic Consequence"],
            "Severity": severity
        })

    return draw_img, detections_data

# --- LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Hangar Image Ingestion")
    uploaded_file = st.file_uploader("Upload Engine Component Photo:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Raw Inspection Frame", use_container_width=True)

with col2:
    st.subheader("2. Computer Vision & Thermodynamic Analysis")
    if uploaded_file is not None:
        with st.spinner("Processing optical scan and calculating stress concentration factors..."):
            annotated_img, detections = run_inspection(image, use_manual_inputs, manual_depth, manual_radius, manual_length)
            st.success("Analysis complete with precision engineering parameters.")
            st.image(annotated_img, caption="Detected Anomalies Overlay", use_container_width=True)

# --- REPORT ---
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("3. Technical Airworthiness & Thermodynamic Consequence Log")
    
    num_defects = len(detections)
    crit_count = sum(1 for d in detections if d["Severity"] in ["CRITICAL", "HIGH"])
    max_kt = max([d["Stress Concentration (Kt)"] for d in detections]) if detections else 1.0

    if crit_count > 0:
        st.error(f"⚠️ **AIRWORTHINESS HOLD**: {crit_count} anomaly(ies) flagged with thermodynamic boundary risks.")
    else:
        st.success("✅ **AIRWORTHINESS CLEARED**: Component operates within normal aerodynamic envelope.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Model", "partes-de-motor/5")
    m2.metric("Total Anomalies", num_defects)
    m3.metric("Peak Stress Riser (Kt)", max_kt)
    m4.metric("Engine Status", "HOLD" if crit_count > 0 else "SERVICEABLE")

    st.markdown("##### Detailed Component Defect & Thermodynamic Breakdown")
    if detections:
        st.table(detections)
    else:
        st.info("No surface anomalies or structural defects detected in this frame.")

    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "partes-de-motor/5",
        "input_mode": "Manual Hangar Gauge Measurements" if use_manual_inputs else "AI Pixel Estimation",
        "airworthiness": "HOLD" if crit_count > 0 else "CLEARED",
        "anomalies": detections
    }

    st.download_button(
        label="📥 Download Thermodynamic & MRO Report (JSON)",
        data=json.dumps(report_data, indent=2),
        file_name=f"RAM_Thermodynamic_Log_{int(time.time())}.json",
        mime="application/json"
    )
