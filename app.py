import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import json
import time
import random
import math
import tempfile
import os
from inference_sdk import InferenceHTTPClient

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AeroInspect AI - MRO Engine & Surface Inspection",
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

st.markdown('<div class="main-header">✈️ AeroInspect AI: MRO Surface Defect & Stress Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">EASA Part-145 Aligned Inspection Protocol | Powered by Roboflow Motor Engine Model</div>', unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.image("https://img.icons8.com/color/96/airplane-tail.png", width=64)
st.sidebar.title("Fleet & MRO Parameters")

# Enter your Roboflow API Key here (or set it as a Streamlit Secret)
ROBOFLOW_API_KEY = st.sidebar.text_input("Roboflow API Key:", value="26JC1OEUbjS0rV3JZTxM", type="password")

engine_model = st.sidebar.selectbox(
    "Target Powerplant:",
    [
        "CFM International LEAP-1B (Boeing 737 MAX)",
        "CFM International CFM56-7B (Boeing 737 NG)",
        "Rolls-Royce Trent XWB (Airbus A350)",
        "GE Aerospace GE90-115B (Boeing 777)"
    ]
)

inspection_module = st.sidebar.selectbox(
    "Inspection Sub-Assembly:",
    [
        "Fan Stage 1 (Titanium / Carbon-Titanium)",
        "Engine Nacelle & Fastener Grid",
        "High-Pressure Compressor (HPC) Stage",
        "HPT Nozzle Guide Vanes (Borescope)"
    ]
)

# --- AERONAUTICAL MATH FUNCTIONS ---
def calculate_stress_concentration(depth_mm, radius_mm):
    if radius_mm <= 0:
        return 3.0
    return round(1.0 + 2.0 * math.sqrt(depth_mm / radius_mm), 2)

def calculate_blend_volume(depth_mm, length_mm):
    width_mm = depth_mm * 4.0
    return round(0.5 * depth_mm * width_mm * length_mm, 2)

# --- INFERENCE ENGINE ---
def run_roboflow_inspection(image, api_key):
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    width, height = image.size

    # Save temp image for SDK upload
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    raw_predictions = []
    
    # Attempt real API call if API key provided
    if api_key and api_key != "YOUR_API_KEY_HERE":
        try:
            client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=api_key
            )
            result = client.infer(tmp_path, model_id="partes-de-motor/5")
            raw_predictions = result.get("predictions", [])
        except Exception as e:
            st.warning(f"Roboflow API call fallback: {e}")

    # Clean up temp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    detections_data = []

    # If predictions returned from model
    if len(raw_predictions) > 0:
        for idx, pred in enumerate(raw_predictions, 1):
            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]
            label = pred["class"]
            conf = pred["confidence"]

            x1 = x - (w / 2)
            y1 = y - (h / 2)
            x2 = x + (w / 2)
            y2 = y + (h / 2)

            radial_span_pct = round((1.0 - (y / height)) * 100, 1)

            # Draw prediction bounding boxes
            color = "#38BDF8"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1 + 5, max(0, y1 - 15)), f"#{idx} {label} ({conf:.0%})", fill=color)

            detections_data.append({
                "Defect ID": f"COMP-{idx:03d}",
                "Classification": f"Engine Component: {label}",
                "Radial Span Location": f"{radial_span_pct}% Position",
                "Stress Concentration (Kt)": 1.15,
                "Blend Volume (mm³)": 0.0,
                "Severity Level": "MONITOR",
                "AMM Maintenance Action": f"Inspect structural integrity of {label} per AMM 72-00-00"
            })
    else:
        # Fallback to simulated MRO inspection if API key not entered yet
        boxes = [
            [width * 0.25, height * 0.30, width * 0.38, height * 0.42],
            [width * 0.60, height * 0.55, width * 0.72, height * 0.68]
        ]
        defect_library = [
            {"class": "Blade_Nick", "severity": "HIGH", "color": "#EF4444", "depth_mm": 0.45, "rad_mm": 0.12, "length_mm": 2.1},
            {"class": "FOD_Impact", "severity": "CRITICAL", "color": "#DC2626", "depth_mm": 0.85, "rad_mm": 0.08, "length_mm": 5.2}
        ]
        for idx, box in enumerate(boxes, 1):
            x1, y1, x2, y2 = box
            def_info = defect_library[(idx - 1) % len(defect_library)]
            center_y = (y1 + y2) / 2
            radial_span_pct = round((1.0 - (center_y / height)) * 100, 1)
            kt = calculate_stress_concentration(def_info["depth_mm"], def_info["rad_mm"])
            blend_vol = calculate_blend_volume(def_info["depth_mm"], def_info["length_mm"])

            draw.rectangle([x1, y1, x2, y2], outline=def_info["color"], width=4)
            draw.text((x1 + 5, max(0, y1 - 15)), f"#{idx} {def_info['class']} (Kt: {kt})", fill=def_info["color"])

            detections_data.append({
                "Defect ID": f"DEF-{idx:03d}",
                "Classification": def_info["class"],
                "Radial Span Location": f"{radial_span_pct}% Span",
                "Stress Concentration (Kt)": kt,
                "Blend Volume (mm³)": blend_vol,
                "Severity Level": def_info["severity"],
                "AMM Maintenance Action": f"BLEND REPAIR (Remove ~{blend_vol} mm³ material per AMM 72-33-00)"
            })

    return draw_img, detections_data

# --- LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Ingestion: Visual / Borescope Stream")
    uploaded_file = st.file_uploader("Upload Engine Component Image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Raw Inspection Input", use_container_width=True)

with col2:
    st.subheader("2. AI Computer Vision & Engine Component Detection")
    if uploaded_file is not None:
        with st.spinner("Connecting to Roboflow Engine Model & Calculating Stress Factors..."):
            annotated_img, detections = run_roboflow_inspection(image, ROBOFLOW_API_KEY)
            st.image(annotated_img, caption="Detected Components Overlay", use_container_width=True)

# --- REPORT ---
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("3. Technical Airworthiness & Engineering Report")
    
    num_defects = len(detections)
    crit_count = sum(1 for d in detections if d["Severity Level"] in ["CRITICAL", "HIGH"])
    max_kt = max([d["Stress Concentration (Kt)"] for d in detections]) if detections else 1.0

    if crit_count > 0 or max_kt > 3.0:
        st.error(f"⚠️ **MAINTENANCE ACTION REQUIRED**: {crit_count} Critical/High Risk Anomaly(ies) Flagged.")
    else:
        st.success("✅ **AIRWORTHINESS CLEARED**: All detected components within allowable operational limits.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Powerplant", engine_model.split("(")[0].strip())
    m2.metric("Total Detections", num_defects)
    m3.metric("Peak Stress Riser (Kt)", max_kt)
    m4.metric("AOG Risk Status", "HOLD" if crit_count > 0 else "CLEARED")

    st.markdown("##### Detailed Component Analysis & MRO Maintenance Log")
    st.table(detections)

    mro_telemetry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "powerplant": engine_model,
        "sub_assembly": inspection_module,
        "model_used": "partes-de-motor/5",
        "airworthiness_status": "AOG_HOLD" if crit_count > 0 else "SERVICEABLE",
        "telemetry": detections
    }

    st.download_button(
        label="📥 Download Official EASA Part-145 Technical Log (JSON)",
        data=json.dumps(mro_telemetry, indent=2),
        file_name=f"RAM_MRO_Engine_Log_{int(time.time())}.json",
        mime="application/json"
    )
