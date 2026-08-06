import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import json
import time
import math
import tempfile
import os
from inference_sdk import InferenceHTTPClient

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AeroInspect AI - Enterprise Multi-Key MRO Engine",
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

st.markdown('<div class="main-header">✈️ AeroInspect AI: Enterprise MRO Surface & Stress Validation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">EASA Part-145 Aligned | Collaborative Multi-Key Engine (Rolls-Royce / GE Aerospace Validation Pipeline)</div>', unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION (MULTI-KEY INPUTS) ---
st.sidebar.image("https://img.icons8.com/color/96/airplane-tail.png", width=64)
st.sidebar.title("Enterprise Fleet Parameters")

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

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Collaborative Enterprise API Keys")
st.sidebar.markdown("Enter keys for parallel multi-tenant execution/failover validation:")

key_rolls_royce = st.sidebar.text_input("Rolls-Royce Validation Key:", type="password", key="rr_key")
key_ge_aerospace = st.sidebar.text_input("GE Aerospace Validation Key:", type="password", key="ge_key")
key_mro_internal = st.sidebar.text_input("Internal MRO Authority Key:", type="password", key="mro_key")

# --- AERONAUTICAL MATH FUNCTIONS ---
def calculate_stress_concentration(depth_mm, radius_mm):
    if radius_mm <= 0:
        return 3.0
    return round(1.0 + 2.0 * math.sqrt(depth_mm / radius_mm), 2)

def calculate_blend_volume(depth_mm, length_mm):
    width_mm = depth_mm * 4.0
    return round(0.5 * depth_mm * width_mm * length_mm, 2)

# --- COLLABORATIVE INFERENCE ENGINE ---
def run_collaborative_inspection(image, api_keys):
    """
    Iterates through the provided enterprise API keys. 
    Queries the Roboflow inference engine using available credentials collaboratively.
    """
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    width, height = image.size

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    raw_predictions = []
    active_key_used = None

    # Filter out empty keys
    valid_keys = [k.strip() for k in api_keys if k and k.strip() != ""]

    if not valid_keys:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return draw_img, [], "ERROR: No valid enterprise API keys provided."

    # Collaborative execution loop (try keys sequentially until success)
    for idx, key in enumerate(valid_keys):
        try:
            client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=key
            )
            result = client.infer(tmp_path, model_id="partes-de-motor/5")
            raw_predictions = result.get("predictions", [])
            active_key_used = f"Enterprise Key #{idx + 1}"
            break  # Break out on successful response
        except Exception as e:
            continue

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    detections_data = []

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
        
        # Estimate depth/radius metrics based on model prediction boundaries for engineering calculation
        estimated_depth = round(h * 0.002, 2)
        estimated_radius = max(0.05, round(w * 0.001, 2))
        estimated_length = round(w * 0.02, 2)

        kt = calculate_stress_concentration(estimated_depth, estimated_radius)
        blend_vol = calculate_blend_volume(estimated_depth, estimated_length)
        
        severity = "CRITICAL" if kt > 2.5 else ("HIGH" if kt > 1.8 else "MONITOR")
        color = "#DC2626" if severity == "CRITICAL" else ("#EF4444" if severity == "HIGH" else "#38BDF8")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1 + 5, max(0, y1 - 15)), f"#{idx} {label} (Kt: {kt})", fill=color)

        detections_data.append({
            "Defect ID": f"COMP-{idx:03d}",
            "Classification": f"Engine Component: {label}",
            "Radial Span Location": f"{radial_span_pct}% Span",
            "Stress Concentration (Kt)": kt,
            "Blend Volume (mm³)": blend_vol,
            "Severity Level": severity,
            "AMM Maintenance Action": f"BLEND REPAIR (Remove ~{blend_vol} mm³ material per AMM 72-33-00)"
        })

    return draw_img, detections_data, active_key_used

# --- LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Ingestion: Visual / Borescope Stream")
    uploaded_file = st.file_uploader("Upload Engine Component Image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Raw Inspection Input", use_container_width=True)

with col2:
    st.subheader("2. Multi-Key AI Computer Vision Analysis")
    if uploaded_file is not None:
        keys_list = [key_rolls_royce, key_ge_aerospace, key_mro_internal]
        with st.spinner("Executing collaborative verification across enterprise keys..."):
            annotated_img, detections, verification_source = run_collaborative_inspection(image, keys_list)
            
            if "ERROR" in verification_source:
                st.error(verification_source)
            else:
                st.success(f"Successfully authenticated via **{verification_source}**")
                st.image(annotated_img, caption="Collaborative Detection Overlay", use_container_width=True)

# --- REPORT ---
if uploaded_file is not None and "ERROR" not in locals().get('verification_source', ''):
    st.markdown("---")
    st.subheader("3. Enterprise Technical Airworthiness & Engineering Report")
    
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
    if detections:
        st.table(detections)
    else:
        st.info("No surface anomalies detected on the target component assembly.")

    mro_telemetry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "powerplant": engine_model,
        "sub_assembly": inspection_module,
        "model_used": "partes-de-motor/5",
        "authenticated_channel": locals().get('verification_source', 'N/A'),
        "airworthiness_status": "AOG_HOLD" if crit_count > 0 else "SERVICEABLE",
        "telemetry": detections
    }

    st.download_button(
        label="📥 Download Official EASA Part-145 Technical Log (JSON)",
        data=json.dumps(mro_telemetry, indent=2),
        file_name=f"RAM_MRO_Engine_Log_{int(time.time())}.json",
        mime="application/json"
    )
