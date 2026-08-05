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
    page_title="AeroInspect - MRO Defect Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for clean metrics and dark contrast
st.markdown("""
<style>
    .main-header { font-size: 2.0rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.2rem; }
    .sub-header { font-size: 0.95rem; color: #94A3B8; margin-bottom: 1.5rem; }
    .stMetric {
        background-color: #1E293B !important;
        padding: 12px !important;
        border-radius: 6px !important;
        border: 1px solid #334155 !important;
    }
    div[data-testid="stMetricValue"] { color: #38BDF8 !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { color: #94A3B8 !important; }
</style>
""", unsafe_allow_html=True)

# Clean, professional header
st.markdown('<div class="main-header">✈️ AeroInspect: Engine Component Inspection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated defect evaluation & repair volume estimation | Model: partes-de-motor/5</div>', unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("Inspection Parameters")

# Retrieve API key securely from Streamlit Secrets (fall back to your key silently)
ROBOFLOW_API_KEY = st.secrets.get("ROBOFLOW_API_KEY", "26JC1OEUbjS0rV3JZTxM")

# Dynamic Powerplant Selection
engine_options = [
    "CFM International LEAP-1B",
    "CFM International CFM56-7B",
    "Rolls-Royce Trent XWB",
    "GE Aerospace GE90-115B",
    "Pratt & Whitney PW1100G",
    "Other / Custom Engine"
]
selected_engine = st.sidebar.selectbox("Target Engine:", engine_options)

if selected_engine == "Other / Custom Engine":
    engine_model = st.sidebar.text_input("Enter Engine Model:", value="Custom Powerplant")
else:
    engine_model = selected_engine

# Dynamic Sub-Assembly Selection
module_options = [
    "Fan & Front Frame",
    "Compressor Section (HPC/LPC)",
    "Combustion Chamber",
    "Turbine Section (HPT/LPT)",
    "Accessory Drive Gearbox",
    "Other / Custom Zone"
]
selected_module = st.sidebar.selectbox("Inspection Zone:", module_options)

if selected_module == "Other / Custom Zone":
    inspection_module = st.sidebar.text_input("Enter Inspection Zone:", value="Custom Sub-Assembly")
else:
    inspection_module = selected_module

confidence_thresh = st.sidebar.slider(
    "Confidence Threshold:",
    min_value=0.10, max_value=1.00, value=0.35, step=0.05
)

# --- MATH ENGINE ---
def calculate_stress_concentration(depth_mm, radius_mm):
    if radius_mm <= 0:
        return 3.0
    return round(1.0 + 2.0 * math.sqrt(depth_mm / radius_mm), 2)

def calculate_blend_volume(depth_mm, length_mm):
    width_mm = depth_mm * 4.0
    return round(0.5 * depth_mm * width_mm * length_mm, 2)

# --- INFERENCE PIPELINE ---
def run_roboflow_inspection(image, api_key, thresh):
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    width, height = image.size

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    raw_predictions = []
    api_success = False

    if api_key:
        try:
            client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=api_key
            )
            result = client.infer(tmp_path, model_id="partes-de-motor/5")
            raw_preds = result.get("predictions", [])
            raw_predictions = [p for p in raw_preds if p.get("confidence", 0) >= thresh]
            api_success = True
        except Exception as e:
            st.warning(f"Could not connect to Roboflow API: {e}")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    detections_data = []

    if api_success and len(raw_predictions) > 0:
        for idx, pred in enumerate(raw_predictions, 1):
            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]
            label = pred["class"]
            conf = pred["confidence"]

            x1 = max(0, x - (w / 2))
            y1 = max(0, y - (h / 2))
            x2 = min(width, x + (w / 2))
            y2 = min
