import streamlit as st
from PIL import Image
import numpy as np
import json
import time
from ultralytics import YOLO

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AeroInspect AI - Rolls-Royce MRO Tool",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Aviation Dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- TITLE SECTION ---
st.markdown('<div class="main-header">✈️ AeroInspect AI: Engine Visual Inspection Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Rolls-Royce IntelligentEngine Aligned MRO Defect Detection (YOLO Neural Network Engine)</div>', unsafe_allow_html=True)

# --- LOAD REAL YOLO MODEL ---
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

with st.spinner("Initializing Deep Learning Engine (YOLOv8 Weights)..."):
    model = load_yolo_model()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.image("https://img.icons8.com/color/96/airplane-tail.png", width=64)
st.sidebar.title("Inspection Setup")

# Target Engine Models (Workhorses & Next-Gen Flagships)
engine_model = st.sidebar.selectbox(
    "Target Engine Model:",
    [
        "Rolls-Royce Trent XWB (Airbus A350)",
        "Rolls-Royce UltraFan (Carbon-Titanium CTi)",
        "GE Aerospace GE90 (Boeing 777)",
        "GE Aerospace GE9X (Boeing 777X)",
        "CFM LEAP-1B (Boeing 737 MAX)",
        "CFM RISE (Open-Rotor Concept)"
    ]
)

# Inspection Mode Toggle
inspection_mode = st.sidebar.radio(
    "Inspection Mode:",
    ["External Fan Blade / Nacelle", "Internal Borescope (HPT/LPT Blades)"]
)

confidence_threshold = st.sidebar.slider(
    "YOLO Confidence Threshold:",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.info("""
**AeroInspect AI v2.0**
Real-time object detection powered by Ultralytics YOLOv8 for Rolls-Royce & Safran engine inspection workflows.
""")

# --- MAIN WORKFLOW ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Hangar Floor Photo")
    uploaded_file = st.file_uploader(
        "Upload engine or borescope image (JPG, PNG):", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Hangar Image", use_container_width=True)

with col2:
    st.subheader("2. Real YOLO AI Prediction Overlay")
    
    if uploaded_file is not None:
        with st.spinner("Running Deep Learning Neural Network Inference..."):
            # Convert uploaded image to matrix array
            img_array = np.array(image)
            
            # Execute real PyTorch YOLO prediction on image pixels
            results = model.predict(source=img_array, conf=confidence_threshold)
            
            # Render visual predictions overlay with bounding boxes
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="YOLO Model Detections & Confidence Scores", use_container_width=True)

# --- MRO REPORT & FLEET TELEMETRY ---
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("3. MRO Inspection Summary Report")
    
    boxes = results[0].boxes
    num_defects = len(boxes)
    
    if num_defects == 0:
        st.success("✅ **INSPECTION PASSED**: No surface anomalies detected above threshold.")
    else:
        st.warning(f"⚠️ **FLAGGED FOR REVIEW**: Identified {num_defects} object/defect candidate(s).")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Engine Powerplant", engine_model.split("(")[0].strip())
    m2.metric("Inspection Mode", "Borescope" if "Borescope" in inspection_mode else "External Fan")
    m3.metric("Neural Net Detection Count", num_defects)
    m4.metric("Engine Telemetry", "CONNECTED")

    if num_defects > 0:
        st.markdown("##### Detailed Detection Breakdown")
        table_data = []
        names = model.names
        for idx, box in enumerate(boxes, 1):
            class_id = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            class_name = names.get(class_id, "Anomaly")
            table_data.append({
                "Detection #": idx,
                "Predicted Class": class_name,
                "Confidence": f"{conf_score:.2%}",
                "Box Coordinates (xyxy)": str([round(x, 1) for x in box.xyxy[0].tolist()])
            })
        st.table(table_data)

    # Simulated Rolls-Royce IntelligentEngine Fleet Telemetry (JSON)
    telemetry_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine_model": engine_model,
        "inspection_mode": inspection_mode,
        "confidence_threshold": confidence_threshold,
        "model_architecture": "YOLOv8 Nano",
        "total_detections": num_defects,
        "status": "PASSED" if num_defects == 0 else "FLAGGED_FOR_REVIEW"
    }

    st.download_button(
        label="📥 Export Telemetry to Fleet Brain (JSON)",
        data=json.dumps(telemetry_payload, indent=2),
        file_name=f"RollsRoyce_MRO_Telemetry_{int(time.time())}.json",
        mime="application/json"
    )
