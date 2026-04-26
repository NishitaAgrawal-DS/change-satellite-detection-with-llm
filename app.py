import streamlit as st
from utils import *
from inference import load_model, predict
from local_llm import generate_local_report
import numpy as np
import cv2
import pandas as pd

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Satellite Change Detection", layout="wide")

# =============================
# CUSTOM CSS (🔥 DESIGN)
# =============================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

.card {
    background-color: #111;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #333;
    margin-bottom: 20px;
}

.card-title {
    font-size: 16px;
    color: #8fd3ff;
    margin-bottom: 10px;
}

.section-title {
    font-size: 26px;
    font-weight: bold;
    margin-top: 30px;
}

.desc {
    color: #aaa;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SESSION STATE
# =============================
if "results" not in st.session_state:
    st.session_state.results = None

# =============================
# HEADER
# =============================
st.title("Satellite Change Detection ")

folder1 = "data/northeast_1995_11_21"
folder2 = "data/northeast_2015_11_21"

model = load_model()

# =============================
# CACHE
# =============================
@st.cache_data
def load_data():
    return (
        load_multiband_from_folder(folder1),
        load_multiband_from_folder(folder2)
    )

@st.cache_data
def load_rgb_images():
    return load_rgb(folder1), load_rgb(folder2)

# =============================
# RUN BUTTON
# =============================
if st.button("Run Analysis"):

    rgb1, rgb2 = load_rgb_images()
    img1, img2 = load_data()

    x1 = preprocess(img1)
    x2 = preprocess(img2)

    global_min = min(x1.min(), x2.min())
    global_max = max(x1.max(), x2.max())

    x1 = (x1 - global_min) / (global_max - global_min + 1e-8)
    x2 = (x2 - global_min) / (global_max - global_min + 1e-8)

    diff = np.abs(x1 - x2).mean(axis=0)

    pred = predict(model, x1, x2)

    threshold = np.percentile(pred, 60)
    mask = (pred.squeeze() > threshold).astype(np.uint8)
    mask = cv2.medianBlur(mask, 5)

    overlay = overlay_change(rgb2, mask)
    change_pct = calculate_change_percentage(mask)
    regions = region_wise_analysis(mask)

    st.session_state.results = {
        "rgb1": rgb1,
        "rgb2": rgb2,
        "diff": diff,
        "mask": mask,
        "overlay": overlay,
        "change_pct": change_pct,
        "regions": regions
    }

# =============================
# OUTPUT SECTION (🔥 DESIGNED)
# =============================
if st.session_state.results:

    data = st.session_state.results

    st.markdown("## 📊 Output Visualization")
    st.markdown("Four key outputs from the change detection pipeline")

    col1, col2 = st.columns(2)

    # -----------------------------
    # T1 vs T2
    # -----------------------------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">T1 vs T2</div>', unsafe_allow_html=True)
        st.image([data["rgb1"], data["rgb2"]], caption=["T1", "T2"])
        st.markdown('<div class="desc">Pre & Post satellite comparison</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # CHANGE MAP
    # -----------------------------
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Change Map</div>', unsafe_allow_html=True)
        st.image(data["mask"] * 255)
        st.markdown('<div class="desc">Detected change regions</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    # -----------------------------
    # OVERLAY
    # -----------------------------
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Overlay</div>', unsafe_allow_html=True)
        st.image(data["overlay"])
        st.markdown('<div class="desc">Changes over original image</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # RAW DIFFERENCE
    # -----------------------------
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Raw Difference</div>', unsafe_allow_html=True)
        st.image(data["diff"])
        st.markdown('<div class="desc">Pixel-level difference</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # =============================
    # METRICS
    # =============================
    st.markdown("### 📈 Analysis")
    st.metric("Change Percentage", f"{data['change_pct']:.2f}%")

    # =============================
    # GIF
    # =============================
    st.markdown("### 🎥 Before vs After")
    gif = save_gif(data["rgb1"], data["overlay"])
    st.image(gif)

    # =============================
    # REGION ANALYSIS
    # =============================
    st.markdown("### 📍 Region-wise Analysis")

    # Convert to DataFrame
    df = pd.DataFrame(data["regions"])

    # Extract row & column numbers
    df["Row"] = df["region"].str.extract(r"R(\d+)").astype(int)
    df["Col"] = df["region"].str.extract(r"C(\d+)").astype(int)

    # Pivot into grid
    pivot = df.pivot(index="Row", columns="Col", values="change_percent")

    # Rename for display
    pivot.index = [f"R{i}" for i in pivot.index]
    pivot.columns = [f"C{j}" for j in pivot.columns]

    # Display as table
    st.dataframe(pivot.style.format("{:.2f}%"), use_container_width=True)

    # =============================
    # AI REPORT
    # =============================
    st.markdown("### 🤖 AI Analysis")

    if st.button("Generate AI Report"):
        report = generate_local_report(
            data["change_pct"],
            data["regions"]
        )
        st.write(report)
