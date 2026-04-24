import streamlit as st
from utils import *
from inference import load_model, predict
from local_llm import generate_local_report

# =============================
# SESSION STATE INIT
# =============================
if "results" not in st.session_state:
    st.session_state.results = None

st.title("⚡ Fast Satellite Change Detection")

folder1 = "data/northeast_1995_11_21"
folder2 = "data/northeast_2015_11_21"

model = load_model()

# =============================
# CACHE LOADING
# =============================
@st.cache_data
def load_data():
    img1 = load_multiband_from_folder(folder1)
    img2 = load_multiband_from_folder(folder2)
    return img1, img2

@st.cache_data
def load_rgb_images():
    rgb1 = load_rgb(folder1)
    rgb2 = load_rgb(folder2)
    return rgb1, rgb2

# =============================
# RUN ANALYSIS BUTTON
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

    import numpy as np
    diff = np.abs(x1 - x2).mean(axis=0)

    pred = predict(model, x1, x2)
    print("Pred min:", pred.min())
    print("Pred max:", pred.max())
    print("Pred mean:", pred.mean())
    threshold = np.percentile(pred, 60)  # try 55–65 range

    mask = (pred.squeeze() > threshold).astype(np.uint8)
    import cv2
    mask = cv2.medianBlur(mask, 5)

    overlay = overlay_change(rgb2, mask)
    change_pct = calculate_change_percentage(mask)
    regions = region_wise_analysis(mask)

    # 🔥 SAVE EVERYTHING
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
# DISPLAY RESULTS (PERSISTENT)
# =============================
if st.session_state.results:

    data = st.session_state.results

    st.subheader("🟦 Before")
    st.image([data["rgb1"], data["rgb2"]], caption=["T1", "T2"])

    st.subheader("📊 Raw Difference")
    st.image(data["diff"], caption="Raw Difference")

    st.subheader("🔴 Change Detection")
    st.image(data["mask"] * 255, caption="Change Map")
    st.image(data["overlay"], caption="Overlay")

    st.metric("Change %", f"{data['change_pct']:.2f}%")

    st.subheader("🎥 Before vs After")
    gif = save_gif(data["rgb1"], data["overlay"])
    st.image(gif)

    st.subheader("📍 Region-wise Analysis")
    for r in data["regions"]:
        st.write(f"{r['region']}: {r['change_percent']:.2f}%")

# =============================
# AI REPORT BUTTON
# =============================
if st.session_state.results:
    st.subheader("🤖 AI Analysis")

    if st.button("AI Report"):
        report = generate_local_report(
            st.session_state.results["change_pct"],
            st.session_state.results["regions"]
        )
        st.write(report)