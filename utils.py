import rasterio
import numpy as np
import os
import cv2
from PIL import Image

# =============================
# MULTIBAND LOADER (B2–B5)
# =============================
def load_multiband_from_folder(folder_path):
    selected_bands = [2, 3, 4, 5]
    band_map = {}

    for file in os.listdir(folder_path):
        for b in selected_bands:
            if f"_B{b}" in file:
                band_map[b] = os.path.join(folder_path, file)

    bands = []
    for b in sorted(band_map.keys()):
        with rasterio.open(band_map[b]) as src:
            band = src.read(1)

            # FIX 1: Consistent rotation
            band = cv2.rotate(band, cv2.ROTATE_90_CLOCKWISE)

            band = cv2.resize(band, (128, 128))
        bands.append(band)

    return np.stack(bands).astype(np.float32)


# =============================
# NDVI
# =============================
def compute_ndvi(img):
    nir = img[2]   # B4
    red = img[1]   # B3
    return (nir - red) / (nir + red + 1e-8)


# =============================
# ROBUST NORMALIZATION 
# =============================
def normalize(img):
    p2, p98 = np.percentile(img, (2, 98))   # better than min-max
    img = np.clip(img, p2, p98)
    return (img - p2) / (p98 - p2 + 1e-8)


# =============================
# PREPROCESS
# =============================
def preprocess(img):
    img = normalize(img)

    ndvi = compute_ndvi(img)
    img = np.concatenate([img, ndvi[np.newaxis, :, :]], axis=0)

    return img  # shape = (5, 128, 128)


# =============================
# RGB LOADER 
# =============================
def load_rgb(folder_path):
    band_map = {}

    for file in os.listdir(folder_path):
        if "_B2" in file: band_map["B2"] = file
        if "_B3" in file: band_map["B3"] = file
        if "_B4" in file: band_map["B4"] = file

    bands = []
    for b in ["B4", "B3", "B2"]:  # RGB order
        with rasterio.open(os.path.join(folder_path, band_map[b])) as src:
            band = src.read(1)

            # FIX 2: Apply SAME rotation as multiband
            band = cv2.rotate(band, cv2.ROTATE_90_CLOCKWISE)

            band = cv2.resize(band, (128, 128))
        bands.append(band)

    rgb = np.stack(bands, axis=-1).astype(np.float32)

    # FIX 3: Robust normalization (solves washed T2)
    rgb = normalize(rgb)

    return rgb


# =============================
# POSTPROCESS
# =============================
def postprocess(mask, threshold=0.3):
    return (mask.squeeze() > threshold).astype(np.uint8)


def calculate_change_percentage(mask):
    return (np.sum(mask) / mask.size) * 100


# =============================
# OVERLAY
# =============================
def overlay_change(rgb, mask):
    overlay = rgb.copy()
    overlay[mask == 1] = [1, 0, 0]  # red
    return overlay


# =============================
# REGION ANALYSIS
# =============================
def region_wise_analysis(mask, grid=4):
    H, W = mask.shape
    hs, ws = H // grid, W // grid

    stats = []
    for i in range(grid):
        for j in range(grid):
            region = mask[i*hs:(i+1)*hs, j*ws:(j+1)*ws]
            pct = (np.sum(region) / region.size) * 100
            stats.append({
                "region": f"R{i+1}C{j+1}",
                "change_percent": pct
            })

    return stats

# =============================
# GIF
# =============================
def save_gif(rgb1, overlay):
    img1 = Image.fromarray((rgb1 * 255).astype(np.uint8))
    img2 = Image.fromarray((overlay * 255).astype(np.uint8))

    path = "change.gif"
    img1.save(path, save_all=True, append_images=[img2], duration=500, loop=0)
    return path