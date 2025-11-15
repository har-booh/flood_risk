#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Flood Viewer — on‑map legends (no colormap selectors)
"""
import json, os, base64
from typing import Optional, Tuple
import numpy as np
import rasterio
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_folium import st_folium
import folium
from branca.element import Element

st.set_page_config(page_title="Flood Risk Viewer", layout="wide")

def read_summary(summary_path: str) -> dict:
    with open(summary_path, "r") as f:
        return json.load(f)

def raster_bounds_latlon(path: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    with rasterio.open(path) as src:
        left, bottom, right, top = src.bounds
    return (bottom, left), (top, right)

def read_raster_array_and_stats(path: str, nodata=None, max_dim: int = 2000):
    with rasterio.open(path) as src:
        height, width = src.height, src.width
        if max(height, width) > max_dim:
            scale = max_dim / float(max(height, width))
            out_h = max(1, int(round(height * scale)))
            out_w = max(1, int(round(width * scale)))
            arr = src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype("float32")
        else:
            arr = src.read(1).astype("float32")
        if nodata is None and src.nodata is not None:
            nodata = src.nodata
    mask = np.zeros_like(arr, dtype=bool)
    if nodata is not None:
        mask |= np.isclose(arr, nodata)
    mask |= ~np.isfinite(arr)
    valid = arr[~mask]
    if valid.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(valid, 2.0))
        vmax = float(np.percentile(valid, 98.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = (float(valid.min()), float(valid.max())) if valid.size else (0.0, 1.0)
            if vmax <= vmin:
                vmin, vmax = 0.0, 1.0
    return arr, mask, vmin, vmax

def raster_to_rgba_image(path: str, cmap_name: str,
                         vmin: Optional[float] = None, vmax: Optional[float] = None,
                         nodata=None, max_dim: int = 2000):
    arr, mask, v_auto_min, v_auto_max = read_raster_array_and_stats(path, nodata=nodata, max_dim=max_dim)
    vmin = vmin if vmin is not None else v_auto_min
    vmax = vmax if vmax is not None else v_auto_max
    normed = (arr - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(arr, dtype="float32")
    normed = np.clip(normed, 0, 1)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(normed)
    rgba[mask, 3] = 0.0
    return (rgba * 255).astype("uint8"), (vmin, vmax)

def add_image_overlay(m, img_rgba: np.ndarray, bounds, name: str, opacity: float = 0.7):
    overlay = folium.raster_layers.ImageOverlay(
        image=img_rgba,
        bounds=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]],
        opacity=opacity,
        name=name,
        interactive=True,
        cross_origin=False,
        zindex=1,
        alt=name
    )
    overlay.add_to(m)

def make_continuous_legend_png(cmap_name: str, vmin: float, vmax: float, title: str, width_px=260) -> bytes:
    fig, ax = plt.subplots(figsize=(3.2, 1.0), dpi=200)
    fig.subplots_adjust(bottom=0.35, top=0.85, left=0.08, right=0.98)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    ax.set_title(title, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    bio.seek(0); img = Image.open(bio).convert("RGBA")
    w, h = img.size; new_h = int(h * (width_px / float(w)))
    img = img.resize((width_px, new_h), Image.LANCZOS)
    bio2 = BytesIO(); img.save(bio2, format="PNG")
    return bio2.getvalue()

LULC_CLASSES = [(1,"Water"),(2,"Urban / Impervious"),(3,"Vegetation"),(4,"Agriculture"),(5,"Wetland"),(6,"Other")]

def make_lulc_legend_png(width_px=240) -> bytes:
    palette = matplotlib.colormaps.get_cmap("tab10")
    row_h = 24
    img_h = row_h * len(LULC_CLASSES) + 16
    img = Image.new("RGBA", (width_px, img_h), (255, 255, 255, 220))
    draw = ImageDraw.Draw(img)
    x0, y = 10, 8; sw = 18
    for code, label in LULC_CLASSES:
        color = tuple(int(c * 255) for c in palette((code - 1) % 10)[:3]) + (255,)
        draw.rectangle([x0, y + 3, x0 + sw, y + 3 + sw], fill=color, outline=(40,40,40,255))
        draw.text((x0 + sw + 10, y + 2), f"{code} — {label}", fill=(10,10,10,255))
        y += row_h
    bio = BytesIO(); img.save(bio, format="PNG"); return bio.getvalue()

def add_onmap_legend(map_obj, img_bytes: bytes, position: str = "bottomright", zindex: int = 1000, width_px: int = 200):
    b64 = base64.b64encode(img_bytes).decode("ascii")
    style_by_pos = {
        "bottomright": "position:absolute; bottom:10px; right:10px;",
        "bottomleft":  "position:absolute; bottom:10px; left:10px;",
        "topright":    "position:absolute; top:10px; right:10px;",
        "topleft":     "position:absolute; top:10px; left:10px;",
    }
    style = style_by_pos.get(position, style_by_pos["bottomright"])
    html = f'<div style="{style} z-index:{zindex}; background: rgba(255,255,255,0.8); padding:6px; border-radius:6px; box-shadow: 0 1px 4px rgba(0,0,0,0.25);"><img src="data:image/png;base64,{b64}" style="width:{width_px}px; height:auto;" /></div>'
    map_obj.get_root().html.add_child(Element(html))

# ---------------- Sidebar ----------------

st.sidebar.title("Layers")
summary_path = st.sidebar.text_input("Summary JSON path", value="data/rasters/prepared_layers_summary.json", key="summary_path_input")
if not os.path.exists(summary_path):
    st.error("Summary JSON not found. Run the fetch step first.")
    st.stop()

meta = read_summary(summary_path)
paths = meta["outputs"]

# Fixed cmaps
CMAPS = {
    "flood_risk_0to1": "viridis",
    "dist_to_river_m": "magma",
    "drainage_density_km_per_km2": "plasma",
    "soil_sand_pct": "cividis",
    "lulc_osm_proxy": "tab10",
}

# Toggles + opacity
show_risk = st.sidebar.checkbox("Flood risk (0–1)", value=True, key="toggle_risk")
risk_opacity = st.sidebar.slider("Risk opacity", 0.0, 1.0, 0.75, 0.05, key="opacity_risk")

st.sidebar.markdown("---")
show_dist = st.sidebar.checkbox("Distance to river (m)", value=False, key="toggle_dist")
dist_opacity = st.sidebar.slider("Distance opacity", 0.0, 1.0, 0.6, 0.05, key="opacity_dist")

st.sidebar.markdown("---")
show_dd = st.sidebar.checkbox("Drainage density (km/km²)", value=False, key="toggle_dd")
dd_opacity = st.sidebar.slider("Drainage density opacity", 0.0, 1.0, 0.6, 0.05, key="opacity_dd")

st.sidebar.markdown("---")
show_soil = st.sidebar.checkbox("Soil sand fraction (%)", value=False, key="toggle_soil")
soil_opacity = st.sidebar.slider("Soil sand opacity", 0.0, 1.0, 0.6, 0.05, key="opacity_soil")

st.sidebar.markdown("---")
show_lulc = st.sidebar.checkbox("LULC (OSM proxy)", value=False, key="toggle_lulc")
lulc_opacity = st.sidebar.slider("LULC opacity", 0.0, 1.0, 0.6, 0.05, key="opacity_lulc")

# --------------- Map Build ----------------

bbox = meta["bbox"]
west, south, east, north = bbox
center_lat = (south + north) / 2.0
center_lon = (west + east) / 2.0
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

# Legend positions cycle
legend_positions = ["bottomright", "bottomleft", "topright", "topleft"]
legend_idx = 0
def next_pos():
    nonlocal_vars = globals().setdefault("_legend_state", {"i":0})
    pos = legend_positions[nonlocal_vars["i"] % len(legend_positions)]
    nonlocal_vars["i"] += 1
    return pos

# Overlays + legends
if show_risk and "flood_risk_0to1" in paths and os.path.exists(paths["flood_risk_0to1"]):
    p = paths["flood_risk_0to1"]
    img, (rvmin, rvmax) = raster_to_rgba_image(p, cmap_name=CMAPS["flood_risk_0to1"])
    add_image_overlay(m, img, raster_bounds_latlon(p), "Flood risk (0–1)", opacity=risk_opacity)
    add_onmap_legend(m, make_continuous_legend_png(CMAPS["flood_risk_0to1"], rvmin, rvmax, "Flood risk (0–1)"), position=next_pos())

if show_dist and "dist_to_river_m" in paths and os.path.exists(paths["dist_to_river_m"]):
    p = paths["dist_to_river_m"]
    img, (dvmin, dvmax) = raster_to_rgba_image(p, cmap_name=CMAPS["dist_to_river_m"])
    add_image_overlay(m, img, raster_bounds_latlon(p), "Distance to river (m)", opacity=dist_opacity)
    add_onmap_legend(m, make_continuous_legend_png(CMAPS["dist_to_river_m"], dvmin, dvmax, "Distance to river (m)"), position=next_pos())

if show_dd and "drainage_density_km_per_km2" in paths and os.path.exists(paths["drainage_density_km_per_km2"]):
    p = paths["drainage_density_km_per_km2"]
    img, (ddvmin, ddvmax) = raster_to_rgba_image(p, cmap_name=CMAPS["drainage_density_km_per_km2"])
    add_image_overlay(m, img, raster_bounds_latlon(p), "Drainage density (km/km²)", opacity=dd_opacity)
    add_onmap_legend(m, make_continuous_legend_png(CMAPS["drainage_density_km_per_km2"], ddvmin, ddvmax, "Drainage density (km/km²)"), position=next_pos())

if show_soil and "soil_sand_pct" in paths and os.path.exists(paths["soil_sand_pct"]):
    p = paths["soil_sand_pct"]
    img, (svmin, svmax) = raster_to_rgba_image(p, cmap_name=CMAPS["soil_sand_pct"])
    add_image_overlay(m, img, raster_bounds_latlon(p), "Soil sand fraction (%)", opacity=soil_opacity)
    add_onmap_legend(m, make_continuous_legend_png(CMAPS["soil_sand_pct"], svmin, svmax, "Soil sand fraction (%)"), position=next_pos())

if show_lulc and "lulc_osm_proxy" in paths and os.path.exists(paths["lulc_osm_proxy"]):
    p = paths["lulc_osm_proxy"]
    img, _ = raster_to_rgba_image(p, cmap_name=CMAPS["lulc_osm_proxy"])
    add_image_overlay(m, img, raster_bounds_latlon(p), "LULC (OSM proxy)", opacity=lulc_opacity)
    add_onmap_legend(m, make_lulc_legend_png(), position=next_pos())

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, use_container_width=True, returned_objects=[])

with st.expander("AOI details"):
    st.write({"aoi_place": meta.get("aoi_place", "<unknown>"), "bbox": bbox})
