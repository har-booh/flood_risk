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
from flood_demo_modular_stable import run as recompute_flood_risk
from branca.element import Element
import pandas as pd
import requests
import hashlib
import time
from pathlib import Path

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

# Contact email used for Nominatim per their usage policy
NOMINATIM_CONTACT_EMAIL = "axumaicollective@gmail.com"

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

def make_continuous_legend_png(cmap_name: str, vmin: float, vmax: float, title: str, width_px=360) -> bytes:
    fig, ax = plt.subplots(figsize=(5.0, 1.4), dpi=200)
    fig.subplots_adjust(bottom=0.35, top=0.85, left=0.08, right=0.98)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    ax.set_title(title, fontsize=10, fontweight='bold')
    cb.ax.tick_params(labelsize=9)
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    bio.seek(0); img = Image.open(bio).convert("RGBA")
    w, h = img.size; new_h = int(h * (width_px / float(w)))
    img = img.resize((width_px, new_h), Image.LANCZOS)
    bio2 = BytesIO(); img.save(bio2, format="PNG")
    return bio2.getvalue()

LULC_CLASSES = [(1,"Water"),(2,"Urban / Impervious"),(3,"Vegetation"),(4,"Agriculture"),(5,"Wetland"),(6,"Other")]

def make_lulc_legend_png(width_px=340) -> bytes:
    palette = matplotlib.colormaps.get_cmap("tab10")
    row_h = 36
    img_h = row_h * len(LULC_CLASSES) + 24
    img = Image.new("RGBA", (width_px, img_h), (255, 255, 255, 220))
    draw = ImageDraw.Draw(img)
    x0, y = 12, 12; sw = 28
    for code, label in LULC_CLASSES:
        color = tuple(int(c * 255) for c in palette((code - 1) % 10)[:3]) + (255,)
        draw.rectangle([x0, y + 4, x0 + sw, y + 4 + sw], fill=color, outline=(40,40,40,255))
        draw.text((x0 + sw + 14, y + 5), f"{code} — {label}", fill=(10,10,10,255), font=None)
        y += row_h
    bio = BytesIO(); img.save(bio, format="PNG"); return bio.getvalue()

def add_onmap_legend(map_obj, img_bytes: bytes, position: str = "bottomright", zindex: int = 1000, width_px: int = 300):
    b64 = base64.b64encode(img_bytes).decode("ascii")
    style_by_pos = {
        "bottomright": "position:absolute; bottom:15px; right:15px;",
        "bottomleft":  "position:absolute; bottom:15px; left:15px;",
        "topright":    "position:absolute; top:15px; right:15px;",
        "topleft":     "position:absolute; top:15px; left:15px;",
    }
    style = style_by_pos.get(position, style_by_pos["bottomright"])
    html = f'<div style="{style} z-index:{zindex}; background: rgba(255,255,255,0.85); padding:10px; border-radius:8px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);"><img src="data:image/png;base64,{b64}" style="width:{width_px}px; height:auto;" /></div>'
    map_obj.get_root().html.add_child(Element(html))

# ---- Location search helpers (lat/lon parse, geocode, nearest-grid, raster sampler)

_GRID_CACHE = {}
_KD = None
_GRID_COORDS = None

def parse_latlon(text: str):
    if not isinstance(text, str):
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except Exception:
        return None
    return None

def load_grid_cells(path: str = "data/grid_cells.csv"):
    global _GRID_CACHE, _KD, _GRID_COORDS
    if path in _GRID_CACHE:
        return _GRID_CACHE[path]
    p = Path(path)
    if not p.exists():
        _GRID_CACHE[path] = None
        return None
    df = pd.read_csv(p)
    coords = np.vstack([df['lat'].values, df['lon'].values]).T
    _GRID_COORDS = coords
    if KDTree is not None:
        try:
            _KD = KDTree(coords)
        except Exception:
            _KD = None
    else:
        _KD = None
    _GRID_CACHE[path] = (df, _KD, coords)
    return _GRID_CACHE[path]

def nearest_grid_point(lat: float, lon: float, grid_path: str = "data/grid_cells.csv"):
    loaded = load_grid_cells(grid_path)
    if not loaded:
        return None
    df, kd, coords = loaded
    if kd is not None:
        dist, idx = kd.query([lat, lon], k=1)
        row = df.iloc[int(idx)].to_dict()
        row['__nn_dist_deg'] = float(dist)
        return row
    # fallback brute-force (euclidean degrees)
    d = np.sqrt((coords[:,0] - lat)**2 + (coords[:,1] - lon)**2)
    idx = int(np.argmin(d))
    row = df.iloc[idx].to_dict()
    row['__nn_dist_deg'] = float(d[idx])
    return row

def geocode_nominatim(query: str, limit: int = 6, user_agent_email: str = None):
    q = query.strip()
    if q == "":
        return []
    if user_agent_email is None:
        user_agent_email = NOMINATIM_CONTACT_EMAIL
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.sha256(q.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"geocode_{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            try:
                cache_file.unlink()
            except Exception:
                pass
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        "User-Agent": f"flood-risk-viewer ({user_agent_email})",
        "Accept": "application/json",
        "Referer": "http://localhost/",
    }
    params = {"q": q, "format": "json", "limit": limit}
    try:
        time.sleep(0.5)
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        # If blocked or forbidden, attempt a gentler fallback UA to see if it's a UA-block
        if resp.status_code == 403 or (isinstance(resp.text, str) and 'Access blocked' in resp.text):
            # fallback to a permissive curl UA (not ideal long-term — set a real contact email to avoid blocks)
            headers_fb = {**headers, 'User-Agent': 'curl/8.7.1'}
            try:
                resp = requests.get(url, headers=headers_fb, params=params, timeout=10)
            except Exception:
                pass
        resp.raise_for_status()
        data = resp.json()
        # normalize lat/lon to floats and small dict keys
        out = []
        for r in data:
            try:
                lat = float(r.get('lat'))
                lon = float(r.get('lon'))
                display = r.get('display_name') or r.get('name') or q
                out.append({'name': display, 'lat': lat, 'lon': lon, 'raw': r})
            except Exception:
                continue
        try:
            cache_file.write_text(json.dumps(out))
        except Exception:
            pass
        return out
    except Exception:
        return []

def sample_raster_at_point(path: str, lat: float, lon: float):
    p = Path(path)
    if not p.exists():
        return None
    try:
        with rasterio.open(str(p)) as src:
            try:
                row, col = src.index(lon, lat)
                arr = src.read(1)
                if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                    val = float(arr[row, col])
                    if src.nodata is not None and np.isclose(val, src.nodata):
                        return None
                    if not np.isfinite(val):
                        return None
                    return val
            except Exception:
                # try sampling as fallback
                for val in rasterio.sample.sample_gen(src, [(lon, lat)]):
                    v = val[0]
                    if np.isfinite(v):
                        return float(v)
    except Exception:
        return None
    return None

# ---------------- Sidebar ----------------

# ---------------- Location search (top of sidebar)
st.sidebar.subheader("Location search")
st.sidebar.write("Enter an address/place name or `lat, lon` (decimal degrees).")
# define the search action before widgets reference it
def _do_loc_search():
    q = st.session_state.get('loc_search_input', '').strip()
    if not q:
        st.session_state['loc_message'] = 'Enter a search query.'
        return
    parsed = parse_latlon(q)
    if parsed:
        lat, lon = parsed
        st.session_state['selected_location'] = {'name': f'{lat:.6f}, {lon:.6f}', 'lat': lat, 'lon': lon}
        st.session_state['loc_candidates'] = []
        st.session_state['loc_message'] = f'Parsed coordinates: {lat:.6f}, {lon:.6f}'
        return
    if st.session_state.get('loc_use_geocode', True):
        res = geocode_nominatim(q)
        st.session_state['loc_candidates'] = res
        if res:
            st.session_state['selected_location'] = {'name': res[0]['name'], 'lat': res[0]['lat'], 'lon': res[0]['lon']}
            st.session_state['loc_message'] = f'Found {len(res)} candidate(s). Selected the first.'
        else:
            st.session_state['loc_message'] = 'No geocoding results.'
    else:
        st.session_state['loc_message'] = 'Geocoding disabled and input is not lat,lon.'
# text input (user types, then clicks Search)
loc_query = st.sidebar.text_input('Search by address or "lat, lon"', placeholder='e.g., Lagos or 6.4285, 3.4795', key='loc_search_input')
use_geocode = st.sidebar.checkbox('Use geocoding (Nominatim)', value=True, key='loc_use_geocode')
st.sidebar.button('Search', on_click=_do_loc_search, key='loc_btn_go')

if st.session_state.get('loc_candidates'):
    candidates = st.session_state['loc_candidates']
    labels = [f"{c['name']} ({c['lat']:.6f}, {c['lon']:.6f})" for c in candidates]
    sel_idx = st.sidebar.selectbox('Choose candidate', list(range(len(labels))), format_func=lambda i: labels[i], key='loc_candidate_sel')
    sel = candidates[sel_idx]
    st.session_state['selected_location'] = {'name': sel['name'], 'lat': sel['lat'], 'lon': sel['lon']}

if 'loc_message' in st.session_state:
    st.sidebar.write(st.session_state.get('loc_message'))

st.sidebar.divider()

# ---------------- Layer controls... ----------------
st.sidebar.title("Layers")
summary_path = st.sidebar.text_input("Summary JSON path", value="data/rasters/prepared_layers_summary.json", key="summary_path_input")
if not os.path.exists(summary_path):
    st.error("Summary JSON not found. Run the fetch step first.")
    st.stop()

meta = read_summary(summary_path)
paths = meta["outputs"]

# Ensure flood_risk_0to1 is in outputs (fallback if not added by flood_demo_modular_stable.py yet)
summary_dir = os.path.dirname(os.path.abspath(summary_path))
flood_risk_path = os.path.join(summary_dir, "flood_risk_0to1.tif")
if "flood_risk_0to1" not in paths and os.path.exists(flood_risk_path):
    paths["flood_risk_0to1"] = flood_risk_path

# Fixed cmaps
CMAPS = {
    "flood_risk_0to1": "viridis",
    "dist_to_river_m": "magma",
    "drainage_density_km_per_km2": "plasma",
    "soil_sand_pct": "cividis",
    "lulc_worldcover_proxy": "tab10",
}

# Controls: toggles + WEIGHTS (replaces per-layer opacity sliders)
st.sidebar.subheader("Flood index weights (used to recompute flood_risk_0to1)")
normalize_weights = st.sidebar.checkbox("Normalize weights to sum to 1", value=True, key="normalize_weights")

# defaults: prefer weights stored in summary, else fall back
w_defaults = {"dist": 0.35, "drainage": 0.25, "soil": 0.20, "lulc": 0.20}
w_used = meta.get("flood_risk_weights_used", {})
for k in w_defaults:
    if k in w_used:
        try:
            w_defaults[k] = float(w_used[k])
        except Exception:
            pass

w_dist = st.sidebar.slider("Weight: distance to waterways", 0.0, 1.0, float(w_defaults["dist"]), 0.01, key="w_dist")
w_dd   = st.sidebar.slider("Weight: drainage density",      0.0, 1.0, float(w_defaults["drainage"]), 0.01, key="w_dd")
w_soil = st.sidebar.slider("Weight: soil infiltration",     0.0, 1.0, float(w_defaults["soil"]), 0.01, key="w_soil")
w_lulc = st.sidebar.slider("Weight: land cover (LULC)",      0.0, 1.0, float(w_defaults["lulc"]), 0.01, key="w_lulc")

if st.sidebar.button("Recompute flood_risk_0to1.tif", key="btn_recompute"):
    try:
        out_path, w_final = recompute_flood_risk(
            summary_path=summary_path,
            w_dist=w_dist, w_drainage=w_dd, w_soil=w_soil, w_lulc=w_lulc,
            out_path_override=None,
            normalize_weights=normalize_weights,
        )
        st.sidebar.success(f"Recomputed: {out_path}")
        st.sidebar.json({"weights_used": w_final})
        # reload meta/paths after recompute
        meta = read_summary(summary_path)
        paths = meta["outputs"]
    except Exception as e:
        st.sidebar.error("Failed to recompute flood risk")
        st.sidebar.exception(e)

st.sidebar.markdown("---")
st.sidebar.subheader("Display settings")
overlay_opacity = st.sidebar.slider("Overlay opacity (visual only)", 0.0, 1.0, 0.65, 0.05, key="overlay_opacity")

st.sidebar.markdown("---")
st.sidebar.subheader("Layers to display")
show_risk = st.sidebar.checkbox("Flood risk (0–1)", value=True, key="toggle_risk")
show_dist = st.sidebar.checkbox("Distance to river (m)", value=False, key="toggle_dist")
show_dd = st.sidebar.checkbox("Drainage density (km/km²)", value=False, key="toggle_dd")
show_soil = st.sidebar.checkbox("Soil sand fraction (%)", value=False, key="toggle_soil")
show_lulc = st.sidebar.checkbox("LULC (worldcover proxy)", value=False, key="toggle_lulc")





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
    add_image_overlay(m, img, raster_bounds_latlon(p), "Flood risk (0–1)", opacity=overlay_opacity)
    add_onmap_legend(m, make_continuous_legend_png(CMAPS["flood_risk_0to1"], rvmin, rvmax, "Flood risk (0–1)"), position=next_pos())

if show_dist and "dist_to_river_m" in paths and os.path.exists(paths["dist_to_river_m"]):
    p = paths["dist_to_river_m"]
    img, (dvmin, dvmax) = raster_to_rgba_image(p, cmap_name=CMAPS["dist_to_river_m"])
    add_image_overlay(m, img, raster_bounds_latlon(p), "Distance to river (m)", opacity=overlay_opacity)
    add_onmap_legend(m, make_continuous_legend_png(CMAPS["dist_to_river_m"], dvmin, dvmax, "Distance to river (m)"), position=next_pos())

if show_dd and "drainage_density_km_per_km2" in paths and os.path.exists(paths["drainage_density_km_per_km2"]):
    p = paths["drainage_density_km_per_km2"]
    img, (ddvmin, ddvmax) = raster_to_rgba_image(p, cmap_name=CMAPS["drainage_density_km_per_km2"])
    add_image_overlay(m, img, raster_bounds_latlon(p), "Drainage density (km/km²)", opacity=overlay_opacity)
    add_onmap_legend(m, make_continuous_legend_png(CMAPS["drainage_density_km_per_km2"], ddvmin, ddvmax, "Drainage density (km/km²)"), position=next_pos())

if show_soil and "soil_sand_pct" in paths and os.path.exists(paths["soil_sand_pct"]):
    p = paths["soil_sand_pct"]
    img, (svmin, svmax) = raster_to_rgba_image(p, cmap_name=CMAPS["soil_sand_pct"])
    add_image_overlay(m, img, raster_bounds_latlon(p), "Soil sand fraction (%)", opacity=overlay_opacity)
    add_onmap_legend(m, make_continuous_legend_png(CMAPS["soil_sand_pct"], svmin, svmax, "Soil sand fraction (%)"), position=next_pos())

if show_lulc and "lulc_worldcover_proxy" in paths and os.path.exists(paths["lulc_worldcover_proxy"]):
    p = paths["lulc_worldcover_proxy"]
    img, _ = raster_to_rgba_image(p, cmap_name=CMAPS["lulc_worldcover_proxy"])
    add_image_overlay(m, img, raster_bounds_latlon(p), "LULC (worldcover proxy)", opacity=overlay_opacity)
    add_onmap_legend(m, make_lulc_legend_png(), position=next_pos())

# Add a marker for any user-selected location and sample nearby values
sel = st.session_state.get('selected_location') if 'selected_location' in st.session_state else None
if sel:
    try:
        lat = float(sel['lat'])
        lon = float(sel['lon'])
        name = sel.get('name', f"{lat:.6f}, {lon:.6f}")
        popup_html = f"<b>{name}</b><br/>{lat:.6f}, {lon:.6f}"
        # sample flood risk raster if available
        if 'flood_risk_0to1' in paths and os.path.exists(paths['flood_risk_0to1']):
            v = sample_raster_at_point(paths['flood_risk_0to1'], lat, lon)
            if v is not None:
                popup_html += f"<br/>Flood risk: {v:.3f}"
        # nearest grid info
        nn = nearest_grid_point(lat, lon)
        if nn is not None:
            nn_info = []
            if 'row' in nn and 'col' in nn:
                nn_info.append(f"r{nn.get('row')},c{nn.get('col')}")
            if 'LULC' in nn:
                nn_info.append(f"LULC:{nn.get('LULC')}")
            if 'DistRiver_m' in nn:
                nn_info.append(f"DistRiver_m:{nn.get('DistRiver_m')}")
            if nn_info:
                popup_html += "<br/>Nearest grid: " + ", ".join(nn_info)
        folium.Marker([lat, lon], popup=folium.Popup(popup_html, max_width=300), icon=folium.Icon(color='red', icon='map-marker')).add_to(m)
        # recenter map
        try:
            m.location = [lat, lon]
            m.zoom_start = 14
        except Exception:
            pass
    except Exception:
        pass

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, use_container_width=True, returned_objects=[])

with st.expander("AOI details"):
    st.write({"aoi_place": meta.get("aoi_place", "<unknown>"), "bbox": bbox})

with st.expander("Debug: Available layers"):
    st.write("**Paths available:**")
    for key, path in paths.items():
        exists = os.path.exists(path)
        st.write(f"  {key}: {path} {'✓' if exists else '✗ (missing)'}")
    st.write("**Summary metadata:**")
    st.json({"aoi_place": meta.get("aoi_place"), "bbox": meta.get("bbox"), "flood_risk_weights_used": meta.get("flood_risk_weights_used", {})})
