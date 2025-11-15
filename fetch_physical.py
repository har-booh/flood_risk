#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_prepare_lagos_data.py
Fetch and prepare realistic layers for Lagos (no index computation).

Outputs (GeoTIFFs in ./data/rasters):
  - dist_to_river_m.tif
  - lulc_osm_proxy.tif
  - drainage_density_km_per_km2.tif
  - soil_sand_pct.tif
And a summary JSON at data/rasters/prepared_layers_summary.json

Run:
  pip install geopandas shapely rasterio osmnx requests numpy scipy pyproj rtree
  python fetch_prepare_lagos_data.py
"""
import os, sys, math, json, warnings
from dataclasses import dataclass
import argparse
from typing import Tuple
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from shapely.geometry import box
from pyproj import CRS
import requests

try:
    import osmnx as ox
except Exception:
    ox = None
try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class Config:
    place_name: str = "Lagos, Nigeria"
    grid_res_deg: float = 0.001         # ≈ 111 m
    fishnet_cell_deg: float = 0.01      # ≈ 1.1 km
    soil_res_m: int = 250
    buffer_deg: float = 0.02
    out_dir: str = "data/rasters"
    tmp_dir: str = "data/tmp"
    crs_epsg: int = 4326

CFG = Config()

# ---------------- Utils ----------------
def ensure_dirs():
    os.makedirs(CFG.out_dir, exist_ok=True)
    os.makedirs(CFG.tmp_dir, exist_ok=True)

def geocode_aoi(place: str) -> gpd.GeoDataFrame:
    if ox is None:
        raise RuntimeError("osmnx is required (pip install osmnx).")
    gdf = ox.geocode_to_gdf(place)
    return gdf.to_crs(epsg=CFG.crs_epsg)

def bbox_from_gdf(gdf: gpd.GeoDataFrame, buffer_deg: float = 0.02):
    minx, miny, maxx, maxy = gdf.total_bounds
    return (minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg)

def make_raster_grid(bbox, res_deg):
    west, south, east, north = bbox
    width = int(math.ceil((east - west) / res_deg))
    height = int(math.ceil((north - south) / res_deg))
    transform = from_origin(west, north, res_deg, res_deg)
    return transform, (height, width)

def save_geotiff(path, array, transform, crs, nodata=None, dtype=None, compress="deflate"):
    if dtype is None: dtype = array.dtype
    profile = {"driver":"GTiff","height":array.shape[0],"width":array.shape[1],
               "count":1,"dtype":dtype,"crs":crs,"transform":transform,
               "compress":compress,"nodata":nodata}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)

def rasterize_geoms(geoms, out_shape, transform, burn_value=1, all_touched=False, dtype="uint8"):
    if len(geoms) == 0:
        return np.zeros(out_shape, dtype=dtype)
    if isinstance(geoms[0], tuple):
        shapes = geoms
    else:
        shapes = [(geom, burn_value) for geom in geoms]
    out = rasterize(shapes=shapes, out_shape=out_shape, transform=transform,
                    fill=0, all_touched=all_touched, dtype=dtype)
    return out

# ------------- OSM fetchers & processors -------------
def fetch_osm_waterways(bbox):
    if ox is None:
        raise RuntimeError("osmnx is required (pip install osmnx).")
    west, south, east, north = bbox
    tags = {"waterway": ["river", "stream", "canal", "drain"]}
    g = None
    # Try v2
    try:
        from osmnx import features as _ox_features
        g = _ox_features.features_from_bbox(north, south, east, west, tags=tags)
    except Exception:
        pass
    # v1 fallback
    if g is None:
        try:
            g = ox.geometries_from_bbox(north, south, east, west, tags=tags)
        except Exception:
            g = None
    # polygon fallback
    if g is None or g.empty:
        aoi = geocode_aoi(CFG.place_name)
        poly = aoi.unary_union
        try:
            from osmnx import features as _ox_features
            g = _ox_features.features_from_polygon(poly, tags=tags)
        except Exception:
            g = ox.geometries_from_polygon(poly, tags=tags)
    g = g[g.geometry.geom_type.isin(["LineString","MultiLineString"])].copy().to_crs(epsg=CFG.crs_epsg)
    g["length_m"] = g.to_crs(epsg=3857).geometry.length
    return g

def fetch_osm_lulc_polys(bbox):
    if ox is None:
        raise RuntimeError("osmnx is required (pip install osmnx).")
    west, south, east, north = bbox
    tags = {"landuse": True, "natural": True, "water": True, "building": True}
    g = None
    try:
        from osmnx import features as _ox_features
        g = _ox_features.features_from_bbox(north, south, east, west, tags=tags)
    except Exception:
        pass
    if g is None:
        try:
            g = ox.geometries_from_bbox(north, south, east, west, tags=tags)
        except Exception:
            g = None
    if g is None or g.empty:
        aoi = geocode_aoi(CFG.place_name)
        poly = aoi.unary_union
        try:
            from osmnx import features as _ox_features
            g = _ox_features.features_from_polygon(poly, tags=tags)
        except Exception:
            g = ox.geometries_from_polygon(poly, tags=tags)
    g = g[g.geometry.geom_type.isin(["Polygon","MultiPolygon"])].copy().to_crs(epsg=CFG.crs_epsg)
    return g

def lulc_class_from_osm(row) -> int:
    low = (lambda x: str(x).lower() if x is not None else "")
    landuse = low(row.get("landuse")); natural = low(row.get("natural"))
    water = low(row.get("water")); building = low(row.get("building"))
    if water in ("reservoir","pond","lake") or natural in ("water","wetland"):
        return 1
    if building and building != "nan": return 2
    if landuse in ("residential","industrial","commercial","retail"): return 2
    if landuse in ("farmland","orchard","vineyard","meadow","grass"): return 4
    if natural in ("wood","forest","grassland","scrub"): return 3
    if natural == "wetland": return 5
    return 6

def build_lulc_raster(polys, transform, out_shape):
    if "lulc_class" not in polys.columns:
        polys = polys.copy(); polys["lulc_class"] = polys.apply(lulc_class_from_osm, axis=1)
    priority = [1,2,5,3,4,6]
    out = np.zeros(out_shape, dtype="uint8")
    for cls in priority:
        geoms = polys.loc[polys["lulc_class"]==cls, "geometry"].values
        if len(geoms)==0: continue
        mask = rasterize_geoms(geoms, out_shape, transform, burn_value=cls, all_touched=True, dtype="uint8")
        out = np.where(mask!=0, mask, out)
    return out

def compute_distance_to_river_raster(waterways, transform, out_shape):
    if distance_transform_edt is None:
        raise RuntimeError("scipy is required for distance transform (pip install scipy).")
    lines_r = rasterize_geoms(waterways.geometry.values, out_shape, transform, burn_value=1, all_touched=True, dtype="uint8")
    dist_px = distance_transform_edt(1 - (lines_r > 0))
    meters_per_deg = 111_320.0
    dist_m = dist_px * CFG.grid_res_deg * meters_per_deg
    dist_m[lines_r > 0] = 0.0
    return dist_m.astype("float32")

def fishnet_grid(bbox, cell_deg):
    west, south, east, north = bbox
    xs = np.arange(west, east, cell_deg); ys = np.arange(south, north, cell_deg)
    polys = [box(x, y, x+cell_deg, y+cell_deg) for x in xs for y in ys]
    return gpd.GeoDataFrame({"geometry": polys}, crs=f"EPSG:{CFG.crs_epsg}")

def compute_drainage_density_grid(waterways, bbox, cell_deg):
    grid = fishnet_grid(bbox, cell_deg)
    metric = 3857
    ww_m = waterways.to_crs(epsg=metric); grid_m = grid.to_crs(epsg=metric)
    sindex = ww_m.sindex
    lengths_km = []; areas_km2 = grid_m.geometry.area / 1e6
    for poly in grid_m.geometry:
        possible = list(sindex.intersection(poly.bounds))
        if not possible: lengths_km.append(0.0); continue
        sub = ww_m.iloc[possible]
        inter = sub.intersection(poly)
        total_len_m = sum(g.length for g in inter if g is not None)
        lengths_km.append(total_len_m / 1000.0)
    grid["dd_km_per_km2"] = np.divide(lengths_km, areas_km2, out=np.zeros_like(areas_km2), where=areas_km2>0)
    return grid

def rasterize_grid_values(grid, value_col, transform, out_shape):
    shapes = list(zip(grid.geometry.values, grid[value_col].values))
    arr = rasterize(shapes=shapes, out_shape=out_shape, transform=transform,
                    fill=0.0, all_touched=True, dtype="float32")
    return arr

# ------------ SoilGrids (sand fraction) via WCS ------------
SOILGRID_WCS = "https://maps.isric.org/mapserv?map=/map/sand.map"

def build_wcs_url(bbox, coverage, res_m, format_str='GEOTIFF_INT16'):
    west, south, east, north = bbox
    params = {
        "SERVICE": "WCS", "VERSION": "2.0.1", "REQUEST": "GetCoverage",
        "COVERAGEID": coverage, "FORMAT": format_str,
        "SUBSET": [f"long({west},{east})", f"lat({south},{north})"],
        "RESX": f"{res_m}m", "RESY": f"{res_m}m", "CRS": "EPSG:4326"
    }
    subset_parts = "&".join([f"SUBSET={p}" for p in params["SUBSET"]])
    url = (f"{SOILGRID_WCS}&SERVICE={params['SERVICE']}&VERSION={params['VERSION']}&REQUEST={params['REQUEST']}"
           f"&COVERAGEID={params['COVERAGEID']}&FORMAT={params['FORMAT']}&{subset_parts}"
           f"&RESX={params['RESX']}&RESY={params['RESY']}&CRS={params['CRS']}"
           f"&SUBSETTINGCRS=EPSG:4326&OUTPUTCRS=EPSG:4326")
    return url

def _looks_like_geotiff(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            head = f.read(4)
        return head in (b'II*\x00', b'MM\x00*')
    except Exception:
        return False

def download_soilgrids_bbox(bbox, out_tif, coverage="sand_0-5cm_Q0.5", res_m=250):
    url = build_wcs_url(bbox, coverage, res_m, format_str='GEOTIFF_INT16')
    print(f"[SoilGrids] Requesting WCS: {url}")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    with open(out_tif, 'wb') as f:
        f.write(r.content)
    if not _looks_like_geotiff(out_tif):
        dbg = out_tif + '.txt'
        with open(dbg, 'wb') as f:
            f.write(r.content)
        raise RuntimeError(f"SoilGrids did not return a GeoTIFF. See debug: {dbg}")
    print(f"[SoilGrids] Saved: {out_tif}")

def resample_to_grid(src_path, target_transform, target_shape, out_path, resampling=Resampling.bilinear):
    with rasterio.open(src_path) as src:
        data = src.read(out_shape=(1, target_shape[0], target_shape[1]), resampling=resampling)
        out_meta = src.meta.copy()
        out_meta.update({"height": target_shape[0], "width": target_shape[1],
                         "transform": target_transform, "crs": src.crs})
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(data)


def parse_args():
    ap = argparse.ArgumentParser(description="Fetch & prepare Lagos-like layers for any AOI")
    ap.add_argument("--place", default="Lagos, Nigeria", help="AOI place name for geocoding (Nominatim)")
    ap.add_argument("--grid-res-deg", type=float, default=0.001, help="Grid resolution in degrees (~0.001 ≈ 111 m)")
    ap.add_argument("--fishnet-deg", type=float, default=0.01, help="Fishnet cell size in degrees for drainage density")
    ap.add_argument("--soil-res-m", type=int, default=250, help="SoilGrids requested resolution in meters")
    ap.add_argument("--buffer-deg", type=float, default=0.02, help="Buffer to expand AOI bbox in degrees")
    ap.add_argument("--out-dir", default="data/rasters", help="Output rasters directory")
    ap.add_argument("--tmp-dir", default="data/tmp", help="Temporary directory")
    return ap.parse_args()

# ---------------------------- Main ----------------------------
def main():
    args = parse_args()
    # Override CFG with CLI
    CFG.place_name = args.place
    CFG.grid_res_deg = args.grid_res_deg
    CFG.fishnet_cell_deg = args.fishnet_deg
    CFG.soil_res_m = args.soil_res_m
    CFG.buffer_deg = args.buffer_deg
    CFG.out_dir = args.out_dir
    CFG.tmp_dir = args.tmp_dir
    ensure_dirs()
    print("=== Fetch & Prepare Layers (no index) ===")
    print(f"AOI: {CFG.place_name}")
    aoi = geocode_aoi(CFG.place_name)
    bbox = bbox_from_gdf(aoi, buffer_deg=CFG.buffer_deg)
    print(f"BBox (W,S,E,N): {bbox}")
    transform, out_shape = make_raster_grid(bbox, CFG.grid_res_deg)
    crs = CRS.from_epsg(CFG.crs_epsg)

    print("Fetching OSM waterways ...")
    waterways = fetch_osm_waterways(bbox)
    print(f"Waterway lines: {len(waterways)}")

    print("Computing distance-to-river (m) ...")
    dist_river = compute_distance_to_river_raster(waterways, transform, out_shape)
    dist_river_path = os.path.join(CFG.out_dir, "dist_to_river_m.tif")
    save_geotiff(dist_river_path, dist_river, transform, crs, nodata=np.nan, dtype="float32")

    print("Fetching OSM LULC polygons ...")
    lulc_polys = fetch_osm_lulc_polys(bbox)
    print(f"LULC polys: {len(lulc_polys)}")
    print("Rasterizing LULC classes ...")
    lulc_arr = build_lulc_raster(lulc_polys, transform, out_shape)
    lulc_path = os.path.join(CFG.out_dir, "lulc_osm_proxy.tif")
    save_geotiff(lulc_path, lulc_arr, transform, crs, nodata=0, dtype="uint8")

    print("Computing drainage density (km/km²) ...")
    dd_grid = compute_drainage_density_grid(waterways, bbox, CFG.fishnet_cell_deg)
    dd_raster = rasterize_grid_values(dd_grid, "dd_km_per_km2", transform, out_shape)
    dd_path = os.path.join(CFG.out_dir, "drainage_density_km_per_km2.tif")
    save_geotiff(dd_path, dd_raster, transform, crs, nodata=0.0, dtype="float32")

    print("Downloading SoilGrids sand fraction (0–5 cm, Q0.5) ...")
    soil_raw = os.path.join(CFG.tmp_dir, "soil_sand_raw.tif")
    download_soilgrids_bbox(bbox, soil_raw, coverage="sand_0-5cm_Q0.5", res_m=CFG.soil_res_m)
    print("Resampling soil layer to target grid ...")
    soil_path = os.path.join(CFG.out_dir, "soil_sand_pct.tif")
    resample_to_grid(soil_raw, transform, out_shape, soil_path, resampling=Resampling.bilinear)

    # Summary (no risk here)
    summary = {
        "aoi_place": CFG.place_name,
        "bbox": bbox,
        "grid_res_deg": CFG.grid_res_deg,
        "fishnet_cell_deg": CFG.fishnet_cell_deg,
        "soil_res_m": CFG.soil_res_m,
        "outputs": {
            "dist_to_river_m": dist_river_path,
            "lulc_osm_proxy": lulc_path,
            "drainage_density_km_per_km2": dd_path,
            "soil_sand_pct": soil_path
        }
    }
    summary_path = os.path.join(CFG.out_dir, "prepared_layers_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary (no risk): {summary_path}")
    print("=== Done (fetch/prepare) ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
