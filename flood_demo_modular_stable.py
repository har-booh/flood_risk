#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_flood_index.py
Read prepared_layers_summary.json and compute a composite flood risk raster (0–1).

Inputs expected from fetch_prepare_lagos_data.py:
  - dist_to_river_m.tif
  - drainage_density_km_per_km2.tif
  - soil_sand_pct.tif
  - lulc_osm_proxy.tif

Output:
  - flood_risk_0to1.tif
  - updated prepared_layers_summary.json including flood_risk_0to1 path

Run:
  pip install rasterio numpy
  python compute_flood_index.py --summary data/rasters/prepared_layers_summary.json \
         --w-dist 0.35 --w-drainage 0.25 --w-soil 0.20 --w-lulc 0.20
"""
import os, sys, json, argparse
import numpy as np
import rasterio

def robust_minmax(x: np.ndarray, lower_q=2.0, upper_q=98.0) -> np.ndarray:
    x = x.astype("float32")
    lo = np.nanpercentile(x, lower_q); hi = np.nanpercentile(x, upper_q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(x); hi = np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype="float32")
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)

def lulc_class_to_risk_weight(lulc_arr: np.ndarray) -> np.ndarray:
    mapping = {1:1.00, 2:0.85, 5:0.75, 4:0.55, 6:0.45, 3:0.35}
    out = np.zeros_like(lulc_arr, dtype="float32")
    for k, v in mapping.items():
        out[lulc_arr == k] = v
    return out

def compute_risk(dist_to_river_m, drainage_density, soil_sand_pct, lulc_arr, weights):
    dist_norm = robust_minmax(dist_to_river_m); dist_risk = 1.0 - dist_norm
    drainage_norm = robust_minmax(drainage_density)
    soil_norm = robust_minmax(soil_sand_pct); soil_risk = 1.0 - soil_norm
    lulc_risk = lulc_class_to_risk_weight(lulc_arr)
    risk = (weights["dist"]*dist_risk + weights["drainage"]*drainage_norm +
            weights["soil"]*soil_risk + weights["lulc"]*lulc_risk)
    return robust_minmax(risk)

def run(summary_path, w_dist, w_drainage, w_soil, w_lulc, out_path_override=None):
    with open(summary_path, "r") as f:
        meta = json.load(f)
    p = meta["outputs"]
    # Read rasters
    with rasterio.open(p["dist_to_river_m"]) as src:
        dist = src.read(1); transform = src.transform; crs = src.crs
    with rasterio.open(p["drainage_density_km_per_km2"]) as src: dd = src.read(1)
    with rasterio.open(p["soil_sand_pct"]) as src: soil = src.read(1)
    with rasterio.open(p["lulc_osm_proxy"]) as src: lulc = src.read(1)

    weights = {"dist": w_dist, "drainage": w_drainage, "soil": w_soil, "lulc": w_lulc}
    risk01 = compute_risk(dist, dd, soil, lulc, weights)

    out_path = out_path_override or os.path.join(os.path.dirname(summary_path), "flood_risk_0to1.tif")
    profile = {"driver":"GTiff","height":risk01.shape[0],"width":risk01.shape[1],
               "count":1,"dtype":"float32","crs":crs,"transform":transform,
               "compress":"deflate","nodata":np.nan}
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(risk01.astype("float32"), 1)

    # Update summary
    meta["outputs"]["flood_risk_0to1"] = out_path
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved flood risk: {out_path}")
    print(f"Updated summary: {summary_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="data/rasters/prepared_layers_summary.json")
    ap.add_argument("--w-dist", type=float, default=0.35)
    ap.add_argument("--w-drainage", type=float, default=0.25)
    ap.add_argument("--w-soil", type=float, default=0.20)
    ap.add_argument("--w-lulc", type=float, default=0.20)
    ap.add_argument("--out", default=None, help="Optional output path for flood_risk_0to1.tif")
    args = ap.parse_args()
    run(args.summary,
        args["w-dist"] if isinstance(args, dict) else args.w_dist,
        args["w-drainage"] if isinstance(args, dict) else args.w_drainage,
        args["w-soil"] if isinstance(args, dict) else args.w_soil,
        args["w-lulc"] if isinstance(args, dict) else args.w_lulc,
        out_path_override=(args.get("out") if isinstance(args, dict) else args.out))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
