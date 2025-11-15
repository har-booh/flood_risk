#!/usr/bin/env python3
"""
fetch_meteorology.py
Fetch daily meteorology for Lagos AOI (from config.yaml) and save:
  data/meteorology_timeseries.csv
Columns: date, rain_mm, temp_c, humid_pct, press_hpa

Sources:
  - NASA POWER (default): no account needed, robust for daily series
  - CHIRPS via Google Earth Engine (optional): needs earthengine-api & auth

Usage:
  python fetch_meteorology.py --start 2023-01-01 --end 2024-12-31
  python fetch_meteorology.py --source power --start 2023-01-01 --end 2024-12-31
  python fetch_meteorology.py --source chirps-gee --start 2023-01-01 --end 2024-12-31

Notes:
  - POWER returns:
      PRECTOT  -> daily precip (mm/day)
      T2M      -> 2m air temperature (°C)
      RH2M     -> 2m relative humidity (%)
      PS       -> surface pressure (kPa)  ==> convert to hPa by *10
  - CHIRPS (via GEE) provides precip only; we synthesize temp/humidity/pressure
    to keep the pipeline running, unless you provide those from another source.
"""

import argparse, os, sys, json, math
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import yaml
except ImportError:
    print("Please install pyyaml: pip install pyyaml")
    sys.exit(1)

DATA_DIR = "data"
OUT_CSV = os.path.join(DATA_DIR, "meteorology_timeseries.csv")

def read_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def aoi_centroid(aoi: dict):
    lat = (float(aoi["min_lat"]) + float(aoi["max_lat"])) / 2.0
    lon = (float(aoi["min_lon"]) + float(aoi["max_lon"])) / 2.0
    return lat, lon

# -------------------------------
# NASA POWER
# -------------------------------
def fetch_power(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    import requests

    # POWER expects YYYYMMDD
    def yyyymmdd(s): return datetime.fromisoformat(s).strftime("%Y%m%d")
    start_s = yyyymmdd(start)
    end_s = yyyymmdd(end)

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOTCORR,T2M,RH2M,PS&community=AG&longitude={lon}&latitude={lat}"
        f"&start={start_s}&end={end_s}&format=JSON"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()

    # Structure: properties.parameter.<VAR> is a dict {YYYYMMDD: value}
    params = js.get("properties", {}).get("parameter", {})

    print("Available parameters:", list(params.keys()))
    
    prectot = params.get("PRECTOTCORR", {})
    t2m     = params.get("T2M", {})
    rh2m    = params.get("RH2M", {})
    ps_kpa  = params.get("PS", {})

    # unify dates
    dates = sorted(set(prectot.keys()) | set(t2m.keys()) | set(rh2m.keys()) | set(ps_kpa.keys()))
    rows = []
    for d in dates:
        # Convert YYYYMMDD -> YYYY-MM-DD
        d_iso = datetime.strptime(d, "%Y%m%d").date().isoformat()
        rain_mm = float(prectot.get(d, np.nan))       # mm/day
        temp_c  = float(t2m.get(d, np.nan))           # °C
        humid   = float(rh2m.get(d, np.nan))          # %
        press_hpa = float(ps_kpa.get(d, np.nan)) * 10 # kPa -> hPa
        rows.append([d_iso, rain_mm, temp_c, humid, press_hpa])

    df = pd.DataFrame(rows, columns=["date","rain_mm","temp_c","humid_pct","press_hpa"])
    return df

# -------------------------------
# CHIRPS via Google Earth Engine (optional)
# -------------------------------
def fetch_chirps_gee(aoi: dict, start: str, end: str) -> pd.DataFrame:
    """
    Requires:
      pip install earthengine-api
      earthengine authenticate
    Pulls daily CHIRPS rainfall averaged over AOI, and synthesizes T/H/PS to keep the columns complete.
    """
    try:
        import ee
    except ImportError:
        raise RuntimeError("earthengine-api not installed. Install with: pip install earthengine-api")

    ee.Initialize()  # must be authenticated beforehand

    # AOI rectangle
    region = ee.Geometry.Rectangle([
        float(aoi["min_lon"]), float(aoi["min_lat"]),
        float(aoi["max_lon"]), float(aoi["max_lat"])
    ])

    ic = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end).filterBounds(region)

    def image_to_feature(img):
        # mean over AOI
        mean_dict = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5500,  # CHIRPS ~5km
            maxPixels=1e9
        )
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        return ee.Feature(None, {"date": date, "rain_mm": mean_dict.get("precipitation")})

    fc = ic.map(image_to_feature).filter(ee.Filter.notNull(["rain_mm"])).distinct("date")
    rows = fc.aggregate_array("date").getInfo()
    rains = fc.aggregate_array("rain_mm").getInfo()

    # Synthesize temp/humidity/pressure (keeps pipeline intact if you only care about rain)
    # You can replace this block with real temp/humidity/pressure later.
    n = len(rows)
    # gentle annual cycle + noise
    t = np.arange(n)
    temp_c  = 28 + 2*np.sin(2*np.pi*t/365.0 + 1.2) + np.random.normal(0, 0.7, n)
    humid   = 80 -10*np.sin(2*np.pi*t/365.0) + np.random.normal(0, 3, n)
    press_hpa = 1012 + 3*np.cos(2*np.pi*t/7.0) + np.random.normal(0, 0.8, n)

    df = pd.DataFrame({
        "date": rows,
        "rain_mm": np.array(rains, dtype=float),
        "temp_c": temp_c,
        "humid_pct": humid,
        "press_hpa": press_hpa
    })
    return df

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="power", choices=["power", "chirps-gee"], help="Data source")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = ap.parse_args()

    cfg = read_config()
    lat, lon = aoi_centroid(cfg["aoi"])
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.source == "power":
        df = fetch_power(lat, lon, args.start, args.end)
    else:
        df = fetch_chirps_gee(cfg["aoi"], args.start, args.end)

    # Sort and save
    df = df.sort_values("date")
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()