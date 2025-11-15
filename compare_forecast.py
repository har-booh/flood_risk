#!/usr/bin/env python3
"""
compare_forecast.py
Walk-forward evaluation of ARX(lags=3) at a chosen horizon (h days).
Outputs:
  - data/predictions_h{h}.csv  (date, actual_rain_mm, pred_rain_mm)
  - outputs/predictions_h{h}.png (plot)
Usage:
  python compare_forecast.py --horizon 1
  python compare_forecast.py --horizon 3 --met-file data/meteorology_timeseries.csv
"""

import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def load_met(met_file):
    df = pd.read_csv(met_file, parse_dates=["date"])
    # basic checks
    if "rain_mm" not in df:
        raise ValueError("CSV must include 'rain_mm' column.")
    for col in ["temp_c", "humid_pct", "press_hpa"]:
        if col not in df:
            # synthesize if missing
            n = len(df); t = np.arange(n)
            if col == "temp_c":
                df[col] = 28 + 2*np.sin(2*np.pi*t/365.0 + 1.2) + np.random.normal(0, 0.7, n)
            elif col == "humid_pct":
                df[col] = 80 - 10*np.sin(2*np.pi*t/365.0) + np.random.normal(0, 3, n)
            elif col == "press_hpa":
                df[col] = 1012 + 3*np.cos(2*np.pi*t/7.0) + np.random.normal(0, 0.8, n)
    return df.sort_values("date").reset_index(drop=True)

def fit_beta(df, lags=3, horizon=1):
    y = df["rain_mm"].values
    X_ex = df[["temp_c", "humid_pct", "press_hpa"]].values
    rows, targets = [], []
    for t in range(lags - 1, len(df) - horizon):
        rain_lag = y[t-(lags-1):t+1]
        ex_t = X_ex[t]
        rows.append(np.concatenate([rain_lag, ex_t, [1.0]]))
        targets.append(y[t + horizon])
    X = np.array(rows); Y = np.array(targets)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta

def predict_from_last(df, beta, lags=3):
    y = df["rain_mm"].values
    X_ex = df[["temp_c", "humid_pct", "press_hpa"]].values
    t = len(df) - 1
    rain_lag = y[t-(lags-1):t+1]
    ex_t = X_ex[t]
    x = np.concatenate([rain_lag, ex_t, [1.0]])
    return float(x @ beta)

def walk_forward(df, horizon=1, lags=3, min_train=60):
    """
    For t in [min_train ... N-horizon-1]:
      fit on df[:t+1], predict for date at t+horizon
    """
    preds = []
    for t in range(max(lags, min_train), len(df) - horizon):
        train = df.iloc[:t+1].copy()
        beta = fit_beta(train, lags=lags, horizon=horizon)
        yhat = predict_from_last(train, beta, lags=lags)
        target_idx = t + horizon
        preds.append((df.loc[target_idx, "date"], df.loc[target_idx, "rain_mm"], max(0.0, yhat)))
    return pd.DataFrame(preds, columns=["date", "actual_rain_mm", "pred_rain_mm"])

def metrics(actual, pred):
    e = pred - actual
    mae = np.mean(np.abs(e))
    rmse = float(np.sqrt(np.mean(e**2)))
    # R2 (handle constant series gracefully)
    denom = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - (np.sum(e**2)/denom) if denom > 0 else np.nan
    return mae, rmse, r2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1, help="Forecast horizon (days ahead)")
    ap.add_argument("--lags", type=int, default=3, help="AR lags on rainfall history")
    ap.add_argument("--min-train", type=int, default=60, help="Minimum initial training window (days)")
    ap.add_argument("--met-file", default="data/meteorology_timeseries.csv", help="Path to met CSV")
    ap.add_argument("--out-csv", default=None, help="Override output CSV path")
    ap.add_argument("--out-png", default=None, help="Override output PNG path")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = load_met(args.met_file)
    dfp = walk_forward(df, horizon=args.horizon, lags=args.lags, min_train=args.min_train)

    # Save CSV
    out_csv = args.out_csv or f"data/predictions_h{args.horizon}.csv"
    dfp.to_csv(out_csv, index=False)

    # Plot
    mae, rmse, r2 = metrics(dfp["actual_rain_mm"].values, dfp["pred_rain_mm"].values)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dfp["date"], dfp["actual_rain_mm"], label="Actual (rain_mm)")
    ax.plot(dfp["date"], dfp["pred_rain_mm"], label=f"Predicted (h={args.horizon})", linestyle="--")
    ax.set_title(f"Rainfall — Actual vs Predicted (h={args.horizon}) | MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")
    ax.set_xlabel("Date"); ax.set_ylabel("mm/day")
    ax.legend()
    fig.tight_layout()
    out_png = args.out_png or f"outputs/predictions_h{args.horizon}.png"
    fig.savefig(out_png, dpi=150)

    print(f"Saved predictions: {out_csv}")
    print(f"Saved plot:        {out_png}")

if __name__ == "__main__":
    main()