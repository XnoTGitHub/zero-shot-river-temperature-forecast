"""XGBoost-Baseline auf denselben Fenstern wie Extended Evaluation.

Features:
  - Bern-Temp Lags h-1, h-3, h-6, h-12, h-24, h-48 (aus Beobachtung)
  - Klimatologie(doy, hour) als starkes Prior
  - Lufttemperatur-Vorhersage(t0+h) aus Open-Meteo
  - DoY sin/cos, hour sin/cos, horizon h

Training: Sommer 2020-2022 (aus 2020-01 bis 2022-12)
Test: Sommer 2023, 2024, 2025, Juli 2026 (dieselben t0-Fenster wie
      extended_evaluation.csv + aare_recursive_thun.csv)

Output: xgboost_baseline.csv mit MAE pro Fenster.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb


BASE = Path(__file__).resolve().parent.parent
CACHE = BASE / "data" / "aare_dashboard_cache"
LAT, LON = 46.948, 7.447
HORIZON = 48


def _get(url, params, key):
    fp = CACHE / f"{key}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    r = requests.get(url, params=params, timeout=90,
                     headers={"User-Agent": "hywa-xgb/1.0"})
    r.raise_for_status()
    d = r.json()
    fp.write_text(json.dumps(d))
    return d


def fetch_meteo(lat, lon, start, end, tag):
    d = _get("https://historical-forecast-api.open-meteo.com/v1/forecast",
             {"latitude": lat, "longitude": lon, "hourly": "temperature_2m",
              "start_date": str(start), "end_date": str(end),
              "timezone": "Europe/Zurich"},
             key=f"meteo_xgb_{tag}_{start}_{end}")
    h = d["hourly"]
    return pd.Series(h["temperature_2m"], index=pd.to_datetime(h["time"]),
                     name="tt")


def load_aare_bern_all():
    df = pd.read_csv(BASE / "data" / "aare_temperature_hourly.csv", parse_dates=["datetime"])
    df = df.set_index("datetime")["temperature"].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def climatology(df_bern):
    """Aus Trainings-Periode (< 2023)."""
    d = df_bern[df_bern.index < "2023-01-01"].to_frame("t")
    d["doy"] = d.index.dayofyear
    d["hour"] = d.index.hour
    clim = d.groupby(["doy", "hour"])["t"].mean()
    return clim


def build_features(t0, h_range, bern, clim, tt):
    """Baut ein DataFrame mit Features + Target fuer Fenster t0.
    Zurueckgabe: X (H rows, F features), y (H targets)."""
    feats = []
    tgts = []
    for h in h_range:
        target_t = t0 + pd.Timedelta(hours=h)
        # Target
        if target_t not in bern.index or np.isnan(bern.loc[target_t]):
            continue
        y = float(bern.loc[target_t])

        # Bern-Temp lags relative zu t0-1
        row = {"h": h}
        for lag in [1, 3, 6, 12, 24, 48]:
            ts = t0 - pd.Timedelta(hours=lag)
            v = bern.get(ts, np.nan)
            row[f"bern_lag{lag}"] = v if not np.isnan(v) else clim.get(
                (ts.dayofyear, ts.hour), np.nan)

        # Klimatologie zum Zeitpunkt t0+h
        row["clim"] = clim.get((target_t.dayofyear, target_t.hour), np.nan)

        # Lufttemperatur-Vorhersage bei t0+h
        row["tt_h"] = tt.get(target_t, np.nan)

        # Zeit-Features
        row["doy_sin"] = np.sin(2 * np.pi * target_t.dayofyear / 365.25)
        row["doy_cos"] = np.cos(2 * np.pi * target_t.dayofyear / 365.25)
        row["hour_sin"] = np.sin(2 * np.pi * target_t.hour / 24)
        row["hour_cos"] = np.cos(2 * np.pi * target_t.hour / 24)

        # Delta zwischen aktuellem Beobachtungswert und Klimatologie
        cur = bern.get(t0 - pd.Timedelta(hours=1), np.nan)
        row["delta_now_clim"] = cur - clim.get((t0.dayofyear, t0.hour), np.nan)

        if any(np.isnan(v) for v in row.values()):
            continue
        feats.append(row)
        tgts.append(y)

    return pd.DataFrame(feats), np.array(tgts)


def generate_training_windows(bern_train, tt_all, clim):
    """Erzeugt Training-Samples aus 2020-2022 Sommer und Uebergangszeiten."""
    X_list, y_list = [], []
    for year in [2020, 2021, 2022]:
        for month in [5, 6, 7, 8, 9]:
            for day in range(1, 29, 3):
                try:
                    t0 = pd.Timestamp(year=year, month=month, day=day, hour=12)
                    if t0 not in bern_train.index:
                        continue
                    X, y = build_features(t0, list(range(1, HORIZON + 1)),
                                          bern_train, clim, tt_all)
                    if len(X) == 0:
                        continue
                    X_list.append(X)
                    y_list.append(y)
                except Exception as e:
                    print(f"    train sample fail {year}-{month:02d}-{day:02d}: {e}")
    X = pd.concat(X_list, ignore_index=True)
    y = np.concatenate(y_list)
    return X, y


def main():
    print("Lade Aare-Bern Historie...")
    bern = load_aare_bern_all()
    print(f"  {len(bern)} h, {bern.index.min()} .. {bern.index.max()}")

    print("Lade Open-Meteo tt (Bern) fuer Trainings- und Test-Zeitraum...")
    tt_train_2020_22 = fetch_meteo(LAT, LON, "2020-01-01", "2022-12-31", "2020-22")
    tt_2023 = fetch_meteo(LAT, LON, "2023-06-01", "2023-07-31", "2023summer")
    tt_2024 = fetch_meteo(LAT, LON, "2024-06-01", "2024-07-31", "2024summer")
    tt_2025 = fetch_meteo(LAT, LON, "2025-06-01", "2025-07-31", "2025summer")
    tt_all = pd.concat([tt_train_2020_22, tt_2023, tt_2024, tt_2025])
    tt_all = tt_all[~tt_all.index.duplicated(keep="last")].sort_index()
    print(f"  tt: {len(tt_all)} h")

    clim = climatology(bern)
    print(f"  Klimatologie: {len(clim)} (DoY,Hour) Zellen")

    print("Erzeuge Trainings-Samples aus 2020-2022 (Mai-Sept)...")
    X_train, y_train = generate_training_windows(bern, tt_all, clim)
    print(f"  n_train = {len(X_train)}   n_features = {X_train.shape[1]}")

    print("Trainiere XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"  Feature importance top 5:")
    imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    for i, (name, val) in enumerate(imp.head(5).items()):
        print(f"    {i+1}. {name:20s}  {val:.4f}")

    # Test auf denselben Fenstern wie extended eval
    ext_df = pd.read_csv(BASE / "src/extended_evaluation.csv",
                         parse_dates=["t0"])
    rt_df = pd.read_csv(BASE / "aare_recursive_thun.csv", parse_dates=["t0"])
    tt_2026 = fetch_meteo(LAT, LON, "2026-06-01", "2026-07-31", "2026summer")
    tt_all_full = pd.concat([tt_all, tt_2026])
    tt_all_full = tt_all_full[~tt_all_full.index.duplicated(keep="last")].sort_index()

    rows = []
    for _, r in ext_df.iterrows():
        t0 = r["t0"]
        X, y = build_features(t0, list(range(1, HORIZON + 1)),
                              bern, clim, tt_all_full)
        if len(X) == 0:
            continue
        pred = model.predict(X)
        mae = float(np.mean(np.abs(pred - y)))
        rows.append({"t0": t0, "dataset": "extended", "year": r.get("year"),
                     "xgb_MAE": mae})

    for _, r in rt_df.iterrows():
        t0 = r["t0"]
        X, y = build_features(t0, list(range(1, HORIZON + 1)),
                              bern, clim, tt_all_full)
        if len(X) == 0:
            continue
        pred = model.predict(X)
        mae = float(np.mean(np.abs(pred - y)))
        rows.append({"t0": t0, "dataset": "juli_2026", "year": 2026,
                     "xgb_MAE": mae})

    df = pd.DataFrame(rows)
    out = BASE / "src/xgboost_baseline.csv"
    df.to_csv(out, index=False)
    print(f"\nOutput: {out}")
    print(f"n_windows = {len(df)}")
    print(f"\nPer-year MAE:")
    for (ds, year), d in df.groupby(["dataset", "year"]):
        print(f"  {ds} {year}   n={len(d)}   XGBoost MAE = {d['xgb_MAE'].mean():.4f}")


if __name__ == "__main__":
    main()
