"""
Zusaetzliche Baselines fuer das HyWa-Paper:
  - Persistenz  y_pred[t0+h] = y[t0-1]
  - Klimatologie  y_pred[t0+h] = mean_over_years(y | DoY, hour)
  - Bootstrap-CIs fuer alle bekannten Modelle aus aare_recursive_thun.csv

Nutzt vorhandene Fenster aus aare_recursive_thun.csv (2026-07-13 bis 2026-07-27).
Holt Wahrheit via Aareguru fuer die Fenster.
Klimatologie aus aare_temperature_hourly.csv (2020-2025).

Output:
  baselines_per_window.csv
  baselines_summary.csv   (mit Bootstrap-CIs)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


HORIZON = 48
BASE = Path(__file__).resolve().parent.parent
CACHE = BASE / "data" / "aare_dashboard_cache"
CACHE.mkdir(exist_ok=True)


def _get(url, params, key):
    fp = CACHE / f"{key}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    r = requests.get(url, params=params, timeout=60,
                     headers={"User-Agent": "hywa-baselines/1.0"})
    r.raise_for_status()
    d = r.json()
    fp.write_text(json.dumps(d))
    return d


def fetch_aare_bern(start, end):
    d = _get("https://aareguru.existenz.ch/v2018/history",
             {"city": "bern", "start": str(start), "end": str(end),
              "app": "tirex_compare", "version": "1.0"},
             key=f"aare_bern_{start}_{end}")
    tp = pd.DataFrame(d["data"]["temperature"], columns=["ts", "temperature"])
    tp["datetime"] = pd.to_datetime(tp["ts"], unit="s")
    tp = tp.sort_values("datetime").set_index("datetime")["temperature"]
    tp.index = tp.index.tz_localize("UTC").tz_convert("Europe/Zurich").tz_localize(None)
    return tp.resample("1h").mean()


def load_climatology():
    """Long-term hourly climatology from 2020-2025 (excluding 2026)."""
    df = pd.read_csv(BASE / "data" / "aare_temperature_hourly.csv", parse_dates=["datetime"])
    df = df[df["datetime"] < "2026-01-01"].copy()
    df["doy"] = df["datetime"].dt.dayofyear
    df["hour"] = df["datetime"].dt.hour
    clim = df.groupby(["doy", "hour"])["temperature"].agg(["mean", "std", "count"])
    return clim


def clim_predict(clim, idx):
    """Return climatology predictions for a DatetimeIndex."""
    doy = pd.Series(idx.dayofyear, index=idx)
    hour = pd.Series(idx.hour, index=idx)
    keys = list(zip(doy.values, hour.values))
    means = np.array([clim.loc[k, "mean"] if k in clim.index else np.nan for k in keys])
    return pd.Series(means, index=idx, name="climatology")


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Non-parametric bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(values)
    n = len(vals)
    boot = rng.choice(vals, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(vals.mean()), float(lo), float(hi), float(vals.std(ddof=1) / np.sqrt(n))


def main():
    rt = pd.read_csv(BASE / "aare_recursive_thun.csv", parse_dates=["t0"])
    print(f"Vorhandene Fenster: {len(rt)}")
    print(f"Zeitraum: {rt['t0'].min()} bis {rt['t0'].max()}")

    aare_start = (rt["t0"].min() - pd.Timedelta(hours=48)).date()
    aare_end = (rt["t0"].max() + pd.Timedelta(hours=HORIZON + 48)).date()
    print(f"Lade Aare-Bern {aare_start} bis {aare_end}")
    aare = fetch_aare_bern(aare_start, aare_end)
    print(f"  {len(aare)} h geladen, min={aare.index.min()} max={aare.index.max()}")

    clim = load_climatology()
    print(f"Klimatologie: {len(clim)} (DoY,Hour)-Zellen aus 2020-2025")

    rows = []
    for _, r in rt.iterrows():
        t0 = r["t0"]
        idx_fc = pd.date_range(t0, t0 + pd.Timedelta(hours=HORIZON - 1), freq="1h")
        truth = aare.reindex(idx_fc)
        if truth.isna().any():
            print(f"  skip {t0} (missing truth)")
            continue

        # Persistence: value at t0-1h broadcast over horizon
        last_obs = aare.reindex([t0 - pd.Timedelta(hours=1)]).iloc[0]
        if np.isnan(last_obs):
            # try t0-2h, t0-3h
            for lag in range(2, 6):
                v = aare.reindex([t0 - pd.Timedelta(hours=lag)]).iloc[0]
                if not np.isnan(v):
                    last_obs = v
                    break
        pred_persist = np.full(HORIZON, last_obs)
        mae_persist = float(np.mean(np.abs(pred_persist - truth.to_numpy())))

        # Climatology
        pred_clim = clim_predict(clim, idx_fc).to_numpy()
        mask = ~np.isnan(pred_clim)
        mae_clim = float(np.mean(np.abs(pred_clim[mask] - truth.to_numpy()[mask])))

        rows.append({"t0": t0, "n": len(idx_fc),
                     "persistence_MAE": mae_persist,
                     "climatology_MAE": mae_clim,
                     "last_obs": last_obs,
                     "truth_mean": float(truth.mean())})

    dfb = pd.DataFrame(rows)
    out_win = BASE / "src/baselines_per_window.csv"
    dfb.to_csv(out_win, index=False)
    print(f"\nBaselines per window -> {out_win}")

    merged = rt.merge(dfb[["t0", "persistence_MAE", "climatology_MAE"]], on="t0", how="left")
    merged_out = BASE / "src/all_models_per_window.csv"
    merged.to_csv(merged_out, index=False)

    metric_cols = [c for c in merged.columns if c.endswith("_MAE") and c != "stage1_thun_MAE"]
    print("\n" + "=" * 78)
    print("Aggregat + 95% Bootstrap-CI (n_boot=10000) ueber Fenster")
    print("=" * 78)
    summary = []
    for c in metric_cols:
        vals = merged[c].dropna().to_numpy()
        m, lo, hi, sem = bootstrap_ci(vals, n_boot=10000, alpha=0.05)
        summary.append({"model": c.replace("_MAE", ""), "n": len(vals), "MAE": m,
                        "CI_lo_95": lo, "CI_hi_95": hi, "SEM": sem})
        print(f"  {c.replace('_MAE',''):20s}  MAE={m:.4f}  95%CI=[{lo:.4f},{hi:.4f}]  n={len(vals)}")

    dfs = pd.DataFrame(summary).sort_values("MAE").reset_index(drop=True)
    out_sum = BASE / "src/baselines_summary.csv"
    dfs.to_csv(out_sum, index=False)
    print(f"\nSummary -> {out_sum}")

    print("\nRangliste (Bootstrap CI):")
    for _, r in dfs.iterrows():
        print(f"  {r['model']:20s}  MAE={r['MAE']:.4f}  [{r['CI_lo_95']:.4f}, {r['CI_hi_95']:.4f}]")


if __name__ == "__main__":
    main()
