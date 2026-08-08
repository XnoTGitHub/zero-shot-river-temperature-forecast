"""Ergaenzt predictions_wide.csv um die 7 fehlenden Fenster (2026-07-13..19).

Ruft dieselbe TiRex-2-Funktion wie context_ablation_and_predictions.py auf,
aber nur Kontext=256 und nur A + T. Ergebnis wird an predictions_wide.csv
angefuegt.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

import numpy as np
import pandas as pd
from context_ablation_and_predictions import (
    fetch_aare_range, fetch_meteo, fetch_wind,
    LAT_BERN, LON_BERN, HORIZON, forecast, wind_alongshore, _fetch_oraku,
)


def main():
    pw = pd.read_csv(BASE / "src/predictions_wide.csv",
                     parse_dates=["t0", "datetime"])
    existing = set(pw["t0"].dt.date.tolist())

    rt = pd.read_csv(BASE / "aare_recursive_thun.csv", parse_dates=["t0"])
    all_starts = rt["t0"].tolist()
    missing = [t0 for t0 in all_starts if t0.date() not in existing]
    print(f"Vorhanden: {len(existing)} Fenster, fehlend: {len(missing)}")
    for t0 in missing:
        print(f"  {t0.date()}")

    if not missing:
        print("Nichts zu tun.")
        return

    aare_start = (missing[0] - pd.Timedelta(hours=600)).date()
    aare_end = (missing[-1] + pd.Timedelta(hours=HORIZON + 24)).date()
    print(f"\nLade Daten {aare_start} .. {aare_end} ...")
    df_bern = fetch_aare_range("bern", aare_start, aare_end)
    s_meteo = fetch_meteo(LAT_BERN, LON_BERN, aare_start, aare_end, "bern_ctx")
    df_wind = fetch_wind(aare_start, aare_end)

    new_rows = []
    for i, t0 in enumerate(missing):
        print(f"\n[{i+1}/{len(missing)}] TiRex-2 fuer t0={t0} (Kontext=256) ...")
        # Rebuild context inline mit erlaubtem laengeren Interpolationsfenster
        context = 256
        ctx_start = t0 - pd.Timedelta(hours=context)
        fc_end = t0 + pd.Timedelta(hours=HORIZON - 1)
        idx_ctx = pd.date_range(ctx_start, t0 - pd.Timedelta(hours=1), freq="1h")
        idx_fc = pd.date_range(t0, fc_end, freq="1h")
        idx_full = idx_ctx.append(idx_fc)

        truth = df_bern["temperature"].reindex(idx_fc)
        if truth.isna().any():
            print("  skip (truth-NaN)")
            continue
        # Lange Luecken (bis 30h) interpolieren fuer die 7 neuen Fenster
        temp_ctx = df_bern["temperature"].reindex(idx_ctx).interpolate(limit=50)
        flow_ctx = df_bern["flow"].reindex(idx_ctx).interpolate(limit=50).ffill().bfill()
        tt_full = s_meteo.reindex(idx_full).interpolate(limit=12).ffill().bfill()
        ws = df_wind["wind_speed"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
        wd = df_wind["wind_dir"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
        if temp_ctx.isna().any() or flow_ctx.isna().any():
            print(f"  skip (ctx-NaN nach Interp: temp={int(temp_ctx.isna().sum())} "
                  f"flow={int(flow_ctx.isna().sum())})")
            continue
        wind_full = wind_alongshore(ws.to_numpy(), wd.to_numpy())

        try:
            bafu = _fetch_oraku(t0, HORIZON, kind="flow").reindex(idx_fc).interpolate(limit=6)
            if bafu.isna().any():
                bafu = pd.Series(np.full(HORIZON, flow_ctx.to_numpy()[-24:].mean()),
                                 index=idx_fc)
        except Exception:
            bafu = pd.Series(np.full(HORIZON, flow_ctx.to_numpy()[-24:].mean()),
                             index=idx_fc)

        flow_past = flow_ctx.to_numpy()
        flow_full = np.concatenate([flow_past, bafu.to_numpy()])
        tt_past = tt_full.iloc[:context].to_numpy()
        tt_all = tt_full.to_numpy()

        pred_T = forecast(temp_ctx.to_numpy(),
                          past_channels=[tt_past],
                          future_channels=[tt_all], horizon=HORIZON)
        pred_A = forecast(temp_ctx.to_numpy(),
                          past_channels=[flow_past, tt_past, wind_full[:context]],
                          future_channels=[flow_full, tt_all, wind_full],
                          horizon=HORIZON)
        mae_T = float(np.mean(np.abs(pred_T - truth.to_numpy())))
        mae_A = float(np.mean(np.abs(pred_A - truth.to_numpy())))
        print(f"  T MAE = {mae_T:.3f}   A MAE = {mae_A:.3f}")
        for j, (dt_j, truth_j, tj, aj) in enumerate(
                zip(idx_fc, truth.to_numpy(), pred_T, pred_A)):
            new_rows.append({
                "t0": t0, "h": j + 1, "datetime": dt_j,
                "truth": truth_j, "T_pred": tj, "A_pred": aj})

    if not new_rows:
        print("Keine neuen Zeilen.")
        return

    df_new = pd.DataFrame(new_rows)
    df_full = pd.concat([pw, df_new], ignore_index=True).sort_values(
        ["t0", "h"]).reset_index(drop=True)
    df_full.to_csv(BASE / "src/predictions_wide.csv", index=False)
    print(f"\npredictions_wide.csv aktualisiert: {len(pw)} -> {len(df_full)} Zeilen")


if __name__ == "__main__":
    main()
