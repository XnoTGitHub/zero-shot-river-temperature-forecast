"""Residuen-Korrelation zwischen A (TiRex-2) und Oraku (Lineare Regression).

Die MAE-Korrelation (r=0.08) ist ein absoluter Fehlerindikator, ergaenzt hier
durch die theoretisch relevante Groesse: die Pearson-Korrelation der vorzeichen-
behafteten Residuen e_A = A - truth und e_O = Oraku - truth.

Die Ensemble-Varianz einer 1/2*(A + O)-Kombination haengt formal von cov(e_A, e_O)
ab, nicht vom MAE. Niedrige Residuen-Korrelation stuetzt daher die Ensemble-These
staerker als eine reine MAE-Korrelation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


BASE = Path(__file__).resolve().parent.parent
CACHE = BASE / "data" / "aare_dashboard_cache"
PW = BASE / "src/predictions_wide.csv"


def load_oraku(t0):
    fp = CACHE / f"oraku_temperature_{t0.isoformat()}_48.json"
    with open(fp) as f:
        d = json.load(f)
    times = pd.to_datetime(d["data"]["time"]).tz_localize(None)
    vals = np.array(d["data"]["value"], dtype=float)
    return pd.Series(vals, index=times)


def main():
    df = pd.read_csv(PW, parse_dates=["t0", "datetime"])
    t0s = sorted(df["t0"].unique())
    print(f"n Fenster in predictions_wide: {len(t0s)}")

    rows = []
    for t0 in t0s:
        sub = df[df["t0"] == t0].copy()
        try:
            oraku = load_oraku(pd.Timestamp(t0))
        except FileNotFoundError:
            print(f"  {t0}: Oraku-Cache fehlt, skip")
            continue
        sub["oraku_pred"] = sub["datetime"].map(oraku)
        if sub["oraku_pred"].isna().any():
            print(f"  {t0}: Oraku-NaN in Fenster, skip")
            continue
        sub["e_A"] = sub["A_pred"] - sub["truth"]
        sub["e_O"] = sub["oraku_pred"] - sub["truth"]
        sub["e_T"] = sub["T_pred"] - sub["truth"]
        rows.append(sub[["t0", "h", "datetime", "truth", "T_pred",
                         "A_pred", "oraku_pred", "e_A", "e_O", "e_T"]])

    if not rows:
        print("Keine Fenster verwertbar.")
        return
    full = pd.concat(rows, ignore_index=True)
    print(f"\nGesamt stuendliche Punkte: {len(full)}   Fenster: {full['t0'].nunique()}")

    print("\n" + "=" * 70)
    print("PEARSON-KORRELATION DER VORZEICHENBEHAFTETEN RESIDUEN (alle Stunden)")
    print("=" * 70)
    for name_a, name_b in [("e_A", "e_O"), ("e_A", "e_T"), ("e_T", "e_O")]:
        r, p = stats.pearsonr(full[name_a], full[name_b])
        print(f"  {name_a:6s} vs {name_b:6s}  r = {r:+.3f}  p = {p:.3e}   n = {len(full)}")

    # Zusaetzlich: robust nach Fenster (mittlere Fenster-Korrelation)
    print("\n" + "=" * 70)
    print("Pro-Fenster-Residuen-Korrelation e_A vs e_O (48h Fenster)")
    print("=" * 70)
    rs = []
    for t0, sub in full.groupby("t0"):
        if len(sub) < 5:
            continue
        r, _ = stats.pearsonr(sub["e_A"], sub["e_O"])
        rs.append(r)
        print(f"  {pd.Timestamp(t0).date()}   r = {r:+.3f}")
    print(f"\n  Mittelwert der Fenster-Korrelationen: {np.mean(rs):+.3f}"
          f"   Median: {np.median(rs):+.3f}"
          f"   IQR: [{np.percentile(rs, 25):+.3f}; {np.percentile(rs, 75):+.3f}]")

    # Kontrolle: Bias, Std der Residuen, cov
    print("\n" + "=" * 70)
    print("Bias / Streuung der Residuen (alle Stunden)")
    print("=" * 70)
    for c in ["e_A", "e_O", "e_T"]:
        e = full[c]
        print(f"  {c}:  mean = {e.mean():+.3f}   std = {e.std():.3f}   "
              f"MAE = {e.abs().mean():.3f}")

    cov = np.cov(full["e_A"], full["e_O"])[0, 1]
    var_A, var_O = full["e_A"].var(), full["e_O"].var()
    var_ens_theoretisch = 0.25 * var_A + 0.25 * var_O + 0.5 * cov
    e_ens = 0.5 * (full["e_A"] + full["e_O"])
    var_ens_beobachtet = e_ens.var()
    print(f"\n  var(e_A)          = {var_A:.4f}")
    print(f"  var(e_O)          = {var_O:.4f}")
    print(f"  cov(e_A, e_O)     = {cov:+.4f}")
    print(f"  var(e_AO) theor.  = {var_ens_theoretisch:.4f}")
    print(f"  var(e_AO) beob.   = {var_ens_beobachtet:.4f}   (Kontrolle)")
    print(f"  var(e_AO) if r=0  = {0.25 * var_A + 0.25 * var_O:.4f}   (Idealfall)")


if __name__ == "__main__":
    main()
