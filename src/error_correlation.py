"""Analyse der Fehler-Korrelation und Bias-Varianz-Decomposition
fuer das Ensemble AO = 0.5*(A + Oraku).

Wenn Fehler-Korrelation zwischen A und Oraku niedrig ist, profitiert das
Ensemble stark. Zusaetzlich zerlegen wir den Fehler in Bias und Varianz.

Nutzt: aare_recursive_thun.csv (enthaelt Fenster-MAEs)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


BASE = Path(__file__).resolve().parent.parent


def main():
    df = pd.read_csv(BASE / "aare_recursive_thun.csv", parse_dates=["t0"])
    print(f"n = {len(df)} Fenster")

    # Fehler-Korrelationen (basierend auf Fenster-MAEs, nicht Residuen)
    pairs = [
        ("A_MAE", "Oraku_MAE"),
        ("A_MAE", "R_recursive_MAE"),
        ("R_recursive_MAE", "Oraku_MAE"),
    ]
    print("\n" + "=" * 70)
    print("PEARSON- und SPEARMAN-KORRELATION der Fenster-MAEs")
    print("=" * 70)
    for a, b in pairs:
        r_p, p_p = stats.pearsonr(df[a], df[b])
        r_s, p_s = stats.spearmanr(df[a], df[b])
        print(f"  {a:20s} vs {b:20s}  Pearson r={r_p:+.3f} (p={p_p:.3f})   "
              f"Spearman r={r_s:+.3f} (p={p_s:.3f})")

    # Ensemble-Effizienz: wenn Fehler unkorreliert, halbiert sich die Varianz.
    # Bei perfekter Korrelation bringt Mittelung nichts.
    print("\n" + "=" * 70)
    print("ENSEMBLE-EFFIZIENZ")
    print("=" * 70)
    var_A = df["A_MAE"].var()
    var_O = df["Oraku_MAE"].var()
    var_AO = df["AO_MAE"].var()
    # Erwartete Varianz bei linearer Kombination 0.5*A + 0.5*O:
    # var(0.5A + 0.5O) = 0.25*var(A) + 0.25*var(O) + 0.5*cov(A,O)
    cov_AO = df[["A_MAE", "Oraku_MAE"]].cov().iloc[0, 1]
    predicted_var = 0.25 * var_A + 0.25 * var_O + 0.5 * cov_AO
    print(f"  var(A)            = {var_A:.5f}")
    print(f"  var(Oraku)        = {var_O:.5f}")
    print(f"  cov(A, Oraku)     = {cov_AO:.5f}")
    print(f"  var(AO) observed  = {var_AO:.5f}")
    print(f"  var(AO) expected  = {predicted_var:.5f}")

    # Bei "keine Korrelation" der Fehler:
    var_AO_ideal = 0.25 * var_A + 0.25 * var_O
    print(f"  var(AO) if r=0    = {var_AO_ideal:.5f}   (Idealfall unabhaengiger Fehler)")

    # Mean-MAE-Vergleich
    print("\n  Mean-MAE:")
    print(f"    A     = {df['A_MAE'].mean():.4f} °C")
    print(f"    Oraku = {df['Oraku_MAE'].mean():.4f} °C")
    print(f"    AO    = {df['AO_MAE'].mean():.4f} °C")

    # Naïve arithmetisches Mittel der individuellen MAEs
    naive_mean_of_maes = 0.5 * (df["A_MAE"].mean() + df["Oraku_MAE"].mean())
    print(f"    (A+O)/2 naïv = {naive_mean_of_maes:.4f} °C")
    ensemble_gain = naive_mean_of_maes - df["AO_MAE"].mean()
    print(f"    Ensemble-Gewinn ggue naiven Mittelwerts: {ensemble_gain:.4f} °C")


if __name__ == "__main__":
    main()
