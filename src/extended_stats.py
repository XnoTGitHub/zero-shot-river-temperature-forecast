"""Statistische Tests auf der Extended Evaluation (40 Fenster, 3 Sommer)."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "src/extended_evaluation.csv")

print(f"n = {len(df)} Fenster ueber {df['year'].nunique()} Sommer (2023-2025)")
print()

pairs = [
    ("T_MAE", "persistence_MAE", "T (tt only) vs. Persistenz"),
    ("A_MAE", "persistence_MAE", "A (tt+flow+wind) vs. Persistenz"),
    ("R_MAE", "persistence_MAE", "R (recursive) vs. Persistenz"),
    ("T_MAE", "climatology_MAE", "T vs. Klimatologie"),
    ("T_MAE", "A_MAE", "T vs. A"),
    ("T_MAE", "R_MAE", "T vs. R"),
    ("R_MAE", "A_MAE", "R vs. A"),
]

print("=" * 90)
print("Wilcoxon signed rank (one-sided, H1: A<B), Cohen's d, Win rates")
print("=" * 90)
for a, b, label in pairs:
    mask = df[a].notna() & df[b].notna()
    va = df.loc[mask, a].to_numpy()
    vb = df.loc[mask, b].to_numpy()
    diff = va - vb
    try:
        w, p = stats.wilcoxon(va, vb, alternative="less")
    except Exception:
        w, p = np.nan, np.nan
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
    wins = int(np.sum(va < vb))
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {label:36s}  wins={wins:2d}/{len(va)}  d={np.mean(diff):+.4f}  p={p:.4f} {sig}  d={d:+.2f}")

print()
print("Per-year mean MAE:")
for year in sorted(df["year"].dropna().unique()):
    d = df[df["year"] == year]
    print(f"  {int(year)}  n={len(d):2d}   T={d['T_MAE'].mean():.3f}  A={d['A_MAE'].mean():.3f}"
          f"  R={d['R_MAE'].mean():.3f}  pers={d['persistence_MAE'].mean():.3f}"
          f"  clim={d['climatology_MAE'].mean():.3f}")
