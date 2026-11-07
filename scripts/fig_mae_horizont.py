# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/phase2b/fig_mae_horizont.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Abb. S3 (Zusatzmaterial): mittlerer absoluter Fehler nach Prognosestunde.
Daten: review_runde1/phase2a/tables/mae_by_horizon.csv (aus frozen_2026-10-14, n = 129).
Aufruf: python3 review_runde1/phase2b/fig_mae_horizont.py
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import TABLES, FIGURES  # noqa: E402
src = TABLES / "mae_by_horizon.csv"
FIGURES.mkdir(parents=True, exist_ok=True)
out = FIGURES / "abb_s3_mae_horizont"

rows = list(csv.DictReader(open(src)))
h = [int(r["h"]) for r in rows]
series = [
    ("Oraku", "Aare-Oraku", "#1b4f72", "--"),
    ("T", "TiRex-2 (Lufttemperatur, L)", "#b03a2e", ":"),
    ("AO", "Ensemble (LAW+Or.)", "#1e8449", "-"),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(6.3, 3.0), dpi=600)
for key, lab, col, ls in series:
    ax.plot(h, [float(r[key]) for r in rows], ls, color=col, lw=1.6, label=lab)
ax.axvline(24.5, color="0.6", lw=0.8)
ax.set_xlim(1, 48); ax.set_ylim(0, 0.65)
ax.set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
ax.set_xlabel("Prognosestunde h nach dem Start (Start 00:00 Uhr Lokalzeit)")
ax.set_ylabel("MAE [°C]")
ax.grid(True, lw=0.4, alpha=0.5)
ax.text(23.5, 0.03, "Tag 1", ha="right", va="bottom", color="0.4")
ax.text(25.5, 0.03, "Tag 2", ha="left", va="bottom", color="0.4")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(out.with_suffix(".png")); fig.savefig(out.with_suffix(".pdf"))
print("written", out)
