"""Erzeugt aus einem vollstaendigen Ergebnisordner den veroeffentlichbaren Referenzstand.

Regel (siehe docs/DATENRECHTE.md): Werte Dritter, deren Weiterveroeffentlichung nicht
geregelt ist, werden nicht auf Stunden- oder Tagesebene weitergegeben. Konkret:

  * Stundendateien: Spalten "truth" (BAFU-Messwert, bezogen ueber aare.guru),
    "Oraku_pred" (Prognose Aare-Oraku) sowie die daraus gemittelten Ensemble-
    Prognosen "AO_pred" und "RO_pred" werden entfernt. Die eigenen TiRex-2-
    Prognosen bleiben vollstaendig erhalten.
  * Dateien mit Tageswerten der Messung oder des Referenzsystems je Tag
    (Tagesextreme je Tag) werden nicht uebernommen.
  * Alle uebrigen Dateien enthalten nur Kennzahlen je Prognosefall oder
    Aggregate (MAE, Bias, Tests, Konfidenzintervalle) und werden unveraendert
    uebernommen.

Aufruf:  python make_public_reference.py <vollstaendiger_ergebnisordner> <zielordner>
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

DROP_COLS = ["truth", "Oraku_pred", "AO_pred", "RO_pred"]
HOURLY_FILES = ["all_seasons_prev_runs_dst_hourly.csv", "all_seasons_hist_dst_hourly.csv"]
EXCLUDE = {"tables/daily_extremes_per_day.csv", "tables/ctrl_tmax_per_day.csv"}


def build(full: Path, out: Path) -> list[str]:
    log = []
    for p in sorted(full.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(full).as_posix()
        if rel in EXCLUDE:
            log.append(f"ausgelassen (Tageswerte Dritter): {rel}")
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.name in HOURLY_FILES:
            # Werte als Text uebernehmen, damit keine Rundung durch Ein-/Auslesen entsteht
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
            drop = [c for c in DROP_COLS if c in df.columns]
            df.drop(columns=drop).to_csv(dst, index=False)
            log.append(f"Spalten entfernt {drop}: {rel}")
        else:
            shutil.copy2(p, dst)
    return log


if __name__ == "__main__":
    full, out = Path(sys.argv[1]), Path(sys.argv[2])
    for line in build(full, out):
        print(line)
