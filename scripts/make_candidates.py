"""Liste aller 136 Kandidaten-Startzeitpunkte mit Status, Abflussquelle und Oraku-Version.

Logik unveraendert aus review_runde1/frozen_2026-10-14/make_frozen.py (Abschnitt
"Fensterliste"), dort beim Einfrieren am 14.10.2026 ausgefuehrt; hier als eigenes
Skript, damit die Datei results/windows_136_candidates.csv reproduzierbar ist.

Liest:   results/all_seasons_prev_runs_dst_summary.csv
         results/final_checks/oraku_model_versions.csv
Schreibt: results/windows_136_candidates.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, FINAL  # noqa: E402

s = pd.read_csv(RESULTS / "all_seasons_prev_runs_dst_summary.csv", parse_dates=["t0"])
mv = pd.read_csv(FINAL / "oraku_model_versions.csv", parse_dates=["t0"])
cand = pd.date_range("2025-11-04", "2026-08-01", freq="2D")
w = pd.DataFrame({"t0_local": cand})
w["status"] = ["ok" if t in set(s.t0) else "excluded_no_truth" for t in cand]
w = w.merge(s[["t0", "flow_source", "fc_start", "fc_end"]].rename(columns={"t0": "t0_local"}), on="t0_local", how="left")
w = w.merge(mv[["t0", "model", "last_updated"]].rename(columns={"t0": "t0_local", "model": "oraku_model", "last_updated": "oraku_last_updated"}), on="t0_local", how="left")
w["cet"] = w.t0_local < "2026-03-29"
w.to_csv(RESULTS / "windows_136_candidates.csv", index=False)
print(f"{len(w)} Kandidaten, davon ok: {(w.status == 'ok').sum()} -> results/windows_136_candidates.csv")
