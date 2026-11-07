"""Tab. 1 des Beitrags: Kenngroessen des Untersuchungszeitraums (Stundenwerte).

Fuer Tab. 1 lag bis v1.0.0 kein eigenes Skript vor; dieses Skript dokumentiert den
Rechenweg nachtraeglich. Es verwendet dieselben Lade- und Aufbereitungsfunktionen wie die
Prognoselaeufe (all_seasons_ensemble_prev_runs_dst.py: tz-bewusste Stundenmittel der
aare.guru-Messwerte, DST-korrigierte Open-Meteo-Vorhersagen) und laeuft deshalb in
Umgebung A (requirements-forecast.txt). TiRex-2 wird nicht geladen.

Zeitraum: 04.11.2025 00:00 bis 31.07.2026 23:00 Lokalzeit (Europe/Zurich), d. h. die
Prognosehorizonte aller Startzeitpunkte. Lufttemperatur = archivierte 24-h-Vorhersage
(Open-Meteo previous_day1) fuer Bern.

Schreibt: results/tables/tab1_kenngroessen.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import TABLES  # noqa: E402
from all_seasons_ensemble_prev_runs_dst import load_inputs, TZ  # noqa: E402

START = pd.Timestamp("2025-11-04 00:00").tz_localize(TZ)
END = pd.Timestamp("2026-07-31 23:00").tz_localize(TZ)

df_bern, df_thun, s_meteo_bern, _s_thun, _wind, _clim = load_inputs()


def stats(series, name, unit, digits, source):
    s = series[(series.index >= START) & (series.index <= END)].dropna()
    return {"groesse": name, "einheit": unit, "n_stunden": int(len(s)),
            "mittel": round(float(s.mean()), digits), "minimum": round(float(s.min()), digits),
            "maximum": round(float(s.max()), digits), "quelle": source}


rows = [
    stats(df_bern["temperature"], "Wassertemperatur Bern", "degC", 1, "BAFU via aare.guru"),
    stats(df_thun["temperature"], "Wassertemperatur Thun", "degC", 1, "BAFU via aare.guru"),
    stats(df_bern["flow"], "Abfluss Bern", "m3/s", 0, "BAFU via aare.guru"),
    stats(df_thun["flow"], "Abfluss Thun", "m3/s", 0, "BAFU via aare.guru"),
    stats(s_meteo_bern["tt_d1"], "Lufttemperatur Bern, 24-h-Vorhersage", "degC", 1, "Open-Meteo-Archiv"),
]
TABLES.mkdir(parents=True, exist_ok=True)
out = pd.DataFrame(rows)
out.to_csv(TABLES / "tab1_kenngroessen.csv", index=False)
print(out.to_string(index=False))
