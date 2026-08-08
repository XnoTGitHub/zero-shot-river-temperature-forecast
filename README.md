# Aare-Wassertemperaturprognose mit TiRex-2 — Reproduktions-Repository

Analyse-Code und gecachte Rohdaten zur Fallstudie:

> Neugebauer, F. (2026): *Zero-Shot-Foundation-Models und heterogene Ensembles
> für die operationelle Wassertemperaturprognose — Fallstudie Aare bei Bern.*
> Hydrologie und Wasserbewirtschaftung.

Alle im Paper zitierten Zahlen, Tabellen und Abbildungen lassen sich mit
diesem Repository ohne erneuten Zugriff auf die externen APIs reproduzieren.

## Verzeichnisstruktur

```
public/
├── README.md
├── LICENSE-CODE            MIT (Code in src/)
├── LICENSE-DATA            CC-BY-4.0 (Daten in data/ und figures/)
├── requirements.txt        Python 3.12.11, exakte Paketversionen
├── src/                    24 Analyse- und Plot-Skripte
├── data/
│   ├── aare_dashboard_cache/       gecachte API-Antworten (Aareguru,
│   │                                Open-Meteo Previous Runs & Historical
│   │                                Forecast, Aare-Oraku)
│   └── aare_temperature_hourly.csv Aareguru-Rohtabelle 2020-2025 (Baseline)
└── figures/                14 im Paper eingebundene PNGs (300 dpi)
```

## Setup

Getestet mit Python 3.12.11 (miniforge3) auf Apple Silicon, CPU-only.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# TiRex-2 Modellgewichte (einmalig, ~200 MB) auf die im Paper verwendete
# Revision festnageln:
huggingface-cli download NX-AI/TiRex-2 \
    --revision 05e5b26db52bfb256f1ae1bdf785589850482de3
```

## Reproduktion der Kern-Ergebnisse

Die Skripte finden Daten und Cache automatisch über
`Path(__file__).resolve().parent.parent` — es sind keine Pfad-Anpassungen
nötig, solange die Verzeichnisstruktur aus diesem Repo erhalten bleibt.

### Primär-Datensatz (Paper, Kap. 3)

```bash
# 1) Vorhersagen für alle 128 Fenster (nutzt gecachte API-Antworten)
python src/all_seasons_ensemble_prev_runs.py
#     -> src/all_seasons_prev_runs_summary.csv
#     -> src/all_seasons_prev_runs_hourly.csv

# 2) Statistische Auswertung (Wilcoxon, Holm, Bootstrap, TOST)
python src/all_seasons_prev_runs_stats.py
#     -> src/prev_runs_ranking.csv, prev_runs_tests.csv,
#        prev_runs_holm.csv, prev_runs_block_bootstrap.csv,
#        prev_runs_tost.csv, prev_runs_bias_by_season.csv,
#        prev_runs_thinning.csv

# 3) Abbildungen 5-8
python src/all_seasons_prev_runs_figures.py
#     -> figures/abb05_ranking_ci.png ...
```

### Vergleich Historical-Forecast (Paper, Kap. 3.1)

```bash
python src/all_seasons_ensemble.py         # Vorhersagen mit HF-Kovariaten
python src/all_seasons_stats.py            # Statistik
python src/all_seasons_figures.py          # Abbildungen
```

### Ergänzende Analysen

| Skript                                | Zweck                                             |
|---------------------------------------|---------------------------------------------------|
| `baselines_and_bootstrap.py`          | Naive Baselines (Persistence, Klimatologie)       |
| `context_ablation_and_predictions.py` | Kovariat-Ablation                                 |
| `delay6h_prev_runs.py`                | Stresstest: künstliche 6-h-Kovariatverzögerung    |
| `xgboost_baseline.py`                 | XGBoost-Vergleichs-Baseline                       |
| `error_correlation.py`                | Fehlerkorrelation zwischen Modellen               |
| `residual_correlation.py`             | Residuen-Analyse                                  |
| `reviewer_robustness.py`              | Reviewer-Sensitivitätsanalysen (Kap. 3, Anhang)   |
| `session17_delay_sensitivity.py`      | Sensitivität der 6-h-Verzögerung                  |
| `session18_holm.py`                   | Holm-Korrektur über Primärtests                   |
| `data_availability_probe.py`          | Ausgeschlossene Fenster identifizieren            |
| `extended_evaluation.py`              | Erweiterte Auswertung (Jahresverlauf)             |
| `make_figures.py` / `make_final_figures.py` / `make_more_figures.py` | Karten- und Schema-Abbildungen |

## Datenquellen und Lizenzen

- **Wasser-Temperatur & Abfluss Aare**: Aareguru-API
  (`aareguru.existenz.ch/v2018/history`), CC-BY 4.0.
- **Meteorologische Vorhersage (archiviert, keine Reanalyse)**: Open-Meteo
  Previous Runs API (`previous-runs-api.open-meteo.com`), zum Vergleich auch
  Historical-Forecast API (`historical-forecast-api.open-meteo.com`), CC-BY 4.0.
- **BAFU-Abflussvorhersage**: bereitgestellt über Aare-Oraku-API
  (`aare-oraku.konzept.space`), öffentlich zugänglich.
- **TiRex-2 Modellgewichte**: `huggingface.co/NX-AI/TiRex-2`, verwendete
  Revision `05e5b26db52bfb256f1ae1bdf785589850482de3` (Stand 2026-07-21).

Alle im Paper zitierten Ergebnisse basieren auf den 128 nicht überlappenden
48-h-Fenstern zwischen 2025-11-04 und 2026-08-01 (Startabstand 2 d). Die
8 wegen Datenlücken ausgeschlossenen Kandidatenfenster sind in
`data_availability_probe.py` dokumentiert.

## Lizenz

- Code (`src/`): MIT — siehe `LICENSE-CODE`.
- Daten (`data/`, `figures/`, alle abgeleiteten CSV/PNG): CC-BY 4.0 — siehe
  `LICENSE-DATA`.

## Zitierung

Bibliographischer Eintrag folgt nach Erscheinen des HyWa-Artikels. Ein
persistenter DOI wird über Zenodo bereitgestellt.
