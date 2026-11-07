#!/usr/bin/env bash
# Vollstaendige Reproduktion der Ergebnisse des Beitrags (v1.1.0, 129 Prognosefaelle).
# Voraussetzungen (README.md, Abschnitt "Reproduktion"):
#   - zwei Umgebungen: .venv-forecast (requirements-forecast.txt) und .venv-analysis (requirements-analysis.txt)
#   - Eingangsdaten Dritter beschafft:  .venv-analysis/bin/python scripts/fetch_inputs.py --download-v100 --api
# Aufruf:  bash run_all.sh            (Rechenzeit ca. 20-25 min auf einem aktuellen Laptop, nur CPU)
set -euo pipefail
cd "$(dirname "$0")"
PYF="${PY_FORECAST:-.venv-forecast/bin/python}"
PYA="${PY_ANALYSIS:-.venv-analysis/bin/python}"
mkdir -p results/logs
run() {
  local py="$1"; shift
  local name; name="$(basename "$1" .py)"
  if [ $# -gt 1 ]; then name="${name}$(printf '_%s' "${@:2}" | tr -d '-')"; fi
  echo ">> $*"; "$py" -I "$@" 2>&1 | tee "results/logs/${name}.log"
}

echo "== 1 Eingangsdaten pruefen (SHA-256 gegen data/MANIFEST_inputs.csv)"
"$PYA" -I scripts/fetch_inputs.py --verify
echo "== 2 Zeitstempelkorrektur testen (Open-Meteo, Sommerzeit)"
run "$PYA" scripts/test_dst_relabel.py
echo "== 3 Prognoselaeufe TiRex-2 (Hauptauswertung, 6-h-Verzoegerung, Historical Forecast)"
run "$PYF" scripts/all_seasons_ensemble_prev_runs_dst.py --tag dst
run "$PYF" scripts/delay6h_prev_runs_dst.py --tag dst
run "$PYF" scripts/all_seasons_ensemble_hist_dst.py --tag dst
run "$PYF" scripts/tab1_kenngroessen.py
echo "== 4 Statistik (n = 129 und n = 128 ohne 28.3.2026)"
run "$PYA" scripts/prev_runs_stats_dst.py dst
run "$PYA" scripts/prev_runs_stats_dst.py dst _n128
echo "== 5 Abschlusspruefungen, Fensterliste, Alt/Neu-Vergleich"
run "$PYA" scripts/final_checks.py
run "$PYA" scripts/make_candidates.py
run "$PYA" scripts/compare_results.py
echo "== 6 Tabellen und Kontrollrechnungen"
run "$PYA" scripts/metrics_hydro.py
run "$PYA" scripts/controls_phase2a.py
run "$PYA" scripts/meteoschweiz_check.py
echo "== 7 Abbildungen"
for f in fig_karte fig_prognoseaufbau fig_hydro_v3 fig_ranking_saison fig_mae_horizont; do run "$PYA" "scripts/$f.py"; done
echo "== 8 Vergleich mit dem Referenzstand"
"$PYA" -I scripts/verify_reproduction.py | tee results/logs/verify_reproduction.log
