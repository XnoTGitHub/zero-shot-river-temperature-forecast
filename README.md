# Wassertemperaturprognosen für die Aare bei Bern – Reproduktions-Repositorium (v1.1.0)

Code, eigene Prognosen und abgeleitete Ergebnisse zum Beitrag

> Neugebauer, F.: *Wassertemperaturprognosen für die Aare bei Bern mit einem vortrainierten
> Zeitreihenmodell: Vergleich und Kombination mit Aare-Oraku.* Hydrologie und
> Wasserbewirtschaftung (eingereicht, überarbeitete Fassung).

| Version | Inhalt | DOI |
|---|---|---|
| **v1.1.0** (diese) | korrigierte Auswertung, **129 Prognosefälle**, eingefrorener Datenstand vom 14.10.2026 | von Zenodo vergeben, über die Concept-DOI auffindbar |
| v1.0.0 (8.8.2026) | Auswertung der Ersteinreichung, 128 Prognosefälle, unverändert als historischer Stand | 10.5281/zenodo.21850779 |
| alle Versionen | Concept-DOI | 10.5281/zenodo.21850778 |

**Was sich gegenüber v1.0.0 geändert hat** (Einzelheiten in `CHANGELOG.md`): Der Archivdienst von
Open-Meteo liefert Zeitstempel mit festem Versatz von 7 200 s zu UTC (ohne Winterzeit). In v1.0.0
wurden sie als Lokalzeit gelesen; in der Winterzeit waren die Wettervorhersagen dadurch um eine
Stunde verschoben, und der Fall vom 28.3.2026 wurde fälschlich als lückenhaft ausgeschlossen. In
v1.1.0 werden die Zeitstempel über UTC rekonstruiert (`scripts/dst_relabel.py`), Eingabezeitraum
und Horizont sind in tatsächlich verstrichenen Stunden definiert, und es ergeben sich 129 Fälle.
Die mittleren MAE ändern sich um höchstens 0,003 °C, keine Schlussfolgerung ändert sich
(`results_reference/compare_summary.txt`).

## Inhalt und Datenrechte

| Ordner | Inhalt | Lizenz |
|---|---|---|
| `scripts/` | alle Skripte der Auswertung (Prognoseläufe, Zeitstempelkorrektur, Statistik, Tabellen, Abbildungen, Prüfungen) | MIT |
| `results_reference/` | eingefrorener Referenzstand: eigene TiRex-2-Prognosen je Fall und Stunde, Kennzahlen je Fall aller Modelle, Tests, Konfidenzintervalle, Tabellen, Abbildungen | CC BY 4.0 |
| `data/cache/` | Open-Meteo-Wettervorhersagen (6 Dateien, unverändert) | CC BY 4.0 (Open-Meteo) |
| `data/meteoschweiz/` | Monats- und Jahreswerte Station Bern/Zollikofen (Tab. S9) | CC BY 4.0 (MeteoSchweiz) |
| `data/osm/` | Gewässergeometrie für Abb. 1 | ODbL 1.0 (OpenStreetMap) |
| `data/MANIFEST_inputs.csv` | alle 431 Eingabedateien mit SHA-256, Herkunft, Rechtestatus und Bezugsweg | CC BY 4.0 |
| `docs/` | Datenrechte, Diff der Skripte gegenüber dem eingefrorenen Stand | – |

**Nicht enthalten** sind die Messwerte des BAFU, bezogen über die Schnittstelle von aare.guru,
und die Prognosen des Referenzsystems Aare-Oraku einschließlich der darüber bezogenen
BAFU-Abflussvorhersage, weil ihre Weiterveröffentlichung nicht geregelt ist
(`docs/DATENRECHTE.md`). Aus demselben Grund fehlen in den Stundendateien von
`results_reference/` die Spalten Messwert, Oraku-Prognose und Ensemble. Für die Reproduktion
werden diese Daten lokal beschafft und gegen die Prüfsummen geprüft (Schritt 1).

**Rechtehinweis zu den Daten Dritter in Version 1.0.0.** Ein Teil dieser fremden Rohdaten ist
bereits in Version 1.0.0 archiviert. Das betrifft die Antworten der Schnittstellen von aare.guru
(BAFU-Messwerte) und Aare-Oraku (Prognosen und BAFU-Abflussvorhersage). Für ihre
Weiterveröffentlichung liegt **keine abschließende Rechteklärung** vor. Die Lizenzangaben in
v1.0.0 (MIT für den Datensatz, CC BY 4.0 in `LICENSE-DATA`) gelten nur für die eigenen Inhalte
des Autors, nicht für diese Daten. Weder v1.0.0 noch dieses Repositorium räumen an den Daten
Dritter Nutzungsrechte ein. Der Abruf mit `scripts/fetch_inputs.py` dient allein der Prüfung und
Reproduktion dieser Auswertung. Für jede weitere Nutzung gelten ausschließlich die Bedingungen der
Anbieter, die die Nutzenden selbst zu prüfen haben; bei aare.guru ist das z. B. nur
nichtkommerzielle Nutzung, und über die Weitergabe der BAFU-Abflussvorhersage entscheidet der
zuständige Kanton (`docs/DATENRECHTE.md`).

## Reproduktion

Getestet auf macOS (Apple Silicon), Python 3.12.11, nur CPU. Rechenzeit insgesamt etwa 20–25 min (Testlauf 7.11.2026: 21 min).

```bash
# 0) Zwei Umgebungen (die Prognoseläufe und die Auswertung wurden mit unterschiedlichen
#    Paketständen gerechnet; beide sind exakt festgehalten)
python3.12 -m venv .venv-forecast && .venv-forecast/bin/pip install -r requirements-forecast.txt
python3.12 -m venv .venv-analysis && .venv-analysis/bin/pip install -r requirements-analysis.txt

# 1) Eingangsdaten Dritter beschaffen und prüfen:
#    287 Dateien aus Zenodo v1.0.0, 131 Dateien über die Schnittstelle von Aare-Oraku
#    (setzt deren Erreichbarkeit voraus, siehe „Abhängigkeit von externen Quellen“)
.venv-analysis/bin/python scripts/fetch_inputs.py --download-v100 --api
#    -> "Eingangsdaten: 431/431 pruefsummengleich"

# 2) Alles rechnen und mit dem Referenzstand vergleichen
bash run_all.sh
#    -> results/ ; am Ende: scripts/verify_reproduction.py
```

Die Modellgewichte von TiRex-2 (NX-AI, Apache-2.0, ca. 380 MB) werden beim ersten Lauf von
Hugging Face geladen. Verwendet wird die Revision `05e5b26db52bfb256f1ae1bdf785589850482de3`;
vor dem Lauf wird die SHA-256 der Gewichtsdatei geprüft (`scripts/paths.py`). Die Rechnung ist
auf der CPU deterministisch.

### Wichtig: Abhängigkeit von externen Quellen

**Die vollständige Reproduktion hängt von der Verfügbarkeit der Schnittstelle von Aare-Oraku ab.**
`run_all.sh` setzt voraus, dass alle 431 Eingabedateien mit der richtigen Prüfsumme vorliegen, und
bricht sonst in Schritt 1 ab. Zwei externe Quellen sind dafür nötig:

| Quelle | Dateien | Verfügbarkeit |
|---|---|---|
| Zenodo v1.0.0 (10.5281/zenodo.21850779) | 287: Messwerte über aare.guru, 256 Oraku-Prognosen, Klimatologie | dauerhaft archiviert |
| Schnittstelle von Aare-Oraku (aare-oraku.konzept.space) | 131: zwei Prognosen für den 28.3.2026, 129 Abfragen der Modellversion | am 7.11.2026 abrufbar und byte-identisch mit dem verwendeten Stand; privat betriebenes Projekt ohne Verfügbarkeitszusage |

Ist die Oraku-Schnittstelle nicht mehr erreichbar oder liefert sie andere Antworten, lässt sich
die Auswertung mit diesem Repositorium **nicht mehr vollständig reproduzieren**. Die Prognoseläufe
brechen dann beim Fall vom 28.3.2026 ab, und die Zuordnung der Oraku-Versionen (Tab. S1, S2) fehlt.
Eine korrigierte Auswertung ohne diesen Fall (n = 128) wäre nur mit angepassten Skripten möglich.
Prüfbar bleiben in diesem Fall:
- alle Kennzahlen je Fall, Tests, Konfidenzintervalle, Tabellen und Abbildungen im eingefrorenen
  Referenzstand `results_reference/`;
- die eigenen TiRex-2-Prognosen je Fall und Stunde, einschließlich der Neuberechnung mit den
  Eingangsdaten aus v1.0.0;
- die ursprüngliche Auswertung der Ersteinreichung (128 Fälle, ohne Zeitstempelkorrektur) mit
  Version 1.0.0.

Die Prüfsummen in `data/MANIFEST_inputs.csv` zeigen bei jedem später beschafften Datenstand
eindeutig, ob er mit dem verwendeten übereinstimmt.

**Weitere Einschränkungen:**
- Die Messwerte sind provisorische Daten zum Stand des Abrufs am 4.8.2026. Ein erneuter Abruf
  über aare.guru (`--refetch-aareguru`) kann wegen nachträglicher Revisionen durch das BAFU
  abweichende Werte liefern. Maßgebend sind die Dateien aus v1.0.0 (Prüfsummen im Manifest).
- Die Statistik- und Tabellenskripte benötigen die vollständigen Stundendateien mit Messwerten und
  Oraku-Prognosen. Mit den veröffentlichten Stundendateien allein (ohne diese Spalten) laufen sie
  nicht; die Tests der Hauptvergleiche lassen sich aber aus `all_seasons_prev_runs_dst_summary.csv`
  (Fehler je Fall) nachrechnen.
- Die exakte Gleichheit der TiRex-2-Prognosen ist für macOS auf Apple Silicon belegt. Auf anderen
  Plattformen sind kleine numerische Abweichungen möglich.

## Zuordnung zu Tabellen und Abbildungen des Beitrags

| Beitrag | Datei in `results/` bzw. `results_reference/` | Skript |
|---|---|---|
| Tab. 1 Kenngrößen | `tables/tab1_kenngroessen.csv` | `tab1_kenngroessen.py` |
| Tab. 2 Modelle | (beschreibend) | – |
| Tab. 3 Prognosegüte, Konfidenzintervalle | `tables/tab1_metrics_by_season.csv`, `tables/tab1b_paired_diffs_vs_oraku.csv`, `stats_dst/prev_runs_ranking.csv`, `stats_dst/prev_runs_block_bootstrap.csv` | `metrics_hydro.py`, `controls_phase2a.py`, `prev_runs_stats_dst.py` |
| Hauptvergleiche (Holm), Wilcoxon, Äquivalenztest | `stats_dst/prev_runs_holm.csv`, `prev_runs_tests.csv`, `prev_runs_tost.csv` | `prev_runs_stats_dst.py` |
| Tab. 4 Tagesmaxima/-minima | `tables/tab2_daily_extremes.csv`, `tables/tab2b_daily_extremes_paired.csv`, `tables/ctrl_tmax_sensitivity.csv` | `metrics_hydro.py`, `controls_phase2a.py` |
| Abb. 1 Karte | `figures/abb01_karte.png` | `fig_karte.py` |
| Abb. 2 Prognoseaufbau | `figures/abb02_prognoseaufbau.png` | `fig_prognoseaufbau.py` |
| Abb. 3–5 Beispiele, Frühlingsverlauf, Bias im Tagesgang | `figures/abb03_*.png` bis `abb05_*.png`, `tables/bias_by_hour_season.csv` | `fig_hydro_v3.py` |
| Abb. S1, S2 Ranking, Verteilung je Jahreszeit | `figures/abb_ranking_n129.png`, `figures/abb_saison_boxplot_n129.png` | `fig_ranking_saison.py` |
| Abb. S3 MAE nach Prognosestunde | `figures/abb_s3_mae_horizont.png`, `tables/mae_by_horizon.csv` | `fig_mae_horizont.py` |
| Tab. S1, S2 Versionen des Referenzsystems, Abflussvorhersage | `tables/tab5_versions_flow.csv`, `tables/ctrl_afternoon_bias_version*.csv`, `final_checks/oraku_model_versions.csv` | `metrics_hydro.py`, `controls_phase2a.py`, `final_checks.py` |
| Tab. S3 Schwellenwerte (illustrativ) | `tables/tab3_thresholds_illustrative.csv`, `tables/ctrl_thresholds_defined.csv` | `metrics_hydro.py`, `controls_phase2a.py` |
| Tab. S4 Zwischenprognose Thun | `tables/tab4_thun_stage1.csv` | `metrics_hydro.py` |
| Tab. S5 Robustheit (Teilmengen) | `final_checks/final_checks.log`, `stats_dst/prev_runs_thinning.csv`, `stats_dst_n128/` | `final_checks.py`, `prev_runs_stats_dst.py` |
| Tab. S6 Verzögerte Wettervorhersagen | `delay6h_prev_runs_dst_summary.csv`, `final_checks/final_checks.log` | `delay6h_prev_runs_dst.py`, `final_checks.py` |
| Tab. S7 Historical Forecast | `all_seasons_hist_dst_summary.csv`, `compare_summary.txt` | `all_seasons_ensemble_hist_dst.py`, `compare_results.py` |
| Tab. S8 Explorative Sommerauswertung | `results_reference/v1.0.0_n128/baselines_summary.csv`, `summer_extra_tests.csv` (unverändert aus v1.0.0) | v1.0.0: `src/baselines_and_bootstrap.py` |
| Tab. S9 MeteoSchweiz | Bildschirmausgabe, `logs/meteoschweiz_check.log` | `meteoschweiz_check.py` |
| Vergleich alt/neu (Zusatzmaterial, Zeitstempel) | `compare_summary.txt`, `compare_tables.json` | `compare_results.py` |

Die Dateinamen der Skripte entsprechen dem eingefrorenen Arbeitsstand; gegenüber diesem sind nur
die Pfade (zentral in `scripts/paths.py`) und die feste Revision der Modellgewichte geändert
(vollständiger Diff: `docs/CODE_CHANGES.diff`). Die Statistikordner heißen hier `stats_dst/`
(n = 129) und `stats_dst_n128/` (ohne 28.3.2026), im eingefrorenen Stand `stats_n129/` und `stats_n128/`.

## Zeitstempel und Sommerzeit

- Messwerte (aare.guru): Unix-Zeit → UTC → Europe/Zurich, Stundenmittel zeitzonenbewusst.
- Open-Meteo: Zeitstempel minus 7 200 s = UTC → Europe/Zurich (`dst_relabel.py`, Tests in `test_dst_relabel.py`).
- Startzeitpunkte 00:00 Lokalzeit; Eingabezeitraum 256 h und Horizont 48 h in verstrichenen Stunden.
  Der Fall vom 28.3.2026 überspannt den Beginn der Sommerzeit und teilt mit dem Fall vom 30.3.2026
  genau einen Messzeitpunkt (30.3. 00:00 MESZ). Die Stundendateien enthalten Lokalzeit mit Versatz
  (`datetime`) und UTC (`datetime_utc`).

## Quellenangaben

Messwerte: Abteilung Hydrologie, Bundesamt für Umwelt BAFU (Bezugsdatum 4.8.2026), über
[aare.guru](https://aare.guru). Wettervorhersagen: Weather data by [Open-Meteo.com](https://open-meteo.com)
(CC BY 4.0), DWD ICON. Lufttemperatur: Quelle: MeteoSchweiz. Karte: © OpenStreetMap-Mitwirkende.
Referenzsystem: [Aare-Oraku](https://aare-oraku.konzept.space). Modell: TiRex-2 (NX-AI).
