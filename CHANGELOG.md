# Änderungen

## v1.1.0 – korrigierte Auswertung für die überarbeitete Fassung des Beitrags

**Wissenschaftlicher Inhalt**
- Zeitstempelkorrektur der Open-Meteo-Wettervorhersagen: Die Archivdienste liefern Zeitstempel mit
  festem Versatz von 7 200 s zu UTC. v1.0.0 las sie als Lokalzeit; in der Winterzeit (MEZ) waren die
  Wettervorhersagen damit um eine Stunde verschoben. v1.1.0 rekonstruiert UTC und rechnet
  zeitzonenbewusst (`scripts/dst_relabel.py`, Tests `scripts/test_dst_relabel.py`).
- Eingabezeitraum (256 h) und Horizont (48 h) sind in tatsächlich verstrichenen Stunden definiert.
  Der Fall vom 28.3.2026 (Horizont über den Beginn der Sommerzeit) war in v1.0.0 fälschlich als
  lückenhaft ausgeschlossen und ist jetzt enthalten: **129 statt 128 Prognosefälle**.
- Auswirkung: Die mittleren MAE ändern sich um höchstens 0,003 °C; keine Schlussfolgerung ändert sich
  (`results_reference/compare_summary.txt`, `compare_tables.json`). Fälle in der Sommerzeit sind
  unverändert (Regressionsprüfung: 54 Fälle identisch).
- Neu ausgewertet: Modellversion des Referenzsystems je Fall, Güte der Abflussvorhersage,
  Robustheitsprüfungen (`scripts/final_checks.py`), Tabellen und Abbildungen der überarbeiteten
  Fassung (`metrics_hydro.py`, `controls_phase2a.py`, `fig_*.py`), Tab. 1 (`tab1_kenngroessen.py`,
  Rechenweg erstmals als Skript dokumentiert).

**Daten und Rechte**
- Die Schnittstellenantworten von aare.guru (BAFU-Messwerte) und Aare-Oraku (Prognosen,
  BAFU-Abflussvorhersage) werden nicht mehr mitveröffentlicht, weil ihre Weitergabe nicht geregelt
  ist; `scripts/fetch_inputs.py` beschafft sie aus v1.0.0 bzw. über die Schnittstellen und prüft
  431 Eingabedateien per SHA-256 (`data/MANIFEST_inputs.csv`).
- Lizenzangaben je Kategorie statt pauschal CC BY 4.0 (`LICENSE-DATA`, `docs/DATENRECHTE.md`).
  Die Angabe „CC BY 4.0“ für Aareguru-Daten in v1.0.0 war nicht belegt.

**Reproduzierbarkeit**
- Zwei exakt festgelegte Umgebungen statt einer unzutreffenden Paketliste: Die `requirements.txt` von
  v1.0.0 nannte `tirex-ts` 1.4.1 und `torch` 2.12.0 und ließ `tirex-2` aus; tatsächlich liefen die
  Prognosen mit `tirex-2` 0.1.1, `tirex-ts` 1.4.2 und `torch` 2.9.1 (`requirements-forecast.txt`),
  die Auswertung mit `pandas` 2.3.2 und `scipy` 1.16.2 (`requirements-analysis.txt`).
- Feste TiRex-2-Revision mit Prüfsummenkontrolle der Gewichte (`scripts/paths.py`); v1.0.0 lud den
  jeweils aktuellen Stand (bisher byte-identisch).
- Keine absoluten Pfade; ein Aufruf (`run_all.sh`) und ein automatischer Abgleich mit dem
  Referenzstand (`scripts/verify_reproduction.py`).

## v1.0.0 – Ersteinreichung (8.8.2026)

Auswertung mit 128 Prognosefällen; unverändert verfügbar unter 10.5281/zenodo.21850779.
