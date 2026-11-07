# Datenrechte und Lizenzen (Stand der Prüfung: 7.11.2026)

Dieses Dokument begründet, welche Dateien das Repositorium v1.1.0 enthält und welche nicht.
Grundsatz: Veröffentlicht werden der eigene Code, die eigenen Prognosen und die abgeleiteten
Kennzahlen sowie fremde Daten nur dann, wenn deren Bedingungen die Weitergabe ausdrücklich
erlauben. Fremde Daten mit ungeklärter Weitergabe bleiben außerhalb des Repositoriums; sie
werden für die Reproduktion über `scripts/fetch_inputs.py` beschafft und per SHA-256 geprüft.

## 1. Übersicht

| Kategorie | Rechteinhaber / Bedingungen (Beleg) | Weitergabe | In v1.1.0 |
|---|---|---|---|
| **Eigener Programmcode** (`scripts/`) | Autor | zulässig | ja, MIT (`LICENSE-CODE`) |
| **Eigene Prognosen und abgeleitete Ergebnisse** (`results_reference/`: TiRex-2-Prognosen je Fall und Stunde, Fehlerkennzahlen je Fall, Tests, Tabellen, Abbildungen) | Autor | zulässig | ja, CC BY 4.0 (`LICENSE-DATA`) |
| **Open-Meteo-Wettervorhersagen** (6 Dateien in `data/cache/`, Modell DWD ICON) | Open-Meteo: „API data are offered under Attribution 4.0 International (CC BY 4.0)“; „copy and redistribute the material in any medium or format“; Namensnennung erforderlich ([Lizenz](https://open-meteo.com/en/licence)). DWD-Vorhersagen ebenfalls CC BY. | zulässig mit Namensnennung | ja, unverändert, CC BY 4.0, „Weather data by Open-Meteo.com“ |
| **MeteoSchweiz-Messdaten** (Station BER, `data/meteoschweiz/`) | MeteoSchweiz Open Data, CC BY 4.0; Weitergabe „only … if the source is acknowledged“ ([Nutzungsbedingungen](https://opendatadocs.meteoswiss.ch/general/terms-of-use)) | zulässig mit Quellenangabe | ja, unverändert, „Quelle: MeteoSchweiz“ |
| **OpenStreetMap-Geometrie** (`data/osm/`, für Abb. 1) | © OpenStreetMap-Mitwirkende, Open Database License 1.0 | zulässig unter ODbL | ja, unverändert, ODbL 1.0 |
| **BAFU-Messwerte via aare.guru** (Wassertemperatur, Abfluss Bern 2135 und Thun 2030; 30 Abschnitte 2025–2026 und Klimatologiedatei 2020–2025) | BAFU: „Die Daten können frei genutzt werden. Die Angabe der Quelle wird empfohlen.“ ([FAQ](https://www.hydrodaten.admin.ch/de/fragen)); AGB BAFU Stand 16.9.2019, Ziff. 8: Nutzung „zu kommerziellen und nicht kommerziellen Zwecken“. **Bezogen wurden die Daten jedoch über aare.guru:** „Diese API ist öffentlich und darf für nicht-kommerzielle Zwecke frei benutzt werden. Sag uns bitte auf aaregurus@existenz.ch Bescheid … setz einen Link zurück auf Aare.guru sowie auf das BAFU.“ ([aare.guru](https://aareguru.existenz.ch/)). Keine Regelung zur Weiterveröffentlichung der Schnittstellenantworten. | **ungeklärt** (Antworten der aare.guru-Schnittstelle) | **nein**; Bezug aus v1.0.0 oder per Schnittstelle; Stundenwerte der Messung auch nicht in den Ergebnisdateien |
| **Prognosen des Referenzsystems Aare-Oraku** (129 Prognosen, 129 Abfragen der Modellversion) | Quellcode AGPL-3.0 ([Repositorium](https://github.com/Joelius300/aare-forecast)): „The deployed API has additional usage restrictions as you can see when you visit it.“ Die Schnittstellenseite ([aare-oraku.konzept.space](https://aare-oraku.konzept.space/)) nennt keine Nutzungs- oder Weitergabebedingungen; Kontakt oraku@aare.guru. | **ungeklärt** | **nein**; Bezug aus v1.0.0 oder per Schnittstelle; Stundenwerte auch nicht in den Ergebnisdateien |
| **BAFU-Abflussvorhersage via Aare-Oraku** (97 Fälle) | AGB BAFU Stand 16.9.2019, Ziff. 8: „Die hydrologischen Vorhersagen dürfen frei verwendet werden. … Über eine Veröffentlichung bzw. Weitergabe entscheidet der für das Gewässer zuständige Kanton.“ | **nicht ohne Zustimmung des Kantons Bern** | **nein**; nur Gütekennzahlen je Fall (`flow_forecast_quality.csv`) |
| **TiRex-2-Modellgewichte** (NX-AI) | Apache-2.0 (Modellkarte auf Hugging Face) | zulässig, aber nicht nötig | nein; Revision und SHA-256 in `scripts/paths.py`, Bezug beim Hersteller |

## 2. Was aus den eigenen Ergebnisdateien entfernt wurde

`scripts/make_public_reference.py` erzeugt den Referenzstand aus dem vollständigen eingefrorenen
Stand vom 14.10.2026 (192 Dateien, SHA-256 gesichert, beim Autor). Entfernt wurden:

- in den Stundendateien `all_seasons_prev_runs_dst_hourly.csv` und `all_seasons_hist_dst_hourly.csv`
  die Spalten `truth` (Messwert), `Oraku_pred` (Prognose Aare-Oraku) sowie `AO_pred` und `RO_pred`
  (Mittel aus TiRex-2 und Aare-Oraku, aus denen sich die Oraku-Prognose zurückrechnen ließe);
- die Dateien mit Tageswerten je Tag: `tables/daily_extremes_per_day.csv`, `tables/ctrl_tmax_per_day.csv`.

Alle Kennzahlen je Prognosefall (z. B. MAE von Aare-Oraku je Fall), alle Aggregate und alle Tests
bleiben enthalten. Damit sind die Aussagen des Beitrags prüfbar; die vollständige Neuberechnung
der Tabellen und Abbildungen benötigt die Eingangsdaten Dritter (`scripts/fetch_inputs.py`).

## 3. Kennzeichnung der Version 1.0.0

Version 1.0.0 (10.5281/zenodo.21850779) enthält die Schnittstellenantworten von aare.guru und
Aare-Oraku und kennzeichnet sie in `LICENSE-DATA` als „CC BY 4.0“; die Zenodo-Metadaten nennen
für den gesamten Datensatz die Softwarelizenz MIT. Beides ist für diese fremden Daten nicht belegt.
**Klarstellung:** Die Angaben MIT und CC BY 4.0 in v1.0.0 gelten nur für die eigenen Inhalte des
Autors (Code bzw. eigene Ergebnisse). Für die dort enthaltenen Daten Dritter gelten allein die
Bedingungen der jeweiligen Anbieter (Abschnitt 1). Für die Weiterveröffentlichung dieser Daten
liegt keine abschließende Rechteklärung vor; weder v1.0.0 noch v1.1.0 räumen an ihnen
Nutzungsrechte ein. Die Dateien einer veröffentlichten
Zenodo-Version lassen sich nicht ändern; v1.0.0 bleibt als historischer Stand der Ersteinreichung
unverändert zugänglich.

## 4. Quellenangaben bei Nachnutzung

- Messwerte: „Daten Oberflächengewässer: Abteilung Hydrologie, Bundesamt für Umwelt BAFU
  (Bezugsdatum 4.8.2026)“, bezogen über aare.guru (https://aare.guru).
- Wettervorhersagen: „Weather data by Open-Meteo.com“ (CC BY 4.0), Modell ICON des Deutschen Wetterdienstes.
- Lufttemperaturmessungen: „Quelle: MeteoSchweiz“ (CC BY 4.0).
- Kartengrundlage: „© OpenStreetMap-Mitwirkende“ (ODbL 1.0).
- Referenzprognosen: Aare-Oraku (https://aare-oraku.konzept.space, Quellcode https://github.com/Joelius300/aare-forecast).
