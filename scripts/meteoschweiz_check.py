# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/phase2b/meteoschweiz_check.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Reproduzierbare Berechnung der Lufttemperatur-Vergleichswerte (Kap. 2.1, Zusatzmaterial).

Quelle: MeteoSchweiz Open Government Data, automatisches Messnetz (ogd-smn), Station BER
(Bern/Zollikofen), Datei ogd-smn_ber_m.csv (Monatswerte; Kopie in literatur/quellen/,
abgerufen 14.10.2026 von https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ber/).
Parameter tre200m0 = Lufttemperatur 2 m über Boden, Monatsmittel (°C).

Aggregation:
  * Untersuchungszeitraum: arithmetisches Mittel der neun Monatsmittel Nov 2025 ... Jul 2026.
  * Vergleich: für jeden der neun Kalendermonate das arithmetische Mittel der Monatsmittel
    der Jahre 1991 bis 2020 (nicht homogenisierte Stationswerte), danach arithmetisches
    Mittel dieser neun Monatswerte.
  * Jahresmittel 1991-2020: Mittel der Jahreswerte tre200y0 aus ogd-smn_ber_y.csv.
Aufruf: python3 -I review_runde1/phase2b/meteoschweiz_check.py
"""
from pathlib import Path
import csv
from collections import defaultdict

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import METEOSCHWEIZ as Q  # noqa: E402


def read(fn, param):
    out = {}
    with open(Q / fn, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            ts = row["reference_timestamp"]  # dd.mm.yyyy HH:MM
            d, m, y = ts.split(" ")[0].split(".")
            v = row[param]
            if v != "":
                out[(int(y), int(m))] = float(v)
    return out


m = read("ogd-smn_ber_m.csv", "tre200m0")
y = read("ogd-smn_ber_y.csv", "tre200y0")

months = [(2025, 11), (2025, 12)] + [(2026, k) for k in range(1, 8)]
names = ["Nov", "Dez", "Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul"]
print("Monat  gemessen  Norm1991-2020  Differenz")
obs, norm = [], []
for (yy, mm), n in zip(months, names):
    o = m[(yy, mm)]
    ref = [m[(k, mm)] for k in range(1991, 2021) if (k, mm) in m]
    assert len(ref) == 30, (mm, len(ref))
    r = sum(ref) / len(ref)
    obs.append(o); norm.append(r)
    print(f"{n:4s} {yy}  {o:6.1f}   {r:6.1f}   {o-r:+6.1f}")
print(f"Mittel der neun Monate: gemessen {sum(obs)/9:.2f} °C, Norm 1991-2020 {sum(norm)/9:.2f} °C, "
      f"Differenz {sum(obs)/9-sum(norm)/9:+.2f} °C")
yr = [y[(k, 1)] for k in range(1991, 2021)]
print(f"Jahresmittel 1991-2020 (Mittel der Jahreswerte, n={len(yr)}): {sum(yr)/len(yr):.2f} °C")
print("Jahresmittel 2020-2025:", ", ".join(f"{k}: {y[(k,1)]:.1f}" for k in range(2020, 2026) if (k, 1) in y))
