"""Vergleicht die reproduzierten Ergebnisse (results/) mit dem Referenzstand (results_reference/).

Fuer jede Referenzdatei wird die gleichnamige Datei in results/ geprueft:
  * CSV: gleiche Zeilenzahl; je Spalte der Referenz Textgleichheit, sonst groesste
    absolute Abweichung (numerische Spalten). Stufen: "identisch" (alle Zellen
    textgleich), "numerisch gleich" (max |Delta| <= 1e-9), "ABWEICHUNG".
  * JSON: rekursiver Vergleich mit derselben Toleranz.
  * TXT/LOG: Textvergleich (Abweichungen werden mit Zeilenzahl gemeldet).
  * PNG: Pruefsumme; bei Unterschied Pixelvergleich (falls Pillow installiert).
PDF-Abbildungen werden nicht verglichen (sie enthalten Erstellungszeitpunkte).

Aufruf: python scripts/verify_reproduction.py [--results results] [--reference results_reference]
Rueckgabewert 0, wenn keine CSV/JSON-Abweichung ueber der Toleranz auftritt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, REFERENCE  # noqa: E402

TOL = 1e-9
SKIP_DIRS = {"v1.0.0_n128"}


def cmp_csv(ref: Path, new: Path):
    a = pd.read_csv(ref, dtype=str, keep_default_na=False)
    b = pd.read_csv(new, dtype=str, keep_default_na=False)
    if len(a) != len(b):
        return "ABWEICHUNG", f"Zeilen {len(a)} vs {len(b)}"
    missing = [c for c in a.columns if c not in b.columns]
    if missing:
        return "ABWEICHUNG", f"Spalten fehlen: {missing}"
    worst, worst_col, textdiff = 0.0, "", []
    for c in a.columns:
        if (a[c].values == b[c].values).all():
            continue
        x = pd.to_numeric(a[c].replace("", np.nan), errors="coerce")
        y = pd.to_numeric(b[c].replace("", np.nan), errors="coerce")
        num_ok = x.notna() | (a[c] == "")
        if num_ok.all() and (x.isna() == y.isna()).all():
            d = float(np.nanmax(np.abs(x - y))) if x.notna().any() else 0.0
            if d > worst:
                worst, worst_col = d, c
        else:
            # nicht numerisch: Zeitstempel o. Ae. nach Normalisierung vergleichen
            try:
                ta = pd.to_datetime(a[c], utc=True, format="mixed")
                tb = pd.to_datetime(b[c], utc=True, format="mixed")
                if (ta == tb).all():
                    continue
            except (ValueError, TypeError):
                pass
            textdiff.append(c)
    if textdiff:
        return "ABWEICHUNG", f"Text abweichend in {textdiff}"
    if worst == 0.0:
        return "identisch", ""
    if worst <= TOL:
        return "numerisch gleich", f"max |Δ| = {worst:.1e} ({worst_col})"
    return "ABWEICHUNG", f"max |Δ| = {worst:.3g} ({worst_col})"


def _jdiff(x, y, path="") -> list[str]:
    if isinstance(x, dict) and isinstance(y, dict):
        out = []
        for k in x:
            out += _jdiff(x[k], y.get(k, "<fehlt>"), f"{path}/{k}")
        return out
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return [f"{path}: Laenge {len(x)} vs {len(y)}"]
        out = []
        for i, (u, v) in enumerate(zip(x, y)):
            out += _jdiff(u, v, f"{path}[{i}]")
        return out
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        if (isinstance(x, float) and math.isnan(x)) and (isinstance(y, float) and math.isnan(y)):
            return []
        return [] if abs(x - y) <= TOL else [f"{path}: {x} vs {y}"]
    return [] if x == y else [f"{path}: {x!r} vs {y!r}"]


def cmp_json(ref: Path, new: Path):
    d = _jdiff(json.loads(ref.read_text()), json.loads(new.read_text()))
    return ("identisch", "") if not d else ("ABWEICHUNG", f"{len(d)} Werte, z. B. {d[0]}")


def cmp_text(ref: Path, new: Path):
    a, b = ref.read_text().splitlines(), new.read_text().splitlines()
    if a == b:
        return "identisch", ""
    n = sum(1 for u, v in zip(a, b) if u != v) + abs(len(a) - len(b))
    return "Text abweichend", f"{n} Zeilen"


def cmp_png(ref: Path, new: Path):
    if hashlib.sha256(ref.read_bytes()).digest() == hashlib.sha256(new.read_bytes()).digest():
        return "identisch", ""
    try:
        from PIL import Image
    except ImportError:
        return "Datei abweichend", "Pillow nicht installiert"
    a = np.asarray(Image.open(ref).convert("RGB"), dtype=int)
    b = np.asarray(Image.open(new).convert("RGB"), dtype=int)
    if a.shape != b.shape:
        return "Bild abweichend", f"Groesse {a.shape} vs {b.shape}"
    diff = np.abs(a - b).max(axis=2)
    share = float((diff > 0).mean())
    return ("pixelgleich", "") if share == 0 else ("Bild abweichend", f"{share:.4%} der Pixel, max Δ {diff.max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    a = ap.parse_args()
    rows, hard_fail = [], 0
    for ref in sorted(a.reference.rglob("*")):
        if ref.is_dir() or ref.suffix == ".pdf":
            continue
        rel = ref.relative_to(a.reference)
        if rel.parts[0] in SKIP_DIRS or ref.name == "README.md":
            continue
        new = a.results / rel
        if not new.exists():
            status, note = "FEHLT", ""
        elif ref.suffix == ".csv":
            status, note = cmp_csv(ref, new)
        elif ref.suffix == ".json":
            status, note = cmp_json(ref, new)
        elif ref.suffix == ".png":
            status, note = cmp_png(ref, new)
        else:
            status, note = cmp_text(ref, new)
        if status in ("ABWEICHUNG", "FEHLT") and ref.suffix in (".csv", ".json"):
            hard_fail += 1
        rows.append((rel.as_posix(), status, note))
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[0]:<{w}}  {r[1]:<17} {r[2]}")
    counts = pd.Series([r[1] for r in rows]).value_counts().to_dict()
    print("\nZusammenfassung:", counts)
    print("ERGEBNIS:", "alle Tabellen/Kennzahlen reproduziert" if not hard_fail else f"{hard_fail} CSV/JSON-Dateien abweichend oder fehlend")
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
