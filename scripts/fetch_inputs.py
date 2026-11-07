"""Beschafft und prueft die Eingangsdaten Dritter, die nicht im Repositorium liegen.

Hintergrund (docs/DATENRECHTE.md): Die Antworten der Schnittstellen von aare.guru
(BAFU-Messwerte) und Aare-Oraku (Prognosen des Referenzsystems, BAFU-Abflussvorhersage)
werden in v1.1.0 nicht weiterveroeffentlicht, weil ihre Weitergabe durch Dritte nicht
geregelt ist. Dieses Skript stellt sie fuer die Reproduktion lokal bereit und prueft
jede Datei gegen die SHA-256-Pruefsummen in data/MANIFEST_inputs.csv.

Quellen (in dieser Reihenfolge):
  1. das Zenodo-Archiv v1.0.0 (10.5281/zenodo.21850779), das 293 der 295 Eingabedateien
     der Prognoseläufe enthaelt (Option --v100-zip PFAD oder --download-v100);
  2. die oeffentliche Schnittstelle von Aare-Oraku fuer die uebrigen Dateien
     (2 Prognosen fuer t0 = 28.3.2026, 129 Abfragen der Modellversion; Option --api).
     Die Parameter entsprechen exakt den urspruenglichen Abrufen.
  3. optional --refetch-aareguru: erneuter Abruf der Messwerte ueber aare.guru, falls
     v1.0.0 nicht verfuegbar ist. Die Messwerte sind provisorisch und werden vom BAFU
     nachtraeglich revidiert; Pruefsummen-Abweichungen sind dann zu erwarten und werden
     gemeldet (die Ergebnisse koennen sich geringfuegig aendern).

Beispiele:
  python scripts/fetch_inputs.py --download-v100 --api      # alles beschaffen und pruefen
  python scripts/fetch_inputs.py --verify                   # nur pruefen
Nutzungshinweis aare.guru: nichtkommerzielle Nutzung, Quellenangabe aare.guru und BAFU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA, MANIFEST  # noqa: E402

V100_URL = ("https://zenodo.org/api/records/21850779/files/"
            "XnoTGitHub/zero-shot-river-temperature-forecast-v1.0.0.zip/content")
V100_MD5 = "558a7a97f70f740546e1d92abcf5f887"
TZ = "Europe/Zurich"
ORAKU = "https://aare-oraku.konzept.space/forecast/{kind}"
AAREGURU = "https://aareguru.existenz.ch/v2018/history"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest() -> list[dict]:
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def from_v100(zip_bytes: bytes, rows: list[dict]) -> int:
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = {n.split("/", 1)[1]: n for n in z.namelist() if "/" in n}
    n = 0
    for r in rows:
        src = r["path_in_v1_0_0"]
        if not src:
            continue
        dst = DATA / r["path"]
        if dst.exists() and sha256(dst) == r["sha256"]:
            continue
        if src not in names:
            print(f"  fehlt im Archiv: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(names[src]))
        n += 1
    return n


def _req(url, params, ua):
    import requests
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=60, headers={"User-Agent": ua})
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:  # noqa: PERF203
            wait = 2 ** attempt
            print(f"    Wiederholung in {wait}s ({type(e).__name__})")
            time.sleep(wait)
    raise RuntimeError(f"Abruf fehlgeschlagen: {url} {params}")


def from_api(rows: list[dict], refetch_aareguru: bool) -> int:
    import pandas as pd
    n = 0
    for r in rows:
        dst = DATA / r["path"]
        if dst.exists() and sha256(dst) == r["sha256"]:
            continue
        name = dst.name
        if name.startswith("oraku_meta_"):
            t0 = pd.Timestamp(name[len("oraku_meta_"):-5])
            resp = _req(ORAKU.format(kind="temperature"),
                        {"city": "bern", "horizon": 1, "from": t0.tz_localize(TZ).isoformat(),
                         "model_info": "true", "app": "hywa-review-audit", "version": "1.0"},
                        "python-requests")
            text = resp.text                      # urspruenglich unveraendert gespeichert
        elif name.startswith("oraku_"):
            kind, t0s, hz = name[len("oraku_"):-5].split("_")
            t0 = pd.Timestamp(t0s)
            if not r["path_in_v1_0_0"]:
                # Nachabruf vom 14.10.2026 (t0 = 28.3.2026): mit model_info=true abgerufen und
                # als Rohtext gespeichert; mit denselben Parametern byte-identisch (geprueft 7.11.2026)
                resp = _req(ORAKU.format(kind=kind),
                            {"city": "bern", "horizon": int(hz), "from": t0.tz_localize(TZ).isoformat(),
                             "model_info": "true"}, "python-requests")
                text = resp.text
            else:
                resp = _req(ORAKU.format(kind=kind),
                            {"city": "bern", "horizon": int(hz), "from": t0.tz_localize(TZ).isoformat(),
                             "app": "hywa-allseasons", "version": "1.0"},
                            "hywa-allseasons-prevruns-dst/1.0")
                text = json.dumps(resp.json())    # Pipeline speichert json.dumps(...)
        elif name.startswith(("aare_bern_", "aare_thun_")) and name.endswith(".json") and refetch_aareguru:
            city, start, end = name[len("aare_"):-5].split("_")
            resp = _req(AAREGURU, {"city": city, "start": start, "end": end,
                                   "app": "hywa-allseasons", "version": "1.0"},
                        "hywa-allseasons-prevruns-dst/1.0")
            text = json.dumps(resp.json())
        else:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        n += 1
        time.sleep(0.3)
    return n


def verify(rows: list[dict]) -> int:
    ok, missing, bad = 0, [], []
    for r in rows:
        p = DATA / r["path"]
        if not p.exists():
            missing.append(r["path"])
        elif sha256(p) != r["sha256"]:
            bad.append(r["path"])
        else:
            ok += 1
    print(f"Eingangsdaten: {ok}/{len(rows)} pruefsummengleich, {len(missing)} fehlen, {len(bad)} abweichend")
    for m in missing[:20]:
        print("  FEHLT     ", m)
    for b in bad[:20]:
        print("  ABWEICHUNG", b)
    return 0 if not missing and not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v100-zip", type=Path, help="lokal gespeichertes Zenodo-Archiv v1.0.0")
    ap.add_argument("--download-v100", action="store_true", help="Archiv v1.0.0 von Zenodo laden (4,3 MB)")
    ap.add_argument("--api", action="store_true", help="fehlende Oraku-Dateien ueber die oeffentliche Schnittstelle abrufen")
    ap.add_argument("--refetch-aareguru", action="store_true", help="Messwerte erneut ueber aare.guru abrufen (revisionsanfaellig)")
    ap.add_argument("--verify", action="store_true", help="nur pruefen")
    a = ap.parse_args()
    rows = manifest()
    if not a.verify:
        if a.v100_zip or a.download_v100:
            if a.v100_zip:
                data = a.v100_zip.read_bytes()
            else:
                import requests
                print("Lade v1.0.0 von Zenodo ...")
                data = requests.get(V100_URL, timeout=300).content
            if hashlib.md5(data).hexdigest() != V100_MD5:
                sys.exit("Archiv v1.0.0: MD5 stimmt nicht mit Zenodo ueberein")
            print(f"  aus v1.0.0 uebernommen: {from_v100(data, rows)} Dateien")
        if a.api or a.refetch_aareguru:
            print(f"  ueber Schnittstellen abgerufen: {from_api(rows, a.refetch_aareguru)} Dateien")
    sys.exit(verify(rows))


if __name__ == "__main__":
    main()
