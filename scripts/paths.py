"""Zentrale, relative Pfade des Reproduktions-Repositoriums (v1.1.0).

Alle Skripte importieren ihre Ein- und Ausgabeorte von hier; es gibt keine
absoluten lokalen Pfade. Die Ordnerstruktur ist in README.md beschrieben.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
CACHE = DATA / "cache"                 # API-Antworten: Open-Meteo (mitgeliefert),
                                       # aare.guru/BAFU und Aare-Oraku (per fetch_inputs.py)
CACHE_NEW = DATA / "cache_new"         # nur fuer unerwartete Nachabrufe der Pipeline; soll leer bleiben
CLIMATOLOGY = CACHE / "aare_temperature_hourly.csv"   # per fetch_inputs.py (aus v1.0.0)
ORAKU_META = DATA / "cache_oraku_meta" # Oraku-Antworten mit model_info (per fetch_inputs.py)
METEOSCHWEIZ = DATA / "meteoschweiz"   # MeteoSchweiz OGD, mitgeliefert (CC BY 4.0)
OSM = DATA / "osm"                     # OpenStreetMap-Geometrie, mitgeliefert (ODbL 1.0)
MANIFEST = DATA / "MANIFEST_inputs.csv"

RESULTS = ROOT / "results"             # wird bei der Reproduktion erzeugt
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
FINAL = RESULTS / "final_checks"
REFERENCE = ROOT / "results_reference" # veroeffentlichter Referenzstand (eingefroren 14.10.2026)
V100_REFERENCE = REFERENCE / "v1.0.0_n128"   # Fenster-Kennzahlen der Ersteinreichung (fuer den Alt/Neu-Vergleich)

# TiRex-2 (NX-AI, Apache-2.0), im Beitrag verwendete Revision und Pruefsumme der Gewichte
TIREX2_REPO = "NX-AI/TiRex-2"
TIREX2_REVISION = "05e5b26db52bfb256f1ae1bdf785589850482de3"
TIREX2_CKPT_SHA256 = "184b160ffbe4c01a26beeba14015ff3507c7497e1f3577114187bbc1d19fcac1"
TIREX2_CKPT_SIZE = 380613375


def verify_tirex2_checkpoint():
    """Laedt (falls noetig) die festgelegte Revision und prueft die Gewichtsdatei per SHA-256."""
    import hashlib
    from huggingface_hub import snapshot_download

    d = Path(snapshot_download(TIREX2_REPO, revision=TIREX2_REVISION))
    ck = d / "model.ckpt"
    h = hashlib.sha256()
    with open(ck, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != TIREX2_CKPT_SHA256 or ck.stat().st_size != TIREX2_CKPT_SIZE:
        raise RuntimeError(f"TiRex-2-Gewichte weichen ab: {ck} sha256={h.hexdigest()}")
    print(f"  TiRex-2 {TIREX2_REVISION[:8]}: model.ckpt SHA-256 ok", flush=True)
    return d
