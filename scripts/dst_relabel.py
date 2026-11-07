# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/frozen_2026-10-14/scripts/dst_relabel.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""DST-korrekte Rekonstruktion der Zeitstempel gecachter Open-Meteo-Antworten.

Hintergrund
-----------
Open-Meteo gibt bei Angabe von ``timezone=Europe/Zurich`` alle Stundenlabels
mit EINEM festen ``utc_offset_seconds`` zurueck ("resolved at request time and
fixed throughout the response, including across daylight saving time
transitions"). In allen Caches dieses Projekts ist das +7200 s (CEST), auch
fuer Winterzeitraeume. Die Labels sind daher im Winter um +1 h gegenueber der
tatsaechlichen Lokalzeit (CET) verschoben.

Vorgehen (robust, UTC-basiert, keine pauschale Verschiebung)
------------------------------------------------------------
  1. UTC-Zeitpunkt = Label (naiv) - utc_offset_seconds
  2. Lokalzeit     = UTC -> Europe/Zurich (DST-bewusst, pytz/zoneinfo)
  3. Fuer die Pipeline-Kompatibilitaet wird zusaetzlich ein naiver lokaler
     Index erzeugt, der die Konventionen des bestehenden Aareguru-Loaders
     nachbildet (tz_convert -> tz_localize(None) -> resample("1h").mean()):
       - DST-Ende (doppelte lokale Stunde 02:00): die zwei Stundenwerte
         werden gemittelt (identisch zu resample.mean auf naivem Index)
       - DST-Beginn (fehlende lokale Stunde 02:00): es entsteht ein NaN-Slot
         (identisch zu resample auf naivem Index); die Pipeline interpoliert
         Luecken < 12 h ohnehin linear.

Diese Datei veraendert keine Originaldateien. Sie liest nur.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Zurich"


def load_open_meteo_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def hourly_frame_raw(d: dict) -> pd.DataFrame:
    """Rohdaten mit den originalen (naiven, fest-offset) Labels als Index."""
    h = d["hourly"]
    df = pd.DataFrame({k: v for k, v in h.items() if k != "time"},
                      index=pd.to_datetime(h["time"]))
    df.index.name = "label_fixed_offset"
    return df


def reconstruct_utc(d: dict) -> pd.DataFrame:
    """Schritt 1: echte UTC-Zeitpunkte aus Label und utc_offset_seconds."""
    df = hourly_frame_raw(d)
    off = int(d["utc_offset_seconds"])
    utc = (df.index - pd.Timedelta(seconds=off)).tz_localize("UTC")
    out = df.copy()
    out.index = utc
    out.index.name = "utc"
    return out


def utc_to_local_aware(df_utc: pd.DataFrame) -> pd.DataFrame:
    """Schritt 2: DST-bewusste Lokalzeit (tz-aware Index, verlustfrei)."""
    out = df_utc.copy()
    out.index = out.index.tz_convert(TZ)
    out.index.name = "local_aware"
    return out


def local_aware_to_naive_pipeline(df_local: pd.DataFrame) -> pd.DataFrame:
    """Schritt 3: naiver lokaler Stundenindex nach Pipeline-Konvention.

    Nachbildung von  idx.tz_localize(None) + resample("1h").mean()  wie im
    Aareguru-Loader (all_seasons_ensemble_prev_runs.py, fetch_aare_city_chunked).
    """
    naive = df_local.copy()
    naive.index = naive.index.tz_localize(None)
    naive.index.name = "local_naive"
    # resample erzeugt bei DST-Ende ein Mittel der doppelten Stunde,
    # bei DST-Beginn einen NaN-Slot fuer die fehlende Stunde.
    return naive.resample("1h").mean()


def relabel_open_meteo(path: str | Path) -> dict[str, pd.DataFrame]:
    """Komplette Kette. Gibt alle Zwischenschritte zurueck (fuer Pruefung)."""
    d = load_open_meteo_json(path)
    raw = hourly_frame_raw(d)
    utc = reconstruct_utc(d)
    loc = utc_to_local_aware(utc)
    naive = local_aware_to_naive_pipeline(loc)
    return {"offset_seconds": int(d["utc_offset_seconds"]), "raw": raw,
            "utc": utc, "local_aware": loc, "local_naive": naive}


def label_shift_hours(res: dict[str, pd.DataFrame]) -> pd.Series:
    """Verschiebung (korrigiertes naives Label - Originallabel) je UTC-Stunde, in h.

    0 h in CEST, -1 h in CET. Dient der Zaehlung betroffener Werte.
    """
    utc_idx = res["utc"].index
    corrected = utc_idx.tz_convert(TZ).tz_localize(None)
    original = res["raw"].index
    return pd.Series((corrected - original) / pd.Timedelta(hours=1), index=utc_idx, name="shift_h")


def windows_affected(res: dict[str, pd.DataFrame], t0s: pd.DatetimeIndex,
                     context_h: int = 256, horizon_h: int = 48) -> pd.DataFrame:
    """Je Prognosefenster: Anzahl verschobener Stunden in Kontext und Horizont.

    t0s sind naive lokale Startzeitpunkte (wie in der Pipeline). Die Zuordnung
    Fenster -> Stunden erfolgt ueber die KORRIGIERTEN lokalen Labels, denn
    diese entsprechen den Aareguru-Labels, gegen die ausgerichtet wird.
    """
    shift = label_shift_hours(res)
    shift_local = shift.copy()
    shift_local.index = shift.index.tz_convert(TZ).tz_localize(None)
    # bei DST-Ende zwei Werte mit gleichem naiven Label: beide zaehlen
    rows = []
    for t0 in t0s:
        ctx = (shift_local.index >= t0 - pd.Timedelta(hours=context_h)) & (shift_local.index < t0)
        fc = (shift_local.index >= t0) & (shift_local.index < t0 + pd.Timedelta(hours=horizon_h))
        rows.append({"t0": t0,
                     "ctx_hours": int(ctx.sum()), "ctx_shifted": int((shift_local[ctx] != 0).sum()),
                     "fc_hours": int(fc.sum()), "fc_shifted": int((shift_local[fc] != 0).sum())})
    df = pd.DataFrame(rows)
    df["affected"] = (df.ctx_shifted + df.fc_shifted) > 0
    df["fully_cet"] = (df.ctx_shifted == df.ctx_hours) & (df.fc_shifted == df.fc_hours)
    return df
