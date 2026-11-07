# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/frozen_2026-10-14/scripts/test_dst_relabel.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Netzwerkfreie Tests fuer dst_relabel (synthetische + echte Cache-Daten).

Aufruf:  python3 -I test_dst_relabel.py     (oder pytest)
Liest nur. Schreibt nichts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dst_relabel import (TZ, hourly_frame_raw, reconstruct_utc, utc_to_local_aware,  # noqa: E402
                         local_aware_to_naive_pipeline, relabel_open_meteo, label_shift_hours)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CACHE  # noqa: E402
PRIMARY = CACHE / "meteo_prevruns_tt_solar_bern_2025-10-23_2026-08-04.json"


def synthetic(start_label: str, hours: int, offset: int) -> dict:
    """Synthetische Open-Meteo-Antwort: Wert = Stundenindex (eindeutig rueckverfolgbar)."""
    idx = pd.date_range(start_label, periods=hours, freq="1h")
    return {"utc_offset_seconds": offset,
            "hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in idx],
                       "x": list(range(hours))}}


def test_utc_reconstruction_is_pure_shift():
    d = synthetic("2026-01-15T00:00", 48, 7200)
    utc = reconstruct_utc(d)
    assert utc.index[0] == pd.Timestamp("2026-01-14 22:00", tz="UTC")
    assert utc.index.is_unique and utc.index.is_monotonic_increasing
    assert list(utc["x"]) == list(range(48)), "Werte duerfen nicht umsortiert werden"


def test_winter_label_minus_one_hour():
    d = synthetic("2026-01-15T00:00", 24, 7200)
    res = relabel_open_meteo.__wrapped__(d) if hasattr(relabel_open_meteo, "__wrapped__") else None
    utc = reconstruct_utc(d)
    loc = utc_to_local_aware(utc)
    naive = local_aware_to_naive_pipeline(loc)
    # Label 2026-01-15 10:00 (fest +2) -> UTC 08:00 -> CET 09:00
    assert naive.loc["2026-01-15 09:00", "x"] == 10


def test_summer_unchanged():
    d = synthetic("2026-07-15T00:00", 24, 7200)
    naive = local_aware_to_naive_pipeline(utc_to_local_aware(reconstruct_utc(d)))
    raw = hourly_frame_raw(d)
    assert np.array_equal(naive["x"].to_numpy(), raw["x"].to_numpy())
    assert (naive.index == raw.index).all()


def test_dst_end_double_hour_averaged_once():
    # 2025-10-26: 03:00 CEST -> 02:00 CET. Labels (fest +2): 00..23 ergeben UTC 22:00(25.) .. 21:00(26.)
    d = synthetic("2025-10-26T00:00", 24, 7200)
    utc = reconstruct_utc(d)
    loc = utc_to_local_aware(utc)
    naive_dup = loc.copy()
    naive_dup.index = naive_dup.index.tz_localize(None)
    assert naive_dup.index.duplicated().sum() == 1, "genau eine doppelte lokale Stunde (02:00)"
    naive = local_aware_to_naive_pipeline(loc)
    assert naive.index.is_unique
    # Werte 2 (02:00 CEST = 00:00 UTC) und 3 (02:00 CET = 01:00 UTC) werden gemittelt -> 2.5
    assert naive.loc["2025-10-26 02:00", "x"] == 2.5
    # keine Luecke: 00:00 .. 23:00 am 26.10. lokal (nach Mittelung 24 Slots, letzter Wert = Label 23 -> 22:00 CET? nein:)
    # Label 23:00 (+2) = 21:00 UTC = 22:00 CET; der lokale Tag endet im Cache-Ausschnitt also bei 22:00 CET.
    assert naive.index[-1] == pd.Timestamp("2025-10-26 22:00")
    # Summe der Werte bleibt erhalten (bis auf die Mittelung der Doppelstunde): 0..23 = 276; naive: 276 - 2 - 3 + 2.5
    assert abs(naive["x"].sum() - (276 - 5 + 2.5)) < 1e-9


def test_dst_start_missing_hour_is_nan_slot():
    d = synthetic("2026-03-29T00:00", 24, 7200)
    naive = local_aware_to_naive_pipeline(utc_to_local_aware(reconstruct_utc(d)))
    # 01:00 CET (UTC 00:00, Label 02:00 -> x=2), dann 03:00 CEST (UTC 01:00, Label 03:00 -> x=3); 02:00 fehlt lokal
    assert naive.loc["2026-03-29 01:00", "x"] == 2
    assert np.isnan(naive.loc["2026-03-29 02:00", "x"]), "fehlende lokale Stunde als NaN-Slot (Pipeline interpoliert)"
    assert naive.loc["2026-03-29 03:00", "x"] == 3
    assert naive["x"].dropna().shape[0] == 24, "kein Wert verloren"


def test_offset_3600_would_be_identity_in_winter():
    """Gegenprobe: waere der Cache mit Offset 3600 (CET) gespeichert, duerfte im Januar nichts verschoben werden."""
    d = synthetic("2026-01-15T00:00", 24, 3600)
    naive = local_aware_to_naive_pipeline(utc_to_local_aware(reconstruct_utc(d)))
    raw = hourly_frame_raw(d)
    assert (naive.index == raw.index).all()


def test_real_cache_consistency():
    res = relabel_open_meteo(PRIMARY)
    assert res["offset_seconds"] == 7200
    sh = label_shift_hours(res)
    assert set(sh.unique()) == {0.0, -1.0}
    assert len(res["utc"]) == len(res["raw"]) == 6864
    assert res["utc"].index.is_unique
    # Kontrollwert: Label 2026-01-15 10:00 -> korrigiert 09:00
    v_raw = res["raw"].loc["2026-01-15 10:00", "temperature_2m_previous_day1"]
    v_new = res["local_naive"].loc["2026-01-15 09:00", "temperature_2m_previous_day1"]
    assert v_raw == v_new
    # Sommer identisch
    a = res["raw"].loc["2026-07-15", "temperature_2m_previous_day1"].to_numpy()
    b = res["local_naive"].loc["2026-07-15", "temperature_2m_previous_day1"].to_numpy()
    assert np.allclose(a, b)


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    tests = [f for n, f in inspect.getmembers(mod, inspect.isfunction) if n.startswith("test_")]
    failed = 0
    for f in tests:
        try:
            f()
            print(f"[OK]   {f.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {f.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[ERR]  {f.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} Tests bestanden")
    sys.exit(1 if failed else 0)
