"""Prueft wie tief die relevanten APIs zurueckreichen fuer eine erweiterte
Evaluation ueber 2023-2026.

- Aareguru history fuer Bern und Thun (temperature, flow)
- Aare-Oraku forecast temperature via `from`-Parameter fuer historische
  Zeitpunkte in 2023, 2024, 2025
- Open-Meteo Historical-Forecast fuer Bern und Thun Region

Output: Report auf stdout.
"""
import datetime as dt
import json
from pathlib import Path
import requests


BASE = Path(__file__).resolve().parent.parent
CACHE = BASE / "data" / "aare_dashboard_cache"
CACHE.mkdir(exist_ok=True)


def _get(url, params, cache_key, timeout=30):
    fp = CACHE / f"probe_{cache_key}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    try:
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "hywa-probe/1.0"})
        r.raise_for_status()
        d = r.json()
        fp.write_text(json.dumps(d))
        return d
    except Exception as e:
        return {"_error": str(e)}


def probe_aareguru(city, start, end):
    d = _get("https://aareguru.existenz.ch/v2018/history",
             {"city": city, "start": str(start), "end": str(end),
              "app": "hywa_probe", "version": "1.0"},
             cache_key=f"aareguru_{city}_{start}_{end}")
    if "_error" in d:
        return {"error": d["_error"]}
    temps = d.get("data", {}).get("temperature", [])
    flows = d.get("data", {}).get("flow", [])
    return {"temperature_pts": len(temps), "flow_pts": len(flows),
            "first_ts": temps[0][0] if temps else None,
            "last_ts": temps[-1][0] if temps else None}


def probe_oraku(from_ts_iso, horizon=48):
    d = _get("https://aare-oraku.konzept.space/forecast/temperature",
             {"city": "bern", "horizon": horizon, "from": from_ts_iso,
              "app": "hywa_probe", "version": "1.0"},
             cache_key=f"oraku_{from_ts_iso.replace(':','-')}_{horizon}")
    if "_error" in d:
        return {"error": d["_error"]}
    vals = d.get("data", {}).get("value", [])
    tims = d.get("data", {}).get("time", [])
    return {"n_forecast": len(vals),
            "first_time": tims[0] if tims else None,
            "last_time": tims[-1] if tims else None,
            "first_value": vals[0] if vals else None}


def probe_openmeteo(lat, lon, start, end):
    d = _get("https://historical-forecast-api.open-meteo.com/v1/forecast",
             {"latitude": lat, "longitude": lon,
              "hourly": "temperature_2m", "start_date": str(start),
              "end_date": str(end), "timezone": "Europe/Zurich"},
             cache_key=f"meteo_{lat}_{lon}_{start}_{end}")
    if "_error" in d:
        return {"error": d["_error"]}
    times = d.get("hourly", {}).get("time", [])
    return {"n_hours": len(times),
            "first": times[0] if times else None,
            "last": times[-1] if times else None}


def main():
    print("=" * 78)
    print("AAREGURU HISTORY: Bern & Thun")
    print("=" * 78)
    # Sonde: verschiedene Jahre
    for year in [2020, 2021, 2022, 2023, 2024, 2025]:
        for month in [1, 7]:
            s = dt.date(year, month, 1)
            e = dt.date(year, month, 5)
            r_bern = probe_aareguru("bern", s, e)
            r_thun = probe_aareguru("thun", s, e)
            print(f"  {s} .. {e}   Bern: {r_bern.get('temperature_pts','?')} temp / "
                  f"{r_bern.get('flow_pts','?')} flow   |   "
                  f"Thun: {r_thun.get('temperature_pts','?')} temp / "
                  f"{r_thun.get('flow_pts','?')} flow")

    print("\n" + "=" * 78)
    print("AARE-ORAKU FORECAST (via `from`-Parameter, historische Zeitpunkte)")
    print("=" * 78)
    for year, month, day in [(2023, 7, 15), (2024, 7, 15), (2025, 7, 15),
                              (2023, 8, 5), (2024, 6, 20), (2025, 5, 10),
                              (2023, 4, 10), (2024, 10, 15)]:
        ts_iso = f"{year:04d}-{month:02d}-{day:02d}T12:00:00+02:00"
        r = probe_oraku(ts_iso)
        n = r.get("n_forecast", 0)
        marker = "OK " if n > 0 else "??"
        print(f"  {marker}  from={ts_iso}   -> {n} h forecast, "
              f"first_time={r.get('first_time')}, first_val={r.get('first_value')}")

    print("\n" + "=" * 78)
    print("OPEN-METEO HISTORICAL-FORECAST")
    print("=" * 78)
    # Bern
    for year in [2022, 2023, 2024, 2025]:
        s = dt.date(year, 7, 1)
        e = dt.date(year, 7, 10)
        r = probe_openmeteo(46.948, 7.447, s, e)
        print(f"  Bern  {s} .. {e}   n_hours={r.get('n_hours', '?')}   "
              f"first={r.get('first', '?')}")

    print("\nEnde.")


if __name__ == "__main__":
    main()
