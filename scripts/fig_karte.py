# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/phase2b/fig_karte.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Abb. 1 (neu): Karte des Untersuchungsgebiets aus OpenStreetMap-Geometrie (Nominatim, ODbL)
und BAFU-Stationskoordinaten (LV95, hydrodaten.admin.ch). Keine Kacheln, keine externen Bilder.
Ausgabe: phase2b/figures_v3/abb01_karte.png/.pdf (Layout-Runde 3; Vorfassung: phase2b/figures/), phase2b/geo/karte_log.txt
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from paths import OSM as GEO, FIGURES as FIG  # noqa: E402
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
CM = 1 / 2.54


def lv95_to_wgs84(E, N):
    """swisstopo-Näherungsformeln (Genauigkeit ~1 m)."""
    y = (E - 2600000) / 1e6; x = (N - 1200000) / 1e6
    lam = 2.6779094 + 4.728982 * y + 0.791484 * y * x + 0.1306 * y * x * x - 0.0436 * y ** 3
    phi = 16.9023892 + 3.238272 * x - 0.270978 * y * y - 0.002528 * x * x - 0.0447 * y * y * x - 0.0140 * x ** 3
    return phi * 100 / 36, lam * 100 / 36


STATIONS = {  # LV95 E/N laut hydrodaten.admin.ch (abgerufen 14.10.2026)
    "Aare – Bern, Schönau (BAFU 2135)": (2600725, 1197996),
    "Aare – Thun (BAFU 2030)": (2613235, 1179279),
}
log = []
pts = {k: lv95_to_wgs84(*v) for k, v in STATIONS.items()}
for k, (lat, lon) in pts.items():
    log.append(f"{k}: LV95 {STATIONS[k]} -> WGS84 {lat:.5f} N, {lon:.5f} E")

thun = json.load(open(GEO / "nominatim_thunersee.json"))["features"][0]["geometry"]
wohlen = json.load(open(GEO / "nominatim_wohlensee.json"))["features"][0]["geometry"]
aare = [f for f in json.load(open(GEO / "nominatim_aare.json"))["features"] if f["properties"]["osm_type"] == "relation"][0]["geometry"]
log.append("Geometrie: OSM relation 1117321 (Thunersee), 2192950 (Aare), 1722165 (Wohlensee); © OpenStreetMap-Mitwirkende, ODbL 1.0")

# Aare-Linien im Kartenausschnitt
W, E, S, N = 7.33, 7.86, 46.63, 47.01
lines = [np.array(l) for l in aare["coordinates"]] if aare["type"] == "MultiLineString" else [np.array(aare["coordinates"])]
clip = [l for l in lines if ((l[:, 0] > W) & (l[:, 0] < E) & (l[:, 1] > S) & (l[:, 1] < N)).any()]


def haversine(a, b):
    R = 6371.0; p1, p2 = math.radians(a[1]), math.radians(b[1]); dp = p2 - p1; dl = math.radians(b[0] - a[0])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


# Flusslaenge Thun -> Bern entlang der OSM-Linie: Linien zu einer Kette verbinden, naechste Punkte zu den Stationen suchen
def chain(lines):
    segs = [l.tolist() for l in lines]
    out = segs.pop(0)
    while segs:
        best = None
        for i, s in enumerate(segs):
            for rev in (False, True):
                ss = s[::-1] if rev else s
                d = haversine(out[-1], ss[0]); db = haversine(out[0], ss[-1])
                if best is None or d < best[0]: best = (d, i, rev, "end")
                if db < best[0]: best = (db, i, rev, "start")
        d, i, rev, where = best
        ss = segs.pop(i); ss = ss[::-1] if rev else ss
        if d > 1.0: continue  # nicht verbunden -> verwerfen
        out = out + ss if where == "end" else ss + out
    return np.array(out)

ch = chain([l for l in lines if ((l[:, 0] > 7.3) & (l[:, 0] < 7.9) & (l[:, 1] > 46.6) & (l[:, 1] < 47.05)).any()])
cum = np.concatenate([[0], np.cumsum([haversine(ch[i], ch[i + 1]) for i in range(len(ch) - 1)])])
def nearest(pt):
    d = [haversine((lon, lat), pt) for lon, lat in ch]; i = int(np.argmin(d)); return i, d[i]
ib, db = nearest((pts["Aare – Bern, Schönau (BAFU 2135)"][1], pts["Aare – Bern, Schönau (BAFU 2135)"][0]))
it, dt = nearest((pts["Aare – Thun (BAFU 2030)"][1], pts["Aare – Thun (BAFU 2030)"][0]))
river_km = abs(cum[ib] - cum[it])
log.append(f"Flusslaenge Thun(2030) -> Bern(2135) entlang OSM-Aare-Linie: {river_km:.1f} km (Abstand Station-Linie: Thun {dt*1000:.0f} m, Bern {db*1000:.0f} m); Luftlinie {haversine((pts['Aare – Thun (BAFU 2030)'][1], pts['Aare – Thun (BAFU 2030)'][0]), (pts['Aare – Bern, Schönau (BAFU 2135)'][1], pts['Aare – Bern, Schönau (BAFU 2135)'][0])):.1f} km")

# ----------------------------------------------------------------- Zeichnen
# Layout-Runde 3 (22.10.2026): Karte links, Stationslegende rechts; Stationen nummeriert mit
# Führungslinien, keine Beschriftung über Achsen/Rahmen; Schriftgrößen für Einbettung in
# Originalgröße (15 cm breit, 100 % = keine Skalierung in Word). Geometrie/Koordinaten unverändert.
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
plt.rcParams.update({"font.size": 8.5})
FIGW, FIGH = 15.0, 9.7                         # cm, entspricht der Einbettungsgröße
fig = plt.figure(figsize=(FIGW * CM, FIGH * CM))
ax = fig.add_axes([0.105, 0.17, 0.50, 0.815])
ax.set_aspect(1 / math.cos(math.radians(46.85)))
S2, N2 = 46.615, 47.015                         # etwas mehr Rand oben/unten
for poly, name in [(thun, "Thunersee"), (wohlen, "Wohlensee")]:
    for ring in (poly["coordinates"] if poly["type"] == "Polygon" else [r for p in poly["coordinates"] for r in p]):
        r = np.array(ring); ax.fill(r[:, 0], r[:, 1], color="#bcd7ea", ec="#6fa3c8", lw=0.6, zorder=1)
for l in clip:
    ax.plot(l[:, 0], l[:, 1], color="#3f7fb5", lw=1.4, zorder=2)
seg = ch[min(ib, it):max(ib, it) + 1]; ax.plot(seg[:, 0], seg[:, 1], color="#1f5f8b", lw=2.6, zorder=3)
BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="#9aa0a6", lw=0.5)
LEAD = dict(arrowstyle="-", color="#444", lw=0.6, shrinkA=0, shrinkB=4)
for num, key, xyt in [("1", "Aare – Bern, Schönau (BAFU 2135)", (7.365, 46.885)),
                      ("2", "Aare – Thun (BAFU 2030)", (7.47, 46.735))]:
    lat, lon = pts[key]
    ax.plot(lon, lat, marker="^", ms=8, color="#CC3311", mec="black", mew=0.6, zorder=5)
    lab = "Bern-Schönau" if num == "1" else "Thun"
    ax.annotate(f"{num}  {lab}", (lon, lat), xytext=xyt, fontsize=8.5, weight="bold", ha="left", va="center",
                bbox=BOX, arrowprops=LEAD, zorder=6)
ax.text(7.715, 46.752, "Thunersee", fontsize=8.5, style="italic", color="#2c5f85", ha="center")
ax.text(7.345, 46.992, "Wohlensee", fontsize=8, style="italic", color="#2c5f85", va="center")
ax.text(7.462, 46.958, "Bern", fontsize=8.5, color="#333", ha="left", va="bottom")
ax.text(7.575, 46.865, "Aare", fontsize=8.5, style="italic", color="#1f5f8b", ha="left")
# Windachse Thunersee (Peilung 114°/294°, Länge ~19 km, aus Manuskript) – Lage unverändert
cx, cy = 7.715, 46.700
ax.annotate("", xy=(cx + 0.11, cy - 0.028), xytext=(cx - 0.11, cy + 0.028), arrowprops=dict(arrowstyle="<->", color="#6a3d9a", lw=1.3), zorder=4)
ax.text(7.61, 46.672, "3", fontsize=8.5, weight="bold", color="#6a3d9a", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#6a3d9a", lw=0.5))
# Maßstab und Nordpfeil mit Abstand zum Rahmen
lat0 = 46.645; lon0 = 7.36; dlon = 10 / (111.32 * math.cos(math.radians(lat0)))
ax.plot([lon0, lon0 + dlon], [lat0, lat0], color="black", lw=2); ax.text(lon0 + dlon / 2, lat0 + 0.007, "10 km", ha="center", fontsize=8.5)
ax.annotate("N", xy=(7.825, 46.995), xytext=(7.825, 46.955), fontsize=9, ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color="black"))
ax.set_xlim(W, E); ax.set_ylim(S2, N2)
ax.set_xticks([7.4, 7.5, 7.6, 7.7, 7.8]); ax.set_yticks([46.65, 46.70, 46.75, 46.80, 46.85, 46.90, 46.95, 47.00])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
ax.tick_params(labelsize=8)
ax.set_xlabel("Länge [°E]", fontsize=8.5); ax.set_ylabel("Breite [°N]", fontsize=8.5)
ax.grid(True, color="#e3e6ea", lw=0.5)
# Stationslegende rechts neben der Karte
lg = fig.add_axes([0.62, 0.17, 0.375, 0.815]); lg.axis("off"); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
def item(y, sym, title, sub):
    sym(0.06, y)
    lg.text(0.16, y, title, fontsize=8.5, weight="bold", va="center")
    lg.text(0.16, y - 0.055, sub, fontsize=8, va="top", linespacing=1.3)
def tri(n):
    def f(x, y):
        lg.plot(x, y, marker="^", ms=8, color="#CC3311", mec="black", mew=0.6)
        lg.text(x - 0.055, y, n, fontsize=8.5, weight="bold", ha="right", va="center")
    return f
lg.text(0.0, 0.985, "Legende", fontsize=8.5, weight="bold", va="top", color="#333")
item(0.88, tri("1"), "Messstelle Bern-Schönau", "BAFU Nr. 2135\nZielgröße der Prognose")
item(0.68, tri("2"), "Messstelle Thun", "BAFU Nr. 2030, Seeausfluss\nZwischenprognose")
def axis_sym(x, y):
    lg.annotate("", xy=(x + 0.05, y - 0.012), xytext=(x - 0.07, y + 0.012), arrowprops=dict(arrowstyle="<->", color="#6a3d9a", lw=1.3))
    lg.text(x - 0.075, y, "3", fontsize=8.5, weight="bold", color="#6a3d9a", ha="right", va="center")
item(0.48, axis_sym, "Seelängsachse", "Thunersee; Projektion des\nWindes (Kap. 3.4)")
def river_sym(x, y):
    lg.plot([x - 0.06, x + 0.05], [y, y], color="#1f5f8b", lw=2.6)
item(0.28, river_sym, "Aare Thun → Bern", f"Fließstrecke ≈ {river_km:.0f} km\n(Luftlinie 22,5 km)")
fig.text(0.995, 0.008, "Gewässer: © OpenStreetMap-Mitwirkende (ODbL), Geometrie vereinfacht;\nStationskoordinaten: BAFU, hydrodaten.admin.ch",
         ha="right", va="bottom", fontsize=6.5, color="#6b7280", linespacing=1.25)
OUT = FIG
fig.savefig(OUT / "abb01_karte.png", dpi=600); fig.savefig(OUT / "abb01_karte.pdf")
(FIG / "abb01_karte_log.txt").write_text("\n".join(log)); print("\n".join(log)); print("->", OUT / "abb01_karte.png")
