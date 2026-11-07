# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/phase2b/fig_prognoseaufbau.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Abb. 2 (neu, v2): Prognoseaufbau und Modelle mit getrennten Eingängen.
Ausgabe: phase2b/figures_v3/abb02_prognoseaufbau.png/.pdf (Layout-Runde 3; Vorfassung: phase2b/figures/)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from paths import FIGURES as FIG  # noqa: E402
FIG.mkdir(parents=True, exist_ok=True)
OUTDIR = FIG
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
CM = 1 / 2.54
C_CTX, C_FC1, C_FC2, C_OBS = "#cfe3f3", "#f7d7c4", "#f3c2a6", "#d9d9d9"
BLUE, ORANGE, RED, GREY = "#0072B2", "#D55E00", "#CC3311", "#6b7280"

# Layout-Runde 3 (22.10.2026): neu gesetzt für Einbettung in Originalgröße (16 cm = volle Textbreite,
# keine Skalierung in Word). Koordinaten in cm. Inhalte (Modellvarianten, Eingänge, Datenwege,
# zeitliche Beziehungen) unverändert; Zeitachse in a) schematisch (nicht maßstäblich).
from matplotlib.patches import Polygon
ENS, ENS_FC = "#AA3377", "#f3dbe8"           # Ensemble-Farbe wie in Abb. 3–5
FW, FH = 16.0, 13.0
fig = plt.figure(figsize=(FW * CM, FH * CM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")
FS, FSS, FSH = 7.8, 7.5, 9.5                 # Box-Text, Hinweise, Panel-Überschriften

def rect(x0, x1, y0, h, fc, txt=None, fs=FS, ec="none", color="#111"):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, h, fc=fc, ec=ec, lw=0.6))
    if txt: ax.text((x0 + x1) / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=fs, color=color, linespacing=1.25)

# ================================================================ a) Ein Prognosefall
top = FH - 0.25
ax.text(0.1, top, "a) Ein Prognosefall", fontsize=FSH, weight="bold", va="top")
ax.text(3.95, top - 0.04, "gleich für alle Modelle: Startzeitpunkt, Zielgröße, Horizont, Bewertungsdaten", fontsize=FSS, color=GREY, va="top")
X0, XT, X24, X48 = 2.25, 9.7, 12.8, 15.9      # t0-256 h, t0, +24 h, +48 h (schematisch)
LBL = dict(ha="right", va="center", fontsize=FSS, color="#333", linespacing=1.2)
# Horizont-Klammer
yb = top - 0.75
ax.plot([XT, XT, X48, X48], [yb - 0.12, yb, yb, yb - 0.12], color="#333", lw=0.7)
ax.text((XT + X48) / 2, yb + 0.08, "Prognosehorizont 48 h, stündlich (h = 0 … 47)", ha="center", va="bottom", fontsize=FSS)
# Zeile 1: Messdaten / Prognose
y1 = top - 1.7
ax.text(X0 - 0.2, y1 + 0.4, "Messdaten", **LBL)
rect(X0, XT, y1, 0.8, C_CTX, "Eingabezeitraum TiRex-2: 256 Messstunden\n(Aare-Oraku: eigener, kürzerer Eingabezeitraum)")
rect(XT, X24, y1, 0.8, C_FC1, "Tag 1"); rect(X24, X48, y1, 0.8, C_FC2, "Tag 2")
ax.plot([X0, X48], [y1, y1], color="#333", lw=0.8)
for xx, lab, ha in [(X0, "t₀ − 256 h", "left"), (XT, "t₀", "center"), (X24, "+24 h", "center"), (X48, "+48 h", "right")]:
    ax.plot([xx, xx], [y1 - 0.1, y1 + 0.8], color="#333", lw=0.8)
    ax.text(xx, y1 - 0.14, lab, ha=ha, va="top", fontsize=FSS)
# Zeile 2: Wettervorhersage der TiRex-2-Varianten
y2 = y1 - 1.1
ax.text(X0 - 0.2, y2 + 0.32, "Wetter-\nvorhersage\n(TiRex-2)", **LBL)
rect(X0, X24, y2, 0.64, "#e8eef5", "Lauf ≥ 24 h vor Gültigkeit (Eingabezeitraum und Tag 1)", fs=FSS)
rect(X24, X48, y2, 0.64, "#d5dde8", "Lauf ≥ 48 h", fs=FSS)
ax.text(X0, y2 - 0.1, "Open-Meteo-Archiv, Vorlaufzeit des Modelllaufs; alle Läufe vor t₀ begonnen", ha="left", va="top", fontsize=FSS, color=GREY, bbox=dict(fc="white", ec="none", pad=0.5), zorder=3)
# Zeile 3: Bewertung
y3 = y2 - 0.9
ax.text(X0 - 0.2, y3 + 0.32, "Bewertung", **LBL)
rect(XT, X48, y3, 0.64, C_OBS, "48 unveränderte Messstunden → MAE je Fall", fs=FSS)
ax.plot([XT, XT], [y3, y1], color="#333", lw=0.7, ls=(0, (2, 2)), zorder=0)
ax.text(X0, y3 - 0.2, "Startzeitpunkte t₀: 00:00 Lokalzeit, alle 2 Tage, 04.11.2025–30.07.2026\n→ 129 Fälle (7 wegen Messlücken ausgeschlossen)", fontsize=FSS, va="top", linespacing=1.3)
# Achsenbruch im Eingabezeitraum (Zeitachse schematisch, nicht maßstäblich)
for dx in (0, 0.12):
    ax.plot([X0 + 2.2 + dx - 0.05, X0 + 2.2 + dx + 0.05], [y1 - 0.09, y1 + 0.09], color="#333", lw=0.8, zorder=4)
ax.add_patch(Rectangle((X0 + 2.18, y1 - 0.03), 0.16, 0.06, fc="white", ec="none", zorder=3))
# Trennlinie zwischen a) und b)
ysep = y3 - 1.1
ax.plot([0.1, FW - 0.1], [ysep, ysep], color="#9aa0a6", lw=0.8)

# ================================================================ b) Modelle und Eingänge
ytb = ysep - 0.25
ax.text(0.1, ytb, "b) Modelle und ihre Eingänge (getrennt je Modell)", fontsize=FSH, weight="bold", va="top")
yc = ytb - 0.6
for x, lab in [(0.3, "Eingänge"), (6.75, "Modelle"), (12.1, "Kombination, Vergleich")]:
    ax.text(x, yc, lab.upper(), fontsize=7, color=GREY, va="center", weight="bold")

def box(x0, x1, y0, y1, title, body, fc, ec, tcolor="#111"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=0.8))
    n = body.count("\n") + 1 if body else 0
    lh = 0.36
    yy = (y0 + y1) / 2 + n * lh / 2
    ax.text(x0 + 0.15, yy, title, fontsize=FS, weight="bold", va="center", ha="left", color=tcolor)
    if body:
        ax.text(x0 + 0.15, yy - lh * 0.55, body, fontsize=FS, va="top", ha="left", linespacing=1.3)

def harrow(x0, x1, y, color="#333"):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9, color=color, lw=0.9, shrinkA=0, shrinkB=0))

def path_arrow(pts, color):
    xs, ys = zip(*pts[:-1]); ax.plot(xs + (pts[-1][0],), ys + (pts[-1][1],), color=color, lw=0.9, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9, color=color, lw=0.9, shrinkA=0, shrinkB=0))

# TiRex-2-Gruppe
gT1 = yc - 0.3                                   # Oberkante Gruppenüberschrift
ax.text(0.3, gT1, "TiRex-2-Varianten  ·  Open-Meteo-Archiv  ·  ohne standortspezifisches Nachtraining", fontsize=FS, weight="bold", color=BLUE, va="top")
fT = gT1 - 0.42                                   # Oberkante Rahmen
IX0, IX1, MX0, MX1, BUS = 0.3, 5.7, 6.75, 11.25, 6.22
b1 = (fT - 1.05, fT - 0.2)
b2 = (b1[0] - 1.55, b1[0] - 0.22)
b3 = (b2[0] - 1.25, b2[0] - 0.22)
box(IX0, IX1, *b1, "Messung Bern", "Wassertemperatur, Abfluss", C_CTX, "#5b8db8")
box(IX0, IX1, *b2, "Vorhersagen", "Lufttemperatur Bern (L, LAW, R)\nWind Thunersee (LAW, R)\nAbflussvorhersage BAFU (LAW, R)", C_CTX, "#5b8db8")
box(IX0, IX1, *b3, "Thun (nur R)", "Messung Wassertemperatur, Abfluss\n+ Wettervorhersage", C_CTX, "#5b8db8")
mL = (b1[0] + 0.05, b1[1] - 0.05)
yLAW = (b2[0] + b2[1]) / 2; mLAW = (yLAW - 0.42, yLAW + 0.42)
mR = (b3[0], b3[1] + 0.45)
box(MX0, MX1, *mL, "TiRex-2 – L", "Lufttemperatur", "#dbe9f6", BLUE)
box(MX0, MX1, *mLAW, "TiRex-2 – LAW", "Lufttemperatur, Abfluss, Wind", "#dbe9f6", BLUE)
box(MX0, MX1, *mR, "TiRex-2 – R (zweistufig)", "Stufe 1: Thun → Stufe 2: Bern", "#dbe9f6", BLUE)
ax.add_patch(Rectangle((0.15, mR[0] - 0.2), 11.25, fT - (mR[0] - 0.2), fc="none", ec="#9bbbd9", lw=0.8, ls="--"))
# Sammelschiene: Messung Bern + Vorhersagen -> L, LAW, R; Thun -> nur R
yi1 = (b1[0] + b1[1]) / 2; yi2 = yLAW; yR_hi = mR[1] - 0.3; yR_lo = (b3[0] + b3[1]) / 2 - 0.15
for y in (yi1, yi2):
    ax.plot([IX1, BUS], [y, y], color="#333", lw=0.9)
ax.plot([BUS, BUS], [yi1, yR_hi], color="#333", lw=0.9)
for y in (yi1, yi2, yR_hi):
    harrow(BUS, MX0, y)
harrow(IX1, MX0, yR_lo)
ax.plot([BUS], [yi2], marker="o", ms=2.5, color="#333")

# Referenzsystem
gO1 = mR[0] - 0.4
ax.text(0.3, gO1, "Referenzsystem  ·  eigene Eingänge und eigenes Verfahren, unabhängig betrieben", fontsize=FS, weight="bold", color=ORANGE, va="top")
fO = gO1 - 0.42
bo = (fO - 1.25, fO - 0.18)
box(IX0, IX1, *bo, "Messung Bern", "Wassertemperatur\n+ Lufttemperaturvorhersage (Meteotest)", "#fbe4d6", ORANGE)
mo = ((bo[0] + bo[1]) / 2 - 0.42, (bo[0] + bo[1]) / 2 + 0.42)
box(MX0, MX1, *mo, "Aare-Oraku", "autoregressive Regression", "#fbe4d6", ORANGE)
harrow(IX1, MX0, (bo[0] + bo[1]) / 2, ORANGE)
ax.add_patch(Rectangle((0.15, bo[0] - 0.2), 11.25, fO - (bo[0] - 0.2), fc="none", ec="#e0a988", lw=0.8, ls="--"))

# Kombination und Vergleich (rechte Spalte)
RX0, RX1, XJ = 12.1, 15.9, 14.0
ens = (gO1 + 0.25, yLAW - 0.45)
assert ens[1] - ens[0] > 1.45, ens
box(RX0, RX1, *ens, "Ensemble", "ungewichtetes Mittel\n½ · TiRex-2 (LAW)\n+ ½ · Aare-Oraku", ENS_FC, ENS, tcolor=ENS)
path_arrow([(MX1, yLAW), (XJ, yLAW), (XJ, ens[1])], BLUE)
yo = (mo[0] + mo[1]) / 2
path_arrow([(MX1, yo), (XJ, yo), (XJ, ens[0])], ORANGE)
box(RX0, RX1, mL[0] - 0.25, mL[1], "Vergleichsmodelle", "Persistenz, Klimatologie,\nThun-Persistenz", "#f1f3f5", GREY)
fig.savefig(OUTDIR / "abb02_prognoseaufbau.png", dpi=600, bbox_inches="tight", pad_inches=0.03); fig.savefig(OUTDIR / "abb02_prognoseaufbau.pdf", bbox_inches="tight", pad_inches=0.03)
print("->", OUTDIR / "abb02_prognoseaufbau.png", "unterster Inhalt y =", round(bo[0] - 0.2, 2), "cm")
