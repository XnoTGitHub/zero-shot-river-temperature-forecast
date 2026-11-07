# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/phase2b/fig_hydro_v3.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Layout-Runde 3 (22.10.2026): Abb. 3, 4 und 5 des Manuskripts, neu gesetzt.

Abgeleitet aus phase2a/figures_hydro.py (Abb. A, B, F); Daten, Fallauswahl, Kurven und MAE-Formel
sind unverändert übernommen. Geändert ist nur die Gestaltung:
- Bildgröße = Einbettungsgröße in Word (14,4 cm breit, entspricht den bisherigen 90 % der Textbreite),
  Schriftgrößen damit in Originalgröße (Achsen 8 pt, Titel 8,5 pt, Legende 8 pt);
- Dezimalkomma wie im Manuskripttext;
- Ensemble violett (#AA3377) statt rot (#CC3311), weil Rot und das Oraku-Orange (#D55E00) bei
  deckungsgleichen Kurven kaum unterscheidbar sind; Linienstile unverändert;
- Abb. 3: Kopfzeile zweizeilig mit Abstand zu den Kurven, t₀-Marke im Diagramm unten, Legende darunter;
- Abb. 4: Versionshinweise lesbar (8 pt, weiß hinterlegt), Oraku-Kurve dünner gestrichelt über der Ensemble-Kurve;
- Abb. 5: Legende, Achsenbeschriftung, Dezimalkomma.
Liest: frozen_2026-10-14/results/, phase2a/tables/. Schreibt: phase2b/figures_v3/.
Keine Modellläufe, keine Neuberechnung von Kennzahlen.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from paths import RESULTS as FROZEN, TABLES as TAB, FIGURES as OUT  # noqa: E402
OUT.mkdir(parents=True, exist_ok=True)
TZ = "Europe/Zurich"
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Fruehling", 4: "Fruehling", 5: "Fruehling",
          6: "Sommer", 7: "Sommer", 8: "Sommer", 9: "Herbst", 10: "Herbst", 11: "Herbst"}
SEASON_DE = {"Winter": "Winter", "Fruehling": "Frühling", "Sommer": "Sommer", "Herbst": "Herbst"}
ENS = "#AA3377"
STYLE = {
    "truth": dict(color="#000000", lw=1.2, ls="-", marker="o", ms=2.4, label="Messung (BAFU, Bern)"),
    "Oraku": dict(color="#D55E00", lw=1.5, ls="--", label="Aare-Oraku (Referenzsystem)"),
    "T":     dict(color="#0072B2", lw=1.6, ls=":", label="TiRex-2 (Lufttemperatur)"),
    "AO":    dict(color=ENS, lw=1.8, ls="-", label="Ensemble ½·TiRex-2 (LAW) + ½·Aare-Oraku"),
}
COL = {k: v["color"] for k, v in STYLE.items()}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid": True,
                     "grid.color": "#e3e6ea", "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 600, "legend.frameon": False})
CM = 1 / 2.54
W = 14.4 * CM          # Abb. 5 (wie bisher 90 % der Textbreite)
WF = 16.0 * CM         # Abb. 3 und 4: volle Textbreite
class CommaFormatter(ScalarFormatter):
    """Standard-Tickpositionen und -Stellenzahl von matplotlib, nur mit Dezimalkomma."""
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def comma_axis(axis):
    f = CommaFormatter(useOffset=False); axis.set_major_formatter(f)
comma = lambda x, n=2: f"{x:.{n}f}".replace(".", ",")

h = pd.read_csv(FROZEN / "all_seasons_prev_runs_dst_hourly.csv", parse_dates=["t0"])
h["dt"] = pd.to_datetime(h["datetime"], utc=True).dt.tz_convert(TZ)
s = pd.read_csv(FROZEN / "all_seasons_prev_runs_dst_summary.csv", parse_dates=["t0"])
s["season"] = s.t0.dt.month.map(SEASON)
meta = json.load(open(TAB / "metrics.json"))
PRED = {"Oraku": "Oraku_pred", "T": "T_pred", "AO": "AO_pred"}
VERSION_CHANGES = [("2026-03-24", "v1.0"), ("2026-05-15", "v1.1"), ("2026-05-17", "v1.2"), ("2026-05-19", "v1.3")]
LOG = []


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  ->", OUT / f"{name}.png")


def window(t0):
    return h[h.t0 == pd.Timestamp(t0)].sort_values("dt")


def context_before(t0, hours=48):
    prev = h[(h.dt < pd.Timestamp(t0, tz=TZ)) & (h.dt >= pd.Timestamp(t0, tz=TZ) - pd.Timedelta(hours=hours))].sort_values("dt")
    return prev if len(prev) else None


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


# ============================================================================= Abb. 3
fig, axes = plt.subplots(2, 2, figsize=(WF, 8.9 * CM), gridspec_kw={"hspace": 0.66, "wspace": 0.22})
SHORT = {"Oraku": "Oraku", "T": "L", "AO": "Ensemble"}
for ax, se in zip(axes.ravel(), ["Winter", "Fruehling", "Sommer", "Herbst"]):
    t0 = meta["example_windows"][se]; w = window(t0); ctx = context_before(t0)
    if ctx is not None:
        ax.plot(ctx.dt, ctx.truth, color="#6b7280", lw=1.0, marker="o", ms=1.6, label="Messung vor t₀ (Kontext)")
    ax.plot(w.dt, w.truth, **STYLE["truth"])
    for m in ["Oraku", "T", "AO"]:
        ax.plot(w.dt, w[PRED[m]], **STYLE[m])
    T0 = pd.Timestamp(t0, tz=TZ)
    ax.axvline(T0, color="#6b7280", lw=0.8, ls=(0, (3, 2)))
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.16 * (hi - lo))
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(T0, 0.985, " t₀", transform=trans, color="#4b5563", fontsize=8, ha="left", va="top")
    vals = {m: mae(w[PRED[m]].to_numpy(), w.truth.to_numpy()) for m in ["Oraku", "T", "AO"]}
    LOG.append(f"Abb3 {se} t0={t0}: " + ", ".join(f"{m} {v:.4f}" for m, v in vals.items()))
    txt = "  ·  ".join(f"{SHORT[m]} {comma(vals[m])}" for m in ["Oraku", "T", "AO"])
    ax.set_title(f"{SEASON_DE[se]}, Start {T0.strftime('%d.%m.%Y')} 00:00", loc="left", fontsize=8.5, weight="bold", pad=14)
    ax.text(0, 1.035, f"MAE [°C]:  {txt}", transform=ax.transAxes, fontsize=7.6, color="#333", va="bottom")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.", tz=TZ))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0], tz=TZ))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12], tz=TZ))
    comma_axis(ax.yaxis)
for ax in axes[:, 0]:
    ax.set_ylabel("Wassertemperatur [°C]")
handles, labels = axes[0, 0].get_legend_handles_labels()
order = [1, 0, 2, 3, 4]   # Spalten: Messung, Messung vor t0, Oraku | TiRex-2, Ensemble
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 0.025), columnspacing=2.0, handlelength=2.6)
fig.subplots_adjust(left=0.085, right=0.985)
save(fig, "abb03_beispielprognosen_saison")


# ============================================================================= Abb. 4
def continuous(start, end):
    seg = h[(h.dt >= pd.Timestamp(start, tz=TZ)) & (h.dt < pd.Timestamp(end, tz=TZ))].sort_values("dt").copy()
    gaps = seg["dt"].diff() > pd.Timedelta(hours=1)
    rows = []
    for i, (idx, r) in enumerate(seg.iterrows()):
        if gaps.iloc[i]:
            nan = r.copy(); nan[["truth"] + list(PRED.values())] = np.nan; nan["dt"] = r["dt"] - pd.Timedelta(minutes=30); rows.append(nan)
        rows.append(r)
    out = pd.DataFrame(rows)
    out["dt"] = pd.to_datetime(out["dt"], utc=True).dt.tz_convert(TZ)
    gap_spans = [(seg["dt"].iloc[i - 1] + pd.Timedelta(hours=1), seg["dt"].iloc[i]) for i in range(1, len(seg)) if gaps.iloc[i]]
    return out, gap_spans


start, end = "2026-03-15", "2026-05-16"
seg, gap_spans = continuous(start, end)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(WF, 8.0 * CM), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
for a, b in gap_spans:
    for ax in (ax1, ax2):
        ax.axvspan(a, b, color="#e5e7eb", zorder=0)
ax1.plot(seg.dt, seg.truth, color="#000", lw=0.9, label=STYLE["truth"]["label"], zorder=3)
ax1.plot(seg.dt, seg[PRED["AO"]], color=ENS, lw=1.2, ls="-", label=STYLE["AO"]["label"], zorder=2)
ax1.plot(seg.dt, seg[PRED["Oraku"]], color=COL["Oraku"], lw=0.9, ls=(0, (3, 1.5)), label=STYLE["Oraku"]["label"], zorder=4)
for vdate, vlabel in VERSION_CHANGES:
    vd = pd.Timestamp(vdate, tz=TZ)
    if pd.Timestamp(start, tz=TZ) <= vd < pd.Timestamp(end, tz=TZ):
        for ax in (ax1, ax2):
            ax.axvline(vd, color="#4b5563", lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax1.text(vd, 0.985, f"Oraku {vlabel}", fontsize=8, color="#374151", va="top", ha="left" if vlabel == "v1.0" else "right",
                 transform=ax1.get_xaxis_transform(), bbox=dict(fc="white", ec="none", pad=1.2), zorder=5)
ax1.set_ylabel("Wassertemperatur [°C]")
ax1.set_title("Frühjahrserwärmung 15.03.–15.05.2026: aneinandergereihte 48-h-Prognosen (Start alle 2 Tage, 00:00)", loc="left", fontsize=8.5)
comma_axis(ax1.yaxis)
lo, hi = ax1.get_ylim(); ax1.set_ylim(lo, hi + 0.12 * (hi - lo))
ax2.plot(seg.dt, seg[PRED["AO"]] - seg.truth, color=ENS, lw=1.0, ls="-", zorder=2)
ax2.plot(seg.dt, seg[PRED["Oraku"]] - seg.truth, color=COL["Oraku"], lw=0.9, ls=(0, (3, 1.5)), zorder=3)
ax2.axhline(0, color="#000", lw=0.6); ax2.set_ylabel("Fehler [°C]")
comma_axis(ax2.yaxis)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.", tz=TZ)); ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7, tz=TZ))
h1, l1 = ax1.get_legend_handles_labels()
H, L = [h1[0], h1[2], h1[1]], [l1[0], l1[2], l1[1]]
if gap_spans:
    H.append(Patch(fc="#e5e7eb", ec="none")); L.append("keine Prognose (Fall wegen Messlücke ausgeschlossen)")
fig.legend(H, L, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.025), columnspacing=2.0, handlelength=2.6)
fig.subplots_adjust(left=0.075, right=0.985)
save(fig, "abb04_verlauf_fruehling")

# ============================================================================= Abb. 5
bh = pd.read_csv(TAB / "bias_by_hour_season.csv")
fig, axes = plt.subplots(2, 2, figsize=(W, 7.3 * CM), sharex=True, sharey=True, gridspec_kw={"hspace": 0.42, "wspace": 0.12})
for ax, se in zip(axes.ravel(), ["Winter", "Fruehling", "Sommer", "Herbst"]):
    g = bh[bh.season == se]; n = int(s[s.season == se].shape[0])
    for m in ["Oraku", "T", "AO"]:
        ax.plot(g.hod, g[m], **STYLE[m])
    ax.axhline(0, color="#000", lw=0.6); ax.set_title(f"{SEASON_DE[se]} (n = {n} Fenster)", loc="left", fontsize=8.5, weight="bold")
    ax.set_xticks(range(0, 24, 6)); ax.set_xlim(0, 23)
    comma_axis(ax.yaxis)
for ax in axes[1]: ax.set_xlabel("Tagesstunde (Lokalzeit)")
for ax in axes[:, 0]: ax.set_ylabel("Fehler [°C]")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.subplots_adjust(left=0.10, right=0.985, bottom=0.17)
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.035), columnspacing=1.4, handlelength=2.6)
save(fig, "abb05_bias_tagesgang")
(OUT / "fig_hydro_v3_log.txt").write_text("\n".join(LOG) + "\n")
print("\n".join(LOG))
