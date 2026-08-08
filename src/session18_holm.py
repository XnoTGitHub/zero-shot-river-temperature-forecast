"""Session 18: Holm-Korrektur ueber die fuenf Primaertests (P6, Reviewer-Runde 8).

Fuer die vorab definierten Primaerhypothesen Q1, Q2a, Q2b, Q3a, Q3b
berechnen wir Holm-adjustierte p-Werte (family-wise error control).

Grundlage: p-Werte aus statistical_tests.csv (all_seasons, n=128).

Ausgabe: src/holm_primary.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent


def holm_adjust(pvals):
    """Holm-Bonferroni step-down: p_(i) * (m - i + 1), monoton nicht-fallend, gekappt bei 1."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, k in enumerate(order):
        val = p[k] * (m - i)
        running = max(running, val)
        adj[k] = min(running, 1.0)
    return adj


def main():
    tests = pd.read_csv(BASE / "all_seasons_tests.csv")
    lookup = {(row["A"], row["B"]): row for _, row in tests.iterrows()}

    primary = [
        ("Q1  T vs Oraku",       ("T",  "Oraku")),
        ("Q2a R vs B_persist",   ("R",  "B_persist")),
        ("Q2b R vs T",           ("R",  "T")),
        ("Q3a AO vs Oraku",      ("AO", "Oraku")),
        ("Q3b AO vs T",          ("AO", "T")),
    ]

    rows = []
    for label, key in primary:
        if key not in lookup:
            print(f"WARN: '{key}' nicht in all_seasons_tests.csv; verfuegbar:")
            print(sorted(lookup.keys()))
            raise KeyError(key)
        r = lookup[key]
        rows.append({
            "question": label,
            "comparison": f"{key[0]} vs {key[1]}",
            "n": int(r["n"]),
            "delta_MAE": float(r["mean_diff"]),
            "p_two_sided_raw": float(r["wilcoxon_p_two_sided"]),
        })

    df = pd.DataFrame(rows)
    df["p_holm_adj"] = holm_adjust(df["p_two_sided_raw"].to_numpy())
    df["sig_raw_alpha05"] = df["p_two_sided_raw"] < 0.05
    df["sig_holm_alpha05"] = df["p_holm_adj"] < 0.05

    df.to_csv(BASE / "holm_primary.csv", index=False)

    print("Holm-Korrektur ueber die 5 Primaertests (Q1, Q2a, Q2b, Q3a, Q3b):")
    with pd.option_context("display.float_format", "{:.3e}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
