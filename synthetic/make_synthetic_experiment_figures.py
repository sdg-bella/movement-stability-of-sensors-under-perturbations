#!/usr/bin/env python3
"""Generate the two publication figures from exported CSVs only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def figure_1(inp: Path, out: Path) -> None:
    T = pd.read_csv(inp / "exp2_certificates.csv")
    T = T[T.smallness_pass.astype(bool) & T.P_cert.notna()].copy()
    grouped = (
        T.groupby(["P_act", "P_cert"], as_index=False)
         .size()
         .rename(columns={"size": "count"})
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sizes = 32 + 24 * np.sqrt(grouped["count"].to_numpy())
    ax.scatter(grouped.P_act, grouped.P_cert, s=sizes, alpha=0.72,
               edgecolors="white", linewidths=0.6)

    for _, row in grouped.iterrows():
        if row["count"] >= 5:
            ax.text(row.P_act, row.P_cert, str(int(row["count"])),
                    ha="center", va="center", fontsize=7.5)

    m = int(max(T.P_act.max(), T.P_cert.max()))
    ax.plot([0, m], [0, m], "--", linewidth=1.3,
            label=r"$P_{\mathrm{cert}}=P_{\mathrm{act}}$")
    ax.set_xlim(-0.25, m + 0.35)
    ax.set_ylim(-0.25, m + 0.35)
    ax.set_xticks(range(m + 1))
    ax.set_yticks(range(m + 1))
    ax.set_xlabel(r"$P_{\mathrm{act}}$")
    ax.set_ylabel(r"$P_{\mathrm{cert}}$")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "Fig1.pdf", bbox_inches="tight")
    fig.savefig(out / "Fig1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_2(inp: Path, out: Path) -> None:
    L = pd.read_csv(inp / "exp5_landscape.csv")
    C = pd.read_csv(inp / "exp5_margin_curve.csv")
    S = pd.read_csv(inp / "exp5_summary.csv").iloc[0]

    finite = L.objective_gap.notna()
    best = L.is_best_at_exact_distance.astype(bool) & finite
    newopt = L.is_new_optimizer.astype(bool) & finite

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax = axes[0]
    ax.scatter(L.loc[finite, "distance"], L.loc[finite, "objective_gap"],
               s=25, alpha=0.48, label="all configurations")
    ax.plot(L.loc[best, "distance"], L.loc[best, "objective_gap"],
            marker="o", linewidth=1.4, label="best loss at exact distance")
    if newopt.any():
        ax.scatter(L.loc[newopt, "distance"], L.loc[newopt, "objective_gap"],
                   s=85, marker="D", facecolors="none", linewidths=1.4,
                   label="new optimizer")
    ax.axhline(S.B_t, linestyle="--", linewidth=1.3, label=r"$B_t$")
    ax.set_xlabel(r"$d_{\mathrm{match}}(S,S_t^\star)$")
    ax.set_ylabel(r"$F_t(S_t^\star)-F_t(S)$")
    ax.set_xticks(sorted(L.distance.unique()))
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    finite_curve = np.isfinite(C.Gamma)
    ax.step(C.loc[finite_curve, "P"], C.loc[finite_curve, "Gamma"],
            where="post", linewidth=1.8, label=r"$M_t(P)$")
    ax.axhline(S.B_t, linestyle="--", linewidth=1.3, label=r"$B_t$")
    ax.axvline(S.P_act, linestyle=":", linewidth=1.4,
               label=rf"$P_{{\mathrm{{act}}}}={int(S.P_act)}$")
    ax.axvline(S.P_cert, linestyle="-.", linewidth=1.4,
               label=rf"$P_{{\mathrm{{cert}}}}={int(S.P_cert)}$")
    ax.set_xlabel(r"radius $P$")
    ax.set_ylabel(r"$M_t(P)$")
    ax.set_xticks(range(int(S.P_max) + 1))
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(out / "Fig2.pdf", bbox_inches="tight")
    fig.savefig(out / "Fig2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="paper_outputs")
    p.add_argument("--output", default="figures")
    args = p.parse_args()
    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    figure_1(inp, out)
    figure_2(inp, out)
    print(f"Wrote {out / 'Fig1.pdf'} and {out / 'Fig2.pdf'}")


if __name__ == "__main__":
    main()
