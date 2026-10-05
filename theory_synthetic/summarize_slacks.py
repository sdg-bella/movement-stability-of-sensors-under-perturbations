"""
Fill the XX placeholders in Section 8.1 from the transition-results CSV.

Usage:
    python summarize_slacks.py path/to/transition_results.csv

The CSV must contain the columns added by safety_margins_patch.py.
Tolerances are the ones used in the experiment code.
"""

import sys

import numpy as np
import pandas as pd

PD_TOL = 1e-11
INEQ_TOL = 1e-12
OPT_ATOL = 1e-9

df = pd.read_csv(sys.argv[1])
e = df[df["eligible"] == True]  # noqa: E712


def report(name, values, tol):
    v = float(np.nanmin(values))
    orders = np.log10(v / tol) if v > 0 else float("nan")
    print(f"{name:<40s} min = {v:.3e}   tol = {tol:.0e}   orders above tol = {orders:.1f}")
    return v


print(f"eligible cases: {len(e)} of {len(df)}")
print(f"moving cases (P_act > 0): {(e['P_act'] > 0).sum()}")
print()
report("PD: lambda_min at reference", e["pd_slack_reference"], PD_TOL)
report("PD: lambda_min at new maximizers", e["pd_slack_new_maximizers"], PD_TOL)
s1 = report("smallness condition 1 slack", e["smallness_slack_1"], INEQ_TOL)
s2 = report("smallness condition 2 slack", e["smallness_slack_2"], INEQ_TOL)
report("crossing slack Gamma(P_cert) - B_t", e["crossing_slack"], INEQ_TOL)
report("preceding slack B_t - Gamma(P_cert-1)",
       e.loc[e["P_cert"] >= 1, "preceding_slack"], INEQ_TOL)
report("loss-bound slack B_t - max loss", e["loss_bound_slack"], OPT_ATOL)
print()
print("cases with a non-singleton numerical maximizer family at t+1:",
      int((e["new_family_size"] > 1).sum()),
      "| largest spread:", float(e["new_family_spread"].max()))
if min(s1, s2) < 0:
    print("WARNING: some eligible case violates an exact smallness condition "
          "(admitted only by the 1e-12 tolerance).")
print()
gap = (e["P_cert"] - e["P_act"])
mv = e["P_act"] > 0
print("median P_cert - P_act, all eligible:", float(gap.median()))
print("median P_cert - P_act, moving cases:", float(gap[mv].median()))
print("sharp moving cases (P_cert == P_act):", int((gap[mv] == 0).sum()))
