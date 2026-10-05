#!/usr/bin/env python3
"""
reads outputs from data_processing.py

2. Fig 1: actual movement P_act versus exact certified radius P_cert, shown as bubble-count panels for the two datasets.
3. Fig 2: one transparently selected METR-LA nontrivial case study showing the complete old-objective landscape and the exact distance-margin curve.
4. Table summarizing preprocessing scale, theorem applicability, nonzero movement, certificate informativeness, and violations.

fixed at...
    q = 48 five-minute samples (4 hours)
    h = 1 five-minute sample
    K = 5
    m = 2
    n = 25 (already frozen by preprocessing)
    raw / uncentered windows


eg. command:
python run_traffic_paper_experiments.py \
    --input traffic_processed \
    --output traffic_paper_outputs
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Q = 48
H = 1
K = 5
M = 2
EXPECTED_N = 25
FIVE_MINUTES_NS = 5 * 60 * 1_000_000_000

PD_TOL = 1e-11
OPT_ATOL = 1e-9
OPT_RTOL = 1e-9
GAP_ZERO_TOL = 1e-10
INEQ_TOL = 1e-12

DATASETS = ["pems_bay", "metr_la"]


#validation log

@dataclass
class ValidationRecord:
    dataset: str
    check: str
    passed: bool
    detail: str


class ValidationLog:
    def __init__(self) -> None:
        self.rows: list[ValidationRecord] = []

    def check(
        self,
        dataset: str,
        condition: bool,
        check: str,
        detail: str = "",
        fatal: bool = True,
    ) -> None:
        passed = bool(condition)
        self.rows.append(ValidationRecord(dataset, check, passed, detail))
        if fatal and not passed:
            raise AssertionError(f"{dataset}: {check}. {detail}")

    def write(self, path: Path) -> None:
        pd.DataFrame([r.__dict__ for r in self.rows]).to_csv(path, index=False)


#prepared-data loading and graph helpers

@dataclass
class PreparedDataset:
    key: str
    label: str
    data: np.ndarray #time x sensor
    timestamps_ns: np.ndarray #int64 nanoseconds
    sensor_ids: list[str]
    metadata: dict[str, object]
    node_hop_dist: np.ndarray #n x n exact integer hop distances


def load_prepared_dataset(root: Path, key: str, vlog: ValidationLog) -> PreparedDataset:
    ddir = root / key
    npz_path = ddir / "traffic_selected.npz"
    meta_path = ddir / "metadata.json"
    edge_path = ddir / "movement_edges.csv"
    for p in (npz_path, meta_path, edge_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing prepared input: {p}")

    with np.load(npz_path, allow_pickle=False) as z:
        data = np.asarray(z["data"], dtype=float)
        timestamps_ns = np.asarray(z["timestamps_ns"], dtype=np.int64)
        sensor_ids = [str(x) for x in z["sensor_ids"].tolist()]
    metadata = json.loads(meta_path.read_text())

    vlog.check(key, data.ndim == 2, "traffic matrix is 2D", f"shape={data.shape}")
    vlog.check(key, data.shape[0] == len(timestamps_ns), "timestamp count matches rows")
    vlog.check(key, data.shape[1] == len(sensor_ids), "sensor-ID count matches columns")
    vlog.check(key, data.shape[1] == EXPECTED_N, "frozen subnetwork has 25 sensors", f"n={data.shape[1]}")
    vlog.check(key, len(set(sensor_ids)) == len(sensor_ids), "selected sensor IDs are unique")
    vlog.check(key, np.all(np.diff(timestamps_ns) > 0), "timestamps are strictly increasing")

    edges = pd.read_csv(edge_path)
    if not {"source", "target"}.issubset(edges.columns):
        raise ValueError(f"{edge_path} must contain source,target columns.")
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    node_hop_dist = compute_hop_distances(sensor_ids, edges)

    vlog.check(key, np.isfinite(node_hop_dist).all(), "selected movement graph is connected")
    vlog.check(
        key,
        int(node_hop_dist.max()) == int(metadata["selected_graph_diameter_hops"]),
        "recomputed graph diameter matches preprocessing metadata",
        f"recomputed={int(node_hop_dist.max())}, metadata={metadata['selected_graph_diameter_hops']}",
    )

    return PreparedDataset(
        key=key,
        label=str(metadata["dataset_label"]),
        data=data,
        timestamps_ns=timestamps_ns,
        sensor_ids=sensor_ids,
        metadata=metadata,
        node_hop_dist=node_hop_dist,
    )


def compute_hop_distances(sensor_ids: list[str], edges: pd.DataFrame) -> np.ndarray:
    pos = {sid: i for i, sid in enumerate(sensor_ids)}
    adj: list[set[int]] = [set() for _ in sensor_ids]
    for u, v in edges[["source", "target"]].itertuples(index=False, name=None):
        u, v = str(u), str(v)
        if u not in pos or v not in pos or u == v:
            continue
        i, j = pos[u], pos[v]
        adj[i].add(j)
        adj[j].add(i)

    n = len(sensor_ids)
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)
    for s in range(n):
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not np.isfinite(D[s, v]):
                    D[s, v] = D[s, u] + 1
                    q.append(v)
    if not np.isfinite(D).all():
        raise ValueError("Prepared selected movement graph is disconnected.")
    return D.astype(int)


#linear algebra

@dataclass
class WindowStats:
    start_row: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    basis: np.ndarray #n x K leading right singular vectors
    objectives: np.ndarray #one per m-subset
    argmax_indices: np.ndarray
    reference_argmax_index: int
    reference_kappa: float
    reference_sigma: float
    captured_energy_fraction: float


def all_configurations(n: int, m: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(n), m))


def window_is_valid(data: np.ndarray, timestamps_ns: np.ndarray, start: int) -> bool:
    stop = start + Q
    X = data[start:stop]
    if X.shape[0] != Q:
        return False
    if not np.isfinite(X).all():
        return False
    t = timestamps_ns[start:stop]
    return bool(np.all(np.diff(t) == FIVE_MINUTES_NS))


def d_optimal_objectives(
    V: np.ndarray,
    configs: Sequence[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    """logdet objectives & smallest compression eigenvalues

    Paper runs use m=2, so use the closed 2x2 formulas for speed and numerical
    transparency.  Q = VV^T is the empirical orthogonal projector.
    """
    if M != 2:
        raise NotImplementedError("This final paper script is intentionally frozen at m=2.")

    Qmat = V @ V.T
    ii = np.asarray([c[0] for c in configs], dtype=int)
    jj = np.asarray([c[1] for c in configs], dtype=int)
    a = Qmat[ii, ii]
    d = Qmat[jj, jj]
    b = Qmat[ii, jj]

    det = a * d - b * b
    disc = np.sqrt(np.maximum(0.0, (a - d) ** 2 + 4.0 * b * b))
    kappa = 0.5 * (a + d - disc)

    obj = np.full(len(configs), -np.inf, dtype=float)
    good = (kappa > PD_TOL) & (det > PD_TOL)
    obj[good] = np.log(det[good])
    return obj, kappa


def optimizer_indices(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("All D-optimal candidates are singular.")
    best = float(values[finite].max())
    tol = OPT_ATOL + OPT_RTOL * max(1.0, abs(best))
    return np.flatnonzero(finite & (values >= best - tol))


def compute_window(
    prepared: PreparedDataset,
    start: int,
    configs: Sequence[tuple[int, ...]],
) -> WindowStats:
    X = prepared.data[start:start + Q]

    #raw / uncentered primary analysis. The columns are sensors, so the leading RIGHT singular vectors define the sensor-coordinate subspace.
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    if K > len(s):
        raise ValueError(f"K={K} exceeds available singular directions {len(s)}.")
    V = Vt[:K].T

    obj, kappas = d_optimal_objectives(V, configs)
    opts = optimizer_indices(obj)
    ref = int(opts[0]) #deterministic reference under any old-time tie
    sigma = float(np.exp(obj[ref]))

    total_energy = float(np.sum(s * s))
    captured = float(np.sum(s[:K] * s[:K]))

    return WindowStats(
        start_row=start,
        start_timestamp_ns=int(prepared.timestamps_ns[start]),
        end_timestamp_ns=int(prepared.timestamps_ns[start + Q - 1]),
        basis=V,
        objectives=obj,
        argmax_indices=opts,
        reference_argmax_index=ref,
        reference_kappa=float(kappas[ref]),
        reference_sigma=sigma,
        captured_energy_fraction=captured / total_energy if total_energy > 0 else np.nan,
    )


def projector_distance(V: np.ndarray, W: np.ndarray) -> float:
    """Spectral norm distance between equal-rank orthogonal projectors."""
    cosines = np.linalg.svd(V.T @ W, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    return float(np.sqrt(max(0.0, 1.0 - float(cosines.min()) ** 2)))


def matching_distance_m2(
    A: tuple[int, int], B: tuple[int, int], node_dist: np.ndarray
) -> int:
    a, b = A
    c, d = B
    direct = max(int(node_dist[a, c]), int(node_dist[b, d]))
    crossed = max(int(node_dist[a, d]), int(node_dist[b, c]))
    return min(direct, crossed)


def configuration_distance_matrix(
    configs: list[tuple[int, ...]], node_dist: np.ndarray
) -> np.ndarray:
    N = len(configs)
    out = np.zeros((N, N), dtype=int)
    for i in range(N):
        Ai = configs[i]
        if len(Ai) != 2:
            raise NotImplementedError("Final paper script is frozen at m=2.")
        for j in range(i + 1, N):
            d = matching_distance_m2(Ai, configs[j], node_dist)
            out[i, j] = out[j, i] = d
    return out


def exact_margin_curve(
    old_objectives: np.ndarray,
    old_ref: int,
    dist_from_ref: np.ndarray,
) -> pd.DataFrame:
    gaps = old_objectives[old_ref] - old_objectives
    gaps[np.abs(gaps) < GAP_ZERO_TOL] = 0.0
    pmax = int(np.max(dist_from_ref))
    rows = []
    for P in range(pmax + 1):
        far = dist_from_ref > P
        candidates = gaps[far]
        finite = np.isfinite(candidates)
        M = float(np.min(candidates[finite])) if finite.any() else np.inf
        rows.append({"P": P, "M": M})
    return pd.DataFrame(rows)


def timestamp_string(ns: int) -> str:
    return pd.Timestamp(ns).isoformat(sep=" ")


#experiments on one dataset

@dataclass
class DatasetRun:
    prepared: PreparedDataset
    configs: list[tuple[int, ...]]
    config_dist: np.ndarray
    transitions: pd.DataFrame
    summary: dict[str, object]


def analyze_pair(
    old: WindowStats,
    new: WindowStats,
    configs: list[tuple[int, ...]],
    config_dist: np.ndarray,
) -> dict[str, object]:
    zeta = projector_distance(old.basis, new.basis)
    kappa = old.reference_kappa
    sigma = old.reference_sigma

    eta1 = 2.0 * M * zeta / kappa
    eta2 = 2.0 * M * zeta * math.exp(eta1) / sigma
    B = eta1 + eta2
    rhs1 = kappa / 2.0
    rhs2 = sigma * math.exp(-eta1) / 2.0
    small1 = zeta <= rhs1 + INEQ_TOL
    small2 = zeta <= rhs2 + INEQ_TOL
    eligible = bool(small1 and small2)

    ref = old.reference_argmax_index
    new_opts = new.argmax_indices
    dist_ref = config_dist[ref]
    pact = int(np.max(dist_ref[new_opts]))
    pmax = int(np.max(dist_ref))

    old_losses_at_new = old.objectives[ref] - old.objectives[new_opts]
    finite_losses = old_losses_at_new[np.isfinite(old_losses_at_new)]
    objective_loss_max = float(np.max(finite_losses)) if len(finite_losses) else np.nan

    pcert = np.nan
    nontrivial = np.nan
    violation = np.nan
    certificate_gap = np.nan
    if eligible:
        curve = exact_margin_curve(old.objectives, ref, dist_ref)
        passed = curve.loc[curve["M"] > B + INEQ_TOL]
        if passed.empty:
            raise RuntimeError("At P_max the far family must be empty, so M=+infinity.")
        pcert = int(passed.iloc[0]["P"])
        nontrivial = bool(pcert < pmax)
        violation = bool(pact > pcert)
        certificate_gap = int(pcert - pact)

    return {
        "old_start_row": old.start_row,
        "new_start_row": new.start_row,
        "old_start_timestamp": timestamp_string(old.start_timestamp_ns),
        "new_start_timestamp": timestamp_string(new.start_timestamp_ns),
        "projector_drift_zeta": zeta,
        "kappa_star": kappa,
        "sigma_star": sigma,
        "eta1": eta1,
        "eta2": eta2,
        "B_t": B,
        "smallness_rhs_1": rhs1,
        "smallness_rhs_2": rhs2,
        "smallness_1": bool(small1),
        "smallness_2": bool(small2),
        "eligible": eligible,
        "old_optimizer_count": int(len(old.argmax_indices)),
        "new_optimizer_count": int(len(new.argmax_indices)),
        "objective_loss_at_new_optimizer_max": objective_loss_max,
        "objective_loss_bound_holds": bool(objective_loss_max <= B + 1e-9) if eligible else np.nan,
        "P_act": pact,
        "P_cert": pcert,
        "P_max": pmax,
        "nontrivial_certificate": nontrivial,
        "certificate_gap": certificate_gap,
        "certificate_violation": violation,
    }


def run_dataset(prepared: PreparedDataset, vlog: ValidationLog) -> DatasetRun:
    n = prepared.data.shape[1]
    configs = all_configurations(n, M)
    config_dist = configuration_distance_matrix(configs, prepared.node_hop_dist)

    candidate_windows = prepared.data.shape[0] - Q + 1
    accepted_windows = 0
    rows: list[dict[str, object]] = []
    captured_energy: list[float] = []

    prev: WindowStats | None = None
    prev_start: int | None = None

    print(f"\nRunning {prepared.label}: {candidate_windows:,} candidate windows ...", flush=True)
    t0 = time.time()

    for start in range(candidate_windows):
        if not window_is_valid(prepared.data, prepared.timestamps_ns, start):
            prev = None
            prev_start = None
            continue

        current = compute_window(prepared, start, configs)
        accepted_windows += 1
        captured_energy.append(current.captured_energy_fraction)

        if prev is not None and prev_start == start - H:
            rows.append(analyze_pair(prev, current, configs, config_dist))

        prev = current
        prev_start = start

        if (start + 1) % 5000 == 0:
            print(
                f"  processed {start + 1:,}/{candidate_windows:,} candidate windows; "
                f"transitions={len(rows):,}",
                flush=True,
            )

    transitions = pd.DataFrame(rows)
    if transitions.empty:
        raise RuntimeError(f"No valid transitions were found for {prepared.label}.")

    elig = transitions["eligible"].astype(bool)
    eligible = transitions.loc[elig].copy()
    movers = eligible["P_act"] > 0
    nontrivial = eligible["nontrivial_certificate"].astype(bool)
    violations = eligible["certificate_violation"].astype(bool)

    #checks
    vlog.check(
        prepared.key,
        bool(eligible["objective_loss_bound_holds"].astype(bool).all()),
        "objective-loss theorem holds on every eligible transition",
        f"eligible={len(eligible):,}",
    )
    vlog.check(
        prepared.key,
        not bool(violations.any()),
        "P_act <= P_cert on every eligible transition",
        f"violations={int(violations.sum())}",
    )
    vlog.check(
        prepared.key,
        bool((eligible["P_cert"] <= eligible["P_max"]).all()),
        "every certificate lies within the maximum feasible matching radius",
    )

    sharp_movers = movers & (eligible["P_act"] == eligible["P_cert"])
    elapsed = time.time() - t0

    summary = {
        "dataset_key": prepared.key,
        "dataset": prepared.label,
        "raw_sensors": int(prepared.metadata["raw_sensors"]),
        "retained_sensors": int(prepared.metadata["retained_after_data_quality"]),
        "selected_sensors": n,
        "graph_diameter_hops": int(prepared.node_hop_dist.max()),
        "candidate_windows": int(candidate_windows),
        "accepted_windows": int(accepted_windows),
        "accepted_window_fraction": float(accepted_windows / candidate_windows),
        "usable_transitions": int(len(transitions)),
        "eligible_transitions": int(len(eligible)),
        "eligibility_rate": float(len(eligible) / len(transitions)),
        "eligible_nonzero_movement": int(movers.sum()),
        "nonzero_movement_rate_among_eligible": float(movers.mean()),
        "nontrivial_certificates": int(nontrivial.sum()),
        "nontrivial_certificate_rate_among_eligible": float(nontrivial.mean()),
        "sharp_nonzero_certificates": int(sharp_movers.sum()),
        "sharp_rate_among_movers": float(sharp_movers.sum() / movers.sum()) if movers.any() else np.nan,
        "certificate_violations": int(violations.sum()),
        "median_certificate_gap_among_movers": float(
            np.median(eligible.loc[movers, "certificate_gap"].to_numpy(dtype=float))
        ) if movers.any() else np.nan,
        "max_P_act_among_eligible": int(eligible["P_act"].max()),
        "median_projector_drift_zeta": float(transitions["projector_drift_zeta"].median()),
        "captured_energy_fraction_min": float(np.min(captured_energy)),
        "captured_energy_fraction_median": float(np.median(captured_energy)),
        "runtime_seconds": float(elapsed),
    }

    print(json.dumps(summary, indent=2), flush=True)
    return DatasetRun(prepared, configs, config_dist, transitions, summary)


#real-data case-study selection & export

@dataclass
class CaseStudy:
    transition: pd.Series
    old: WindowStats
    new: WindowStats
    landscape: pd.DataFrame
    margin_curve: pd.DataFrame


def choose_metr_case(run: DatasetRun, vlog: ValidationLog) -> CaseStudy:
    """Choose the case without manual timestamp selection.

    Rule fixed in code: among METR-LA transitions that are theorem-eligible, have P_act>0, are integer-radius sharp (P_cert=P_act), and are genuinely nontrivial (P_cert<P_max), choose the largest P_act; break ties by earliest time.

    This emphasizes a large, exact, non-vacuous movement event while avoiding
    visual cherry-picking.
    """
    T = run.transitions.copy()
    candidates = T[
        T["eligible"].astype(bool)
        & (T["P_act"] > 0)
        & (T["P_act"] == T["P_cert"])
        & (T["P_cert"] < T["P_max"])
    ].copy()
    vlog.check(
        run.prepared.key,
        not candidates.empty,
        "at least one sharp nontrivial moving transition exists for the case study",
    )

    chosen = candidates.sort_values(
        ["P_act", "old_start_row"], ascending=[False, True]
    ).iloc[0]

    old_start = int(chosen["old_start_row"])
    new_start = int(chosen["new_start_row"])
    old = compute_window(run.prepared, old_start, run.configs)
    new = compute_window(run.prepared, new_start, run.configs)

    ref = old.reference_argmax_index
    dist = run.config_dist[ref]
    loss = old.objectives[ref] - old.objectives
    loss[np.abs(loss) < GAP_ZERO_TOL] = 0.0

    new_opt_set = set(int(i) for i in new.argmax_indices)
    best_at_distance: dict[int, float] = {}
    for d in sorted(set(int(x) for x in dist)):
        vals = loss[(dist == d) & np.isfinite(loss)]
        if len(vals):
            best_at_distance[d] = float(np.min(vals))

    landscape_rows = []
    for i, config in enumerate(run.configs):
        finite = bool(np.isfinite(loss[i]))
        landscape_rows.append(
            {
                "config_index": i,
                "sensor_1": run.prepared.sensor_ids[config[0]],
                "sensor_2": run.prepared.sensor_ids[config[1]],
                "distance": int(dist[i]),
                "objective_gap": float(loss[i]) if finite else np.nan,
                "is_new_optimizer": i in new_opt_set,
                "is_best_at_exact_distance": (
                    finite
                    and int(dist[i]) in best_at_distance
                    and abs(float(loss[i]) - best_at_distance[int(dist[i])]) <= GAP_ZERO_TOL
                ),
            }
        )
    landscape = pd.DataFrame(landscape_rows)
    margin = exact_margin_curve(old.objectives, ref, dist)
    margin["B_t"] = float(chosen["B_t"])

    return CaseStudy(chosen, old, new, landscape, margin)


#outputs

def figure_certificates(runs: list[DatasetRun], out_pdf: Path, out_png: Path) -> None:
    """Two-panel bubble-count version of P_act versus P_cert."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))

    grouped_by_run = []
    global_max_count = 1
    for run in runs:
        E = run.transitions.loc[run.transitions["eligible"].astype(bool)].copy()
        E = E[np.isfinite(pd.to_numeric(E["P_cert"], errors="coerce"))]
        grouped = (
            E.groupby(["P_act", "P_cert"], as_index=False)
             .size()
             .rename(columns={"size": "count"})
        )
        grouped_by_run.append((run, E, grouped))
        if len(grouped):
            global_max_count = max(global_max_count, int(grouped["count"].max()))

    for ax, (run, E, grouped) in zip(axes, grouped_by_run):
        counts = grouped["count"].to_numpy(dtype=float)
        sizes = 34.0 + 250.0 * np.sqrt(counts / global_max_count)
        ax.scatter(
            grouped["P_act"], grouped["P_cert"],
            s=sizes, alpha=0.72, edgecolors="white", linewidths=0.6,
        )

        #labels are the multiplicities at each integer pair. Suppress only singletons to keep the figure readable.
        for _, row in grouped.iterrows():
            if int(row["count"]) >= 2:
                ax.text(
                    float(row["P_act"]), float(row["P_cert"]), str(int(row["count"])),
                    ha="center", va="center", fontsize=7.0,
                )

        lim = int(max(E["P_act"].max(), E["P_cert"].max()))
        ax.plot([0, lim], [0, lim], "--", linewidth=1.2,
                label=r"$P_{\rm cert}=P_{\rm act}$")
        ax.set_xlim(-0.35, lim + 0.35)
        ax.set_ylim(-0.35, lim + 0.35)
        ax.set_xticks(range(lim + 1))
        ax.set_yticks(range(lim + 1))
        ax.set_xlabel(r"actual movement $P_{\rm act}$")
        ax.set_ylabel(r"certified radius $P_{\rm cert}$")
        violations = int(E["certificate_violation"].astype(bool).sum())
        ax.set_title(f"{run.prepared.label}  (eligible: {len(E):,}; violations: {violations})")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.20)
        ax.legend(frameon=False, loc="lower right", fontsize=8)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_case_study(case: CaseStudy, out_pdf: Path, out_png: Path) -> None:
    L = case.landscape
    C = case.margin_curve
    row = case.transition

    finite = L["objective_gap"].notna()
    best = L["is_best_at_exact_distance"].astype(bool) & finite
    newopt = L["is_new_optimizer"].astype(bool) & finite

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.35))

    ax = axes[0]
    ax.scatter(
        L.loc[finite, "distance"], L.loc[finite, "objective_gap"],
        s=25, alpha=0.45, label="all configurations",
    )
    best_curve = (
        L.loc[finite]
         .groupby("distance", as_index=False)["objective_gap"]
         .min()
         .sort_values("distance")
    )
    ax.plot(
        best_curve["distance"], best_curve["objective_gap"],
        marker="o", linewidth=1.4, label="minimum loss at each distance",
    )
    if newopt.any():
        ax.scatter(
            L.loc[newopt, "distance"], L.loc[newopt, "objective_gap"],
            s=90,
            marker="D",
            color="#40E0D0",
            edgecolors="black",
            linewidths=0.8,
            label="new optimizer",
        )
    ax.axhline(float(row["B_t"]), linestyle="--", linewidth=1.3, label=r"$B_t$")
    ax.set_xlabel(r"$d_{\rm match}(S,S_t^\star)$")
    ax.set_ylabel(r"$F_t(S_t^\star)-F_t(S)$")
    ax.set_xticks(range(int(L["distance"].max()) + 1))
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Objective loss by matching distance")

    ax = axes[1]
    finite_curve = np.isfinite(C["M"].to_numpy(dtype=float))

    #continuous-looking piecewise-constant distance-margin curve
    ax.step(
        C.loc[finite_curve, "P"],
        C.loc[finite_curve, "M"],
        where="post",
        linewidth=1.8,
        label=r"$M_t(P)$",
    )

    #perturbation threshold
    ax.axhline(
        float(row["B_t"]),
        linestyle="--",
        linewidth=1.3,
        label=r"$B_t$",
    )

    pact = int(row["P_act"])
    pcert = int(row["P_cert"])

    #actual and certified movement radius
    if pact == pcert:
        ax.axvline(
            pact,
            linestyle=":",
            linewidth=1.5,
            label=rf"$P_{{\rm act}}=P_{{\rm cert}}={pact}$",
        )
    else:
        ax.axvline(
            pact,
            linestyle=":",
            linewidth=1.4,
            label=rf"$P_{{\rm act}}={pact}$",
        )
        ax.axvline(
            pcert,
            linestyle="-.",
            linewidth=1.4,
            label=rf"$P_{{\rm cert}}={pcert}$",
        )

    #explanation
    ax.annotate(
        r"$M_t(P)\approx 0.054$ for $P\leq 5$",
        xy=(3, 0.054),
        xytext=(1.3, 0.55),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.8,
        ),
        fontsize=9,
    )

    ax.annotate(
        r"$M_t(6)\approx 2.822>B_t$",
        xy=(6, 2.822),
        xytext=(3.25, 2.73),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.8,
        ),
        fontsize=9,
    )

    ax.set_xlabel(r"radius $P$")
    ax.set_ylabel(r"$M_t(P)$")

    ax.set_xlim(0, int(row["P_max"]))

    ax.grid(alpha=0.20)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Distance margin by radius")

    old_time = pd.Timestamp(row["old_start_timestamp"]).strftime("%b %d, %Y, %H:%M")
    new_time = pd.Timestamp(row["new_start_timestamp"]).strftime("%H:%M")

    date_text = f"METR-LA transition: {old_time}–{new_time}"
    fig.suptitle(date_text, fontsize=10)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_paper_table(runs: list[DatasetRun], out_csv: Path, out_tex: Path) -> pd.DataFrame:
    rows = []
    for run in runs:
        s = run.summary
        rows.append(
            {
                "Dataset": s["dataset"],
                "Sensors (raw/retained/used)": f"{s['raw_sensors']}/{s['retained_sensors']}/{s['selected_sensors']}",
                "Usable transitions": int(s["usable_transitions"]),
                "Eligible": f"{s['eligible_transitions']:,} ({100*s['eligibility_rate']:.1f}%)",
                "P_act>0": f"{s['eligible_nonzero_movement']:,} ({100*s['nonzero_movement_rate_among_eligible']:.1f}%)",
                "Nontrivial cert.": f"{s['nontrivial_certificates']:,} ({100*s['nontrivial_certificate_rate_among_eligible']:.1f}%)",
                "Violations": int(s["certificate_violations"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)

    def tex_escape_percent(text: object) -> str:
        return str(text).replace("%", r"\%")

    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Exact certification on real traffic data. Percentages in the $P_{\rm act}>0$ and nontrivial-certificate columns are among theorem-eligible transitions. A certificate is nontrivial when $P_{\rm cert}<P_{\max}$, where $P_{\max}=\max_S d_{\rm match}(S_t^\star,S)$.}",
        r"\label{tab:traffic_results}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"Dataset & Sensors & Usable & Eligible & $P_{\rm act}>0$ & Nontrivial cert. & Violations \\",
        r" & raw/ret./used & transitions & & & & \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['Dataset']} & {row['Sensors (raw/retained/used)']} & {int(row['Usable transitions']):,} & "
            f"{tex_escape_percent(row['Eligible'])} & {tex_escape_percent(row['P_act>0'])} & "
            f"{tex_escape_percent(row['Nontrivial cert.'])} & {int(row['Violations'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_tex.write_text("\n".join(lines) + "\n")
    return table


#main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("traffic_processed"))
    parser.add_argument("--output", type=Path, default=Path("traffic_paper_outputs"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    figdir = args.output / "figures"
    figdir.mkdir(exist_ok=True)
    vlog = ValidationLog()

    # Confirm that the preprocessing protocol is the one this paper script expects.
    protocol_path = args.input / "protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(f"Missing preprocessing protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    vlog.check("ALL", int(protocol["movement_knn_k"]) == 3, "movement graph uses frozen k=3")
    vlog.check("ALL", int(protocol["selected_subnetwork_size"]) == EXPECTED_N, "preprocessing froze n=25")
    vlog.check("ALL", abs(float(protocol["metr_la_missingness_threshold"]) - 0.10) < 1e-15, "METR-LA threshold is 10%")

    prepared = [load_prepared_dataset(args.input, key, vlog) for key in DATASETS]
    runs = [run_dataset(d, vlog) for d in prepared]

    #save tables
    for run in runs:
        run.transitions.to_csv(args.output / f"{run.prepared.key}_transition_results.csv", index=False)
        (args.output / f"{run.prepared.key}_summary.json").write_text(
            json.dumps(run.summary, indent=2) + "\n"
        )

    #fig 1
    figure_certificates(
        runs,
        figdir / "TrafficFig1_certificates.pdf",
        figdir / "TrafficFig1_certificates.png",
    )

    #fig 2
    metr_run = next(r for r in runs if r.prepared.key == "metr_la")
    case = choose_metr_case(metr_run, vlog)
    case.landscape.to_csv(args.output / "case_study_landscape.csv", index=False)
    case.margin_curve.to_csv(args.output / "case_study_margin_curve.csv", index=False)
    case_summary = pd.DataFrame([{
        "dataset": "METR-LA",
        "selection_rule": "largest P_act among eligible nonzero transitions with P_cert=P_act<P_max; earliest time breaks ties",
        **case.transition.to_dict(),
    }])
    case_summary.to_csv(args.output / "case_study_summary.csv", index=False)
    figure_case_study(
        case,
        figdir / "TrafficFig2_case_study.pdf",
        figdir / "TrafficFig2_case_study.png",
    )

    #table
    make_paper_table(
        runs,
        args.output / "Table1_real_traffic.csv",
        args.output / "Table1_real_traffic.tex",
    )

    #validations
    vlog.write(args.output / "validation_log.csv")
    manifest = {
        "paper_protocol": {
            "q": Q,
            "h": H,
            "rank_K": K,
            "sensor_budget_m": M,
            "subnetwork_n": EXPECTED_N,
            "centered_within_window": False,
            "projector": "leading K right singular vectors of each q x n raw traffic window",
            "projector_drift": "spectral norm between empirical orthogonal projectors, evaluated by principal angles",
            "P_act": "worst bottleneck matching distance from the deterministic old reference optimizer to any new optimizer",
            "P_cert": "smallest integer P with exact M_t^Q(P) > B_t",
            "nontrivial_certificate": "P_cert < P_max, where P_max=max_S d_match(S_t^*,S)",
            "case_study_selection": "METR-LA; largest P_act among eligible moving sharp nontrivial transitions; earliest time breaks ties",
        },
        "numerical_conventions": {
            "PD_TOL": PD_TOL,
            "OPT_ATOL": OPT_ATOL,
            "OPT_RTOL": OPT_RTOL,
            "GAP_ZERO_TOL": GAP_ZERO_TOL,
            "INEQ_TOL": INEQ_TOL,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n" + "=" * 78)
    print("FINAL PAPER OUTPUTS WRITTEN")
    print(f"Figure 1: {figdir / 'TrafficFig1_certificates.pdf'}")
    print(f"Figure 2: {figdir / 'TrafficFig2_case_study.pdf'}")
    print(f"Table:    {args.output / 'Table1_real_traffic.tex'}")
    print(f"Audits:   {args.output / 'validation_log.csv'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
