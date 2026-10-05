#!/usr/bin/env python3
"""Reproducibility code for Experiments 2 and 5 in
"Metric--Margin Stability of D-Optimal Subset Selection".

Experiment 2: exact actual movement P_act versus exact certificate P_cert.
Experiment 5: full objective landscape and exact distance-margin curve.

The code exhaustively enumerates all m-subsets in the reported finite instances.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import eigh
from scipy.sparse.csgraph import shortest_path

# -----------------------------------------------------------------------------
# Fixed numerical conventions used by the paper run
# -----------------------------------------------------------------------------

GLOBAL_SEED = 20260825
PD_TOL = 1e-11
OPT_ATOL = 1e-9
OPT_RTOL = 1e-9
GAP_ZERO_TOL = 1e-10
INEQ_TOL = 1e-12

PROFILES = {
    "quick": {"exp2_seeds": 1},
    "paper": {"exp2_seeds": 2},
}

EXPERIMENT_INFO = {
    2: "Exact finite-instance movement certificates",
    5: "Full objective landscape and exact distance margin",
}


# -----------------------------------------------------------------------------
# Validation / consistency log
# -----------------------------------------------------------------------------

@dataclass
class ValidationRecord:
    experiment: int
    check: str
    passed: bool
    detail: str


class ValidationLog:
    def __init__(self) -> None:
        self.rows: List[ValidationRecord] = []

    def check(self, experiment: int, condition: bool, check: str,
              detail: str = "", fatal: bool = True) -> None:
        passed = bool(condition)
        self.rows.append(ValidationRecord(experiment, check, passed, detail))
        if fatal and not passed:
            raise AssertionError(f"Experiment {experiment}: {check}. {detail}")

    def write(self, out: Path) -> None:
        pd.DataFrame([r.__dict__ for r in self.rows]).to_csv(
            out / "validation_log.csv", index=False
        )


# -----------------------------------------------------------------------------
# Linear algebra and combinatorial helpers
# -----------------------------------------------------------------------------

def symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def stable_logdet_psd(A: np.ndarray, pd_tol: float = PD_TOL) -> float:
    """Return log(det(A)) for a numerically positive-definite PSD matrix."""
    ev = np.linalg.eigvalsh(symmetrize(A))
    if ev.size == 0 or ev[0] <= pd_tol:
        return -np.inf
    return float(np.log(ev).sum())


def laplacian_from_graph(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    A = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        A[u, v] = A[v, u] = w
    return np.diag(A.sum(axis=1)) - A


def spectral_projector(L: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    evals, evecs = eigh(symmetrize(L))
    U = evecs[:, :K]
    P = symmetrize(U @ U.T)
    return P, evals, U


def all_sets(n: int, m: int) -> List[Tuple[int, ...]]:
    return list(itertools.combinations(range(n), m))


def objective(P: np.ndarray, S: Sequence[int]) -> float:
    return stable_logdet_psd(P[np.ix_(S, S)])


def objectives(P: np.ndarray, sets: Sequence[Tuple[int, ...]]) -> np.ndarray:
    return np.array([objective(P, S) for S in sets], dtype=float)


def optimizers_from_values(
    sets: Sequence[Tuple[int, ...]], vals: np.ndarray,
    atol: float = OPT_ATOL, rtol: float = OPT_RTOL,
) -> Tuple[List[Tuple[int, ...]], float]:
    finite = np.isfinite(vals)
    if not finite.any():
        raise ValueError("No finite objective values.")
    best = float(vals[finite].max())
    tol = atol + rtol * max(1.0, abs(best))
    opts = [
        sets[i] for i, v in enumerate(vals)
        if np.isfinite(v) and abs(float(v) - best) <= tol
    ]
    return opts, best


def optimizers(P: np.ndarray, sets: Sequence[Tuple[int, ...]]):
    vals = objectives(P, sets)
    opts, best = optimizers_from_values(sets, vals)
    return opts, vals, best


def reference_optimizer(opts: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
    # Deterministic reference when the baseline has multiple maximizers.
    return min(opts)


def unweighted_distance_matrix(G: nx.Graph) -> np.ndarray:
    A = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()), weight=None)
    D = shortest_path(A, directed=False, unweighted=True)
    if not np.isfinite(D).all():
        raise ValueError("Movement graph must be connected.")
    return np.rint(D).astype(int)


def matching_distance(S: Sequence[int], T: Sequence[int], D: np.ndarray) -> int:
    """Exact bottleneck matching distance (brute force; m=2 in paper runs)."""
    if len(S) != len(T):
        raise ValueError("Sets must have equal cardinality.")
    best = math.inf
    for perm in itertools.permutations(T):
        radius = max(int(D[S[i], perm[i]]) for i in range(len(S)))
        best = min(best, radius)
    return int(best)


def replacement_count(S: Sequence[int], T: Sequence[int]) -> int:
    return len(set(S) - set(T))


def distances_from_reference(
    sets: Sequence[Tuple[int, ...]], Sstar: Tuple[int, ...], D: np.ndarray
) -> np.ndarray:
    return np.array([matching_distance(Sstar, S, D) for S in sets], dtype=int)


def objective_gaps(
    vals: np.ndarray, sets: Sequence[Tuple[int, ...]], Sstar: Tuple[int, ...]
) -> np.ndarray:
    idx = sets.index(Sstar)
    fstar = float(vals[idx])
    gaps = np.full(len(vals), np.inf, dtype=float)
    finite = np.isfinite(vals)
    gaps[finite] = fstar - vals[finite]
    gaps[np.abs(gaps) < GAP_ZERO_TOL] = 0.0
    return gaps


def exact_margin_from_arrays(gaps: np.ndarray, dists: np.ndarray, radius: int) -> float:
    far = dists > radius
    if not far.any():
        return np.inf
    candidates = gaps[far]
    finite = np.isfinite(candidates)
    if not finite.any():
        return np.inf
    return float(candidates[finite].min())


def margin_curve(gaps: np.ndarray, dists: np.ndarray, max_radius: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"P": P, "Gamma": exact_margin_from_arrays(gaps, dists, P)}
        for P in range(max_radius + 1)
    ])


def exact_pcert_from_arrays(
    gaps: np.ndarray, dists: np.ndarray, B: float, max_radius: int
) -> int:
    """Smallest integer P with Gamma_t(P) > B.

    max_radius is P_max := max_S d_match(Sstar,S), not the graph diameter.
    At P_max the far family is empty, so Gamma_t(P_max)=+inf.
    """
    for P in range(max_radius + 1):
        if exact_margin_from_arrays(gaps, dists, P) > B:
            return P
    raise RuntimeError(
        "At the maximum feasible matching radius the far family must be empty."
    )


def theorem_quantities(
    Pt: np.ndarray, Pt1: np.ndarray, Sstar: Tuple[int, ...], m: int,
    r: Optional[float] = None,
) -> Dict[str, float]:
    rho = float(np.linalg.norm(Pt - Pt1, 2))
    if r is None:
        r = rho
    if r + INEQ_TOL < rho:
        raise ValueError("Certified upper bound r must satisfy r >= rho.")

    Cstar = symmetrize(Pt[np.ix_(Sstar, Sstar)])
    ev = np.linalg.eigvalsh(Cstar)
    kappa = float(ev[0])
    if kappa <= PD_TOL:
        raise ValueError("Reference optimizer compression is not positive definite.")
    sigma = float(np.exp(np.log(ev).sum()))

    eta1 = float(2 * m * r / kappa)
    eta2 = float(2 * m * r * np.exp(eta1) / sigma)
    B = eta1 + eta2

    rhs1 = kappa / 2
    rhs2 = sigma * np.exp(-eta1) / 2
    smallness_1 = r <= rhs1 + INEQ_TOL
    smallness_2 = r <= rhs2 + INEQ_TOL

    return {
        "rho": rho,
        "r": float(r),
        "kappa_star": kappa,
        "sigma_star": sigma,
        "eta1": eta1,
        "eta2": eta2,
        "B_t": float(B),
        "smallness_rhs_1": float(rhs1),
        "smallness_rhs_2": float(rhs2),
        "smallness_slack_1": float(rhs1 - r),
        "smallness_slack_2": float(rhs2 - r),
        "smallness_1": bool(smallness_1),
        "smallness_2": bool(smallness_2),
        "smallness_pass": bool(smallness_1 and smallness_2),
    }


def laplacian_perturbation_quantities(
    L0: np.ndarray, L1: np.ndarray, evals0: np.ndarray, K: int
) -> Dict[str, float]:
    epsilon = float(np.linalg.norm(L1 - L0, 2))
    delta = float(evals0[K] - evals0[K - 1])
    dk = np.inf if epsilon >= delta else float(epsilon / (delta - epsilon))
    return {
        "epsilon": epsilon,
        "delta": delta,
        "epsilon_over_delta": epsilon / delta if delta > 0 else np.inf,
        "DK_bound": dk,
        "DK_assumption_pass": bool(epsilon < delta / 4),
    }


def worst_new_optimizer_movement(
    Sstar: Tuple[int, ...], opts1: Sequence[Tuple[int, ...]], D: np.ndarray
) -> int:
    return max(matching_distance(Sstar, S, D) for S in opts1)


def graph_diameter(D: np.ndarray) -> int:
    return int(D.max())


# -----------------------------------------------------------------------------
# Graph generation and perturbation (exact paper protocol)
# -----------------------------------------------------------------------------

def random_weighted_copy(
    G: nx.Graph, rng: np.random.Generator,
    low: float = 0.7, high: float = 1.3,
) -> nx.Graph:
    H = G.copy()
    for u, v in H.edges():
        H[u][v]["weight"] = float(rng.uniform(low, high))
    return H


def _connect_rgg_components_nearest(G: nx.Graph, pos: Dict[int, np.ndarray]) -> None:
    while not nx.is_connected(G):
        comps = [list(c) for c in nx.connected_components(G)]
        c0 = comps[0]
        best = None
        for j in range(1, len(comps)):
            for u in c0:
                for v in comps[j]:
                    d = float(np.linalg.norm(np.asarray(pos[u]) - np.asarray(pos[v])))
                    if best is None or d < best[0]:
                        best = (d, u, v)
        assert best is not None
        _, u, v = best
        G.add_edge(u, v)


def make_graph(
    family: str, n: int, seed: int
) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    rng = np.random.default_rng(seed)

    if family == "grid":
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise ValueError("Grid experiments require n to be a perfect square.")
        base = nx.grid_2d_graph(side, side)
        mapping = {node: i for i, node in enumerate(sorted(base.nodes()))}
        G = nx.relabel_nodes(base, mapping)
        pos = {
            mapping[(i, j)]: (float(j), float(-i))
            for i in range(side) for j in range(side)
        }

    elif family == "rgg":
        G = nx.random_geometric_graph(n, radius=0.46, seed=seed)
        raw_pos = nx.get_node_attributes(G, "pos")
        _connect_rgg_components_nearest(G, raw_pos)
        pos = {i: (float(raw_pos[i][0]), float(raw_pos[i][1])) for i in range(n)}

    elif family == "sbm":
        sizes = [n // 2, n - n // 2]
        G = nx.stochastic_block_model(
            sizes, [[0.50, 0.08], [0.08, 0.50]], seed=seed
        )
        while not nx.is_connected(G):
            comps = sorted(
                (sorted(c) for c in nx.connected_components(G)),
                key=lambda c: c[0],
            )
            G.add_edge(comps[0][0], comps[1][0])
        spring = nx.spring_layout(G, seed=seed)
        pos = {i: (float(spring[i][0]), float(spring[i][1])) for i in range(n)}

    else:
        raise ValueError(f"Unknown graph family: {family}")

    G = random_weighted_copy(G, rng)
    nx.set_node_attributes(G, {i: pos[i] for i in G.nodes()}, "pos")
    return G, pos


def perturb_single_edge_fraction(
    G: nx.Graph, edge: Tuple[int, int], frac: float
) -> nx.Graph:
    H = G.copy()
    u, v = edge
    w0 = float(G[u][v]["weight"])
    w1 = w0 * (1.0 + frac)
    if w1 <= 0:
        raise ValueError("Perturbation changed an edge weight to nonpositive.")
    H[u][v]["weight"] = w1
    return H


def save_graph_nodes(
    path: Path, pos: Dict[int, Tuple[float, float]],
    P: Optional[np.ndarray] = None,
    Sstar: Optional[Tuple[int, ...]] = None,
) -> None:
    rows = []
    for i in sorted(pos):
        rows.append({
            "node": i,
            "x": pos[i][0],
            "y": pos[i][1],
            "leverage": float(P[i, i]) if P is not None else np.nan,
            "is_reference_optimizer": int(Sstar is not None and i in Sstar),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# -----------------------------------------------------------------------------
# Experiment 2
# -----------------------------------------------------------------------------

def experiment_2(out: Path, profile: Dict, v: ValidationLog) -> None:
    exp = 2
    configs = [("grid", 16, 500), ("rgg", 12, 509), ("sbm", 12, 503)]
    K, m = 4, 2
    fracs = [-0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20]
    rows = []

    for family, n, base_seed in configs:
        for seed_offset in range(profile["exp2_seeds"]):
            seed = base_seed + 37 * seed_offset
            G0, _ = make_graph(family, n, seed)
            D = unweighted_distance_matrix(G0)
            diameter = graph_diameter(D)
            L0 = laplacian_from_graph(G0)
            P0, evals0, _ = spectral_projector(L0, K)
            sets = all_sets(n, m)
            opts0, vals0, _ = optimizers(P0, sets)
            Sstar = reference_optimizer(opts0)
            dists = distances_from_reference(sets, Sstar, D)
            P_max = int(dists.max())
            gaps = objective_gaps(vals0, sets, Sstar)

            finite_pd = []
            for S in sets:
                ev = np.linalg.eigvalsh(symmetrize(P0[np.ix_(S, S)]))
                if ev[0] > PD_TOL:
                    finite_pd.append(float(ev[0]))
            min_positive_compression_eig = min(finite_pd)

            for edge_index, edge in enumerate(G0.edges()):
                for frac in fracs:
                    G1 = perturb_single_edge_fraction(G0, edge, frac)
                    L1 = laplacian_from_graph(G1)
                    P1, _, _ = spectral_projector(L1, K)
                    opts1, vals1, _ = optimizers(P1, sets)
                    tq = theorem_quantities(P0, P1, Sstar, m)
                    lap = laplacian_perturbation_quantities(L0, L1, evals0, K)
                    P_act = worst_new_optimizer_movement(Sstar, opts1, D)

                    finite1 = np.sort(vals1[np.isfinite(vals1)])[::-1]
                    second_best_gap_t1 = (
                        float(finite1[0] - finite1[1]) if finite1.size >= 2 else np.nan
                    )
                    perturbed_pd = []
                    for S in sets:
                        ev1 = np.linalg.eigvalsh(symmetrize(P1[np.ix_(S, S)]))
                        if ev1[0] > PD_TOL:
                            perturbed_pd.append(float(ev1[0]))
                    min_positive_compression_eig_perturbed = min(perturbed_pd)

                    # Numerical safety quantities used in Section 8.1.
                    new_opt_indices = [sets.index(S) for S in opts1]
                    
                    new_opt_kappas = []
                    for S in opts1:
                        C1 = symmetrize(P1[np.ix_(S, S)])
                        new_opt_kappas.append(float(np.linalg.eigvalsh(C1)[0]))
                    pd_slack_new_maximizers = float(min(new_opt_kappas))
                    
                    new_vals = vals1[new_opt_indices]
                    new_family_spread = float(np.max(new_vals) - np.min(new_vals))
                    
                    ref_index = sets.index(Sstar)
                    losses = vals0[ref_index] - vals0[new_opt_indices]
                    losses = losses[np.isfinite(losses)]
                    loss_bound_slack = (
                        float(tq["B_t"] - np.max(losses))
                        if len(losses) else np.nan
                    )
                    
                    P_cert = np.nan
                    crossing_slack = np.nan
                    preceding_slack = np.nan
                    if tq["smallness_pass"]:
                        P_cert = exact_pcert_from_arrays(gaps, dists, tq["B_t"], P_max)
                        gamma_cert = exact_margin_from_arrays(gaps, dists, int(P_cert))
                        crossing_slack = (
                            float(gamma_cert - tq["B_t"])
                            if np.isfinite(gamma_cert) else np.inf
                        )
                        if P_cert >= 1:
                            gamma_before = exact_margin_from_arrays(
                                gaps, dists, int(P_cert) - 1
                            )
                            preceding_slack = float(tq["B_t"] - gamma_before)
                        v.check(
                            exp, P_act <= P_cert,
                            "Computed certificate implication is satisfied",
                            f"family={family}, seed={seed}, edge={edge}, frac={frac}, "
                            f"Pact={P_act}, Pcert={P_cert}",
                        )

                    rows.append({
                        "family": family,
                        "n": n,
                        "seed": seed,
                        "K": K,
                        "m": m,
                        "edge_index": edge_index,
                        "u": edge[0],
                        "v": edge[1],
                        "fractional_weight_change": frac,
                        **lap,
                        **tq,
                        "P_act": P_act,
                        "P_cert": P_cert,
                        "P_max": P_max,
                        "diameter": diameter,
                        "nontrivial_certificate": bool(
                            tq["smallness_pass"] and P_cert < P_max
                        ),
                        "integer_radius_sharp": bool(
                            tq["smallness_pass"] and P_cert == P_act
                        ),
                        "certificate_slack_hops": (
                            float(P_cert - P_act) if tq["smallness_pass"] else np.nan
                        ),
                        "certificate_crossing_slack": crossing_slack,
                        "min_positive_compression_eig_baseline": min_positive_compression_eig,
                        "min_positive_compression_eig_perturbed": min_positive_compression_eig_perturbed,
                        "perturbed_top_two_objective_gap": second_best_gap_t1,
                        "eligible": bool(tq["smallness_pass"]),
                        "pd_slack_reference": float(tq["kappa_star"]),
                        "pd_slack_new_maximizers": pd_slack_new_maximizers,
                        "crossing_slack": crossing_slack,
                        "preceding_slack": preceding_slack,
                        "new_family_size": int(len(opts1)),
                        "new_family_spread": new_family_spread,
                        "loss_bound_slack": loss_bound_slack,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(out / "exp2_certificates.csv", index=False)

    cert = df[df.smallness_pass].copy()
    cert["implication_check"] = cert.P_act <= cert.P_cert

    def summarize(group: pd.DataFrame, family: str) -> dict:
        moving = group[group.P_act > 0]
        return {
            "family": family,
            "attempted_cases": int(len(df) if family == "ALL" else len(df[df.family == family])),
            "eligible_cases": int(len(group)),
            "eligibility_rate": float(len(group) / (len(df) if family == "ALL" else len(df[df.family == family]))),
            "implication_check_rate": float(group.implication_check.mean()),
            "moving_cases": int((group.P_act > 0).sum()),
            "nontrivial_certificates": int(group.nontrivial_certificate.sum()),
            "nontrivial_rate_among_eligible": float(group.nontrivial_certificate.mean()),
            "integer_radius_sharp_cases": int(group.integer_radius_sharp.sum()),
            "integer_radius_sharp_rate": float(group.integer_radius_sharp.mean()),
            "moving_sharp_cases": int((moving.P_cert == moving.P_act).sum()),
            "moving_sharp_rate": float((moving.P_cert == moving.P_act).mean()) if len(moving) else np.nan,
            "mean_certificate_slack_hops": float(group.certificate_slack_hops.mean()),
            "median_certificate_slack_hops": float(group.certificate_slack_hops.median()),
            "max_certificate_slack_hops": float(group.certificate_slack_hops.max()),
        }

    rows_summary = [summarize(cert, "ALL")]
    for family in ["grid", "rgg", "sbm"]:
        rows_summary.append(summarize(cert[cert.family == family], family))
    pd.DataFrame(rows_summary).to_csv(out / "exp2_summary.csv", index=False)

    v.check(exp, len(cert) > 0, "At least one perturbation case is eligible",
            f"count={len(cert)}")
    v.check(exp, bool(cert.implication_check.all()),
            "All eligible cases satisfy P_act <= P_cert",
            f"violations={(~cert.implication_check).sum()}")


# -----------------------------------------------------------------------------
# Representative random-geometric instance used by Experiment 5
# -----------------------------------------------------------------------------

def representative_rgg_instance(
    seed: int = 911, n: int = 12, K: int = 4, m: int = 2
):
    G0, pos = make_graph("rgg", n, seed)
    D = unweighted_distance_matrix(G0)
    L0 = laplacian_from_graph(G0)
    P0, evals0, U0 = spectral_projector(L0, K)
    sets = all_sets(n, m)
    opts0, vals0, _ = optimizers(P0, sets)
    Sstar = reference_optimizer(opts0)
    dists = distances_from_reference(sets, Sstar, D)
    P_max = int(dists.max())
    gaps = objective_gaps(vals0, sets, Sstar)

    candidates = []
    fracs = [0.01, -0.01, 0.02, -0.02, 0.05, -0.05, 0.10, -0.10]
    for edge in G0.edges():
        for frac in fracs:
            G1 = perturb_single_edge_fraction(G0, edge, frac)
            L1 = laplacian_from_graph(G1)
            P1, _, _ = spectral_projector(L1, K)
            opts1, _, _ = optimizers(P1, sets)
            tq = theorem_quantities(P0, P1, Sstar, m)
            P_act = worst_new_optimizer_movement(Sstar, opts1, D)
            if tq["smallness_pass"]:
                pcert = exact_pcert_from_arrays(gaps, dists, tq["B_t"], P_max)
                # Deterministic selection rule stated in the manuscript:
                # prefer positive movement, then a nontrivial certificate,
                # then the largest observed projector drift.
                score = (int(P_act > 0), int(pcert < P_max), tq["rho"])
                candidates.append(
                    (score, edge, frac, G1, L1, P1, opts1, tq, P_act, pcert)
                )
    if not candidates:
        raise RuntimeError("No representative perturbation passed sharp smallness.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[0]
    _, edge, frac, G1, L1, P1, opts1, tq, P_act, pcert = chosen
    return {
        "G0": G0, "G1": G1, "pos": pos, "D": D, "L0": L0, "L1": L1,
        "P0": P0, "P1": P1, "evals0": evals0, "U0": U0, "sets": sets,
        "opts0": opts0, "opts1": opts1, "vals0": vals0, "Sstar": Sstar,
        "dists": dists, "gaps": gaps, "tq": tq, "P_act": P_act,
        "P_cert": pcert, "P_max": P_max, "edge": edge, "frac": frac,
        "K": K, "m": m,
    }


# -----------------------------------------------------------------------------
# Experiment 5
# -----------------------------------------------------------------------------

def experiment_5(out: Path, profile: Dict, v: ValidationLog) -> None:
    exp = 5
    R = representative_rgg_instance()
    P0, P1, sets, Sstar = R["P0"], R["P1"], R["sets"], R["Sstar"]
    vals0 = R["vals0"]
    vals1 = objectives(P1, sets)
    dists, gaps = R["dists"], R["gaps"]
    tq = R["tq"]

    rows = []
    for i, S in enumerate(sets):
        C = symmetrize(P0[np.ix_(S, S)])
        ev = np.linalg.eigvalsh(C)
        finite0 = np.isfinite(vals0[i])
        finite1 = np.isfinite(vals1[i])
        rows.append({
            "set_id": i,
            "set": str(S),
            "distance": int(dists[i]),
            "replacement_count": replacement_count(Sstar, S),
            "objective_t": float(vals0[i]) if finite0 else np.nan,
            "objective_t1": float(vals1[i]) if finite1 else np.nan,
            "objective_gap": float(gaps[i]) if np.isfinite(gaps[i]) else np.nan,
            "lambda_min": float(ev[0]),
            "is_reference_optimizer": int(S == Sstar),
            "is_new_optimizer": int(S in R["opts1"]),
            "B_t": tq["B_t"],
        })
    landscape = pd.DataFrame(rows)

    landscape["is_best_at_exact_distance"] = 0
    for d in sorted(landscape.distance.unique()):
        sub = landscape[(landscape.distance == d) & landscape.objective_gap.notna()]
        if len(sub):
            idx = sub.objective_gap.idxmin()
            landscape.loc[idx, "is_best_at_exact_distance"] = 1
    landscape.to_csv(out / "exp5_landscape.csv", index=False)

    curve = margin_curve(gaps, dists, R["P_max"])
    curve["B_t"] = tq["B_t"]
    curve["P_act"] = R["P_act"]
    curve["P_cert"] = R["P_cert"]
    curve["P_max"] = R["P_max"]
    curve.to_csv(out / "exp5_margin_curve.csv", index=False)

    save_graph_nodes(out / "exp5_nodes.csv", R["pos"], P0, Sstar)

    finite_vals1 = np.sort(vals1[np.isfinite(vals1)])[::-1]
    perturbed_top_two_gap = (
        float(finite_vals1[0] - finite_vals1[1]) if finite_vals1.size >= 2 else np.nan
    )
    gamma_cert = exact_margin_from_arrays(gaps, dists, R["P_cert"])
    summary = {
        "seed": 911,
        "n": 12,
        "rgg_radius": 0.46,
        "num_edges": R["G0"].number_of_edges(),
        "K": R["K"],
        "m": R["m"],
        "reference_optimizer": str(R["Sstar"]),
        "num_baseline_optimizers": len(R["opts0"]),
        "num_new_optimizers": len(R["opts1"]),
        "perturbed_edge_u": R["edge"][0],
        "perturbed_edge_v": R["edge"][1],
        "fractional_change": R["frac"],
        "P_act": R["P_act"],
        "P_cert": R["P_cert"],
        "P_max": R["P_max"],
        "certificate_crossing_slack": (
            float(gamma_cert - tq["B_t"]) if np.isfinite(gamma_cert) else np.inf
        ),
        "perturbed_top_two_objective_gap": perturbed_top_two_gap,
        **tq,
    }
    pd.DataFrame([summary]).to_csv(out / "exp5_summary.csv", index=False)

    v.check(exp, tq["smallness_pass"],
            "Representative landscape perturbation is eligible")
    v.check(exp, R["P_act"] <= R["P_cert"],
            "Representative certificate implication is satisfied")
    v.check(exp, R["P_cert"] < R["P_max"],
            "Representative certificate is nontrivial")


# -----------------------------------------------------------------------------
# Summary / manifest
# -----------------------------------------------------------------------------

def write_accuracy_summary(out: Path) -> None:
    T = pd.read_csv(out / "exp2_certificates.csv")
    eligible = T[T.smallness_pass.astype(bool)].copy()

    positive_eigs = pd.concat([
        T.min_positive_compression_eig_baseline,
        T.min_positive_compression_eig_perturbed,
    ]).replace([np.inf, -np.inf], np.nan).dropna()
    top_two = T.perturbed_top_two_objective_gap
    top_two = top_two[(top_two > 0) & np.isfinite(top_two)]
    slack2 = eligible.smallness_slack_2
    crossing = eligible.certificate_crossing_slack
    crossing = crossing[np.isfinite(crossing)]

    rows = [
        {"quantity": "minimum positive compression eigenvalue encountered",
         "value": float(positive_eigs.min())},
        {"quantity": "minimum positive perturbed top-two objective gap",
         "value": float(top_two.min())},
        {"quantity": "minimum eligible second-smallness slack",
         "value": float(slack2.min())},
        {"quantity": "minimum finite certificate-crossing slack",
         "value": float(crossing.min())},
        {"quantity": "PD_TOL", "value": PD_TOL},
        {"quantity": "OPT_ATOL", "value": OPT_ATOL},
        {"quantity": "OPT_RTOL", "value": OPT_RTOL},
        {"quantity": "GAP_ZERO_TOL", "value": GAP_ZERO_TOL},
        {"quantity": "INEQ_TOL", "value": INEQ_TOL},
    ]
    pd.DataFrame(rows).to_csv(out / "numerical_accuracy.csv", index=False)


def write_manifest(out: Path, args, selected: List[int], elapsed: float) -> None:
    manifest = {
        "suite": "reproducibility package: MMSF Experiments 2 and 5",
        "profile": args.profile,
        "selected_experiments": selected,
        "global_seed": GLOBAL_SEED,
        "numerical_tolerances": {
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
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "networkx": nx.__version__,
        },
        "elapsed_seconds": elapsed,
        "interpretation": (
            "The experiments numerically illustrate the proved certificate and "
            "check implementation consistency; they are not statistical validation "
            "of the theorem."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_experiments(text: str) -> List[int]:
    out = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    bad = [x for x in out if x not in EXPERIMENT_INFO]
    if bad:
        raise ValueError(f"Only experiments 2 and 5 are included; got {bad}.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paper Experiments 2 and 5 and export exact finite-instance data."
    )
    parser.add_argument("--profile", choices=PROFILES, default="paper")
    parser.add_argument("--experiments", default="2,5")
    parser.add_argument("--output", default="paper_outputs")
    args = parser.parse_args()

    selected = parse_experiments(args.experiments)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    v = ValidationLog()
    profile = PROFILES[args.profile]

    print(f"MMSF experiments | profile={args.profile} | experiments={selected}")
    print(f"Output: {out.resolve()}")
    start = time.time()
    timing_rows = []

    funcs = {2: experiment_2, 5: experiment_5}
    for num in selected:
        print(f"\n[{num}] {EXPERIMENT_INFO[num]}")
        t0 = time.time()
        funcs[num](out, profile, v)
        dt = time.time() - t0
        timing_rows.append({"experiment": num, "title": EXPERIMENT_INFO[num], "seconds": dt})
        print(f"    completed in {dt:.2f} s")

    elapsed = time.time() - start
    pd.DataFrame(timing_rows).to_csv(out / "timings.csv", index=False)
    v.write(out)
    if 2 in selected:
        write_accuracy_summary(out)
    write_manifest(out, args, selected, elapsed)

    checks = pd.DataFrame([r.__dict__ for r in v.rows])
    failed = int((~checks.passed).sum()) if len(checks) else 0
    print(f"\nDone in {elapsed:.2f} s. Consistency checks: {len(checks)}, failures: {failed}")
    if failed:
        raise SystemExit(1)

    if 2 in selected:
        summary = pd.read_csv(out / "exp2_summary.csv")
        overall = summary[summary.family == "ALL"].iloc[0]
        print(
            "Experiment 2 overall: "
            f"attempted={int(overall.attempted_cases)}, "
            f"eligible={int(overall.eligible_cases)}, "
            f"nontrivial={int(overall.nontrivial_certificates)}, "
            f"moving={int(overall.moving_cases)}, "
            f"integer-sharp={int(overall.integer_radius_sharp_cases)}"
        )
    if 5 in selected:
        s5 = pd.read_csv(out / "exp5_summary.csv").iloc[0]
        print(
            "Experiment 5 representative: "
            f"edge=({int(s5.perturbed_edge_u)},{int(s5.perturbed_edge_v)}), "
            f"fraction={s5.fractional_change:+.2f}, "
            f"P_act={int(s5.P_act)}, P_cert={int(s5.P_cert)}, P_max={int(s5.P_max)}"
        )

    print(
        "\nNext: python make_synthetic_experiment_figures.py "
        "--input paper_outputs --output figures"
    )


if __name__ == "__main__":
    main()
