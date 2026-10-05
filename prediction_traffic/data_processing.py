#!/usr/bin/env python3
"""
inputs:
pems-bay.h5
metr-la.h5
distances_bay_2017.csv
distances_la_2012.csv

steps:
- preserve original timestamps and sensor IDs
- interpret exact zero values and nonfinite values as missing
- PEMS-BAY: keep every sensor at the data-quality stage
- METR-LA: remove sensors with >10% zero/nonfinite values
- no interpolation, fill, normalization, or smoothing
- build fixed movement graph:
    - only with data-eligible benchmark sensors
    - connect each sensor to its 3 nearest sensors by road-network distance
    - symmetrize by taking the union of those local edges
    - no added artificial bridged edges
    - keep largest naturally connected component
    - take a deterministic BFS subnetwork of 25 sensors, seeded by the first retained HDF5 sensor (in original column order) in that component
- export only the fixed 25-sensor analysis matrix plus enough metadata to reproduce the experiment
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# paper choices

FIVE_MINUTES_NS = 5 * 60 * 1_000_000_000
METR_MAX_MISSING = 0.10
MOVEMENT_K = 3
N_SELECTED = 25

DATASET_CONFIG = {
    "pems_bay": {
        "label": "PEMS-BAY",
        "preferred_hdf_key": "/speed",
        "remove_high_missing_sensors": False,
    },
    "metr_la": {
        "label": "METR-LA",
        "preferred_hdf_key": "/df",
        "remove_high_missing_sensors": True,
    },
}


def normalize_id(value: object) -> str:
    """Normalize sensor IDs without changing ordinary integer-looking IDs."""
    s = str(value).strip()
    if s.endswith(".0"):
        try:
            return str(int(float(s)))
        except ValueError:
            pass
    return s


def discover_hdf_key(path: Path, preferred: str) -> str:
    with pd.HDFStore(path, mode="r") as store:
        keys = list(store.keys())
    if preferred in keys:
        return preferred
    if len(keys) == 1:
        return keys[0]
    raise ValueError(
        f"Could not choose a unique HDF5 key in {path}. Found {keys}; "
        f"expected {preferred!r}."
    )


def load_and_validate_h5(path: Path, dataset_key: str) -> tuple[pd.DataFrame, str]:
    cfg = DATASET_CONFIG[dataset_key]
    key = discover_hdf_key(path, cfg["preferred_hdf_key"])
    frame = pd.read_hdf(path, key=key)

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{path} must use a pandas DatetimeIndex.")
    if not frame.index.is_unique:
        raise ValueError(f"{path} contains duplicate timestamps.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path} timestamps are not increasing.")

    frame = frame.copy()
    frame.columns = [normalize_id(c) for c in frame.columns]
    if pd.Index(frame.columns).duplicated().any():
        raise ValueError("Sensor IDs are not unique after normalization.")
    try:
        numeric = frame.to_numpy(dtype=float)
    except Exception as exc:
        raise TypeError(f"{path} contains nonnumeric traffic values.") from exc
    frame = pd.DataFrame(numeric, index=frame.index.copy(), columns=frame.columns.copy())
    return frame, key


def cadence_break_count(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 0
    delta = np.diff(index.asi8)
    return int(np.count_nonzero(delta != FIVE_MINUTES_NS))


def clean_data_quality(
    raw: pd.DataFrame, dataset_key: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    #non-finite values and zeros are treated as missing in METR-LA
    cfg = DATASET_CONFIG[dataset_key]
    values = raw.to_numpy(dtype=float, copy=True)
    bad = (~np.isfinite(values)) | (values == 0.0)
    missing_rate = bad.mean(axis=0)

    values[bad] = np.nan
    cleaned_all = pd.DataFrame(values, index=raw.index.copy(), columns=raw.columns.copy())

    if cfg["remove_high_missing_sensors"]:
        keep_mask = missing_rate <= METR_MAX_MISSING
    else:
        keep_mask = np.ones(len(raw.columns), dtype=bool)

    retained_ids = [sid for sid, keep in zip(raw.columns, keep_mask) if keep]
    cleaned = cleaned_all.loc[:, retained_ids].copy()

    quality = pd.DataFrame(
        {
            "sensor_id": list(raw.columns),
            "zero_or_nonfinite_rate": missing_rate,
            "retained_by_data_quality": keep_mask,
        }
    )

    stats = {
        "raw_time_steps": int(raw.shape[0]),
        "raw_sensors": int(raw.shape[1]),
        "zero_or_nonfinite_count": int(bad.sum()),
        "zero_or_nonfinite_rate": float(bad.mean()),
        "retained_after_data_quality": int(len(retained_ids)),
        "removed_by_data_quality": int(raw.shape[1] - len(retained_ids)),
        "non_5_minute_timestamp_gaps": cadence_break_count(raw.index),
        "first_timestamp": raw.index[0].isoformat(),
        "last_timestamp": raw.index[-1].isoformat(),
    }
    return cleaned, quality, stats


def read_distance_csv(path: Path) -> pd.DataFrame:
    first = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in first.columns}
    from_col = next((lower[k] for k in ("from", "source", "src") if k in lower), None)
    to_col = next((lower[k] for k in ("to", "target", "dst") if k in lower), None)
    dist_col = next((lower[k] for k in ("distance", "cost", "dist") if k in lower), None)

    if from_col is None or to_col is None or dist_col is None:
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 3:
            raise ValueError(f"{path} must contain at least three columns.")
        df = df.iloc[:, :3].copy()
        df.columns = ["from", "to", "distance"]
    else:
        df = first[[from_col, to_col, dist_col]].copy()
        df.columns = ["from", "to", "distance"]

    df["from"] = df["from"].map(normalize_id)
    df["to"] = df["to"].map(normalize_id)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df[np.isfinite(df["distance"]) & (df["distance"] >= 0)].copy()
    df = df[df["from"] != df["to"]].copy()
    if df.empty:
        raise ValueError(f"No usable road-distance rows were found in {path}.")
    return df


def pair_key(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def symmetrized_pair_distances(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    #for each unordered sensor pair, keep the smaller available road distance
    pair_dist: dict[tuple[str, str], float] = {}
    for u, v, d in df[["from", "to", "distance"]].itertuples(index=False, name=None):
        key = pair_key(u, v)
        d = float(d)
        if key not in pair_dist or d < pair_dist[key]:
            pair_dist[key] = d
    return pair_dist


def adjacency_from_edges(nodes: Iterable[str], edges: set[tuple[str, str]]) -> dict[str, set[str]]:
    adj = {u: set() for u in nodes}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def connected_components(nodes: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    adj = adjacency_from_edges(nodes, edges)
    seen: set[str] = set()
    comps: list[list[str]] = []
    for root in nodes:
        if root in seen:
            continue
        queue = deque([root])
        seen.add(root)
        comp: list[str] = []
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        comps.append(comp)
    return sorted(comps, key=lambda c: (-len(c), c[0]))


def bfs_selected_subnetwork(
    data_order: list[str],
    largest_component: set[str],
    native_edges: set[tuple[str, str]],
    n: int,
) -> tuple[str, list[str]]:
    #BFS
    seed = next((sid for sid in data_order if sid in largest_component), None)
    if seed is None:
        raise ValueError("No retained benchmark sensor lies in the largest movement component.")

    adj = adjacency_from_edges(largest_component, native_edges)
    order = {sid: i for i, sid in enumerate(data_order)}
    queue = deque([seed])
    seen = {seed}
    chosen: list[str] = []

    while queue and len(chosen) < n:
        u = queue.popleft()
        chosen.append(u)
        for v in sorted(adj[u], key=lambda x: order.get(x, 10**12)):
            if v not in seen:
                seen.add(v)
                queue.append(v)

    if len(chosen) < n:
        raise ValueError(
            f"Largest native movement component yielded only {len(chosen)} BFS sensors; "
            f"requested {n}."
        )
    return seed, chosen


def hop_distance_matrix(selected: list[str], edges: set[tuple[str, str]]) -> np.ndarray:
    adj = adjacency_from_edges(selected, edges)
    n = len(selected)
    pos = {sid: i for i, sid in enumerate(selected)}
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)

    for s, sid in enumerate(selected):
        queue = deque([sid])
        while queue:
            u = queue.popleft()
            ui = pos[u]
            for v in adj[u]:
                vi = pos[v]
                if not np.isfinite(D[s, vi]):
                    D[s, vi] = D[s, ui] + 1
                    queue.append(v)

    if not np.isfinite(D).all():
        raise ValueError("The selected 25-sensor movement graph is not connected.")
    return D.astype(int)


def build_fixed_movement_subnetwork(
    retained_ids_in_data_order: list[str],
    distance_csv: Path,
) -> tuple[list[str], pd.DataFrame, dict[str, object], set[str]]:
    #build the fixed k=3 native road-distance graph and select 25 sensors
    distances = read_distance_csv(distance_csv)
    retained = set(retained_ids_in_data_order)
    distance_nodes = set(distances["from"]) | set(distances["to"])
    present = set(retained) & distance_nodes
    absent = set(retained) - distance_nodes

    filtered = distances[
        distances["from"].isin(present) & distances["to"].isin(present)
    ].copy()
    pair_dist = symmetrized_pair_distances(filtered)

    represented: set[str] = set()
    for u, v in pair_dist:
        represented.add(u)
        represented.add(v)
    if len(represented) < N_SELECTED:
        raise ValueError(
            f"Only {len(represented)} retained sensors have road-distance pairs; "
            f"need at least {N_SELECTED}."
        )

    nbr_dist: dict[str, list[tuple[float, str]]] = {u: [] for u in represented}
    for (u, v), d in pair_dist.items():
        nbr_dist[u].append((d, v))
        nbr_dist[v].append((d, u))

    native_edges: set[tuple[str, str]] = set()
    for u in represented:
        nearest = sorted(nbr_dist[u], key=lambda x: (x[0], x[1]))[:MOVEMENT_K]
        for _, v in nearest:
            native_edges.add(pair_key(u, v))

    nodes = sorted(represented)
    comps = connected_components(nodes, native_edges)
    largest = set(comps[0])

    seed, selected = bfs_selected_subnetwork(
        retained_ids_in_data_order, largest, native_edges, N_SELECTED
    )
    selected_set = set(selected)
    selected_edges = {e for e in native_edges if e[0] in selected_set and e[1] in selected_set}
    D = hop_distance_matrix(selected, selected_edges)

    edge_rows = [
        {"source": u, "target": v, "road_distance": pair_dist[(u, v)]}
        for u, v in sorted(selected_edges)
    ]
    edge_df = pd.DataFrame(edge_rows)

    degree = {sid: 0 for sid in selected}
    for u, v in selected_edges:
        degree[u] += 1
        degree[v] += 1
    deg_values = np.asarray(list(degree.values()), dtype=float)
    edge_distances = edge_df["road_distance"].to_numpy(dtype=float)

    graph_stats = {
        "movement_knn_k": MOVEMENT_K,
        "retained_ids_present_in_distance_file": int(len(present)),
        "retained_ids_absent_from_distance_file": int(len(absent)),
        "absent_sensor_ids": sorted(absent),
        "native_component_count": int(len(comps)),
        "native_component_sizes": [int(len(c)) for c in comps],
        "largest_native_component_size": int(len(largest)),
        "artificial_bridge_edges_added": 0,
        "graph_seed_sensor": seed,
        "selected_sensor_count": int(len(selected)),
        "selected_edge_count": int(len(selected_edges)),
        "selected_graph_diameter_hops": int(D.max()),
        "selected_degree_min": int(deg_values.min()),
        "selected_degree_median": float(np.median(deg_values)),
        "selected_degree_max": int(deg_values.max()),
        "selected_road_edge_distance_min": float(edge_distances.min()),
        "selected_road_edge_distance_median": float(np.median(edge_distances)),
        "selected_road_edge_distance_max": float(edge_distances.max()),
    }
    return selected, edge_df, graph_stats, largest


def save_dataset(
    dataset_key: str,
    h5_path: Path,
    distance_path: Path,
    out_root: Path,
) -> dict[str, object]:
    cfg = DATASET_CONFIG[dataset_key]
    raw, hdf_key = load_and_validate_h5(h5_path, dataset_key)
    cleaned, quality, data_stats = clean_data_quality(raw, dataset_key)

    retained_ids = list(cleaned.columns)
    selected_ids, movement_edges, graph_stats, largest_component = build_fixed_movement_subnetwork(
        retained_ids, distance_path
    )

    distance_frame = read_distance_csv(distance_path)
    distance_nodes = set(distance_frame["from"]) | set(distance_frame["to"])
    selected_set = set(selected_ids)
    quality["present_in_distance_file"] = quality["sensor_id"].isin(distance_nodes)
    quality["in_largest_native_component"] = quality["sensor_id"].isin(largest_component)
    quality["selected_for_paper"] = quality["sensor_id"].isin(selected_set)

    selected_frame = cleaned.loc[:, selected_ids].copy()
    out = out_root / dataset_key
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / "traffic_selected.npz",
        data=selected_frame.to_numpy(dtype=np.float64),
        timestamps_ns=selected_frame.index.asi8.astype(np.int64),
        sensor_ids=np.asarray(selected_ids, dtype="U32"),
    )
    movement_edges.to_csv(out / "movement_edges.csv", index=False)
    quality.to_csv(out / "sensor_quality.csv", index=False)

    quality_lookup = quality.set_index("sensor_id")
    selected_quality = pd.DataFrame({"sensor_id": selected_ids})
    selected_quality["bfs_order"] = np.arange(len(selected_ids), dtype=int)
    selected_quality["zero_or_nonfinite_rate"] = [
        float(quality_lookup.loc[sid, "zero_or_nonfinite_rate"]) for sid in selected_ids
    ]
    selected_quality.to_csv(out / "selected_sensors.csv", index=False)

    metadata = {
        "dataset_key": dataset_key,
        "dataset_label": cfg["label"],
        "source_h5_file": h5_path.name,
        "source_hdf_key": hdf_key,
        "source_distance_file": distance_path.name,
        **data_stats,
        **graph_stats,
        "selected_sensor_ids": selected_ids,
        "preprocessing_protocol": {
            "zero_values_treated_as_missing": True,
            "nonfinite_values_treated_as_missing": True,
            "interpolation": False,
            "normalization": False,
            "smoothing": False,
            "metr_la_max_missing_rate": METR_MAX_MISSING if dataset_key == "metr_la" else None,
            "movement_graph": "undirected union of each eligible sensor's 3 nearest road-distance neighbors; largest native component only; no artificial bridges",
            "subnetwork_selection": "deterministic BFS from first retained HDF5 sensor in largest native movement component",
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("=" * 78)
    print(cfg["label"])
    print(f"Raw matrix: {raw.shape[0]:,} time steps x {raw.shape[1]:,} sensors")
    print(
        f"Zero/nonfinite rate: {data_stats['zero_or_nonfinite_rate']:.6%} "
        f"({data_stats['zero_or_nonfinite_count']:,} entries)"
    )
    print(f"Retained after data-quality rule: {data_stats['retained_after_data_quality']:,} sensors")
    print(f"Largest native k=3 movement component: {graph_stats['largest_native_component_size']:,} sensors")
    print(f"Frozen BFS seed: {graph_stats['graph_seed_sensor']}")
    print(f"Frozen paper subnetwork: {len(selected_ids)} sensors, {len(movement_edges)} edges")
    print(f"Selected graph diameter: {graph_stats['selected_graph_diameter_hops']} hops")
    print("Selected IDs: " + ", ".join(selected_ids))
    print(f"Wrote: {out}")

    return metadata


def write_top_level_summary(metadata: list[dict[str, object]], out_root: Path) -> None:
    rows = []
    for m in metadata:
        rows.append(
            {
                "dataset": m["dataset_label"],
                "time_steps": m["raw_time_steps"],
                "raw_sensors": m["raw_sensors"],
                "zero_or_nonfinite_rate": m["zero_or_nonfinite_rate"],
                "retained_after_data_quality": m["retained_after_data_quality"],
                "largest_native_component": m["largest_native_component_size"],
                "selected_sensors": m["selected_sensor_count"],
                "selected_edges": m["selected_edge_count"],
                "selected_graph_diameter_hops": m["selected_graph_diameter_hops"],
                "graph_seed_sensor": m["graph_seed_sensor"],
            }
        )
    pd.DataFrame(rows).to_csv(out_root / "preprocessing_summary.csv", index=False)

    protocol = {
        "paper_preprocessing_version": 1,
        "metr_la_missingness_threshold": METR_MAX_MISSING,
        "movement_knn_k": MOVEMENT_K,
        "selected_subnetwork_size": N_SELECTED,
        "datasets": [m["dataset_key"] for m in metadata],
    }
    (out_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pems-h5", type=Path, default=Path("pems-bay.h5"))
    parser.add_argument("--metr-h5", type=Path, default=Path("metr-la.h5"))
    parser.add_argument("--pems-distances", type=Path, default=Path("distances_bay_2017.csv"))
    parser.add_argument("--metr-distances", type=Path, default=Path("distances_la_2012.csv"))
    parser.add_argument("--output", type=Path, default=Path("traffic_processed"))
    args = parser.parse_args()

    for path in (args.pems_h5, args.metr_h5, args.pems_distances, args.metr_distances):
        if not path.exists():
            raise FileNotFoundError(f"Required raw input not found: {path}")

    args.output.mkdir(parents=True, exist_ok=True)
    metas = [
        save_dataset("pems_bay", args.pems_h5, args.pems_distances, args.output),
        save_dataset("metr_la", args.metr_h5, args.metr_distances, args.output),
    ]
    write_top_level_summary(metas, args.output)

    print("=" * 78)
    print("Preprocessing complete.")
    print(f"Downstream experiment input: {args.output.resolve()}")
    print("The experiment script no longer needs the four raw files once this folder exists.")


if __name__ == "__main__":
    main()
