import math
import random
import networkx as nx
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple
from scipy.stats import spearmanr


def _get_topological_depths(dag: nx.DiGraph) -> Dict[Any, int]:
    depths: Dict[Any, int] = {}
    for node in nx.topological_sort(dag):
        preds = list(dag.predecessors(node))
        depths[node] = 0 if not preds else max(depths[p] for p in preds) + 1
    return depths


def _get_extensions(dag: nx.DiGraph, elements_set: Set[Any]) -> Dict[Any, Set[Any]]:
    ext: Dict[Any, Set[Any]] = {node: set() for node in dag.nodes()}
    for node in reversed(list(nx.topological_sort(dag))):
        if node in elements_set:
            ext[node].add(node)
        for child in dag.successors(node):
            ext[node].update(ext[child])
    return ext


def _clean_transitive_reduction(dag: nx.DiGraph) -> nx.DiGraph:
    tr = nx.transitive_reduction(dag)
    result = nx.DiGraph()
    result.add_nodes_from(dag.nodes(data=True))
    result.add_edges_from(tr.edges(data=True))
    return result


def _get_lca_vector(
    dag: nx.DiGraph,
    elements: List[Any],
    sample_pairs: Optional[List[Tuple[Any, Any]]] = None,
) -> Dict[Tuple[Any, Any], int]:
    elements_list = list(elements)
    n = len(elements_list)
    elem_to_idx = {elem: i for i, elem in enumerate(elements_list)}
    elem_bits = [1 << i for i in range(n)]

    ext_mask: Dict[Any, int] = {node: 0 for node in dag.nodes()}
    for node in reversed(list(nx.topological_sort(dag))):
        mask = 0
        if node in elem_to_idx:
            mask |= elem_bits[elem_to_idx[node]]
        for child in dag.successors(node):
            mask |= ext_mask[child]
        ext_mask[node] = mask

    branching_nodes: Dict[Any, List[int]] = {
        node: [ext_mask[c] for c in dag.successors(node)]
        for node in dag.nodes()
        if dag.out_degree(node) >= 2
    }

    leaf_ancestors: Dict[int, List[Any]] = {
        i: [
            node
            for node in branching_nodes
            if (ext_mask[node] & elem_bits[i])
        ]
        for i in range(n)
    }

    if sample_pairs is None:
        pairs_to_process = [
            (elements_list[i], elements_list[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
    else:
        pairs_to_process = sample_pairs

    pairs_by_u: Dict[Any, List[Tuple[Any, int]]] = {}
    for u, v in pairs_to_process:
        pairs_by_u.setdefault(u, []).append((v, elem_to_idx[v]))

    lca_vec: Dict[Tuple[Any, Any], int] = {}
    for u, v_list in pairs_by_u.items():
        u_idx = elem_to_idx[u]
        u_bit = elem_bits[u_idx]
        u_anc = leaf_ancestors[u_idx]
        if not u_anc:
            for v, _ in v_list:
                lca_vec[(u, v)] = 0
            continue

        for v, v_idx in v_list:
            v_bit = elem_bits[v_idx]
            both_bits = u_bit | v_bit
            union_mask = 0
            for a in u_anc:
                if ext_mask[a] & v_bit:
                    for c_mask in branching_nodes[a]:
                        hit = c_mask & both_bits
                        if hit and hit != both_bits:
                            union_mask |= ext_mask[a]
                            break
            lca_vec[(u, v)] = union_mask.bit_count()

    return lca_vec


def _try_sample_merge(
    dag: nx.DiGraph, internal_nodes: Set[Any], ext: Dict[Any, Set[Any]], epsilon: float
) -> Optional[nx.DiGraph]:
    internal_edges = [
        (u, v) for u, v in dag.edges() if u in internal_nodes and v in internal_nodes
    ]
    if not internal_edges:
        return None
    u, v = random.choice(internal_edges)
    diff_len = len(ext[u] - ext[v])
    if 1 <= diff_len <= max(1, math.ceil(epsilon * len(ext[u]))):
        if dag.out_degree(u) > 1 or dag.in_degree(v) > 1:
            new_dag = dag.copy()
            in_edges_v = list(new_dag.in_edges(v))
            out_edges_v = list(new_dag.out_edges(v))
            new_dag.remove_edge(u, v)
            for p, _ in in_edges_v:
                if p != u:
                    new_dag.add_edge(p, u)
            for _, c in out_edges_v:
                new_dag.add_edge(u, c)
            new_dag.remove_node(v)
            return _clean_transitive_reduction(new_dag)
    return None


def _try_sample_split(
    dag: nx.DiGraph, internal_nodes: Set[Any], ext: Dict[Any, Set[Any]], epsilon: float
) -> Optional[nx.DiGraph]:
    eligible_w = [n for n in internal_nodes if dag.out_degree(n) >= 3]
    if not eligible_w:
        return None
    w = random.choice(eligible_w)
    children = list(dag.successors(w))
    k = len(children)
    max_r = min(4, k - 1)
    r = random.randint(2, max_r)
    c_sub = random.sample(children, r)
    union_sub: Set[Any] = set()
    for c in c_sub:
        union_sub.update(ext[c])
    if 1 <= len(union_sub) <= max(1, math.ceil(epsilon * len(ext[w]))):
        c_tag = "_".join(sorted(str(c) for c in c_sub))
        new_node = f"v_split_{w}_{c_tag}"
        new_dag = dag.copy()
        new_dag.add_node(new_node)
        new_dag.add_edge(w, new_node)
        for c in c_sub:
            new_dag.remove_edge(w, c)
            new_dag.add_edge(new_node, c)
        return _clean_transitive_reduction(new_dag)
    return None


def _try_sample_add_reticulation(
    dag: nx.DiGraph,
    internal_nodes: Set[Any],
    ext: Dict[Any, Set[Any]],
    depths: Dict[Any, int],
    epsilon: float,
) -> Optional[nx.DiGraph]:
    internal_list = list(internal_nodes)
    if len(internal_list) < 2:
        return None
    u = random.choice(internal_list)
    d_u = depths[u]
    eligible_v = [
        v
        for v in internal_list
        if v != u and not dag.has_edge(u, v) and abs(depths[v] - d_u) <= 1
    ]
    if not eligible_v:
        return None
    v = random.choice(eligible_v)
    diff_len = len(ext[v] - ext[u])
    if 1 <= diff_len <= max(1, math.ceil(epsilon * len(ext[u]))):
        if not nx.has_path(dag, v, u) and not nx.has_path(dag, u, v):
            new_dag = dag.copy()
            new_dag.add_edge(u, v)
            return _clean_transitive_reduction(new_dag)
    return None


def _try_sample_delete_reticulation(
    dag: nx.DiGraph,
    internal_nodes: Set[Any],
    ext: Dict[Any, Set[Any]],
    elements_set: Set[Any],
    epsilon: float,
) -> Optional[nx.DiGraph]:
    mp_nodes = [v for v in internal_nodes if dag.in_degree(v) >= 2]
    if not mp_nodes:
        return None
    v = random.choice(mp_nodes)
    valid_parents = [u for u in dag.predecessors(v) if u in internal_nodes and dag.out_degree(u) > 1]
    if not valid_parents:
        return None
    u = random.choice(valid_parents)
    test_dag = dag.copy()
    test_dag.remove_edge(u, v)
    roots = [n for n in test_dag.nodes if test_dag.in_degree(n) == 0]
    reachable = set()
    for r in roots:
        reachable.update(nx.descendants(test_dag, r) & elements_set)
        if r in elements_set:
            reachable.add(r)
    if elements_set.issubset(reachable):
        ext_u_new = (
            nx.descendants(test_dag, u) & elements_set
        ) | ({u} if u in elements_set else set())
        lost_len = len(ext[u] - ext_u_new)
        if 1 <= lost_len <= max(1, math.ceil(epsilon * len(ext[u]))):
            return _clean_transitive_reduction(test_dag)
    return None


def _try_sample_leaf_transfer(
    dag: nx.DiGraph, internal_nodes: Set[Any], elements_set: Set[Any]
) -> Optional[nx.DiGraph]:
    leaf_parents = [
        p
        for p in internal_nodes
        if any(c in elements_set for c in dag.successors(p)) and dag.out_degree(p) > 1
    ]
    if not leaf_parents:
        return None
    p1 = random.choice(leaf_parents)
    direct_leaves = [c for c in dag.successors(p1) if c in elements_set]
    l = random.choice(direct_leaves)
    local_targets = set()
    for parent in dag.predecessors(p1):
        local_targets.add(parent)
        for sib in dag.successors(parent):
            if sib != p1 and sib in internal_nodes:
                local_targets.add(sib)
    for child in dag.successors(p1):
        if child in internal_nodes:
            local_targets.add(child)
    valid_targets = [
        p2 for p2 in local_targets if not dag.has_edge(p2, l) and p2 in internal_nodes
    ]
    if not valid_targets:
        return None
    other_parents = [
        q for q in dag.predecessors(l) if q != p1 and q in internal_nodes
    ]
    safe_targets = [
        p2
        for p2 in valid_targets
        if all(not nx.has_path(dag, p2, q) for q in other_parents)
    ]
    if not safe_targets:
        return None
    p2 = random.choice(safe_targets)
    new_dag = dag.copy()
    new_dag.remove_edge(p1, l)
    new_dag.add_edge(p2, l)
    return _clean_transitive_reduction(new_dag)


def do_random_local_change(
    dag: nx.DiGraph, elements: List[Any], epsilon: float = 0.15, max_attempts: int = 300
) -> nx.DiGraph:
    elements_set = set(elements)
    internal_nodes = set(dag.nodes()) - elements_set
    ext = _get_extensions(dag, elements_set)
    depths = _get_topological_depths(dag)

    op_types = [
        "MERGE",
        "SPLIT",
        "ADD_RETICULATION",
        "DELETE_RETICULATION",
        "LEAF_TRANSFER",
    ]

    for _ in range(max_attempts):
        chosen_op = random.choice(op_types)
        res = None
        if chosen_op == "MERGE":
            res = _try_sample_merge(dag, internal_nodes, ext, epsilon)
        elif chosen_op == "SPLIT":
            res = _try_sample_split(dag, internal_nodes, ext, epsilon)
        elif chosen_op == "ADD_RETICULATION":
            res = _try_sample_add_reticulation(
                dag, internal_nodes, ext, depths, epsilon
            )
        elif chosen_op == "DELETE_RETICULATION":
            res = _try_sample_delete_reticulation(
                dag, internal_nodes, ext, elements_set, epsilon
            )
        elif chosen_op == "LEAF_TRANSFER":
            res = _try_sample_leaf_transfer(dag, internal_nodes, elements_set)

        if res is not None:
            current_leaves = {n for n, d in res.out_degree() if d == 0}
            if current_leaves != elements_set:
                continue
            return res

    raise RuntimeError("Could not find a valid local change within max attempts.")


def generate_perturbation_trajectory(
    dag: nx.DiGraph,
    elements: List[Any],
    k: int = 20,
    epsilon: float = 0.15,
    num_samples: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[List[nx.DiGraph], List[float]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    elements_list = list(elements)
    n = len(elements_list)
    total_pairs = n * (n - 1) // 2
    m = min(num_samples, total_pairs)

    if m >= total_pairs:
        sample_pairs = [
            (elements_list[i], elements_list[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
    else:
        rng = np.random.default_rng(seed)
        sampled_indices: Set[Tuple[int, int]] = set()
        while len(sampled_indices) < m:
            needed = m - len(sampled_indices)
            batch = max(needed * 2, 1000)
            i_arr = rng.integers(0, n, size=batch)
            j_arr = rng.integers(0, n, size=batch)
            valid = i_arr != j_arr
            i_v = np.maximum(i_arr[valid], j_arr[valid])
            j_v = np.minimum(i_arr[valid], j_arr[valid])
            for i_idx, j_idx in zip(i_v, j_v):
                sampled_indices.add((int(j_idx), int(i_idx)))
                if len(sampled_indices) == m:
                    break
        sample_pairs = [(elements_list[i], elements_list[j]) for i, j in sampled_indices]

    dags = [dag.copy()]
    distances = [0.0]

    current_dag = dag.copy()
    base_lca_vec = _get_lca_vector(dag, elements_list, sample_pairs)

    for _ in range(k):
        current_dag = do_random_local_change(current_dag, elements_list, epsilon=epsilon)
        curr_lca_vec = _get_lca_vector(current_dag, elements_list, sample_pairs)

        dist = float(
            sum(abs(curr_lca_vec[p] - base_lca_vec[p]) for p in sample_pairs)
        )
        dags.append(current_dag.copy())
        distances.append(dist)

    return dags, distances


def simulate_single_trajectory(
    round_idx: int,
    dag: nx.DiGraph,
    elements: List[Any],
    W: np.ndarray,
    evaluator: Any,
    k: int = 20,
    epsilon: float = 0.15,
    num_samples: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    r"""
    Executes a single independent structural perturbation trajectory:
    1. Generates trajectory of k local perturbations and records cumulative topological distance.
    2. Evaluates costs for all metrics on each DAG in the trajectory.
    3. Calculates Spearman rank correlation \rho against topological distance from base DAG.
    4. Explicitly detects and flags flatlines (zero variance across trajectory) without artificial zero-padding.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    dags, distances = generate_perturbation_trajectory(
        dag, elements, k=k, epsilon=epsilon, num_samples=num_samples, seed=seed
    )

    trajectory_costs: Dict[str, List[float]] = {m.short_name: [] for m in evaluator.metrics}
    for idx, current_dag in enumerate(dags):
        print(f"Evaluating DAG {idx} of {len(dags)}")
        costs, _ = evaluator.evaluate(elements, current_dag, W)
        for m in evaluator.metrics:
            trajectory_costs[m.short_name].append(float(costs[m]))

    correlations: Dict[str, float] = {}
    is_flat: Dict[str, bool] = {}

    dist_arr = np.array(distances, dtype=float)
    dist_has_var = np.ptp(dist_arr) > 1e-12

    for m in evaluator.metrics:
        name = m.short_name
        c_series = np.array(trajectory_costs[name], dtype=float)
        c_has_var = np.ptp(c_series) > 1e-12

        if not dist_has_var or not c_has_var:
            correlations[name] = np.nan
            is_flat[name] = True
        else:
            rho, _ = spearmanr(c_series, dist_arr)
            if np.isnan(rho):
                correlations[name] = np.nan
                is_flat[name] = True
            else:
                correlations[name] = float(rho)
                is_flat[name] = False

    return {
        "round_idx": round_idx,
        "distances": distances,
        "trajectory_costs": trajectory_costs,
        "correlations": correlations,
        "is_flat": is_flat,
    }
