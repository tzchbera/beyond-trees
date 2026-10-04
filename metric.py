import networkx as nx
import numpy as np
import pandas as pd

from typing import List, Dict, Tuple
from scipy.linalg import solve_triangular

from matrix import WeightMatrix, DataFrameWeightMatrix, to_dataframe


class Metric:
    @property
    def name(self):
        return NotImplemented

    @property
    def type(self):
        return NotImplemented

    @property
    def short_name(self):
        return NotImplemented

    @property
    def paper_name(self):
        return self.name

    @property
    def latex_name(self):
        return self.name


class MinDasguptaMetric(Metric):
    @property
    def name(self):
        return "Min Dasgupta"

    @property
    def short_name(self):
        return "min"

    @property
    def paper_name(self):
        return "c_min (MCA-min)"

    @property
    def latex_name(self):
        return r"$c_{\min}$ (MCA-min)"

    @property
    def type(self):
        return "min"


class AverageDasguptaMetric(Metric):
    def __init__(self):
        super(AverageDasguptaMetric, self).__init__()

    @property
    def name(self):
        return "Weighted Average Dasgupta"

    @property
    def short_name(self):
        return "avg"

    @property
    def paper_name(self):
        return "c_trans (Pan et al.)"

    @property
    def latex_name(self):
        return r"$c_{\mathrm{trans}}$ (Pan et al.)"

    @property
    def type(self):
        return "avg"


class MeanDasguptaMetric(Metric):
    def __init__(self, p: int):
        super(MeanDasguptaMetric, self).__init__()
        self.p = p

    @property
    def name(self):
        return f"{self.p}-Mean Dasgupta"

    @property
    def short_name(self):
        return f"p={self.p}"

    @property
    def paper_name(self):
        descriptors = {
            1: "c_1 (Arithmetic)",
            0: "c_0 (Geometric)",
            -1: "c_-1 (Harmonic)",
            -2: "c_-2 (Inv-Quad)",
        }
        return descriptors.get(self.p, f"c_{self.p} (p={self.p})")

    @property
    def latex_name(self):
        descriptors = {
            1: r"$c_{1}$ (Arithmetic)",
            0: r"$c_{0}$ (Geometric)",
            -1: r"$c_{-1}$ (Harmonic)",
            -2: r"$c_{-2}$ (Inv-Quad)",
        }
        return descriptors.get(self.p, rf"$c_{{{self.p}}}$ ($p={self.p}$)")

    @property
    def type(self):
        return "mean"


class MetricEvaluator:
    def __init__(self, metrics):
        self.metrics = metrics

    @property
    def names(self):
        return [metric.name for metric in self.metrics]

    @property
    def short_names(self):
        return [metric.short_name for metric in self.metrics]

    def evaluate(self, elements: List[str], dag: nx.DiGraph, matrix: WeightMatrix) -> Tuple[Dict[object, float], Dict[str, Dict[int, np.ndarray]]]:
        n = len(elements)
        total_leaves = [node for node in dag.nodes() if dag.out_degree(node) == 0]
        assert len(total_leaves) == n, len(total_leaves)
        assert self.metrics

        elem_to_idx = {elem: i for i, elem in enumerate(elements)}
        topo_nodes = list(nx.topological_sort(dag))
        node_to_idx = {node: i for i, node in enumerate(topo_nodes)}
        m_len = len(topo_nodes)

        leaf_indices = {}
        for node in reversed(topo_nodes):
            children = list(dag.successors(node))
            if not children:
                leaf_indices[node] = np.array([elem_to_idx[node]], dtype=np.int32)
            elif len(children) == 1:
                leaf_indices[node] = leaf_indices[children[0]]
            else:
                leaf_indices[node] = np.unique(np.concatenate([leaf_indices[c] for c in children]))

        leaf_counts = np.array([len(leaf_indices[u]) for u in topo_nodes], dtype=float)

        if isinstance(matrix, np.ndarray):
            W = np.tril(matrix, k=-1) if matrix.shape == (n, n) else matrix
        elif hasattr(matrix, "to_dataframe"):
            mat_df = matrix.to_dataframe(elements)
            W = np.tril(mat_df.to_numpy(dtype=float), k=-1)
        elif isinstance(matrix, pd.DataFrame):
            W = np.tril(matrix.loc[elements, elements].to_numpy(dtype=float), k=-1)
        else:
            W = np.zeros((n, n))
            for i in range(n):
                for j in range(i):
                    W[i, j] = matrix.at(elements[i], elements[j])

        A_elem = None
        A = np.zeros((m_len, m_len), dtype=float)
        np.fill_diagonal(A, 1.0)
        for i, node in enumerate(topo_nodes):
            parents = list(dag.predecessors(node))
            if parents:
                p_indices = [node_to_idx[p] for p in parents]
                A[i, :] = np.mean(A[p_indices, :], axis=0)
                A[i, i] = 1.0

        elem_indices = [node_to_idx[u] for u in elements]
        A_elem = A[elem_indices, :]

        mean_results: Dict[object, float] = {}
        hp_matrices: Dict[int, np.ndarray] = {}
        K = A ** 2
        mean_metrics = [m for m in self.metrics if m.type == "mean"]
        p_values = {m.p for m in mean_metrics}

        for p in p_values:
            if p == 0:
                h = np.log(leaf_counts)
            elif p == 1:
                h = leaf_counts
            else:
                h = leaf_counts ** p

            if solve_triangular is not None:
                g = solve_triangular(K, h, lower=True, unit_diagonal=True)
            else:
                g = np.linalg.solve(K, h)

            E = (A_elem * g) @ A_elem.T

            if p == 0:
                hp = np.exp(E)
            elif p == 1:
                hp = E
            else:
                hp = E ** (1.0 / p)

            mean_results[p] = float(np.sum(W * hp))
            hp_matrices[p] = hp

        min_lca_len = np.full((n, n), np.inf, dtype=float)
        norm_sum = np.zeros((n, n), dtype=float)
        weighted_sum = np.zeros((n, n), dtype=float)

        pos = np.full(n, -1, dtype=np.int32)

        for node in topo_nodes:
            children = list(dag.successors(node))
            if len(children) < 2:
                continue

            w_leaves = leaf_indices[node]
            k = len(w_leaves)
            if k < 2:
                continue

            pos[w_leaves] = np.arange(k, dtype=np.int32)
            covered = np.zeros((k, k), dtype=bool)

            for child in children:
                c_leaves = leaf_indices[child]
                if len(c_leaves) >= 2:
                    c_local = pos[c_leaves]
                    covered[np.ix_(c_local, c_local)] = True

            tril_i, tril_j = np.tril_indices(k, k=-1)
            is_lca = ~covered[tril_i, tril_j]

            if np.any(is_lca):
                pair_i = w_leaves[tril_i[is_lca]]
                pair_j = w_leaves[tril_j[is_lca]]

                min_lca_len[pair_i, pair_j] = np.minimum(min_lca_len[pair_i, pair_j], k)

                w_idx = node_to_idx[node]
                a = A_elem[:, w_idx]
                val = a[pair_i] * a[pair_j]
                norm_sum[pair_i, pair_j] += val
                weighted_sum[pair_i, pair_j] += val * k

            pos[w_leaves] = -1

        finite_mask = np.isfinite(min_lca_len)
        product = np.zeros_like(W)
        np.multiply(W, min_lca_len, out=product, where=finite_mask)
        val_min = float(np.sum(product))

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_matrix = np.where(norm_sum > 0, weighted_sum / norm_sum, 0.0)
        val_avg = float(np.sum(W * avg_matrix))

        results = {}

        for metric in self.metrics:
            if metric.type == "min":
                results[metric] = val_min
            elif metric.type == "avg":
                results[metric] = val_avg
            elif metric.type == "mean":
                results[metric] = mean_results[metric.p]

        dist_matrices = {}
        for m in self.metrics:
            if m.type == "min":
                dist_matrices[m] = np.where(np.isfinite(min_lca_len), min_lca_len, 0.0)
            elif m.type == "avg":
                dist_matrices[m] = avg_matrix
            elif m.type == "mean":
                dist_matrices[m] = hp_matrices[m.p]

        return results, {
            "W": W,
            "Hp": hp_matrices,
            "min_lca": np.where(np.isfinite(min_lca_len), min_lca_len, 0.0),
            "avg_lca": avg_matrix,
            "dist_matrices": dist_matrices,
        }


###############################################################################
# TESTING
###############################################################################


def create_input_1() -> Tuple[List[str], nx.DiGraph, WeightMatrix]:
    elements = ['l1', 'l2', 'l3', 'l4']

    dag = nx.DiGraph()
    edges = [
        ('root', 'n1'), ('root', 'n2'),
        ('n1', 'l1'), ('n1', 'l2'),
        ('n2', 'l2'), ('n2', 'n3'),
        ('n3', 'l3'), ('n3', 'l4'),
    ]
    dag.add_edges_from(edges)

    data = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.6, 0.6],
        [0.0, 0.6, 1.0, 1.0],
        [0.0, 0.6, 1.0, 1.0],
    ]
    matrix = DataFrameWeightMatrix(pd.DataFrame(data, index=elements, columns=elements))

    return elements, dag, matrix


def create_input_2() -> Tuple[List[str], nx.DiGraph, WeightMatrix]:
    elements = ['l1', 'l2', 'l3', 'l4']

    dag = nx.DiGraph()
    edges = [
        ('root', 'n1'), ('root', 'n2'),
        ('n1', 'l4'), ('n1', 'n3'),
        ('n2', 'l1'), ('n2', 'n4'),
        ('n3', 'l1'), ('n3', 'l2'),
        ('n4', 'l2'), ('n4', 'l3'),
    ]
    dag.add_edges_from(edges)

    data = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.6, 0.6],
        [0.0, 0.6, 1.0, 1.0],
        [0.0, 0.6, 1.0, 1.0],
    ]

    matrix = DataFrameWeightMatrix(pd.DataFrame(data, index=elements, columns=elements))

    return elements, dag, matrix


def render(d):
    import json
    print(json.dumps({x.short_name: d[x] for x in d}, indent=2))


def main():
    metric_evaluator = MetricEvaluator([
        MinDasguptaMetric(), AverageDasguptaMetric(), MeanDasguptaMetric(1),
        MeanDasguptaMetric(0), MeanDasguptaMetric(-1), MeanDasguptaMetric(-2),
    ])
    render(metric_evaluator.evaluate(*create_input_1())[0])
    render(metric_evaluator.evaluate(*create_input_2())[0])


if __name__ == '__main__':
    main()
