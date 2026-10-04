import igraph as ig
import numpy as np
import pandas as pd

from typing import Iterable, Optional, Tuple


class WeightMatrix:
    def at(self, u, v) -> float:
        raise NotImplementedError()

    def couples(self) -> Tuple:
        raise NotImplementedError()

    def to_dataframe(self, elements: Optional[Iterable[str]] = None) -> pd.DataFrame:
        raise NotImplementedError()


class IGraphWeightMatrix(WeightMatrix):
    def __init__(self, graph: ig.Graph):
        self._graph = graph

    def at(self, u, v) -> float:
        eid = self._graph.get_eid(u, v, directed=False, error=False)

        if eid != -1:
            weight = self._graph.es[eid]["weight"]
            return weight
        else:
            return 0.0

    def couples(self) -> Tuple:
        raise NotImplementedError()

    def to_dataframe(self, elements: Optional[Iterable[str]] = None) -> pd.DataFrame:
        attr = "weight" if "weight" in self._graph.es.attributes() else None
        if elements is None:
            names = self._graph.vs["name"] if "name" in self._graph.vs.attributes() else list(range(self._graph.vcount()))
            adj = np.array(self._graph.get_adjacency(attribute=attr, default=0.0).data, dtype=float)
            return pd.DataFrame(adj, index=names, columns=names)

        elem_list = list(elements)
        if "name" in self._graph.vs.attributes() and list(self._graph.vs["name"]) == elem_list:
            adj = np.array(self._graph.get_adjacency(attribute=attr, default=0.0).data, dtype=float)
            return pd.DataFrame(adj, index=elem_list, columns=elem_list)

        name_to_idx = {name: i for i, name in enumerate(self._graph.vs["name"])} if "name" in self._graph.vs.attributes() else {}
        sub_indices = [name_to_idx[e] for e in elem_list]
        subgraph = self._graph.induced_subgraph(sub_indices)
        adj = np.array(subgraph.get_adjacency(attribute=attr, default=0.0).data, dtype=float)
        return pd.DataFrame(adj, index=elem_list, columns=elem_list)


class DataFrameWeightMatrix(WeightMatrix):
    def __init__(self, data_frame: pd.DataFrame):
        self._data_frame = data_frame

    def at(self, u, v) -> float:
        return self._data_frame.loc[u, v]

    def couples(self) -> Tuple:
        _couples = []
        raise NotImplementedError()

    def to_dataframe(self, elements: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if elements is not None:
            elem_list = list(elements)
            return self._data_frame.loc[elem_list, elem_list]
        return self._data_frame


def to_dataframe(matrix: WeightMatrix, elements: Optional[Iterable[str]] = None) -> pd.DataFrame:
    try:
        return matrix.to_dataframe(elements)
    except (NotImplementedError, AttributeError):
        if elements is None:
            raise ValueError("elements must be provided for generic WeightMatrix")
        elem_list = list(elements)
        n = len(elem_list)
        arr = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i):
                val = matrix.at(elem_list[i], elem_list[j])
                arr[i, j] = val
                arr[j, i] = val
        return pd.DataFrame(arr, index=elem_list, columns=elem_list)
