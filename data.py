import gzip
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import obonet
import pandas as pd

from matrix import IGraphWeightMatrix, WeightMatrix                     


DATA_FILES = {
    "HUMAN": {
        "GAF": "HUMAN-uniprot.gaf",
        "BIOGRID": "BIOGRID-ORGANISM-Homo_sapiens-5.0.262.tab3.txt",
        "TAXON": "9606",
    },
    "YEAST": {
        "GAF": "YEAST-uniprot.gaf",
        "BIOGRID": "BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-5.0.262.tab3.txt",
        "TAXON": "559292",
    },
    "FRUIT FLY": {
        "GAF": "DROME-uniprot.gaf",
        "BIOGRID": "BIOGRID-ORGANISM-Drosophila_melanogaster-5.0.262.tab3.txt",
        "TAXON": "7227",
    },
    "MOUSE": {
        "GAF": "MOUSE-uniprot.gaf",
        "BIOGRID": "BIOGRID-ORGANISM-Mus_musculus-5.0.262.tab3.txt",
        "TAXON": "10090",
    },
}


@dataclass
class Dataset:
    name: str
    elements: List[str]
    matrix: WeightMatrix
    dag: nx.DiGraph


def _resource_path(resources_dir: Path, filename: str) -> Path:
    """Return a file inside the given resources directory."""
    path = resources_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Could not find resource file: {path}")
    return path.resolve()


def _load_go_graph(resources_dir: Path) -> nx.DiGraph:
    """Load GO parent-to-child edges for is_a and part_of relationships."""
    path = _resource_path(resources_dir, "go-basic.obo")
    print(f"Loading {path.name}...")
    obo_graph = obonet.read_obo(str(path))
    graph = nx.DiGraph()

    for child, parent, relation in obo_graph.edges(keys=True):
        if relation in {"is_a", "part_of"} and child.startswith("GO:") and parent.startswith("GO:"):
            graph.add_edge(parent, child)

    graph = nx.transitive_reduction(graph)

    for node, data in obo_graph.nodes(data=True):
        if node in graph:
            graph.nodes[node]["name"] = data.get("name", data.get("label", node))

    print(f"Graph built with {graph.number_of_nodes()} nodes.")
    return graph


def parse_gaf_annotations(gaf_path: str, aspect: str = "P", allowed_evidence: Optional[Set[str]] = None, excluded_evidence: Optional[Set[str]] = None) -> Dict[str, Set[str]]:

    gene_to_go: Dict[str, Set[str]] = defaultdict(set)
    open_file = gzip.open if gaf_path.endswith(".gz") else open

    with open_file(gaf_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15:
                continue

            # GAF columns (1-based): symbol 3, qualifier 4, GO ID 5, evidence 7, aspect 9.
            gene, qualifier, go_id, evidence, row_aspect = (
                fields[2], fields[3], fields[4], fields[6], fields[8]
            )
            if "NOT" in qualifier.split("|") or row_aspect != aspect:
                continue
            if allowed_evidence and evidence not in allowed_evidence:
                continue
            if excluded_evidence and evidence in excluded_evidence:
                continue

            gene_to_go[gene].add(go_id)

    return dict(gene_to_go)


def _resolve_go_id(term: str, valid_nodes: Set[str]) -> Optional[str]:

    term = str(term).strip()
    if not term or term == "nan":
        return None
    if term in valid_nodes:
        return term

    if term.startswith("GO:"):
        candidate = term.removeprefix("GO:")
    else:
        candidate = f"GO:{term}"
    return candidate if candidate in valid_nodes else None


def build_gene_ontology_dag(go_graph: nx.DiGraph, gene_to_terms: Dict[str, Set[str]], elements: List[str]) -> Tuple[nx.DiGraph, List[str]]:

    valid_nodes = set(go_graph)
    root = _resolve_go_id("GO:0008150", valid_nodes)

    subtree_nodes = nx.descendants(go_graph, root) | {root}
    terms_by_gene: Dict[str, Set[str]] = {}
    for gene in elements:
        terms = {
            resolved
            for term in gene_to_terms.get(gene, ())
            if (resolved := _resolve_go_id(term, valid_nodes)) in subtree_nodes
        }
        if not terms:
            raise ValueError(f"Gene {gene} has no annotation under GO root {root}.")
        terms_by_gene[gene] = terms

    annotated_terms = set().union(*terms_by_gene.values()) if terms_by_gene else set()
    relevant_terms = {root} | annotated_terms
    ancestors_by_term: Dict[str, Set[str]] = {}
    for term in annotated_terms:
        ancestors = nx.ancestors(go_graph, term) & subtree_nodes
        ancestors_by_term[term] = ancestors
        relevant_terms.update(ancestors)

    ontology = go_graph.subgraph(relevant_terms)
    dag = nx.DiGraph()
    for node in sorted(ontology):
        dag.add_node(
            node,
            type="root" if node == root else "ontology",
            name=ontology.nodes[node].get("name", node),
        )
    dag.add_edges_from(sorted(ontology.edges()))

    for gene, terms in terms_by_gene.items():
        ancestors = set().union(*(ancestors_by_term[term] for term in terms))
        specific_terms = sorted(terms - ancestors)
        dag.add_node(gene, type="element")
        dag.add_edges_from((term, gene) for term in specific_terms)

    sources = {node for node, degree in dag.in_degree() if degree == 0}
    if sources != {root}:
        raise ValueError(f"DAG rooted at {root} has unexpected sources: {sorted(sources)}")
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Resulting ontology graph is not a DAG.")

    leaves = {node for node, degree in dag.out_degree() if degree == 0}
    if leaves != set(elements):
        raise ValueError("DAG leaves do not match the requested elements.")

    return dag, elements


def parse_biogrid_table(biogrid_path: str, taxon_id: str) -> Tuple[Set[str], Set[Tuple[str, str]]]:

    path = Path(biogrid_path)
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=['Official Symbol Interactor A', 'Official Symbol Interactor B', 'Experimental System Type', 'Organism ID Interactor A', 'Organism ID Interactor B'],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    symbol_a = df['Official Symbol Interactor A']
    symbol_b = df['Official Symbol Interactor B']
    df = df[
        (df['Experimental System Type'] == 'physical') &
        (df['Organism ID Interactor A'] == str(taxon_id)) &
        (df['Organism ID Interactor B'] == str(taxon_id)) &
        (symbol_a != symbol_b) &
        (symbol_a != '') &
        (symbol_b != '') &
        (symbol_a != '-') &
        (symbol_b != '-')
    ]

    g1 = df[['Official Symbol Interactor A', 'Official Symbol Interactor B']].min(axis=1)
    g2 = df[['Official Symbol Interactor A', 'Official Symbol Interactor B']].max(axis=1)

    edges_df = pd.DataFrame({'u': g1, 'v': g2}).drop_duplicates()

    genes_list = sorted(list(set(edges_df['u']).union(set(edges_df['v']))))
    edges_list = list(edges_df.itertuples(index=False, name=None))

    return genes_list, edges_list


def validate_dataset_quality(elements: List[str], edges: list, go_graph: nx.DiGraph, gene_to_go: Dict[str, Set[str]], dag: Optional[nx.DiGraph] = None, min_annotation_depth: int = 3, go_root: str = "GO:0008150"):
    n_elem = len(elements)
    if n_elem == 0:
        print("[FAIL] Elements list is empty.")
        return

    elem_set = set(elements)
    if len(elements) != n_elem:
        print(f"[WARNING] Duplicate genes found: {len(elements) - n_elem}")

    if dag is not None:

        if not nx.is_directed_acyclic_graph(dag):
            print("[FAIL] Augmented graph is not a DAG.")
            return

        missing_genes = elem_set - set(dag.nodes())

        if missing_genes:
            print(
                f"[FAIL] {len(missing_genes)} genes "
                "are missing from the augmented DAG."
            )
            return

        invalid_leaves = [
            gene for gene in elem_set
            if dag.in_degree(gene) == 0
            or dag.out_degree(gene) > 0
        ]

        if invalid_leaves:
            print(
                f"[FAIL] {len(invalid_leaves)} genes "
                "are not valid leaves."
            )
            return

    g_ppi = nx.Graph()
    g_ppi.add_nodes_from(elem_set)
    g_ppi.add_edges_from((u, v) for u, v, *rest in edges if u in elem_set and v in elem_set and u != v)
    
    lcc = max(nx.connected_components(g_ppi), key=len) if g_ppi.number_of_edges() > 0 else set()
    lcc_fraction = len(lcc) / n_elem

    print(f"LCC size: {len(lcc)}/{n_elem} ({lcc_fraction:.1%})")

    root_id = go_root if go_root in go_graph else "GO:0008150"
    depths = nx.single_source_shortest_path_length(go_graph, root_id) if root_id in go_graph else {}

    deep_annotated_genes = 0
    for gene in elements:
        terms = gene_to_go.get(gene, set())
        max_depth = max((depths.get(t, 0) for t in terms), default=0)
        if max_depth >= min_annotation_depth:
            deep_annotated_genes += 1

    depth_fraction = deep_annotated_genes / n_elem

    print(f"GO Annotation Specificity: {deep_annotated_genes}/{n_elem} ({depth_fraction:.1%})")

    term_counts: Dict[str, int] = defaultdict(int)
    for gene in elements:
        for t in gene_to_go.get(gene, set()):
            if depths.get(t, 0) >= min_annotation_depth:
                term_counts[t] += 1

    top_term, top_count = max(term_counts.items(), key=lambda x: x[1]) if term_counts else ("None", 0)
    top_fraction = top_count / n_elem

    term_name = go_graph.nodes.get(top_term, {}).get("name", top_term)
    print(f"Biological Diversity: {top_term} ('{term_name}') {top_count}/{n_elem} ({top_fraction:.1%})")

    branch_depth = 4
    frontier = [t for t, d in depths.items() if d == branch_depth]

    term_to_genes = defaultdict(set)
    for gene in elem_set:
        for t in gene_to_go.get(gene, set()):
            term_to_genes[t].add(gene)

    branch_genes = {}
    for t in frontier:
        terms = nx.descendants(go_graph, t) | {t}
        branch_genes[t] = set().union(*(term_to_genes[x] for x in terms if x in term_to_genes))

    top_branch, top_genes = max(
        branch_genes.items(), key=lambda x: len(x[1]), default=(None, set())
    )

    membership = defaultdict(int)
    for genes in branch_genes.values():
        for g in genes:
            membership[g] += 1

    top_fraction = len(top_genes) / n_elem
    multi_fraction = sum(v > 1 for v in membership.values()) / n_elem
    unassigned_fraction = sum(membership.get(g, 0) == 0 for g in elem_set) / n_elem

    branch_name = go_graph.nodes[top_branch].get("name", top_branch) if top_branch else "None"

    print(f"Biological Branch Coverage: {len(frontier)} branches, {len(top_genes)}/{n_elem} ({top_fraction:.1%}) genes in dominant branch '{top_branch} ('{branch_name}')")
    print(f"Genes in multiple branches: {multi_fraction:.1%}")
    print(f"Genes not assigned to a branch: {unassigned_fraction:.1%}")


def select_kcore(common_genes: List[str], edges: list, target_size: int = 1000) -> List[str]:
    if target_size <= 0:
        return []

    c_set = set(common_genes)
    g = nx.Graph()
    g.add_nodes_from(c_set)
    g.add_edges_from((u, v) for u, v, *rest in edges if u in c_set and v in c_set and u != v)

    if g.number_of_edges() == 0:
        return []

    core_map = nx.core_number(g)
    max_k = max(core_map.values())

    if max_k == 0:
        return []

    for k in range(max_k, 0, -1):
        selected = {n for n, core in core_map.items() if core >= k}
        if len(selected) >= target_size:
            print(f"[Full K-core] k={k}, Selected {len(selected)}/{target_size} target genes.")
            return sorted(selected)

    selected = {n for n, core in core_map.items() if core >= 1}
    print(f"[Full K-core] Only {len(selected)} non-isolated genes available.")

    return sorted(selected)


def load_datasets(resources_dir: str, model: Optional[str] = None, max_genes: int = 1000) -> List[Dataset]:
    resources = Path(resources_dir)
    if not resources.is_dir():
        raise FileNotFoundError(f"Resources directory not found: {resources}")

    if model is None:
        models = ["HUMAN", "YEAST", "MOUSE", "FRUIT FLY"]
    else:
        models = [model]

    go_graph = _load_go_graph(resources)

    datasets = []

    for model in models:
        biogrid_path = _resource_path(resources, DATA_FILES[model]["BIOGRID"])
        gaf_path = _resource_path(resources, DATA_FILES[model]["GAF"])
        taxon_id = DATA_FILES[model]["TAXON"]

        genes, edges = parse_biogrid_table(str(biogrid_path), taxon_id)
        genes = set(genes)
        edges = [(u, v, 1.0) for u, v in edges]

        gene_to_go = parse_gaf_annotations(str(gaf_path), aspect="P")
        common_genes = sorted(list(genes & set(gene_to_go.keys())))
        print(f"Found {len(common_genes)} genes in intersection of BioGrid and GO-BP.")

        if not common_genes:
            raise ValueError("Intersection between BioGrid and GO annotations is empty!")

        elements = select_kcore(common_genes, edges, max_genes)

        elements_set = set(elements)
        elem_to_idx = {elem: i for i, elem in enumerate(elements)}
        filtered_edges = [(elem_to_idx[u], elem_to_idx[v], w) for u, v, w in edges if u in elements_set and v in elements_set]

        g_sim = ig.Graph(n=len(elements), directed=False)
        g_sim.vs["name"] = elements
        if filtered_edges:
            g_sim.add_edges([(u, v) for u, v, _ in filtered_edges])
            g_sim.es["weight"] = [w for _, _, w in filtered_edges]
        else:
            g_sim.es["weight"] = []

        matrix = IGraphWeightMatrix(g_sim)
        dag, elements = build_gene_ontology_dag(go_graph, gene_to_go, elements)
        print(f"Successfully constructed DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges, {len(elements)} leaves.")

        validate_dataset_quality(elements, edges, go_graph, gene_to_go, dag)

        n_v = g_sim.vcount()
        m_1 = g_sim.ecount()
        n_pairs = n_v * (n_v - 1) // 2
        print(f"{model}: |V|={n_v} | interacting pairs m_1={m_1:,} | pairs={n_pairs:,} | density={100.0 * m_1 / n_pairs:.2f}% (igraph density: {100.0 * g_sim.density():.2f}%)")

        datasets.append(Dataset(name=model, elements=elements, matrix=matrix, dag=dag))

    return datasets
