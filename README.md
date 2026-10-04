# beyond-trees

Official implementation and benchmarks for "Beyond Trees - A Generalized Dasgupta Metric for Overlapping Hierarchical Clustering".

The code builds a Gene Ontology DAG over a BioGRID physical-interaction network and evaluates the power-mean Dasgupta cost \(H_p\). Two experiments reproduce the biological figures and the perturbation tables in the paper.

## Setup

Python 3.10 or newer. Install the dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn networkx python-igraph obonet tqdm
```

Run commands from this directory:

```bash
python . --help
```

## Data

Pass the data directory with `--resources` / `-r`. It must contain:

- `go-basic.obo` (Gene Ontology basic ontology)
- a UniProt GAF and a BioGRID organism table (`.tab3.txt`) for each species

The default run uses human, yeast, mouse, and fruit fly. Arabidopsis, *C. elegans*, and *E. coli* K-12 W3110 are also accepted. The expected filenames are listed in `DATA_FILES` in `data.py`.

Each dataset keeps the genes that appear in both BioGRID and GO Biological Process, then takes a k-core of at most `--max-genes` genes (default 1000).

## Experiments

Biological contrast. For each species, average precision of \(1/H_p\) against physical interactions, for \(p\) from \(-2\) to \(1\) in steps of \(0.5\). The figure is a 2×2 panel when all four default species are loaded. A PNG is written next to the PDF.

```bash
python . biological-contrast --r /path/to/Resources
python . biological-contrast --r /path/to/Resources -m YEAST -g 1000 -b 50 -s results/biological_contrast.pdf
```

Perturbation monotonicity. Repeated local damage of the DAG, summarized by the Spearman correlation between topological distance from the original DAG and the cost. Each finished species is appended to the text file, so a stopped run keeps the species that already completed. The file is cleared at the start of a new run.

```bash
python . perturbation-monotonicity --r /path/to/Resources
python . perturbation-monotonicity --r /path/to/Resources -m "FRUIT FLY" -r 50 -k 20 -j 4 -s results/perturbation_monotonicity.txt
```

`FRUIT FLY` must be quoted because the name contains a space.

## Common options

| Option | Meaning |
| --- | --- |
| `-r`, `--resources` | Directory with `go-basic.obo`, the GAF files, and the BioGRID tables. Required. |
| `-m`, `--model` | `HUMAN`, `YEAST`, `MOUSE`, `FRUIT FLY`, etc. Default: the four species above. |
| `-g`, `--max-genes` | Maximum number of genes (default 1000). |
| `--seed` | Random seed (default 42). |

`biological-contrast` also takes `-b` / `--bootstrap-reps` (default 50; `0` disables the confidence band) and `-s` / `--save`.

`perturbation-monotonicity` also takes `-R` / `--rounds`, `-k` / `--steps`, `-j` / `--jobs`, `--samples`, `--epsilon`, and `-s` / `--save`.
