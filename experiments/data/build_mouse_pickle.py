"""
Build experiments/data/tasic2018.pickle from the raw Tasic et al. (2018)
mouse-cortex single-cell RNA-seq data.

The mouse experiments load a preprocessed pickle (see load_data.load_mouse_data).
That pickle is git-ignored because it is large, so this script reproduces it
from the raw downloads.

Required downloads
------------------
1. VISp exon matrix + gene rows (Allen Institute):
     http://celltypes.brain-map.org/api/v2/well_known_file_download/694413985
   gives  mouse_VISp_2018-06-14_exon-matrix.csv
          mouse_VISp_2018-06-14_genes-rows.csv

2. ALM exon matrix (Allen Institute):
     http://celltypes.brain-map.org/api/v2/well_known_file_download/694413179
   gives  mouse_ALM_2018-06-14_exon-matrix.csv

   (browse all files at http://celltypes.brain-map.org/rnaseq)

3. Cluster / colour annotations (Kobak & Berens, "The art of using t-SNE"):
     https://github.com/berenslab/rna-seq-tsne/blob/master/data/tasic-sample_heatmap_plot_data.csv
   file  tasic-sample_heatmap_plot_data.csv

Layout expected by the defaults below::

    <raw_dir>/
        mouse_VISp_2018-06-14_exon-matrix.csv
        mouse_VISp_2018-06-14_genes-rows.csv
        mouse_ALM_2018-06-14_exon-matrix.csv
        tasic-sample_heatmap_plot_data.csv

Usage
-----
    python experiments/data/build_mouse_pickle.py \
        --raw_dir path/to/tasic-nature \
        --out experiments/data/tasic2018.pickle
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import sparse

# rnaseqTools.py lives alongside this file.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import rnaseqTools  # noqa: E402


def build(raw_dir, out_path):
    visp = os.path.join(raw_dir, "mouse_VISp_2018-06-14_exon-matrix.csv")
    alm = os.path.join(raw_dir, "mouse_ALM_2018-06-14_exon-matrix.csv")
    genes_rows = os.path.join(raw_dir, "mouse_VISp_2018-06-14_genes-rows.csv")
    cluster_info = os.path.join(raw_dir, "tasic-sample_heatmap_plot_data.csv")

    for p in (visp, alm, genes_rows, cluster_info):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing required input: {p}\n"
                "See the module docstring for download links."
            )

    counts1, genes1, cells1 = rnaseqTools.sparseload(visp)
    counts2, genes2, cells2 = rnaseqTools.sparseload(alm)

    counts = sparse.vstack((counts1, counts2), format="csc")
    cells = np.concatenate((cells1, cells2))

    assert np.all(genes1 == genes2), "VISp and ALM gene rows do not match"
    genes = np.copy(genes1)

    # Map Entrez gene IDs -> symbols.
    genesDF = pd.read_csv(genes_rows)
    id2symbol = dict(zip(genesDF["gene_entrez_id"].tolist(),
                         genesDF["gene_symbol"].tolist()))
    genes = np.array([id2symbol[g] for g in genes])

    # Cluster / colour annotations.
    clusterInfo = pd.read_csv(cluster_info)
    goodCells = clusterInfo["sample_name"].values
    ids = clusterInfo["cluster_id"].values
    labels = clusterInfo["cluster_label"].values
    colors = clusterInfo["cluster_color"].values

    clusterNames = np.array([labels[ids == i + 1][0] for i in range(np.max(ids))])
    clusterColors = np.array([colors[ids == i + 1][0] for i in range(np.max(ids))])
    clusters = np.copy(ids)

    # Restrict to the annotated "good" cells, in annotation order.
    ind = np.array([np.where(cells == c)[0][0] for c in goodCells])
    counts = counts[ind, :]

    areas = (ind < cells1.size).astype(int)  # 0 = VISp, 1 = ALM
    clusters = clusters - 1  # 0-based cluster indices

    tasic2018 = {
        "counts": counts,
        "genes": genes,
        "clusters": clusters,
        "areas": areas,
        "clusterColors": clusterColors,
        "clusterNames": clusterNames,
    }

    print("counts shape:        ", tasic2018["counts"].shape)
    print("cells in VISp (0):   ", np.sum(tasic2018["areas"] == 0))
    print("cells in ALM  (1):   ", np.sum(tasic2018["areas"] == 1))
    print("n clusters:          ", np.unique(tasic2018["clusters"]).size)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(tasic2018, f)
    print(f"\nWrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", default=os.path.join(_THIS_DIR, "tasic-nature"),
                   help="Directory containing the raw downloaded CSVs.")
    p.add_argument("--out", default=os.path.join(_THIS_DIR, "tasic2018.pickle"),
                   help="Output path for the built pickle.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.raw_dir, args.out)
