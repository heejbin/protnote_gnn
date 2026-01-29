"""
Build combined graph+PLM pkl for structural encoder.
Expects: FASTA (with GO labels), structure_index.json (sequence_id -> pdb_path), PLM cache (sequence_id -> per-residue embeddings).
Output: combined_graph_plm.pkl mapping sequence_id -> (x, plm, edge_index, edge_s).
Usage:
  python bin/make_structural_dataset.py --fasta-path data/swissprot/.../train_GO.fasta \\
    --structure-index data/structure_index.json --plm-cache-dir data/plm_cache/ \\
    --output data/structure_graphs/combined_graph_plm.pkl
"""
import argparse
import json
import os
import pickle
import torch

from protnote.utils.data import read_fasta
from protnote.utils.structure_graph import pdb_to_graph


def main():
    parser = argparse.ArgumentParser(description="Build combined graph+PLM pkl for structural encoder.")
    parser.add_argument("--fasta-path", type=str, required=True, help="FASTA with sequences and GO labels.")
    parser.add_argument("--structure-index", type=str, required=True, help="JSON: sequence_id -> pdb_path.")
    parser.add_argument("--plm-cache-dir", type=str, required=True, help="Dir with {sequence_id}.pt per-residue PLM embeddings.")
    parser.add_argument("--output", type=str, required=True, help="Output pkl path (e.g. data/structure_graphs/combined_graph_plm.pkl).")
    parser.add_argument("--edge-cutoff", type=float, default=8.0, help="Radius for radius_graph.")
    parser.add_argument("--num-rbf", type=int, default=16, help="RBF edge features.")
    args = parser.parse_args()

    with open(args.structure_index) as f:
        structure_index = json.load(f)

    data = read_fasta(args.fasta_path)
    plm_dir = args.plm_cache_dir.rstrip("/")
    results = {}
    skipped = 0
    for seq, sid, labels in data:
        if sid not in structure_index:
            skipped += 1
            continue
        pdb_path = structure_index[sid]
        if not os.path.isfile(pdb_path):
            skipped += 1
            continue
        plm_path = os.path.join(plm_dir, f"{sid}.pt")
        if not os.path.isfile(plm_path):
            skipped += 1
            continue
        try:
            coords, edge_index, edge_s = pdb_to_graph(
                pdb_path, edge_cutoff=args.edge_cutoff, num_rbf=args.num_rbf
            )
            plm = torch.load(plm_path)
            if not isinstance(plm, torch.Tensor):
                plm = torch.tensor(plm, dtype=torch.float32)
            if plm.shape[0] != coords.shape[0]:
                skipped += 1
                continue
            results[sid] = (coords, plm, edge_index, edge_s)
        except Exception as e:
            skipped += 1
            continue

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(results)} samples to {args.output} (skipped {skipped}).")


if __name__ == "__main__":
    main()
