"""
Create structure_index.json: sequence_id -> pdb_path.
Expects a directory of PDB files named {sequence_id}.pdb, or a CSV/JSON mapping.
Usage:
  python bin/fetch_structure_mapping.py --fasta-path data/.../train_GO.fasta \\
    --pdb-dir data/pdb/ --output data/structure_index.json
  (Scans pdb-dir for *.pdb and matches by filename stem to FASTA sequence IDs.)
"""
import argparse
import json
import os

from protnote.utils.data import read_fasta


def main():
    parser = argparse.ArgumentParser(description="Build structure_index.json (sequence_id -> pdb_path).")
    parser.add_argument("--fasta-path", type=str, required=True, help="FASTA to get sequence IDs from.")
    parser.add_argument("--pdb-dir", type=str, required=True, help="Directory containing {sequence_id}.pdb files.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path.")
    args = parser.parse_args()

    data = read_fasta(args.fasta_path)
    valid_ids = {sid for _, sid, _ in data}
    pdb_dir = args.pdb_dir.rstrip("/")
    index = {}
    for f in os.listdir(pdb_dir):
        if f.endswith(".pdb") or f.endswith(".cif"):
            stem = f.rsplit(".", 1)[0]
            if stem in valid_ids:
                index[stem] = os.path.join(pdb_dir, f)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(index)} entries to {args.output}.")


if __name__ == "__main__":
    main()
