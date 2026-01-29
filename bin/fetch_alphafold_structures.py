"""
Download protein structures from AlphaFoldDB (https://alphafold.ebi.ac.uk/) to a local directory.
Proposal: "The input sequence is first searched from AlphaFoldDB, or processed by AlphaFold 3."
AlphaFoldDB is an online database; this script downloads structures so they can be used locally.

Usage:
  # From FASTA (sequence IDs = UniProt IDs):
  python bin/fetch_alphafold_structures.py --fasta-path data/.../train_GO.fasta --output-dir data/pdb/

  # From a list of UniProt IDs:
  python bin/fetch_alphafold_structures.py --id-list uniprot_ids.txt --output-dir data/pdb/

Output: {output_dir}/{uniprot_id}.cif (or .pdb). Then run fetch_structure_mapping.py with --pdb-dir {output_dir}.
"""
import argparse
import json
import os
import time
import urllib.request
import urllib.error

from protnote.utils.data import read_fasta

ALPHAFOLD_API_BASE = "https://alphafold.ebi.ac.uk/api/prediction"
# Fallback URL pattern when API doesn't return a direct link (AlphaFold DB v4)
ALPHAFOLD_FILE_URL_CIF = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.cif"
ALPHAFOLD_FILE_URL_PDB = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"


def get_prediction_url(uniprot_id):
    """Get model file URL from AlphaFoldDB API for a UniProt accession."""
    url = f"{ALPHAFOLD_API_BASE}/{uniprot_id}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        # API may return a list of entries; take first (best) model
        if isinstance(data, list) and len(data) > 0:
            entry = data[0]
        elif isinstance(data, dict):
            entry = data
        else:
            return None
        # Prefer cifUrl or pdbUrl; fallback to known pattern
        for key in ("cifUrl", "pdbUrl", "cif_url", "pdb_url"):
            if key in entry and entry[key]:
                return entry[key], "cif" if "cif" in key.lower() else "pdb"
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError):
        pass
    return None


def download_fallback(uniprot_id, prefer_pdb=True):
    """Return (url, ext) using known AlphaFoldDB file URL pattern."""
    if prefer_pdb:
        return ALPHAFOLD_FILE_URL_PDB.format(uniprot_id=uniprot_id), "pdb"
    return ALPHAFOLD_FILE_URL_CIF.format(uniprot_id=uniprot_id), "cif"


def download_file(url, out_path):
    """Download URL to out_path. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ProtNote-Structural/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(out_path, "wb") as f:
                f.write(resp.read())
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download structures from AlphaFoldDB (EBI) to a local directory. "
        "AlphaFoldDB is online; structures are saved under --output-dir (e.g. data/pdb/)."
    )
    parser.add_argument("--fasta-path", type=str, default=None, help="FASTA file; sequence IDs will be used as UniProt IDs.")
    parser.add_argument("--id-list", type=str, default=None, help="Text file with one UniProt ID per line.")
    parser.add_argument("--output-dir", type=str, required=True, help="Local directory to save PDB/mmCIF files (e.g. data/pdb/).")
    parser.add_argument("--prefer-pdb", action="store_true", help="Prefer PDB over mmCIF when using fallback URL.")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests to avoid rate limiting.")
    args = parser.parse_args()

    if args.fasta_path:
        data = read_fasta(args.fasta_path)
        uniprot_ids = list({sid for _, sid, _ in data})
    elif args.id_list:
        with open(args.id_list) as f:
            uniprot_ids = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Provide --fasta-path or --id-list.")

    os.makedirs(args.output_dir, exist_ok=True)
    done = 0
    skipped = 0
    for i, uid in enumerate(uniprot_ids):
        out_cif = os.path.join(args.output_dir, f"{uid}.cif")
        out_pdb = os.path.join(args.output_dir, f"{uid}.pdb")
        if os.path.isfile(out_cif) or os.path.isfile(out_pdb):
            skipped += 1
            continue
        result = get_prediction_url(uid)
        if result:
            url, ext = result
            out_path = os.path.join(args.output_dir, f"{uid}.{ext}")
        else:
            url, ext = download_fallback(uid, prefer_pdb=args.prefer_pdb)
            out_path = os.path.join(args.output_dir, f"{uid}.{ext}")
        if download_file(url, out_path):
            done += 1
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(uniprot_ids)} (downloaded {done}, skipped {skipped}).")
        time.sleep(args.delay)

    print(f"Done. Downloaded {done} structures to {args.output_dir} (already present: {skipped}).")
    print("Next: run fetch_structure_mapping.py with --pdb-dir", args.output_dir)


if __name__ == "__main__":
    main()
