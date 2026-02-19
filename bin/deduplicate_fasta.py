"""Deduplicate a FASTA file by amino acid sequence, merging GO labels.

When multiple entries share the same amino acid sequence, keeps the first
encountered ID as canonical and unions all GO labels across the group.
Optionally re-splits into train/dev/test and updates the graph index.

Example Usage:
    # Dry run (stats only, no files written)
    python bin/deduplicate_fasta.py \
        --input-path data/sequences/.../full_GO.fasta \
        --splits-dir data/sequences/.../splits \
        --prefix GO \
        --dry-run

    # Full dedup + re-split + graph index update
    python bin/deduplicate_fasta.py \
        --input-path data/sequences/.../full_GO.fasta \
        --splits-dir data/sequences/.../splits \
        --prefix GO \
        --seed 42 \
        --graph-index-path data/processed/graph_index.json \
        --graph-dir data/processed

    # Skip splitting, skip backups
    python bin/deduplicate_fasta.py \
        --input-path data/sequences/.../full_GO.fasta \
        --no-split --no-backup
"""

import argparse
import importlib.util
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Allow imports from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from protnote.utils.data import read_fasta, save_to_fasta

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _import_split_dataset():
    """Import split_dataset.py from bin/ (not a Python package)."""
    spec = importlib.util.spec_from_file_location(
        "split_dataset",
        str(Path(__file__).resolve().parent / "split_dataset.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def deduplicate_entries(data):
    """Deduplicate FASTA entries by sequence, merging labels.

    Args:
        data: List of (sequence, sequence_id, labels) tuples from read_fasta().

    Returns:
        Tuple of:
            - deduped: list of (sequence, canonical_id, merged_labels) tuples
            - stats: dict with deduplication statistics
    """
    # Pass 1: group by sequence, track per-entry label sets
    # seq_groups maps sequence -> (canonical_id, merged_label_set, [all_ids])
    seq_groups = {}
    # Track distinct label combos per sequence (for same-vs-diff stats)
    seq_label_combos = {}
    # Fast lookup: seq_id -> original labels
    id_to_labels = {}

    for sequence, seq_id, labels in data:
        label_key = tuple(sorted(labels))
        id_to_labels[seq_id] = labels

        if sequence in seq_groups:
            canonical_id, label_set, group_ids = seq_groups[sequence]
            label_set.update(labels)
            group_ids.append(seq_id)
        else:
            seq_groups[sequence] = (seq_id, set(labels), [seq_id])

        if sequence not in seq_label_combos:
            seq_label_combos[sequence] = set()
        seq_label_combos[sequence].add(label_key)

    # Pass 2: build deduped list and stats
    deduped = []
    same_label_groups = 0
    diff_label_groups = 0
    total_labels_gained = 0
    largest_group_size = 0
    largest_group_id = None
    dropped_ids = set()

    for sequence, (canonical_id, label_set, group_ids) in seq_groups.items():
        merged_labels = sorted(label_set)
        deduped.append((sequence, canonical_id, merged_labels))

        group_size = len(group_ids)
        if group_size > largest_group_size:
            largest_group_size = group_size
            largest_group_id = canonical_id

        if group_size > 1:
            # Collect dropped IDs
            for gid in group_ids[1:]:
                dropped_ids.add(gid)

            # Check if label sets differ across the group
            distinct_combos = seq_label_combos[sequence]
            if len(distinct_combos) == 1:
                same_label_groups += 1
            else:
                diff_label_groups += 1
                canonical_original_count = len(id_to_labels[canonical_id])
                total_labels_gained += len(label_set) - canonical_original_count

    stats = {
        "original_count": len(data),
        "deduped_count": len(deduped),
        "removed_count": len(data) - len(deduped),
        "duplicate_groups": same_label_groups + diff_label_groups,
        "same_label_groups": same_label_groups,
        "diff_label_groups": diff_label_groups,
        "total_labels_gained": total_labels_gained,
        "largest_group_size": largest_group_size,
        "largest_group_id": largest_group_id,
        "dropped_ids": dropped_ids,
    }

    return deduped, stats


def backup_file(path, timestamp):
    """Create a timestamped backup of a file if it exists."""
    if os.path.exists(path):
        bak_path = f"{path}.{timestamp}.bak"
        shutil.copy2(path, bak_path)
        logger.info(f"  Backed up {path} -> {bak_path}")
        return bak_path
    return None


def update_graph_index(graph_index_path, graph_dir, dropped_ids, canonical_ids, cleanup_graphs=False):
    """Update graph_index.json after deduplication.

    - Recovers orphan .pt files (on disk but not in index)
    - Remaps graph entries for dropped IDs to canonical IDs
    - Removes entries for dropped IDs
    - Optionally deletes orphaned .pt files

    Args:
        graph_index_path: Path to graph_index.json
        graph_dir: Directory containing .pt graph files
        dropped_ids: Set of sequence IDs that were removed during dedup
        canonical_ids: Dict mapping dropped_id -> canonical_id
        cleanup_graphs: If True, delete .pt files for dropped IDs

    Returns:
        Dict with graph index update statistics.
    """
    import torch

    with open(graph_index_path, "r") as f:
        graph_index = json.load(f)

    graph_dir = Path(graph_dir)
    stats = {
        "original_index_entries": len(graph_index),
        "orphans_recovered": 0,
        "entries_remapped": 0,
        "entries_removed": 0,
        "files_deleted": 0,
    }

    # Graph index entries are dicts: {"filename": "...", "n_atoms": N}
    def _get_filename(entry):
        if isinstance(entry, dict):
            return entry["filename"]
        return entry  # fallback for plain string entries

    # Step 1: Find orphan .pt files (on disk but missing from index)
    indexed_files = {_get_filename(entry) for entry in graph_index.values()}
    on_disk = {pt_file.name for pt_file in graph_dir.glob("*.pt")}

    orphan_files = on_disk - indexed_files
    logger.info(f"  Found {len(orphan_files)} orphan .pt files on disk not in index")

    # Recover orphans by extracting seq_id from filename ({seq_id}.pt)
    for orphan_name in orphan_files:
        seq_id = orphan_name.removesuffix(".pt")
        if seq_id and seq_id not in graph_index:
            # Load to get n_atoms for a complete index entry
            orphan_path = graph_dir / orphan_name
            try:
                graph_data = torch.load(orphan_path, map_location="cpu", weights_only=False)
                n_atoms = graph_data.get("n_atoms") if isinstance(graph_data, dict) else None
                graph_index[seq_id] = {"filename": orphan_name, "n_atoms": n_atoms}
                stats["orphans_recovered"] += 1
            except Exception as e:
                logger.warning(f"  Could not load orphan {orphan_name}: {e}")

    # Step 2: For each duplicate group, ensure canonical ID has an entry
    # If canonical has no .pt but a dropped ID does, remap
    for did in dropped_ids:
        cid = canonical_ids[did]
        if did in graph_index:
            if cid not in graph_index:
                # Remap: canonical gets the dropped ID's graph file
                graph_index[cid] = graph_index[did]
                stats["entries_remapped"] += 1
                logger.debug(f"  Remapped graph entry {did} -> {cid}")

    # Step 3: Remove entries for dropped IDs
    for did in dropped_ids:
        if did in graph_index:
            dropped_entry = graph_index.pop(did)
            dropped_file = _get_filename(dropped_entry)
            stats["entries_removed"] += 1

            # Optionally delete the .pt file if it's now orphaned
            # (only if no other entry points to it)
            if cleanup_graphs:
                still_referenced = {_get_filename(e) for e in graph_index.values()}
                if dropped_file not in still_referenced:
                    dropped_path = graph_dir / dropped_file
                    if dropped_path.exists():
                        dropped_path.unlink()
                        stats["files_deleted"] += 1

    # Step 4: Write updated index
    with open(graph_index_path, "w") as f:
        json.dump(graph_index, f)

    stats["final_index_entries"] = len(graph_index)
    return stats


def print_report(dedup_stats, split_sizes=None, graph_stats=None):
    """Print a summary report of the deduplication."""
    print("\n" + "=" * 60)
    print("FASTA DEDUPLICATION REPORT")
    print("=" * 60)

    print(f"\nSequence counts:")
    print(f"  Original entries:    {dedup_stats['original_count']:>10,}")
    print(f"  Deduplicated entries: {dedup_stats['deduped_count']:>9,}")
    print(f"  Removed:             {dedup_stats['removed_count']:>10,}")

    print(f"\nDuplicate groups:      {dedup_stats['duplicate_groups']:>10,}")
    print(f"  Same labels:         {dedup_stats['same_label_groups']:>10,}")
    print(f"  Different labels:    {dedup_stats['diff_label_groups']:>10,}")
    print(f"  Labels gained:       {dedup_stats['total_labels_gained']:>10,}")
    print(f"  Largest group:       {dedup_stats['largest_group_size']:>10,} copies (ID: {dedup_stats['largest_group_id']})")

    if split_sizes:
        print(f"\nSplit sizes (80/10/10):")
        for name, size in split_sizes.items():
            print(f"  {name:>5s}: {size:>10,}")
        print(f"  Total: {sum(split_sizes.values()):>10,}")

    if graph_stats:
        print(f"\nGraph index updates:")
        print(f"  Original entries:    {graph_stats['original_index_entries']:>10,}")
        print(f"  Orphans recovered:   {graph_stats['orphans_recovered']:>10,}")
        print(f"  Entries remapped:    {graph_stats['entries_remapped']:>10,}")
        print(f"  Entries removed:     {graph_stats['entries_removed']:>10,}")
        print(f"  Final entries:       {graph_stats['final_index_entries']:>10,}")
        if graph_stats['files_deleted'] > 0:
            print(f"  Files deleted:       {graph_stats['files_deleted']:>10,}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate a FASTA file by amino acid sequence, merging GO labels."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the input FASTA file (e.g., full_GO.fasta).",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Output directory for train/dev/test split files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="GO",
        help="Label prefix for output files (default: GO).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train fraction (default: 0.8)."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation fraction (default: 0.1)."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test fraction (default: 0.1)."
    )
    parser.add_argument(
        "--graph-index-path",
        type=str,
        default=None,
        help="Path to graph_index.json (optional).",
    )
    parser.add_argument(
        "--graph-dir",
        type=str,
        default=None,
        help="Directory containing .pt graph files (optional).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only, do not write any files.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip re-splitting into train/dev/test.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating .bak backups of original files.",
    )
    parser.add_argument(
        "--cleanup-graphs",
        action="store_true",
        help="Delete orphaned .pt files for dropped IDs.",
    )

    args = parser.parse_args()

    # Validate
    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        raise SystemExit(1)

    if not args.no_split and not args.dry_run and args.splits_dir is None:
        parser.error("--splits-dir is required unless --no-split or --dry-run is set.")

    if args.graph_index_path and not args.graph_dir:
        parser.error("--graph-dir is required when --graph-index-path is set.")

    ratio_total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_total - 1.0) > 1e-6:
        parser.error(f"Ratios must sum to 1.0, got {ratio_total:.4f}")

    # Step 1: Read FASTA
    logger.info(f"Reading {input_path} ...")
    data = read_fasta(str(input_path))
    logger.info(f"Loaded {len(data)} entries.")

    # Step 2: Deduplicate
    logger.info("Deduplicating by sequence ...")
    deduped, dedup_stats = deduplicate_entries(data)
    logger.info(
        f"Deduplication complete: {dedup_stats['original_count']} -> {dedup_stats['deduped_count']} "
        f"({dedup_stats['removed_count']} removed)"
    )

    if args.dry_run:
        print_report(dedup_stats)
        logger.info("Dry run complete. No files were modified.")
        return

    # Step 3: Backup originals
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.no_backup:
        logger.info("Creating backups ...")
        backup_file(str(input_path), timestamp)
        if args.splits_dir:
            splits_dir = Path(args.splits_dir)
            for split_name in ["train", "dev", "test"]:
                split_file = splits_dir / f"{split_name}_{args.prefix}.fasta"
                backup_file(str(split_file), timestamp)

    # Step 4: Write deduplicated FASTA
    logger.info(f"Writing deduplicated FASTA to {input_path} ...")
    save_to_fasta(deduped, str(input_path))

    # Step 5: Re-split
    split_sizes = None
    if not args.no_split:
        _split_mod = _import_split_dataset()
        split_dataset = _split_mod.split_dataset
        log_split_stats = _split_mod.log_split_stats

        logger.info("Re-splitting into train/dev/test ...")
        train, val, test = split_dataset(
            deduped,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        all_labels = set()
        for _, _, labels in deduped:
            all_labels.update(labels)

        log_split_stats("train", train, all_labels)
        log_split_stats("dev", val, all_labels)
        log_split_stats("test", test, all_labels)

        splits_dir = Path(args.splits_dir)
        splits_dir.mkdir(parents=True, exist_ok=True)

        save_to_fasta(train, str(splits_dir / f"train_{args.prefix}.fasta"))
        save_to_fasta(val, str(splits_dir / f"dev_{args.prefix}.fasta"))
        save_to_fasta(test, str(splits_dir / f"test_{args.prefix}.fasta"))

        split_sizes = {
            "train": len(train),
            "dev": len(val),
            "test": len(test),
        }

    # Step 6: Graph index dedup
    graph_stats = None
    if args.graph_index_path:
        if not Path(args.graph_index_path).exists():
            logger.error(f"Graph index not found: {args.graph_index_path}")
            raise SystemExit(1)
        if not args.no_backup:
            backup_file(args.graph_index_path, timestamp)
        logger.info("Updating graph index ...")
        # Build canonical mapping: dropped_id -> canonical_id
        seq_to_canonical = {}
        for sequence, seq_id, labels in data:
            if sequence not in seq_to_canonical:
                seq_to_canonical[sequence] = seq_id

        canonical_map = {}
        for sequence, seq_id, labels in data:
            canonical_id = seq_to_canonical[sequence]
            if seq_id != canonical_id:
                canonical_map[seq_id] = canonical_id

        graph_stats = update_graph_index(
            graph_index_path=args.graph_index_path,
            graph_dir=args.graph_dir,
            dropped_ids=dedup_stats["dropped_ids"],
            canonical_ids=canonical_map,
            cleanup_graphs=args.cleanup_graphs,
        )

    # Step 7: Print report
    print_report(dedup_stats, split_sizes=split_sizes, graph_stats=graph_stats)
    logger.info("Done.")


if __name__ == "__main__":
    main()
