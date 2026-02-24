"""Split a full FASTA dataset into train/dev/test splits.

Performs a random or stratified split of a FASTA file into training,
validation (dev), and test sets. Outputs separate FASTA files suitable
for use with the training pipeline.

Example Usage:
    # Random split with default 80/10/10 ratio
    python bin/split_dataset.py \
        --input-path-name FULL_EC_DATA_PATH \
        --output-dir data/swissprot/proteinfer_splits/random \
        --prefix EC

    # Custom ratios
    python bin/split_dataset.py \
        --input-path-name FULL_EC_DATA_PATH \
        --output-dir data/swissprot/proteinfer_splits/random \
        --prefix EC \
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

    # Absolute input path instead of config key
    python bin/split_dataset.py \
        --input-path data/my_dataset.fasta \
        --output-dir data/splits \
        --prefix GO
"""

import argparse
import logging
import os
import random
from collections import Counter
from pathlib import Path

from protnote.utils.configs import get_project_root, register_resolvers
from protnote.utils.data import read_fasta, save_to_fasta

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def split_dataset(data, train_ratio, val_ratio, test_ratio, seed=42):
    """Split data into train/val/test by random shuffle.

    Args:
        data: List of (sequence, sequence_id, labels) tuples.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) lists.
    """
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = [data[i] for i in indices[:n_train]]
    val = [data[i] for i in indices[n_train : n_train + n_val]]
    test = [data[i] for i in indices[n_train + n_val :]]

    return train, val, test


def log_split_stats(name, split_data, all_labels):
    """Log statistics for a dataset split."""
    label_counter = Counter()
    for _, _, labels in split_data:
        label_counter.update(labels)
    logger.info(
        f"  {name:>5s}: {len(split_data):>7d} sequences, "
        f"{len(label_counter):>6d} unique labels "
        f"({len(label_counter) / max(len(all_labels), 1) * 100:.1f}% of total)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a FASTA dataset into train/dev/test sets."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-path-name",
        type=str,
        help="Config key name from paths/default.yaml (e.g., FULL_EC_DATA_PATH).",
    )
    input_group.add_argument(
        "--input-path",
        type=str,
        help="Direct path to the input FASTA file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for split FASTA files (relative to project root or absolute).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Label prefix for output files (e.g., EC → train_EC.fasta, dev_EC.fasta, test_EC.fasta).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train fraction (default: 0.8).")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation fraction (default: 0.1).")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test fraction (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error(f"Ratios must sum to 1.0, got {total_ratio:.4f}")

    project_root = get_project_root()

    # Resolve input path
    if args.input_path_name:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        register_resolvers()
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(project_root / "configs")):
            cfg = compose(config_name="config")
        input_path = str(project_root / "data" / cfg.paths.data_paths[args.input_path_name])
    else:
        input_path = args.input_path
        if not os.path.isabs(input_path):
            input_path = str(project_root / input_path)

    # Resolve output dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        raise SystemExit(1)

    # Read input
    logger.info(f"Reading {input_path} ...")
    data = read_fasta(input_path)
    logger.info(f"Loaded {len(data)} sequences.")

    # Collect all labels for stats
    all_labels = set()
    for _, _, labels in data:
        all_labels.update(labels)
    logger.info(f"Total unique labels: {len(all_labels)}")

    # Split
    train, val, test = split_dataset(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    logger.info("Split statistics:")
    log_split_stats("train", train, all_labels)
    log_split_stats("dev", val, all_labels)
    log_split_stats("test", test, all_labels)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        f"train_{args.prefix}.fasta": train,
        f"dev_{args.prefix}.fasta": val,
        f"test_{args.prefix}.fasta": test,
    }

    for filename, split_data in splits.items():
        out_path = str(output_dir / filename)
        save_to_fasta(split_data, out_path)

    logger.info("Done.")
