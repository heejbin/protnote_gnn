#!/usr/bin/env python
"""
Structural ProtNote hyperparameter search (Optuna)

This script does two things using protnote_gnn (structure/hybrid encoder):

1. Create a cluster_id stratified 10% subset of the existing FASTA splits
   (train/dev/test) and save them next to the originals.
2. Run Optuna: for each trial, train for 5 epochs and pick the best
   hyperparameters by validation macro mAP (final_validation_map_macro).

Paths (data_root, optuna_output_dir) and structure index are read from
config (configs/config.yaml: structural_optuna.*, paths.data_paths.STRUCTURE_INDEX_PATH).
Logs are written to a .txt file under the Optuna output directory.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import optuna
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from omegaconf import DictConfig
from protnote.utils.configs import get_project_root, register_resolvers
from protnote.utils.data import read_fasta, save_to_fasta

# =====================
# Constants (paths come from config in __main__)
# =====================
SEED = 42
SUBSET_FRACTION = 0.10
N_TRIALS = 15

random.seed(SEED)
np.random.seed(SEED)


class Tee:
    """Write to both file and stdout."""

    def __init__(self, file_path: Path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data: str):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def setup_logging(log_file: Path):
    """Redirect stdout to both log file and console."""
    tee = Tee(log_file)
    sys.stdout = tee
    return tee


def multilabel_stratified_subsample(
    data: Sequence[Tuple[str, str, Sequence[str]]],
    fraction: float,
    seed: int = 42,
) -> List[Tuple[str, str, Sequence[str]]]:
    """Greedy multi-label stratified subsample."""

    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0,1], got {fraction}")

    n = len(data)
    n_target = max(1, int(round(n * fraction)))
    if n_target >= n:
        return list(data)

    rng = random.Random(seed)

    labels_per_i: List[Tuple[str, ...]] = []
    label_counts = Counter()
    for _seq, _sid, labels in data:
        labs = tuple(labels) if labels else ("__NO_LABEL__",)
        labels_per_i.append(labs)
        label_counts.update(labs)

    desired = {l: fraction * c for l, c in label_counts.items()}
    selected = set()
    selected_label_counts = Counter()

    label_to_indices = defaultdict(list)
    for i, labs in enumerate(labels_per_i):
        for l in set(labs):
            label_to_indices[l].append(i)
    for l in label_to_indices:
        rng.shuffle(label_to_indices[l])

    def remaining_need(l: str) -> float:
        return desired[l] - selected_label_counts[l]

    def score_index(i: int) -> float:
        return sum(max(0.0, remaining_need(l)) for l in set(labels_per_i[i]))

    all_indices = list(range(n))
    rng.shuffle(all_indices)

    while len(selected) < n_target:
        labels_sorted = sorted(desired.keys(), key=lambda l: remaining_need(l), reverse=True)
        chosen_i = None

        for l in labels_sorted:
            if remaining_need(l) <= 0:
                break
            candidates = [i for i in label_to_indices[l] if i not in selected]
            if not candidates:
                continue
            chosen_i = max(candidates, key=score_index)
            break

        if chosen_i is None:
            for i in all_indices:
                if i not in selected:
                    chosen_i = i
                    break

        if chosen_i is None:
            break

        selected.add(chosen_i)
        selected_label_counts.update(set(labels_per_i[chosen_i]))

    subset = [data[i] for i in sorted(selected)]
    return subset


def label_histogram(data: Sequence[Tuple[str, str, Sequence[str]]]) -> Counter:
    c = Counter()
    for _seq, _sid, labels in data:
        c.update(labels)
    return c


def print_top_labels(c: Counter, k: int = 15):
    print(f"Top-{k} labels:")
    for lab, cnt in c.most_common(k):
        print(f"  {lab}: {cnt}")


def build_cluster_stratification_data(
    data: Sequence[Tuple[str, str, Sequence[str]]],
    structure_index: Dict,
) -> List[Tuple[str, str, Sequence[str]]]:
    """Build (seq, sid, [cluster_id]) for stratification."""
    return [
        (seq, sid, [structure_index.get(sid, {}).get("cluster_id", "__NO_CLUSTER__")])
        for (seq, sid, _) in data
    ]


def make_subset_path(original: Path, fraction: float) -> Path:
    tag = f"sub{int(round(fraction * 100))}"
    return original.with_name(f"{original.stem}_{tag}{original.suffix}")


def make_and_save_subset(
    original_path: Path, fraction: float, seed: int, structure_index: Dict
) -> Path:
    data = read_fasta(str(original_path))
    sid_to_go_labels = {sid: labels for (_seq, sid, labels) in data}
    data_for_strat = build_cluster_stratification_data(data, structure_index)
    subset_strat = multilabel_stratified_subsample(
        data_for_strat, fraction=fraction, seed=seed
    )
    subset = [
        (seq, sid, sid_to_go_labels[sid]) for (seq, sid, _) in subset_strat
    ]

    out_path = make_subset_path(original_path, fraction)
    save_to_fasta(subset, str(out_path))

    print("=" * 80)
    print("Original:", original_path)
    print("Subset  :", out_path)
    print(f"Sizes   : {len(data)} -> {len(subset)} ({len(subset)/len(data)*100:.2f}%)")

    c_full = label_histogram(data_for_strat)
    c_sub = label_histogram(subset_strat)
    print(f"Unique cluster_ids: {len(c_full)} -> {len(c_sub)}")
    print_top_labels(c_full, k=10)
    print("--")
    print_top_labels(c_sub, k=10)

    return out_path


@dataclass(frozen=True)
class TrialResult:
    trial_number: int
    metric: float
    metrics_path: Path
    trial_dir: Path


def run_one_trial(
    *,
    trial_number: int,
    overrides: List[str],
    data_root: Path,
    trial_root: Path,
    project_root: Path,
    train_sub_rel: Path,
    val_sub_rel: Path,
    test_sub_rel: Path,
    master_port: int,
) -> TrialResult:
    """Run `python bin/main.py ...` in a subprocess and return validation macro mAP (final_validation_map_macro)."""

    trial_dir = trial_root / f"trial_{trial_number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = trial_dir / "metrics.json"

    env = os.environ.copy()
    env["AMLT_DATA_DIR"] = str(data_root)
    env["AMLT_OUTPUT_DIR"] = str(trial_dir)
    env["NODE_RANK"] = "0"
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = str(master_port)

    cmd = [
        sys.executable,
        "bin/main.py",
        "run.amlt=true",
        "run.gpus=1",
        "run.nodes=1",
        "run.nr=0",
        "run.test_paths_names=null",
        "run.save_val_test_metrics=true",
        f"run.save_val_test_metrics_file={metrics_path}",
        "params.NUM_EPOCHS=5",
        "params.EPOCHS_PER_VALIDATION=1",
        "run.use_sequence_encoder=false",
        f"paths.data_paths.TRAIN_DATA_PATH={train_sub_rel.as_posix()}",
        f"paths.data_paths.VAL_DATA_PATH={val_sub_rel.as_posix()}",
        f"paths.data_paths.TEST_DATA_PATH={test_sub_rel.as_posix()}",
        *overrides,
    ]

    # Stream output in real-time (avoids "stuck" feeling; subprocess.run+PIPE buffers until done)
    log_path = trial_dir / "run.log"
    with open(log_path, "w") as log_file:
        p = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in iter(p.stdout.readline, ""):
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        p.wait()

    if p.returncode != 0:
        raise RuntimeError(
            f"Trial failed (exit={p.returncode}). See log: {trial_dir / 'run.log'}"
        )

    if not metrics_path.exists():
        raise RuntimeError(
            f"Metrics file not written: {metrics_path}. See log: {trial_dir/'run.log'}"
        )

    metrics_list = json.loads(metrics_path.read_text())
    if not metrics_list:
        raise RuntimeError(f"Empty metrics file: {metrics_path}")

    last = metrics_list[-1]
    key = "final_validation_map_macro"
    if key not in last:
        raise KeyError(
            f"Missing '{key}' in metrics. Available keys: {sorted(last.keys())}"
        )

    metric = float(last[key])
    return TrialResult(
        trial_number=trial_number,
        metric=metric,
        metrics_path=metrics_path,
        trial_dir=trial_dir,
    )


def main(cfg: DictConfig, log_file: Path):
    data_root = Path(cfg.structural_optuna.data_root)
    optuna_output_dir = str(cfg.structural_optuna.optuna_output_dir)
    optuna_root = Path(optuna_output_dir) if Path(optuna_output_dir).is_absolute() else data_root / optuna_output_dir

    print("=" * 80)
    print("Structural ProtNote Optuna Hyperparameter Search")
    print("=" * 80)
    print(f"LOG_FILE: {log_file}")
    print(f"DATA_ROOT: {data_root}")
    print(f"OPTUNA_ROOT: {optuna_root}")
    print(f"SUBSET_FRACTION: {SUBSET_FRACTION}")
    print(f"N_TRIALS: {N_TRIALS}")
    print("(Optimization metric: validation macro mAP)")
    print("=" * 80)

    project_root = get_project_root()

    train_rel = Path(cfg.paths.data_paths.TRAIN_DATA_PATH)
    val_rel = Path(cfg.paths.data_paths.VAL_DATA_PATH)
    test_rel = Path(cfg.paths.data_paths.TEST_DATA_PATH)

    train_path = data_root / train_rel
    val_path = data_root / val_rel
    test_path = data_root / test_rel

    print("TRAIN_PATH:", train_path)
    print("VAL_PATH  :", val_path)
    print("TEST_PATH :", test_path)

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing expected FASTA: {p}")

    print("OK: found the default FASTA split files under DATA_ROOT")

    # Use existing subset if present; otherwise create cluster_id stratified subset
    train_sub_path = make_subset_path(train_path, SUBSET_FRACTION)
    val_sub_path = make_subset_path(val_path, SUBSET_FRACTION)
    test_sub_path = make_subset_path(test_path, SUBSET_FRACTION)

    if train_sub_path.exists() and val_sub_path.exists() and test_sub_path.exists():
        print("\nUsing existing subset files (skip creation):")
        print("  TRAIN_SUB:", train_sub_path)
        print("  VAL_SUB  :", val_sub_path)
        print("  TEST_SUB :", test_sub_path)
    else:
        print("\nCreating cluster_id stratified subsets...")
        structure_index_rel = cfg.paths.data_paths.STRUCTURE_INDEX_PATH
        structure_index_path = data_root / structure_index_rel
        with open(structure_index_path) as f:
            structure_index = json.load(f)

        train_sub_path = make_and_save_subset(
            train_path, SUBSET_FRACTION, SEED, structure_index
        )
        val_sub_path = make_and_save_subset(val_path, SUBSET_FRACTION, SEED, structure_index)
        test_sub_path = make_and_save_subset(test_path, SUBSET_FRACTION, SEED, structure_index)

    train_sub_rel = train_sub_path.relative_to(data_root)
    val_sub_rel = val_sub_path.relative_to(data_root)
    test_sub_rel = test_sub_path.relative_to(data_root)

    print("\nHydra overrides (relative under DATA_ROOT):")
    print("  paths.data_paths.TRAIN_DATA_PATH=", train_sub_rel.as_posix())
    print("  paths.data_paths.VAL_DATA_PATH=", val_sub_rel.as_posix())
    print("  paths.data_paths.TEST_DATA_PATH=", test_sub_rel.as_posix())

    # Optuna: 5-epoch training, optimize by validation macro mAP
    def objective(trial: optuna.Trial) -> float:
        print(f"\n[Optuna] Trial {trial.number}/{N_TRIALS} starting...")
        lr = trial.suggest_float("LEARNING_RATE", 1e-5, 5e-4, log=True)
        wd = trial.suggest_float("WEIGHT_DECAY", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("OUTPUT_MLP_DROPOUT", 0.0, 0.3)
        train_bs = trial.suggest_categorical("TRAIN_BATCH_SIZE", [4, 8, 16])
        focal_gamma = trial.suggest_float("FOCAL_LOSS_GAMMA", 0.5, 5.0)
        focal_alpha = trial.suggest_categorical("FOCAL_LOSS_ALPHA", [-1, 0.25, 0.5, 0.75])

        overrides = [
            f"params.LEARNING_RATE={lr}",
            f"params.WEIGHT_DECAY={wd}",
            f"params.OUTPUT_MLP_DROPOUT={dropout}",
            f"params.TRAIN_BATCH_SIZE={train_bs}",
            f"params.FOCAL_LOSS_GAMMA={focal_gamma}",
            f"params.FOCAL_LOSS_ALPHA={focal_alpha}",
            "params.NUM_WORKERS=2",
            "params.WEIGHTED_SAMPLING=false",
        ]

        master_port = 12000 + (trial.number % 2000)

        try:
            result = run_one_trial(
                trial_number=trial.number,
                overrides=overrides,
                data_root=data_root,
                trial_root=optuna_root,
                project_root=project_root,
                train_sub_rel=train_sub_rel,
                val_sub_rel=val_sub_rel,
                test_sub_rel=test_sub_rel,
                master_port=master_port,
            )
        except Exception as e:
            # If a trial fails (e.g., CUDA OOM / memory error), prune the trial
            # instead of crashing the whole Optuna study.
            print(f"[Optuna] Trial {trial.number} failed with error: {e}")
            print("[Optuna] Pruning this trial due to failure (likely OOM or resource issue).")
            raise optuna.TrialPruned() from e

        trial.set_user_attr("trial_dir", str(result.trial_dir))
        return result.metric

    print("\n" + "=" * 80)
    print("Starting Optuna study with", N_TRIALS, "trials")
    print("=" * 80)

    def progress_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        """Print trial progress (start/complete) for visibility."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"[Optuna] Trial {trial.number} completed: value={trial.value:.6f}")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        callbacks=[progress_callback],
    )

    print("\n" + "=" * 80)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    print("Best trial dir:", study.best_trial.user_attrs.get("trial_dir"))
    print("=" * 80)
    print(f"\nLog saved to: {log_file}")


# --- Setup logging and run ---
if __name__ == "__main__":
    project_root = get_project_root()
    GlobalHydra.instance().clear()
    register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(project_root / "configs")):
        cfg = compose(config_name="config")

    data_root = Path(cfg.structural_optuna.data_root)
    optuna_output_dir = str(cfg.structural_optuna.optuna_output_dir)
    optuna_root = Path(optuna_output_dir) if Path(optuna_output_dir).is_absolute() else data_root / optuna_output_dir
    optuna_root.mkdir(parents=True, exist_ok=True)
    log_file = optuna_root / f"structural_optuna_hparam_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    _tee = setup_logging(log_file)
    try:
        main(cfg, log_file)
    finally:
        sys.stdout = _tee.stdout
        _tee.close()
