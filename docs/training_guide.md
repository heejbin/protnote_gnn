# ToxinNote Training Guide

This guide walks through the complete pipeline for training a ToxinNote model — from environment setup and data preparation through training, evaluation, and inference.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Data Preparation](#3-data-preparation)
4. [Configuration System](#4-configuration-system)
5. [Training](#5-training)
6. [Evaluation and Testing](#6-evaluation-and-testing)
7. [Inference](#7-inference)
8. [Advanced Topics](#8-advanced-topics)

---

## 1. Prerequisites

- **Hardware**: NVIDIA GPU with CUDA 11.8+
- **Software**: Python 3.10+, [Pixi](https://pixi.sh/) package manager
- **Platform**: Linux (x86_64)

## 2. Environment Setup

```bash
# Clone the repository
git clone <repo-url> && cd toxinnote-dev

# Activate the environment (installs all dependencies automatically)
pixi shell
```

This installs PyTorch 2.4.0, Hydra, OmegaConf, transformers, ESM-C, AtomWorks, and all other dependencies defined in `pyproject.toml`.

## 3. Data Preparation

Data preparation has several stages. The quickstart path downloads pre-processed data. For custom datasets, follow the full pipeline.

### 3.1 Download Pre-processed Data (Quickstart)

```bash
pixi run dataprep
```

This downloads the complete dataset from HuggingFace (`cs527-toxinnote/protnote_data`) into the `data/` directory, including:
- FASTA files (train/val/test splits)
- GO/EC annotations
- Pre-computed label embeddings
- ProteInfer model weights
- Vocabularies

After this step you can skip directly to [Section 3.6](#36-generate-esm-c-sequence-embeddings) if using the hybrid encoder, or [Section 5](#5-training) if using the legacy ProteInfer encoder.

### 3.2 Create Datasets from Raw Sources (Optional)

If you need to build datasets from scratch:

**From ProteInfer TFRecords:**

```bash
python bin/make_proteinfer_dataset.py \
    --dataset-type random \
    --annotation-types GO
```

This converts TFRecord files into FASTA format with GO annotations in the sequence headers.

**From SwissProt releases (for temporal splits / new GO terms):**

```bash
python bin/make_dataset_from_swissprot.py \
    --latest-swissprot-file uniprot_sprot_jul_2024.dat \
    --output-file-path data/swissprot/test_2024.fasta \
    --label-vocabulary new \
    --sequence-vocabulary new
```

**For zero-shot evaluation (label-disjoint splits):**

```bash
python bin/make_zero_shot_datasets_from_proteinfer.py
```

Creates 80/10/10 train/val/test splits where labels are disjoint across splits.

### 3.3 Generate Parenthood JSON (Optional)

Generate the GO/EC label hierarchy used for label normalization during evaluation:

```bash
# Auto-detect source files (latest .obo + enzyme files in data/annotations/)
python bin/generate_parenthood.py

# Or specify files explicitly
python bin/generate_parenthood.py \
    --go-obo data/annotations/go_2025-10-10.obo \
    --enzyme-dat data/annotations/enzyme_251015.dat \
    --enzclass-txt data/annotations/enzclass_251015.txt \
    --output data/vocabularies/parenthood_2025_10.json
```

This replaces the proteinfer `parenthood_bin.py` dependency. Source files needed:
- **GO**: A `.obo` file — download with `python bin/download_GO_annotations.py`
- **EC**: `enzyme.dat` + `enzclass.txt` from [ExPASy](https://ftp.expasy.org/databases/enzyme/) (optional, use `--skip-ec` for GO only)

Output: `data/vocabularies/parenthood_{date}.json` — referenced in config as `PARENTHOOD_LIB_PATH`.

### 3.4 Download Protein Structures (Optional)

Required only if you need structure files for new proteins not already in `data/structures/`.

```bash
python bin/download_structures.py \
    --fasta-path data/swissprot/proteinfer_splits/random/train_GO.fasta \
    --source alphafolddb
```

Downloads CIF/PDB files from AlphaFold DB or RCSB PDB into `data/structures/`.

### 3.5 Generate Label Embeddings

Pre-compute text embeddings for all GO term descriptions using Multilingual E5:

```bash
python bin/generate_label_embeddings.py \
    --add-instruction \
    --account-for-sos
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--pooling-method` | `mean` | Embedding pooling: `mean`, `last_token`, or `all` |
| `--label-encoder-checkpoint` | `intfloat/multilingual-e5-large-instruct` | HuggingFace model |
| `--base-label-embedding-path` | `GO_BASE_LABEL_EMBEDDING_PATH` | Config key for output path |
| `--annotations-path-name` | `GO_ANNOTATIONS_2019_UPDATED_PATH` | Config key for annotations |
| `--add-instruction` | off | Format input for instruction-tuned model |
| `--account-for-sos` | off | Ignore SOS token in pooling |

**For EC labels:**

```bash
python bin/generate_label_embeddings.py \
    --base-label-embedding-path EC_BASE_LABEL_EMBEDDING_PATH \
    --annotations-path-name EC_ANNOTATIONS_PATH \
    --add-instruction --account-for-sos
```

Output: `data/embeddings/frozen_E5_multiling_inst_label_embeddings_mean.pt` (+ corresponding index file).

### 3.6 Generate ESM-C Sequence Embeddings

Required for the hybrid (ESM-C + EGNN) encoder. Generates per-residue embeddings (960-dim) for each protein:

```bash
python bin/generate_sequence_embeddings.py \
    --fasta-path-names TRAIN_DATA_PATH VAL_DATA_PATH TEST_DATA_PATH \
    --batch-size 16
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `esmc_300m` | ESM-C model variant |
| `--batch-size` | `8` | Inference batch size |
| `--max-sequence-length` | `5000` | Skip longer sequences |

Output: Individual `.pt` files in `data/embeddings/esmc/` plus `index.json`.

### 3.7 Build Atom-Level Graphs

Combines structure files + ESM-C embeddings into atom-level graphs for the EGNN encoder:

```bash
python bin/prepare_graph_data.py \
    --fasta-path-names TRAIN_DATA_PATH VAL_DATA_PATH TEST_DATA_PATH \
    --num-workers 8 \
    --knn-k 20 \
    --consolidate
```

**What it does:**
1. Parses CIF/PDB structure files (via AtomWorks)
2. Loads pre-computed ESM-C per-residue embeddings
3. Builds atom-level graphs (covalent bonds + k-NN spatial edges)
4. Aligns ESM-C residue embeddings to structure atoms
5. Saves as individual `.pt` graphs or a consolidated `.pngrph` archive

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-workers` | `8` | Parallel processing workers |
| `--knn-k` | `20` | Number of k-nearest-neighbor edges per atom |
| `--consolidate` | off | Merge all graphs into a single `.pngrph` archive |

Output: `data/processed/graphs.pngrph` (archive) + `data/processed/graph_index.json`.

### Data Preparation Summary

| Step | Script | Required For | Output |
|------|--------|--------------|--------|
| Download data | `pixi run dataprep` | All | `data/` directory |
| Parenthood JSON | `generate_parenthood.py` | Evaluation (label normalization) | `data/vocabularies/parenthood_*.json` |
| Label embeddings | `generate_label_embeddings.py` | All | `data/embeddings/*.pt` |
| ESM-C embeddings | `generate_sequence_embeddings.py` | Hybrid encoder | `data/embeddings/esmc/*.pt` |
| Atom graphs | `prepare_graph_data.py` | Hybrid encoder | `data/processed/graphs.pngrph` |

---

## 4. Configuration System

ToxinNote uses [Hydra](https://hydra.cc/) for configuration management.

### 4.1 Config Structure

```
configs/
  config.yaml                  # Root config: defaults + runtime settings
  params/
    default.yaml               # Training hyperparameters
  encoder/
    proteinfer.yaml            # ProteInfer CNN constants
    structural.yaml            # EGNN structural encoder settings
  paths/
    default.yaml               # Data and output paths (relative)
  remote/
    default.yaml               # Remote URLs (HuggingFace, AlphaFold DB)
```

### 4.2 Runtime Settings (`config.yaml` → `run:`)

These control what the training script does:

| Setting | Default | Description |
|---------|---------|-------------|
| `run.name` | `ProtNote` | Experiment name (used in checkpoints, logs, W&B) |
| `run.train_path_name` | `null` | Config key for training FASTA (e.g., `TRAIN_DATA_PATH`) |
| `run.validation_path_name` | `null` | Config key for validation FASTA |
| `run.test_paths_names` | `null` | List of config keys for test FASTAs |
| `run.full_path_name` | `null` | Config key for full dataset (vocabulary generation) |
| `run.annotations_path_name` | `GO_ANNOTATIONS_PATH` | Config key for GO/EC annotations |
| `run.base_label_embedding_name` | `GO_BASE_LABEL_EMBEDDING_PATH` | Config key for label embeddings |
| `run.wandb_project` | `null` | W&B project name (null = no logging) |
| `run.model_file` | `null` | Checkpoint to load (filename in `data/models/ProtNote/`) |
| `run.from_checkpoint` | `false` | Resume training from checkpoint (restores optimizer state) |
| `run.save_prediction_results` | `false` | Save predictions as HDF5 |
| `run.save_embeddings` | `false` | Save joint embeddings |
| `run.use_sequence_encoder` | `false` | Use legacy ProteInfer CNN (default: hybrid EGNN) |
| `run.gpus` | `1` | Number of GPUs per node |
| `run.nodes` | `1` | Number of nodes |

### 4.3 Key Hyperparameters (`params/default.yaml`)

**Batch sizes and sampling:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_BATCH_SIZE` | `2` | Per-GPU training batch size |
| `VALIDATION_BATCH_SIZE` | `2` | Per-GPU validation batch size |
| `TEST_BATCH_SIZE` | `8` | Per-GPU test batch size |
| `WEIGHTED_SAMPLING` | `true` | Over-sample rare sequences by inverse label frequency |
| `INV_FREQUENCY_POWER` | `0.5` | Power for inverse frequency weighting |
| `TRAIN_LABEL_SAMPLE_SIZE` | `null` | Sample K labels per batch (null = all) |

**Optimization:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | `0.0003` | Learning rate |
| `OPTIMIZER` | `Adam` | Optimizer (Adam or AdamW) |
| `WEIGHT_DECAY` | `0.001` | Weight decay (AdamW/SGD only) |
| `NUM_EPOCHS` | `46` | Total training epochs |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | Accumulate gradients over N steps |
| `CLIP_VALUE` | `1` | Gradient clipping max norm (null = off) |

**Architecture:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PROTEIN_EMBEDDING_DIM` | `1100` | Protein encoder output dimension |
| `LABEL_EMBEDDING_DIM` | `1024` | Label encoder output dimension |
| `LATENT_EMBEDDING_DIM` | `1024` | Shared latent space dimension |
| `FEATURE_FUSION` | `concatenation` | Fusion method: `concatenation`, `similarity`, `concatenation_diff`, `concatenation_prod` |
| `OUTPUT_MLP_NUM_LAYERS` | `3` | Output MLP depth |
| `PROJECTION_HEAD_NUM_LAYERS` | `4` | Projection head depth |

**Encoder settings:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_PLM_EMBEDDINGS` | `true` | Use ESM-C embeddings (false = learned AA embeddings) |
| `EGNN_HIDDEN_DIM` | `256` | EGNN hidden dimension |
| `EGNN_N_LAYERS` | `4` | Number of EGNN layers |
| `EGNN_OUT_DIM` | `256` | EGNN output dimension |
| `LORA` | `true` | Enable LoRA for label encoder |
| `LORA_RANK` | `4` | LoRA rank |
| `LABEL_ENCODER_NUM_TRAINABLE_LAYERS` | `0` | Trainable layers in E5 (0 = frozen) |

**Loss function:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LOSS_FN` | `FocalLoss` | Loss: `BCE`, `FocalLoss`, `WeightedBCE`, `BatchWeightedBCE`, `RGDBCE`, `CBLoss` |
| `FOCAL_LOSS_GAMMA` | `2` | Focal loss focusing parameter |
| `FOCAL_LOSS_ALPHA` | `-1` | Focal loss balancing (-1 = no balancing) |

**Augmentation:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `AUGMENT_RESIDUE_PROBABILITY` | `0.1` | Probability of BLOSUM62-based AA substitution per residue |
| `LABEL_EMBEDDING_NOISING_ALPHA` | `20.0` | Additive noise on label embeddings during training |

**Evaluation:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPTIMIZATION_METRIC_NAME` | `f1_macro` | Metric to optimize for model selection |
| `DECISION_TH` | `0.5` | Decision threshold (null = tune on validation set) |
| `ESTIMATE_MAP` | `false` | Compute mean average precision |

### 4.4 Data Path Keys (`paths/default.yaml`)

Path names referenced in CLI arguments map to relative paths under `data/`:

| Key | Path |
|-----|------|
| `TRAIN_DATA_PATH` | `swissprot/proteinfer_splits/random/train_GO.fasta` |
| `VAL_DATA_PATH` | `swissprot/proteinfer_splits/random/dev_GO.fasta` |
| `TEST_DATA_PATH` | `swissprot/proteinfer_splits/random/test_GO.fasta` |
| `FULL_DATA_PATH` | `swissprot/proteinfer_splits/random/full_GO.fasta` |
| `TRAIN_DATA_PATH_ZERO_SHOT` | `swissprot/proteinfer_splits/random/fake_train_GO_zero_shot.fasta` |
| `VAL_DATA_PATH_ZERO_SHOT` | `swissprot/proteinfer_splits/random/fake_dev_GO_zero_shot.fasta` |
| `TEST_DATA_PATH_ZERO_SHOT` | `zero_shot/GO_swissprot_jul_2024.fasta` |
| `GO_ANNOTATIONS_PATH` | `annotations/go_annotations_jul_2024.pkl` |
| `GO_BASE_LABEL_EMBEDDING_PATH` | `embeddings/frozen_label_embeddings.pt` |

### 4.5 Overriding Config at the Command Line

All parameters can be overridden via Hydra CLI syntax:

```bash
# Override a single parameter
python bin/main.py params.LEARNING_RATE=0.001

# Override multiple parameters
python bin/main.py params.TRAIN_BATCH_SIZE=8 params.NUM_EPOCHS=100

# Override run settings
python bin/main.py run.name=my_experiment run.wandb_project=my_project

# Print the resolved config without running
python bin/main.py --cfg job
```

---

## 5. Training

### 5.1 Supervised Training (Hybrid Encoder — Default)

The default encoder is the hybrid ESM-C + EGNN structure encoder. This requires pre-computed ESM-C embeddings and atom-level graphs (see [Section 3](#3-data-preparation)).

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    run.test_paths_names='[TEST_DATA_PATH]' \
    run.full_path_name=FULL_DATA_PATH \
    run.annotations_path_name=GO_ANNOTATIONS_PATH \
    run.base_label_embedding_name=GO_BASE_LABEL_EMBEDDING_PATH \
    run.name=toxinnote_hybrid \
    run.wandb_project=toxinnote
```

**What happens:**
1. Config is loaded and paths resolved
2. DDP process group is initialized (one process per GPU)
3. Label tokenizer and encoder (E5) are loaded
4. Graph index and archive are loaded
5. `ProteinStructureDataset` is created for train/val/test
6. `StructuralProteinEncoder` (ESM-C + EGNN) is initialized
7. `ProtNote` model is assembled and wrapped in DDP
8. Loss function and weighted sampler are configured
9. Training loop runs for `NUM_EPOCHS` epochs
10. Best checkpoint is saved based on `OPTIMIZATION_METRIC_NAME`
11. Final evaluation on validation and test sets

### 5.2 Supervised Training (Legacy ProteInfer Encoder)

For sequence-only training without structure data:

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    run.test_paths_names='[TEST_DATA_PATH]' \
    run.full_path_name=FULL_DATA_PATH \
    run.use_sequence_encoder=true \
    run.name=protnote_cnn
```

### 5.3 Zero-Shot Training

Train on a label-disjoint split to evaluate zero-shot generalization:

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH_ZERO_SHOT \
    run.validation_path_name=VAL_DATA_PATH_ZERO_SHOT \
    run.test_paths_names='[TEST_DATA_PATH_ZERO_SHOT]' \
    run.annotations_path_name=GO_ANNOTATIONS_PATH \
    run.base_label_embedding_name=GO_BASE_LABEL_EMBEDDING_PATH \
    run.name=zero_shot_experiment \
    params.EXTRACT_VOCABULARIES_FROM=null
```

Note: `EXTRACT_VOCABULARIES_FROM=null` ensures vocabularies are built from the training set rather than the full dataset (important for proper zero-shot evaluation).

### 5.4 Multi-GPU Training

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    run.gpus=4 \
    run.name=multi_gpu_experiment
```

For multi-node training, also set `run.nodes` and `run.nr` (or use AMLT with `run.amlt=true`).

### 5.5 Resume from Checkpoint

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    run.model_file=2024-01-15_toxinnote_best_val_metric.pt \
    run.from_checkpoint=true \
    run.name=resumed_experiment
```

The `model_file` path is relative to `data/models/ProtNote/`. Setting `from_checkpoint=true` restores the optimizer state and epoch counter.

### 5.6 Checkpoints

During training, the trainer saves checkpoints to `outputs/checkpoints/`:

| Checkpoint | Filename Pattern | When |
|------------|-----------------|------|
| Best metric | `{timestamp}_{name}_best_val_metric.pt` | When `OPTIMIZATION_METRIC_NAME` improves |
| Best loss | `{timestamp}_{name}_best_val_loss.pt` | When validation loss improves |
| Last epoch | `{timestamp}_{name}_last_epoch.pt` | After every epoch |
| Periodic | `{timestamp}_{name}_epoch_{N}.pt` | Every 10 epochs |

Each checkpoint contains: `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_val_metric`.

### 5.7 Logging

- **Console + file**: Logs are written to `outputs/logs/{timestamp}_{name}.log`
- **Weights & Biases**: Set `run.wandb_project=<project>` to enable
- **MLFlow**: Set `run.mlflow=true` (requires AMLT)

---

## 6. Evaluation and Testing

### 6.1 Evaluate a Trained Model

Run inference and compute metrics on a test set:

```bash
python bin/main.py \
    run.test_paths_names='[TEST_DATA_PATH]' \
    run.model_file=best_model.pt \
    run.annotations_path_name=GO_ANNOTATIONS_PATH \
    run.base_label_embedding_name=GO_BASE_LABEL_EMBEDDING_PATH \
    run.save_prediction_results=true \
    run.name=eval_run
```

Include a validation set to automatically tune the decision threshold:

```bash
python bin/main.py \
    run.validation_path_name=VAL_DATA_PATH \
    run.test_paths_names='[TEST_DATA_PATH]' \
    run.model_file=best_model.pt \
    run.save_prediction_results=true \
    params.DECISION_TH=null \
    run.name=eval_with_threshold_tuning
```

Setting `DECISION_TH=null` triggers threshold optimization on the validation set before testing.

### 6.2 Metrics

The following metrics are computed:

| Metric | Description |
|--------|-------------|
| `f1_micro` | F1 score, micro-averaged across all labels |
| `f1_macro` | F1 score, macro-averaged (mean per-label F1) |
| `f1_weighted` | F1 score, weighted by label support |
| `precision_micro/macro` | Precision |
| `recall_micro/macro` | Recall |
| `map_micro` | Mean average precision (micro) |
| `map_macro` | Mean average precision (macro) |
| `bce_loss` | Binary cross-entropy loss |
| `focal_loss` | Focal loss |

### 6.3 ProteInfer Baseline

Test the standalone ProteInfer CNN baseline (no text encoder):

```bash
python bin/test_proteinfer.py \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --proteinfer-weights GO \
    --threshold 0.5
```

### 6.4 BLAST Baseline

Run sequence similarity baseline using BLAST:

```bash
python bin/run_blast.py \
    --test-paths-name TEST_DATA_PATH \
    --num-threads 8
```

### 6.5 Batch Model Testing

Test multiple models across datasets using pre-defined test commands:

```bash
python bin/test_models.py \
    --model-files model1.pt model2.pt \
    --test-names GO_zero_shot GO_supervised
```

### 6.6 Save Predictions

Predictions are saved as HDF5 files in `outputs/results/`:

```
outputs/results/{data_loader_name}_results_{run_name}.h5
```

Each file contains:
- `sequence_ids`: Protein identifiers
- `logits`: Raw model outputs
- `labels`: Ground truth multi-hot vectors
- `label_vocabulary`: Ordered label list

---

## 7. Inference

### 7.1 Inference on New Proteins

To run inference on a custom FASTA file:

1. Add the FASTA path to `configs/paths/default.yaml` under `data_paths`:
   ```yaml
   MY_CUSTOM_DATA: path/to/my_proteins.fasta
   ```

2. If using the hybrid encoder, generate ESM-C embeddings and atom graphs for the new proteins (see Sections 3.6 and 3.7).

3. Run inference:
   ```bash
   python bin/main.py \
       run.test_paths_names='[MY_CUSTOM_DATA]' \
       run.model_file=best_model.pt \
       run.save_prediction_results=true \
       run.name=custom_inference
   ```

### 7.2 FASTA Format

Input FASTA files should have GO/EC annotations in the header:

```
>SEQUENCE_ID GO:0006412 GO:0003735
MAKQKTEVVRIVGRPFAYTLKDSQAKLR...
```

For inference-only (no ground truth), annotations in headers are optional.

---

## 8. Advanced Topics

### 8.1 Model Architecture Overview

ToxinNote uses a two-tower architecture:

```
Protein branch                         Text branch
──────────────                         ───────────
Structure + Sequence                   GO term description
        │                                      │
   ESM-C (frozen)                    Multilingual E5
   960-dim/residue                   (frozen + LoRA)
        │                                      │
   + atom-type one-hot (37)                    │
   = 997-dim/atom                              │
        │                                      │
   EGNN (4 layers)                             │
   256-dim/atom                                │
        │                                      │
   Global pooling                              │
   → 1100-dim                          → 1024-dim
        │                                      │
   Projection head                    Projection head
   → 1024-dim                         → 1024-dim
        │                                      │
        └──────── Concatenation ───────────────┘
                        │
                   Output MLP (3 layers)
                        │
                   sigmoid → probability
```

### 8.2 Feature Fusion Options

| Method | Description |
|--------|-------------|
| `concatenation` | `[protein_emb, label_emb]` → MLP (default) |
| `concatenation_diff` | `[protein_emb, label_emb, protein_emb - label_emb]` → MLP |
| `concatenation_prod` | `[protein_emb, label_emb, protein_emb * label_emb]` → MLP |
| `similarity` | Cosine similarity / temperature |

### 8.3 Loss Functions

| Loss | When to Use |
|------|-------------|
| `FocalLoss` | Default. Handles extreme class imbalance in GO annotation |
| `BCE` | Standard binary cross-entropy baseline |
| `WeightedBCE` | Per-label weighting by inverse frequency |
| `CBLoss` | Class-balanced loss for long-tailed distributions |
| `RGDBCE` | Reweighted gradient descent BCE |

### 8.4 Data Augmentation

**Sequence augmentation** (`AUGMENT_RESIDUE_PROBABILITY=0.1`): Each residue has a 10% chance of being substituted according to BLOSUM62 substitution matrix probabilities. Only applied during training.

**Label embedding noising** (`LABEL_EMBEDDING_NOISING_ALPHA=20.0`): Additive Gaussian noise is applied to label embeddings during training, scaled by `alpha / sqrt(embedding_dim)`. This regularizes the text encoder pathway.

### 8.5 Interpretability

ToxinNote supports residue-level interpretability via Integrated Gradients:

```bash
python bin/main.py \
    run.test_paths_names='[TEST_DATA_PATH]' \
    run.model_file=best_model.pt \
    params.INTERPRETABILITY_METHOD=IntegratedGradient \
    params.INTERPRETABILITY_WEIGHT=0.1 \
    params.INTERPRETABILITY_N_STEPS=50
```

This computes gradient-input attributions and can be trained jointly with a site ground truth MSE loss.

### 8.6 Rapid Prototyping

Use subset fractions to train on smaller data for quick iteration:

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    params.TRAIN_SUBSET_FRACTION=0.1 \
    params.VALIDATION_SUBSET_FRACTION=0.1 \
    params.NUM_EPOCHS=5 \
    run.name=quick_test
```

### 8.7 Common Override Recipes

**Increase batch size with gradient accumulation (for limited GPU memory):**
```bash
params.TRAIN_BATCH_SIZE=2 params.GRADIENT_ACCUMULATION_STEPS=4
# Effective batch size = 2 * 4 = 8
```

**Disable weighted sampling:**
```bash
params.WEIGHTED_SAMPLING=false
```

**Use BioGPT as label encoder:**
```bash
params.LABEL_ENCODER_CHECKPOINT=microsoft/biogpt
```

**Unfreeze label encoder layers:**
```bash
params.LABEL_ENCODER_NUM_TRAINABLE_LAYERS=2 params.LORA=false
```

**Use learned AA embeddings instead of ESM-C (for de novo proteins):**
```bash
params.USE_PLM_EMBEDDINGS=false params.LEARNED_AA_EMBEDDING_DIM=128
```
