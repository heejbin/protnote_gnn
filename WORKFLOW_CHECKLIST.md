# ToxinNote Workflow Checklist

## 1. Entry Point and Configuration

| Step | Location | Status | Notes |
|------|----------|--------|-------|
| CLI | `bin/main.py` | ✅ | Hydra CLI — all args are overrides: `run.train_path_name=X`, `params.KEY=VALUE`, etc. |
| Config load | `@hydra.main(config_path="../configs", config_name="config")` → `get_setup(cfg: DictConfig, is_master)` | ✅ | Returns dict with `params`, `paths`, `dataset_paths`, `timestamp`, `logger`, `LABEL_EMBEDDING_PATH` |
| Path resolution | `resolve_paths(paths_cfg, data_root, output_root)` in `protnote/utils/configs.py` | ✅ | Flattens `data_paths` / `output_paths`, prepends `data/` and `outputs/` roots respectively |
| model_file | `main.py` L78-79 | ✅ | When `run.model_file` is set, resolves to `{project_root}/data/models/ProtNote/{run.model_file}` |

---

## 2. Data Pipeline

### 2.1 Mode Branch

Controlled by `run.use_sequence_encoder` (default: `false`). In `main.py`: `use_hybrid = not run.use_sequence_encoder`.

- **Hybrid mode** (default, `use_hybrid=True`): `ProteinStructureDataset` + `create_structural_loaders` → atom-level `graph_data` dict, `sequence_ids`, `label_multihots`, `label_embeddings`, `label_token_counts`
- **Sequence mode** (`run.use_sequence_encoder=true`): `ProteinDataset` + `create_multiple_loaders` → `sequence_onehots`, `sequence_lengths`, `label_multihots`, `label_embeddings`, `label_token_counts`

### 2.2 Prerequisites for Hybrid (Atom-Level) Mode

| Step | Script / Config | Description |
|------|-----------------|-------------|
| 1) Structures | `paths["STRUCTURE_DIR"]` | PDB/CIF files downloaded via `bin/download_structures.py` or local AFDB |
| 2) ESM-C embeddings | `paths["ESMC_EMBEDDING_DIR"]` + `paths["ESMC_INDEX_PATH"]` | `bin/generate_sequence_embeddings.py` → per-residue 960-dim .pt files + index.json |
| 3) Atom graphs | `paths["GRAPH_INDEX_PATH"]` + `paths["GRAPH_ARCHIVE_PATH"]` | `bin/prepare_graph_data.py` → `graph_index.json` (seq_id → `{"filename", "n_atoms"}`) + `graphs.pngrph` archive |

When using hybrid mode, **you must** prepare data in the order above.

### 2.3 Dataset / Collator

| Item | Status | Notes |
|------|--------|-------|
| `ProteinStructureDataset` | ✅ | Uses FASTA + `graph_index` (JSON) + `graph_archive_path` (.pngrph) + `config["LABEL_EMBEDDING_PATH"]` |
| `__getitem__` (atom-level) | ✅ | Returns dict: `atom_coords`, `atom_types`, `atom_to_residue`, `esmc_embeddings`, `edge_index`, `num_residues`, `num_atoms`, `residue_indices`, `sequence_id`, `sequence_str`, `label_multihots`, `label_embeddings`, `label_token_counts` |
| `__getitem__` (legacy) | ✅ | Returns dict: `structure_batch` (PyG Data with `.x`, `.plm`, `.edge_index`, `.edge_s`), `sequence_id`, `label_multihots`, `label_embeddings`, `label_token_counts` |
| `collate_structure_batch` (atom-level) | ✅ | Returns dict: `graph_data` (concatenated atoms with offset `edge_index` + `atom_to_protein` batch tensor), `sequence_ids`, `label_multihots`, `label_embeddings`, `label_token_counts` |
| `collate_structure_batch` (legacy) | ✅ | Returns dict: `structure_batch` (PyG Batch), `sequence_ids`, `label_multihots`, `label_embeddings`, `label_token_counts` |
| `create_structural_loaders` | ✅ | `observation_sampler_factory` + `collate_structure_batch` + `drop_last=(train)`. When `MAX_ATOMS_PER_BATCH` is set, wraps element sampler with `DynamicBatchSampler` for atom-budget-based batching |
| `get_atom_counts` | ✅ | Returns `np.ndarray` of per-sample atom counts from `graph_index[seq_id]["n_atoms"]` |

---

## 3. Model

| Component | Condition | Status |
|-----------|-----------|--------|
| **Protein encoder** | `use_hybrid=True` (default) | `StructuralProteinEncoder` with `use_atom_level=True`. Params: `ESMC_EMBEDDING_DIM`, `PROTEIN_EMBEDDING_DIM`, `EGNN_HIDDEN_DIM`, `EGNN_N_LAYERS` |
| **Protein encoder** | `run.use_sequence_encoder=true` | `ProteInfer` (pretrained or random init) |
| **Label encoder** | Shared | Multilingual E5 (frozen + optional LoRA), cached from `config["LABEL_EMBEDDING_PATH"]` |
| **ProtNote** | Shared | Branches on input type: `graph_data` dict (atom-level), `structure_batch` (legacy PyG), or `sequence_onehots` + `sequence_lengths` (ProteInfer) |

`StructuralProteinEncoder` input expectations:
- **Atom-level** (default): Dict with `esmc_embeddings` [N_atoms, 960], `atom_coords` [N_atoms, 3], `atom_types` [N_atoms], `edge_index` [2, E], `atom_to_protein` [N_atoms], `num_proteins` (int)
- **Legacy PyG**: Batch with `.x` (coords), `.plm` (embeddings), `.edge_index`, `.edge_s` (edge attrs), `.batch`

---

## 4. Trainer

| Feature | Status | Notes |
|---------|--------|-------|
| `_to_device` | ✅ | Moves Tensor, dict (recursively), BatchEncoding, and PyG Batch to device |
| `train_one_epoch` (atom-level) | ✅ | Unpacks `graph_data` dict from batch → moves tensors to device → forward → loss |
| `train_one_epoch` (legacy structural) | ✅ | Unpacks `structure_batch` (PyG Batch) → forward → loss + (optional) interpretability |
| `train_one_epoch` (sequence) | ✅ | Unpacks `sequence_onehots`, `sequence_lengths` → forward → loss |
| `train_one_epoch` (interpretability) | ✅ | Legacy structural path only: when `INTERPRETABILITY_METHOD=IntegratedGradient` and `_site_ground_truth` loaded: `structure_batch.plm.requires_grad_(True)` → `gradient_input_node_attribution_from_logits` → MSE loss added |
| `evaluation_step` | ✅ | Mirrors `train_one_epoch` batch unpacking for all three code paths |
| Gradient accumulation / scaler / clip | ✅ | `autocast()` + `GradScaler`, configurable via `GRADIENT_ACCUMULATION_STEPS` and `CLIP_VALUE` |
| `set_epoch` | ✅ | Checks `batch_sampler.set_epoch()` first (for `DynamicBatchSampler`), then falls back to `sampler.set_epoch()` |

---

## 5. Interpretability

| Item | Status |
|------|--------|
| `INTERPRETABILITY_METHOD` | `IntegratedGradient` or `none` (config `params/default.yaml`; compared case-insensitively) |
| `INTERPRETABILITY_SITE_TSV_PATH` | In `paths/default.yaml`; TSV with `Entry` / `Site` columns (UniProt format) |
| `load_site_ground_truth_tsv` | ✅ Parses `SITE N` entries → dict of Entry → [1-based positions] (`protnote/utils/interpretability.py`) |
| `site_positions_to_target_vector` | ✅ Converts 1-based site positions to normalized per-residue target vector |
| `gradient_input_node_attribution_from_logits` | ✅ Uses `torch.autograd.grad()` with `retain_graph=True`; compatible with single backward |
| `interpretability_loss_mse` | ✅ MSE between node attribution and site-based target; scales by `logits_per_sample` |
| Trainer integration | ✅ Active only in legacy structural mode (`structure_batch.plm`), not yet wired for atom-level `graph_data` path |

---

## 6. Validation / Testing

| Item | Status |
|------|--------|
| Validation loader | Runs every `EPOCHS_PER_VALIDATION` epochs |
| Optimal threshold | `find_optimal_threshold` (on validation set when `DECISION_TH=null`) |
| Test loop | Evaluates for each path in `run.test_paths_names`, collects metrics |
| `eval_metrics` | F1, precision, recall, MAP; respects `label_sample_sizes` |

---

## 7. Checkpoint Paths

| Action | Path | Notes |
|--------|------|-------|
| **Save** | `config["paths"]["OUTPUT_MODEL_DIR"]` → `outputs/checkpoints/` | Saved by `ProtNoteTrainer`: best_val_metric, best_val_loss, last_epoch, periodic |
| **Load** | `{project_root}/data/models/ProtNote/{run.model_file}` | `run.model_file` is a filename; resolved to absolute path at `main.py` L78-79 |

Save and load locations differ. To resume or evaluate: pass `run.model_file=<filename>` (file must be under `data/models/ProtNote/`), or copy checkpoint from `outputs/checkpoints/`.

---

## 8. Configuration Consistency Checklist

- [ ] If using hybrid mode (default), `GRAPH_INDEX_PATH` JSON and `GRAPH_ARCHIVE_PATH` (.pngrph) exist and contain entries matching the FASTA sequences
- [ ] Graph index entries have `{"filename": "...", "n_atoms": N}` format (required for `DynamicBatchSampler`)
- [ ] `ESMC_EMBEDDING_DIM` (960) matches the ESM-C model used in `generate_sequence_embeddings.py`
- [ ] `PROTEIN_EMBEDDING_DIM` (1100) matches ProtNote constructor and `StructuralProteinEncoder` output dimension
- [ ] In structural mode with interpretability, `INTERPRETABILITY_SITE_TSV_PATH` TSV exists with `Entry` / `Site` columns
- [ ] Label side: `LABEL_EMBEDDING_PATH` (generated by `get_setup` from base path + params), annotations pickle, and `VOCABULARIES_DIR` all point to valid data
- [ ] If using `MAX_ATOMS_PER_BATCH`, dataset is atom-level (`use_atom_level=True`) and graph index has `n_atoms`

---

## 9. Recommended Run Order (Hybrid Encoder — Default)

1. **Data preparation**
   FASTA splits, GO/EC annotations, label embeddings (`bin/generate_label_embeddings.py`).

2. **Structure pipeline**
   - Download structures: `bin/download_structures.py` (or configure local AFDB via `LOCAL_AFDB_DIR`).
   - Generate ESM-C embeddings: `bin/generate_sequence_embeddings.py`.
   - Build atom-level graphs: `bin/prepare_graph_data.py` → `graph_index.json` + `graphs.pngrph`.

3. **Interpretability (optional)**
   Place site TSV at `INTERPRETABILITY_SITE_TSV_PATH`, set `params.INTERPRETABILITY_METHOD=IntegratedGradient`.

4. **Training**
   ```bash
   python bin/main.py \
       run.train_path_name=TRAIN_DATA_PATH \
       run.validation_path_name=VAL_DATA_PATH \
       run.test_paths_names='[TEST_DATA_PATH]' \
       run.name=experiment_name
   ```

5. **Evaluation / testing**
   ```bash
   python bin/main.py \
       run.test_paths_names='[TEST_DATA_PATH]' \
       run.model_file=checkpoint_filename.pt \
       run.save_prediction_results=true
   ```

This document is a checklist for verifying the workflow end-to-end.
