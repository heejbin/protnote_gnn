# ProtNote-GNN Workflow Checklist

## 1. Entry Point and Configuration

| Step | Location | Status | Notes |
|------|----------|--------|-------|
| CLI | `bin/main.py` | ✅ | `--train-path-name`, `--validation-path-name`, `--test-paths-names`, `--model-file`, `--override`, etc. |
| Config load | `get_setup(config_path=project_root/'configs'/'base_config.yaml', ...)` | ✅ | `configs/base_config.yaml` + overrides |
| Path resolution | `configs.py`: `paths` = flat paths from DATA_PATH / output_paths | ✅ | `paths["STRUCTURE_GRAPH_PKL"]`, `paths["INTERPRETABILITY_SITE_TSV_PATH"]`, etc. |
| **model_file None** | `main.py` L219 | ✅ Fixed | When `--model-file` is not set, skip reassigning `args.model_file` (previously: Path/None caused an error) |

---

## 2. Data Pipeline

### 2.1 Mode Branch: `USE_STRUCTURAL_ENCODER`

- **False**: `ProteinDataset` + `create_multiple_loaders` → `sequence_onehots`, `sequence_lengths`, `label_multihots`, `label_embeddings`, `label_token_counts`
- **True**: `ProteinStructureDataset` + `create_structural_loaders` → `structure_batch` (PyG Batch), `sequence_ids`, `label_multihots`, `label_embeddings`, `label_token_counts`

### 2.2 Prerequisites for Structural Mode

| Step | Script / Config | Description |
|------|-----------------|--------------|
| 1) Structures | `paths["PDB_DIR"]` | PDB/mmCIF files present locally or downloaded via `bin/fetch_alphafold_structures.py` |
| 2) Mapping | `paths["STRUCTURE_INDEX_PATH"]` | `bin/fetch_structure_mapping.py` → sequence_id → local PDB path JSON |
| 3) Graph + PLM | `paths["STRUCTURE_GRAPH_PKL"]` | `bin/make_structural_dataset.py` → sequence_id → (x, plm, edge_index, edge_s) pkl |

When using structural mode, **you must** prepare data in the order above.

### 2.3 Dataset / Collator

| Item | Status | Notes |
|------|--------|-------|
| `ProteinStructureDataset` | ✅ | Uses FASTA + `graph_plm_pkl_path`, `config["LABEL_EMBEDDING_PATH"]`, vocab, etc. |
| `__getitem__` | ✅ | Returns `structure_batch` (PyG Data), `sequence_id`, `label_multihots`, `label_embeddings`, `label_token_counts` |
| `collate_structure_batch` | ✅ | PyG Batch + `sequence_ids` list, label tensors |
| `create_structural_loaders` | ✅ | `observation_sampler_factory`, `collate_structure_batch`, `drop_last=(train)` |

---

## 3. Model

| Component | Condition | Status |
|-----------|-----------|--------|
| **Protein encoder** | `USE_STRUCTURAL_ENCODER=True` | `StructuralProteinEncoder` (EGNN), uses `structural_encoder_params` |
| **Protein encoder** | `USE_STRUCTURAL_ENCODER=False` | `ProteInfer` (pretrained or random) |
| **Label encoder** | Shared | E5/BioGPT etc., cache from `LABEL_EMBEDDING_PATH` |
| **ProtNote** | Shared | Branches on `structure_batch` or `(sequence_onehots, sequence_lengths)`; noising with `label_token_counts` |

`StructuralProteinEncoder` expects a PyG Batch with `.x`, `.plm`, `.edge_index`, `.edge_s`, `.batch`.

---

## 4. Trainer

| Feature | Status | Notes |
|---------|--------|-------|
| `_to_device` | ✅ | Moves Tensor, BatchEncoding, PyG Batch, etc. to device |
| `train_one_epoch` (structural) | ✅ | Unpacks `structure_batch`, `sequence_ids`, `label_*` → forward → loss + (optional) interpretability |
| `train_one_epoch` (interpretability) | ✅ | When `INTERPRETABILITY_METHOD=IntegratedGradient` and `_site_ground_truth` present: `plm.requires_grad_(True)` → gradient-input attribution → add MSE loss |
| `evaluation_step` (structural) | ✅ | `structure_batch`, `sequence_ids` → forward → loss |
| Gradient accumulation / scaler / clip | ✅ | Behaves according to config |

---

## 5. Interpretability (Phase 5)

| Item | Status |
|------|--------|
| `INTERPRETABILITY_METHOD` | none / IntegratedGradient (config; compared case-insensitively) |
| `INTERPRETABILITY_SITE_TSV_PATH` | Included in `paths`; TSV with Entry/Site columns |
| `load_site_ground_truth_tsv` | ✅ Parses SITE N → Entry → [1-based positions] |
| Site ground-truth vector | ✅ (1/|sites|)*logit_value on sites when `logits_per_sample` is passed |
| `gradient_input_node_attribution_from_logits` | ✅ Uses retain_graph; compatible with a single backward |
| `interpretability_loss_mse` | ✅ When `logits_per_sample` is used, scales then MSE |
| Trainer usage | ✅ Add interpretability term to loss only in structural mode + IntegratedGradient + TSV loaded |

---

## 6. Validation / Testing

| Item | Status |
|------|--------|
| Validation loader | Runs every `EPOCHS_PER_VALIDATION` |
| Optimal threshold | `find_optimal_threshold` (on validation) |
| Test loop | Evaluates for `test_paths_names`, collects metrics |
| `eval_metrics` | F1 etc., respects `label_sample_sizes` |

---

## 7. Checkpoint Paths (Important)

| Action | Path | Notes |
|--------|------|-------|
| **Save** | `config["paths"]["OUTPUT_MODEL_DIR"]` → `{OUTPUT_PATH}/checkpoints/` | e.g. `outputs/checkpoints/` |
| **Load** | `os.path.join(config["DATA_PATH"], args.model_file)` | `args.model_file` is already set to `project_root/data/models/ProtNote/<filename>` |

So **save location** and **default load location** differ.

- New training → checkpoints are saved under `outputs/checkpoints/`.
- To resume or evaluate: either place the checkpoint under `data/models/ProtNote/` and pass `--model-file <filename>`, or change the code so `args.model_file` uses the output-directory path.

---

## 8. Configuration Consistency Checklist

- [ ] If `USE_STRUCTURAL_ENCODER=True`, the `STRUCTURE_GRAPH_PKL` pkl exists and its sequence_ids match the FASTA.
- [ ] In structural mode with interpretability, the `INTERPRETABILITY_SITE_TSV_PATH` TSV exists and its Entry/Site column format matches the code.
- [ ] `structural_encoder_params.PLM_MODEL` (ProtT5 / ESM-C) matches `PLM_EMBEDDING_DIM` (or null for auto).
- [ ] `PROTEIN_EMBEDDING_DIM` (1100) matches ProtNote and EGNN output dimension.
- [ ] Label side: `LABEL_EMBEDDING_PATH`, `annotations_path`, and `vocabularies_dir` all point to valid data paths.

---

## 9. Recommended Run Order (Structural Mode)

1. **Data preparation**  
   FASTA, GO annotations, label embedding generation, etc. (same as original ProtNote).

2. **Structure pipeline**  
   - Obtain PDB/mmCIF (locally or via `fetch_alphafold_structures.py`).  
   - Run `fetch_structure_mapping.py` → `STRUCTURE_INDEX_PATH`.  
   - Run `make_structural_dataset.py` → `STRUCTURE_GRAPH_PKL`.

3. **Interpretability (optional)**  
   Prepare the Site TSV, set `INTERPRETABILITY_SITE_TSV_PATH`, and set `INTERPRETABILITY_METHOD: IntegratedGradient` (or similar).

4. **Training**  
   `python bin/main.py --train-path-name TRAIN_DATA_PATH --validation-path-name VAL_DATA_PATH ...`

5. **Evaluation / testing**  
   Use `--test-paths-names TEST_DATA_PATH`; if needed, `--model-file <checkpoint path or filename under data/models/ProtNote/>`.

This document is a checklist for verifying the workflow end-to-end.
