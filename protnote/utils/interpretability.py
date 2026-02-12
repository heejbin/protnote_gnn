"""
Interpretability: Integrated Gradients and site-based ground truth for structural encoder.
Phase 5 of proposal: align IG attribution with SwissProt Site annotations (MSE loss).
Alternative: maximize attribution on functional residues (from Swiss-Prot .dat).
"""
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import torch

try:
    import torch_scatter
except ImportError:
    torch_scatter = None

# Feature keys that denote functional residue sites (UniProt FT table)
DEFAULT_SITE_FEATURE_KEYS = frozenset(
    {"SITE", "ACT_SITE", "BINDING", "METAL", "DISULFID", "LIPID", "CARBOHYD", "MOD_RES"}
)


def load_site_from_swissprot_dat(
    dat_path: str,
    feature_keys: Optional[Set[str]] = None,
    cache_path: Optional[str] = None,
) -> Dict[str, List[int]]:
    """
    Load functional residue positions from UniProt/Swiss-Prot .dat file.

    Extracts FT (feature) lines for SITE, ACT_SITE, BINDING, METAL, etc.
    Returns: dict[Entry] -> list of 1-based residue positions (unique, sorted).

    feature_keys: Set of FT keys to include (default: SITE, ACT_SITE, BINDING, METAL, ...).
    cache_path: If set, save/load pickled dict for faster reuse.
    """
    from Bio import SwissProt

    if feature_keys is None:
        feature_keys = DEFAULT_SITE_FEATURE_KEYS

    if cache_path and os.path.isfile(cache_path):
        import pickle
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    out: Dict[str, List[int]] = {}
    dat_path = os.path.expanduser(dat_path)
    if not os.path.isfile(dat_path):
        return out

    with open(dat_path, "r") as f:
        for record in SwissProt.parse(f):
            acc = record.accessions[0] if record.accessions else None
            if not acc:
                continue
            positions: set = set()
            for feat in getattr(record, "features", []) or []:
                # Support both tuple (key, from, to, desc) and FeatureTable objects
                try:
                    if hasattr(feat, "type"):
                        key = (feat.type or "").strip().upper()
                        if key not in feature_keys:
                            continue
                        loc = getattr(feat, "location", None)
                        if loc is None:
                            continue
                        start = getattr(loc, "start", None)
                        end = getattr(loc, "end", None)
                        if start is not None and isinstance(start, int):
                            start_1based = start + 1
                            if end is not None and isinstance(end, int):
                                for p in range(start_1based, end + 1):
                                    positions.add(p)
                            else:
                                positions.add(start_1based)
                        elif end is not None and isinstance(end, int):
                            positions.add(end + 1)
                    else:
                        if len(feat) < 3:
                            continue
                        key = (feat[0] or "").strip().upper()
                        if key not in feature_keys:
                            continue
                        start, end = feat[1], feat[2]
                        if isinstance(start, int) and isinstance(end, int):
                            for p in range(start, end + 1):
                                positions.add(p)
                        elif isinstance(start, int):
                            positions.add(start)
                        elif isinstance(end, int):
                            positions.add(end)
                except (TypeError, ValueError, AttributeError):
                    continue
            out[acc] = sorted(positions)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        import pickle
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)

    return out


def load_site_ground_truth_tsv(tsv_path: str) -> Dict[str, List[int]]:
    """
    Load UniProt TSV with Entry and Site columns.
    Site column format: "SITE 29; /note=...; SITE 30; ..." (1-based residue positions).
    Returns: dict[Entry] -> list of 1-based site positions (unique, sorted).
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    if "Entry" not in df.columns or "Site" not in df.columns:
        raise ValueError(f"TSV must have 'Entry' and 'Site' columns. Found: {list(df.columns)}")
    out = {}
    for _, row in df.iterrows():
        entry = row.get("Entry", "")
        site_str = row.get("Site", "")
        if pd.isna(site_str) or not str(site_str).strip():
            out[entry] = []
            continue
        # Match "SITE 29" or "SITE 96" etc (1-based positions)
        positions = list(set(int(m) for m in re.findall(r"SITE\s+(\d+)", str(site_str), re.IGNORECASE)))
        out[entry] = sorted(positions)
    return out


def site_positions_to_target_vector(
    num_residues: int,
    site_positions_1based: List[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    logit_value: Optional[float] = None,
) -> torch.Tensor:
    """
    Ground truth: each site residue gets (1/|sites|) * logit_value (or 1/|sites| if logit_value is None), others 0.
    site_positions_1based: 1-based residue indices from UniProt (e.g. [29, 30, 31]).
    logit_value: if given, target at sites = (1/|sites|)*logit_value so vector sums to logit_value; else sum = 1.0.
    Returns: vector of shape (num_residues,).
    """
    v = torch.zeros(num_residues, device=device, dtype=dtype)
    if not site_positions_1based:
        if logit_value is None:
            v += 1.0 / max(num_residues, 1)
        return v
    n_sites = len(site_positions_1based)
    scale = (float(logit_value) / n_sites) if logit_value is not None else (1.0 / n_sites)
    for pos_1 in site_positions_1based:
        i = pos_1 - 1  # 0-based
        if 0 <= i < num_residues:
            v[i] = scale
    if logit_value is None and v.sum() < 1e-8:
        v += 1.0 / max(num_residues, 1)
    elif logit_value is None and v.sum() > 1e-8:
        v = v / v.sum()
    return v


def integrated_gradients_node_attribution(
    model: torch.nn.Module,
    structure_batch,
    inputs_for_forward: dict,
    n_steps: int = 50,
    baseline_plm: Optional[torch.Tensor] = None,
    target_logit_selector: str = "sum",  # "sum" = sum over labels for each sample
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute Integrated Gradients attribution of the scalar output (per-sample logit sum)
    w.r.t. node features (batch.plm). Returns node-level attribution, shape (num_nodes,).
    We sum over feature dim to get per-node importance, then normalize per graph.
    Calls backward() in the loop but zero_grad after each step so model params are clean for the main backward.
    """
    if device is None:
        device = next(model.parameters()).device
    plm = structure_batch.plm.detach()
    if baseline_plm is None:
        baseline_plm = torch.zeros_like(plm, device=plm.device, dtype=plm.dtype)
    baseline_plm = baseline_plm.to(device)
    plm = plm.to(device)
    attribution = torch.zeros_like(plm, device=device, dtype=plm.dtype)
    for k in range(n_steps):
        t = (k + 0.5) / n_steps
        plm_t = (baseline_plm + t * (plm - baseline_plm)).detach().requires_grad_(True)
        batch_t = structure_batch.clone()
        batch_t.plm = plm_t
        batch_t = batch_t.to(device)
        inp = {**inputs_for_forward, "structure_batch": batch_t}
        logits, _ = model(**inp)
        scalar = logits.sum()
        model.zero_grad(set_to_none=True)
        scalar.backward()
        if plm_t.grad is not None:
            attribution = attribution + (plm - baseline_plm) * plm_t.grad.detach() / n_steps
        model.zero_grad(set_to_none=True)
    # Per-node importance: sum over feature dim
    node_attr = attribution.sum(dim=-1)
    # Normalize per graph so each graph's attribution sums to 1
    if torch_scatter is not None and hasattr(structure_batch, "batch"):
        batch_idx = structure_batch.batch.to(device)
        sums = torch_scatter.scatter_add(node_attr, batch_idx, dim=0)
        sums = sums[batch_idx].clamp(min=1e-8)
        node_attr = node_attr / sums
    else:
        s = node_attr.sum()
        if s > 1e-8:
            node_attr = node_attr / s
    return node_attr


def gradient_input_atom_attribution_from_logits(
    logits: torch.Tensor,
    node_features: torch.Tensor,
    atom_to_protein: torch.Tensor,
    device: torch.device = None,
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Differentiable attribution for atom-level graph: (x - baseline) * grad(logit sum, x).
    node_features: [N_atoms, feat_dim] with requires_grad=True (e.g. esmc_embeddings).
    atom_to_protein: [N_atoms] batch index per atom.
    Returns per-atom attribution (num_atoms,) summing to 1 per protein (normalized).
    """
    if device is None:
        device = logits.device
    x = node_features
    if baseline is None:
        baseline = torch.zeros_like(x, device=x.device, dtype=x.dtype)
    scalar = logits.sum()
    (grad_x,) = torch.autograd.grad(scalar, x, retain_graph=True, allow_unused=True)
    if grad_x is None:
        attr = torch.zeros(x.shape[0], device=device, dtype=x.dtype)
    else:
        attr = (x - baseline).detach() * grad_x
    attr = attr.sum(dim=-1)
    if torch_scatter is not None:
        batch_idx = atom_to_protein.to(device)
        sums = torch_scatter.scatter_add(attr, batch_idx, dim=0)
        sums = sums[batch_idx].clamp(min=1e-8)
        attr = attr / sums
    else:
        s = attr.sum()
        if s > 1e-8:
            attr = attr / s
    return attr


def gradient_input_node_attribution_from_logits(
    logits: torch.Tensor,
    structure_batch,
    baseline_plm: Optional[torch.Tensor] = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Differentiable attribution from already-computed logits: (plm - baseline) * grad(logit sum, plm).
    Uses autograd.grad with retain_graph so the same graph can be used for main_loss.backward().
    structure_batch.plm must have requires_grad=True when logits were computed.
    Returns normalized per-node importance (num_nodes,) summing to 1 per graph.
    """
    if device is None:
        device = logits.device
    plm = structure_batch.plm
    if baseline_plm is None:
        baseline_plm = torch.zeros_like(plm, device=plm.device, dtype=plm.dtype)
    baseline_plm = baseline_plm.to(device)
    plm = plm.to(device)
    scalar = logits.sum()
    (grad_plm,) = torch.autograd.grad(scalar, plm, retain_graph=True, allow_unused=True)
    if grad_plm is None:
        attribution = torch.zeros_like(plm, device=device, dtype=plm.dtype)
    else:
        attribution = (plm - baseline_plm).detach() * grad_plm
    node_attr = attribution.sum(dim=-1)
    if torch_scatter is not None and hasattr(structure_batch, "batch"):
        batch_idx = structure_batch.batch.to(device)
        sums = torch_scatter.scatter_add(node_attr, batch_idx, dim=0)
        sums = sums[batch_idx].clamp(min=1e-8)
        node_attr = node_attr / sums
    else:
        s = node_attr.sum()
        if s > 1e-8:
            node_attr = node_attr / s
    return node_attr


def interpretability_loss_mse(
    node_attribution: torch.Tensor,
    batch_idx: torch.Tensor,
    sequence_ids: List[str],
    site_dict: Dict[str, List[int]],
    device: torch.device,
    logits_per_sample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    """
    MSE between node attribution and site-based ground truth.
    node_attribution: (num_nodes,) per-node importance summing to 1 per graph (normalized).
    batch_idx: (num_nodes,) graph index for each node.
    sequence_ids: list of length batch_size (Entry / sequence id per graph).
    site_dict: Entry -> list of 1-based site positions.
    logits_per_sample: (batch_size,) logit value per graph (e.g. logits.sum(dim=1)).
        If given, ground truth at sites = (1/|sites|)*logit so target sums to logit per graph,
        and we scale node_attr by logit so both have same scale before MSE.
    Returns: (mse_loss, count of graphs with non-empty site used).
    """
    if torch_scatter is None:
        return torch.tensor(0.0, device=device), 0
    unique_batch = batch_idx.unique(sorted=True)
    losses = []
    count = 0
    for b in unique_batch:
        bi = int(b)
        mask = batch_idx == b
        node_attr_g = node_attribution[mask]
        L = node_attr_g.shape[0]
        sid = sequence_ids[bi] if bi < len(sequence_ids) else None
        if sid is None:
            continue
        sites = site_dict.get(sid, [])
        if not sites:
            continue
        logit_g = None
        if logits_per_sample is not None and bi < logits_per_sample.shape[0]:
            logit_g = logits_per_sample[bi].item()
            if logit_g <= 0:
                continue
        target = site_positions_to_target_vector(L, sites, device=device, logit_value=logit_g)
        if logits_per_sample is not None and logit_g is not None:
            node_attr_scaled = node_attr_g * logit_g
        else:
            node_attr_scaled = node_attr_g
        count += 1
        losses.append(((node_attr_scaled - target) ** 2).mean())
    if not losses:
        return torch.tensor(0.0, device=device), 0
    return torch.stack(losses).mean(), count


def interpretability_loss_maximize_site_attribution(
    atom_attribution: torch.Tensor,
    atom_to_protein: torch.Tensor,
    atom_to_residue: torch.Tensor,
    sequence_ids: List[str],
    site_dict: Dict[str, List[int]],
    device: torch.device,
    normalize_per_protein: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Maximize IG attribution on atoms belonging to functional site residues.

    Loss = -mean(sum of attribution on site atoms per protein).
    Minimizing this loss maximizes the attribution on functional residues.

    atom_attribution: (num_atoms,) per-atom importance (e.g. from gradient*input).
    atom_to_protein: (num_atoms,) batch index for each atom.
    atom_to_residue: (num_atoms,) 0-based residue index for each atom.
    sequence_ids: list of Entry IDs per protein in batch.
    site_dict: Entry -> list of 1-based residue positions from Swiss-Prot.
    normalize_per_protein: If True, divide by total attr per protein before summing (focus on fraction on sites).
    Returns: (loss, count of proteins with non-empty sites used).
    """
    if torch_scatter is None:
        return torch.tensor(0.0, device=device), 0

    unique_batch = atom_to_protein.unique(sorted=True)
    losses = []
    count = 0

    for b in unique_batch:
        bi = int(b)
        mask = atom_to_protein == b
        attr_g = atom_attribution[mask]
        res_g = atom_to_residue[mask]  # 0-based

        sid = sequence_ids[bi] if bi < len(sequence_ids) else None
        if sid is None:
            continue
        sites_1based = site_dict.get(sid, [])
        if not sites_1based:
            continue

        # 1-based -> 0-based
        sites_0based = [p - 1 for p in sites_1based if p >= 1]
        if not sites_0based:
            continue
        sites_t = torch.tensor(sites_0based, device=res_g.device, dtype=res_g.dtype)
        site_atom_mask = (res_g.unsqueeze(1) == sites_t).any(dim=1)
        if not site_atom_mask.any():
            continue

        sum_on_sites = attr_g[site_atom_mask].sum()
        if normalize_per_protein:
            total = attr_g.sum().clamp(min=1e-8)
            term = sum_on_sites / total
        else:
            term = sum_on_sites

        # Maximize term -> minimize -term
        losses.append(-term)
        count += 1

    if not losses:
        return torch.tensor(0.0, device=device), 0
    return torch.stack(losses).mean(), count
