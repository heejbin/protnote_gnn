"""
Structure graph building: PDB -> Cα coords, edge_index, RBF edge_s.
Ported from StrucToxNet Preprocessing/4_feature_all.py for use in structural data pipeline.
"""
import torch
import numpy as np

try:
    import torch_cluster
except ImportError:
    torch_cluster = None

try:
    from Bio.PDB import PDBParser
except ImportError:
    PDBParser = None


def _normalize(tensor, dim=-1):
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device="cpu"):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf, device=device)
    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))
    return edge_s, edge_v


def extract_ca_coordinates(file_path):
    """Extract Cα coordinates and residue names from a PDB file."""
    if PDBParser is None:
        raise ImportError("Biopython is required for PDB parsing: pip install biopython")
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure("structure", file_path)
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_atom = residue["CA"]
                    resname = residue.get_resname()
                    ca_coords.append([resname, ca_atom.coord[0], ca_atom.coord[1], ca_atom.coord[2]])
    import pandas as pd
    return pd.DataFrame(ca_coords, columns=["resname", "x", "y", "z"])


def pdb_to_graph(
    pdb_path,
    edge_cutoff=8,
    num_rbf=16,
    connection="radius",
    device="cpu",
):
    """
    Build a residue-level graph from a PDB file (Cα atoms).
    Returns: (coords, edge_index, edge_s) as tensors; coords are (N, 3), edge_s is (E, num_rbf).
    """
    if torch_cluster is None:
        raise ImportError("torch_cluster is required: pip install torch-cluster")
    df = extract_ca_coordinates(pdb_path)
    coords = torch.as_tensor(df[["x", "y", "z"]].to_numpy(), dtype=torch.float32, device=device)
    if connection == "knn":
        edge_index = torch_cluster.knn_graph(coords, k=10)
    else:
        edge_index = torch_cluster.radius_graph(coords, edge_cutoff)
    edge_s, _ = _edge_features(coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf, device=device)
    return coords, edge_index, edge_s


def coords_to_graph(coords, edge_cutoff=8, num_rbf=16, connection="radius", device="cpu"):
    """
    Build graph from coordinate array (N, 3).
    Returns: (coords, edge_index, edge_s).
    """
    if torch_cluster is None:
        raise ImportError("torch_cluster is required: pip install torch-cluster")
    if not isinstance(coords, torch.Tensor):
        coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
    if connection == "knn":
        edge_index = torch_cluster.knn_graph(coords, k=10)
    else:
        edge_index = torch_cluster.radius_graph(coords, edge_cutoff)
    edge_s, _ = _edge_features(coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf, device=device)
    return coords, edge_index, edge_s
