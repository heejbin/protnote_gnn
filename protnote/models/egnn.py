"""
E(3) Equivariant GNN for structural protein encoding.
Ported from StrucToxNet (Jiao et al.) for use as ProtNote's protein branch.
"""
import torch
from torch import nn

try:
    import torch_scatter
except ImportError:
    torch_scatter = None


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL(nn.Module):
    """E(3) Equivariant Graph Convolutional Layer."""

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
    ):
        super(E_GCL, self).__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        # edge_mlp input: [source, target, radial, edge_attr] -> 2*hidden_nf + 1 + edges_in_d
        edge_in_dim = 2 * hidden_nf + 1 + (edges_in_d if edges_in_d else 0)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        layer = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = out.to(torch.float32)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index[0], edge_index[1]
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        agg = agg.to(torch.float32)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index[0], edge_index[1]
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise ValueError(f"Wrong coords_agg parameter: {self.coords_agg}")
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index[0], edge_index[1]
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[edge_index[0]], h[edge_index[1]], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, _ = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class StructuralProteinEncoder(nn.Module):
    """
    Structural-Sequential encoder: PLM node features + EGNN -> graph-level embedding.
    Interface compatible with ProtNote's sequence_encoder: get_embeddings(batch) -> (N, protein_embedding_dim).
    """

    def __init__(
        self,
        plm_embedding_dim=1069,
        protein_embedding_dim=1100,
        hidden_nf=256,
        num_layers=3,
        in_edge_nf=16,
        dropout=0.5,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
    ):
        super().__init__()
        if torch_scatter is None:
            raise ImportError("StructuralProteinEncoder requires torch_scatter. Install with: pip install torch_scatter")
        self.plm_embedding_dim = plm_embedding_dim
        self.protein_embedding_dim = protein_embedding_dim
        self.hidden_nf = hidden_nf
        self.num_layers = num_layers

        self.W_v = nn.Linear(plm_embedding_dim, hidden_nf, bias=True)
        self.gcl_layers = nn.ModuleList([
            E_GCL(
                hidden_nf,
                hidden_nf,
                hidden_nf,
                edges_in_d=in_edge_nf,
                act_fn=act_fn,
                residual=residual,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
            )
            for _ in range(num_layers)
        ])
        # Map pooled hidden_nf -> protein_embedding_dim (e.g. 1100 for ProtNote W_p)
        self.projection = nn.Sequential(
            nn.Linear(hidden_nf, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, protein_embedding_dim),
        )

    def forward(self, batch):
        """
        batch: PyG Batch with .x (coords), .plm (node PLM embeddings), .edge_index, .edge_s, .batch
        """
        x = batch.x
        h = batch.plm
        h = self.W_v(h)
        for gcl in self.gcl_layers:
            h, x, _ = gcl(h, batch.edge_index, x, edge_attr=batch.edge_s)
        out = torch_scatter.scatter_max(h, batch.batch, dim=0)[0].float()
        return self.projection(out)

    def get_embeddings(self, batch):
        """
        Same interface as ProteInfer.get_embeddings for drop-in replacement in ProtNote.
        batch: PyG Batch (structure_batch) with .x, .plm, .edge_index, .edge_s, .batch
        Returns: (batch_size, protein_embedding_dim)
        """
        return self.forward(batch)
