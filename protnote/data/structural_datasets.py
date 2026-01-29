"""
Structural dataset: same GO/label format as ProteinDataset but returns PyG Data (graph + PLM) per sample.
Expects precomputed graph+PLM pkl: dict[sequence_id] -> (x, plm, edge_index, edge_s).
"""
import pickle
import logging
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except ImportError:
    Data = None
    DataLoader = None

from protnote.utils.data import read_fasta, get_vocab_mappings
from protnote.utils.data import generate_vocabularies
import blosum as bl


class ProteinStructureDataset(Dataset):
    """
    Dataset for structural encoder: each sample is (graph + PLM, labels).
    Labels and vocabulary match ProteinDataset (GO annotations). Graph+PLM loaded from precomputed pkl.
    """

    def __init__(
        self,
        data_paths: dict,
        config: dict,
        logger=None,
        graph_plm_pkl_path: str = None,
    ):
        """
        data_paths: same as ProteinDataset (data_path, dataset_type, annotations_path, vocabularies_dir).
        graph_plm_pkl_path: path to pkl mapping sequence_id -> (x, plm, edge_index, edge_s).
                            Only sequences present in this pkl are kept.
        """
        self.logger = logger or logging.getLogger(__name__)
        required_keys = ["data_path", "dataset_type"]
        for key in required_keys:
            if key not in data_paths:
                raise ValueError(f"Missing required key in data_paths: {key}")
        if graph_plm_pkl_path is None:
            graph_plm_pkl_path = data_paths.get("graph_plm_pkl_path")
        if not graph_plm_pkl_path:
            raise ValueError("ProteinStructureDataset requires graph_plm_pkl_path (in data_paths or argument).")

        self.dataset_type = data_paths["dataset_type"]
        self.data_path = data_paths["data_path"]
        self.graph_plm_pkl_path = graph_plm_pkl_path

        self.augment_residue_probability = config["params"].get("AUGMENT_RESIDUE_PROBABILITY", 0.0)
        self.label_augmentation_descriptions = config["params"].get("LABEL_AUGMENTATION_DESCRIPTIONS", "name+label").split("+")
        self.inference_go_descriptions = config["params"].get("INFERENCE_GO_DESCRIPTIONS", "name+label").split("+")

        blosum62 = bl.BLOSUM(62)
        self.blosum62 = defaultdict(
            dict,
            {aa1: {aa2: blosum62[aa1][aa2] for aa2 in blosum62.keys()} for aa1 in blosum62.keys()},
        )

        # Load graph+PLM pkl and restrict to sequences that have structure
        with open(graph_plm_pkl_path, "rb") as f:
            self.graph_plm_pkl = pickle.load(f)
        self.valid_ids = set(self.graph_plm_pkl.keys())

        # Load FASTA and filter to valid sequence_ids
        raw_data = read_fasta(data_paths["data_path"])
        self.data = [(seq, sid, labels) for (seq, sid, labels) in raw_data if sid in self.valid_ids]
        self.logger.info(
            f"ProteinStructureDataset: {len(self.data)} samples with structure (from {len(raw_data)} in FASTA)."
        )

        subset_fraction = config["params"].get(f"{self.dataset_type.upper()}_SUBSET_FRACTION", 1.0)
        if subset_fraction < 1.0:
            self.data = self.data[: int(subset_fraction * len(self.data))]

        extract_vocabularies_from = config["params"].get("EXTRACT_VOCABULARIES_FROM")
        vocabulary_path = (
            config["paths"][extract_vocabularies_from]
            if extract_vocabularies_from
            else self.data_path
        )
        self._preprocess_data(
            deduplicate=config["params"].get("DEDUPLICATE", True),
            max_sequence_length=config["params"].get("MAX_SEQUENCE_LENGTH"),
            vocabulary_path=vocabulary_path,
        )

        # Label embeddings (same as ProteinDataset)
        INDEX_OUTPUT_PATH = config["LABEL_EMBEDDING_PATH"].split(".")
        INDEX_OUTPUT_PATH = "_".join([INDEX_OUTPUT_PATH[0], "index"]) + "." + INDEX_OUTPUT_PATH[1]
        index_mapping = torch.load(INDEX_OUTPUT_PATH)
        emb_tensor = torch.load(config["LABEL_EMBEDDING_PATH"])
        (
            self.label_embeddings_index,
            self.label_embeddings,
            self.label_token_counts,
            self.label_descriptions,
        ) = self._process_label_embedding_mapping(mapping=index_mapping, embeddings=emb_tensor)
        self.sorted_label_embeddings, self.sorted_label_token_counts = self._sort_label_embeddings()

    def _preprocess_data(self, deduplicate, max_sequence_length, vocabulary_path):
        self.logger.info("Cleaning structural data...")
        df = pd.DataFrame(self.data, columns=["sequence", "sequence_id", "labels"])
        if deduplicate:
            df = df.drop_duplicates(subset="sequence", keep="first")
        if max_sequence_length is not None and self.dataset_type == "train":
            df = df[df["sequence"].apply(len) <= max_sequence_length]
        self.data = list(df.itertuples(index=False, name=None))

        from collections import Counter
        label_freq = Counter()
        for _, _, labels in self.data:
            for lab in labels:
                label_freq[lab] += 1
        self.label_frequency = label_freq

        vocabularies = generate_vocabularies(data=self.data)
        self.amino_acid_vocabulary = vocabularies["amino_acid_vocab"]
        self.label_vocabulary = vocabularies["label_vocab"]
        self.sequence_id_vocabulary = vocabularies["sequence_id_vocab"]
        self.represented_vocabulary_mask = [label in self.label_frequency for label in self.label_vocabulary]
        self._process_amino_acid_vocab()
        self._process_label_vocab()
        self._process_sequence_id_vocab()

    def _process_amino_acid_vocab(self):
        self.aminoacid2int, self.int2aminoacid = get_vocab_mappings(self.amino_acid_vocabulary)

    def _process_label_vocab(self):
        self.label2int, self.int2label = get_vocab_mappings(self.label_vocabulary)

    def _process_sequence_id_vocab(self):
        self.sequence_id2int, self.int2sequence_id = get_vocab_mappings(self.sequence_id_vocabulary)

    def _process_label_embedding_mapping(self, mapping, embeddings):
        import pandas as pd
        if not isinstance(mapping, pd.DataFrame):
            mapping = pd.DataFrame(mapping)
        descriptions_considered = (
            self.label_augmentation_descriptions
            if self.dataset_type == "train"
            else self.inference_go_descriptions
        )
        mask = (mapping["description_type"].isin(descriptions_considered)) & (
            mapping["id"].isin(self.label_vocabulary)
        )
        if hasattr(mask, "values"):
            mask = mask.values
        mapping = mapping.loc[mask].reset_index(drop=True).reset_index()
        embeddings = embeddings[mask]
        token_counts = mapping["token_count"].values
        mapping = (
            mapping.groupby("id")
            .agg(min_idx=("index", "min"), max_idx=("index", "max"))
            .to_dict(orient="index")
        )
        return mapping, embeddings, token_counts, None

    def _sort_label_embeddings(self):
        idx_list = []
        for go_term in self.label_vocabulary:
            idx_list.extend(
                range(
                    self.label_embeddings_index[go_term]["min_idx"],
                    self.label_embeddings_index[go_term]["max_idx"] + 1,
                )
            )
        return (
            self.label_embeddings[idx_list],
            self.label_token_counts[idx_list],
        )

    def __len__(self):
        return len(self.data)

    def calculate_label_weights(self, inv_freq=True, normalize=True, return_list=True, power=0.5):
        """Same interface as ProteinDataset.calculate_label_weights for weighted sampling/loss."""
        import numpy as np
        counts = np.array([self.label_frequency.get(lab, 0) for lab in self.label_vocabulary], dtype=np.float64)
        counts = np.clip(counts, 1e-6, None)
        if inv_freq:
            weights = 1.0 / (counts ** power)
        else:
            weights = counts
        if normalize:
            weights = weights / weights.sum()
        if return_list:
            return torch.tensor(weights, dtype=torch.float32)
        return weights

    def calculate_pos_weight(self):
        """Same interface as ProteinDataset.calculate_pos_weight for BCE."""
        num_pos = sum(self.label_frequency.get(lab, 0) for lab in self.label_vocabulary)
        num_neg = len(self.data) * len(self.label_vocabulary) - num_pos
        return torch.tensor(num_neg / max(num_pos, 1e-6), dtype=torch.float32)

    def __getitem__(self, idx):
        sequence, sequence_id, labels = self.data[idx]
        x, plm, edge_index, edge_s = self.graph_plm_pkl[sequence_id]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if not isinstance(plm, torch.Tensor):
            plm = torch.as_tensor(plm, dtype=torch.float32)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        if not isinstance(edge_s, torch.Tensor):
            edge_s = torch.as_tensor(edge_s, dtype=torch.float32)
        if edge_index.dim() == 1:
            edge_index = edge_index.view(2, -1)

        graph_data = Data(x=x, plm=plm, edge_index=edge_index, edge_s=edge_s)

        labels_ints = [self.label2int[label] for label in labels]
        label_multihots = torch.nn.functional.one_hot(
            torch.tensor(labels_ints, dtype=torch.long),
            num_classes=len(self.label_vocabulary),
        ).sum(dim=0)

        label_embeddings = self.sorted_label_embeddings
        label_token_counts = self.sorted_label_token_counts

        return {
            "structure_batch": graph_data,
            "sequence_id": sequence_id,
            "label_multihots": label_multihots,
            "label_embeddings": label_embeddings,
            "label_token_counts": label_token_counts,
        }


def create_structural_loaders(
    datasets: dict,
    params: dict,
    label_sample_sizes: dict = None,
    shuffle_labels: bool = False,
    in_batch_sampling: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    world_size: int = 1,
    rank: int = 0,
    sequence_weights=None,
):
    """Create DataLoaders for structural encoder; uses collate_structure_batch."""
    from functools import partial
    from collections import defaultdict
    from torch.utils.data import DataLoader
    from protnote.data.samplers import observation_sampler_factory
    from protnote.data.collators import collate_structure_batch

    loaders = defaultdict(list)
    for dataset_type, dataset_list in datasets.items():
        batch_size_for_type = params[f"{dataset_type.upper()}_BATCH_SIZE"]
        label_sample_size = label_sample_sizes.get(dataset_type) if label_sample_sizes else None
        for dataset in dataset_list:
            sequence_sampler = observation_sampler_factory(
                distribute_labels=params.get("DISTRIBUTE_LABELS", False),
                weighted_sampling=params.get("WEIGHTED_SAMPLING", False) and (dataset_type == "train"),
                dataset=dataset,
                world_size=world_size,
                rank=rank,
                sequence_weights=sequence_weights,
                shuffle=(dataset_type == "train"),
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size_for_type,
                shuffle=False,
                collate_fn=partial(
                    collate_structure_batch,
                    label_sample_size=label_sample_size,
                    distribute_labels=params.get("DISTRIBUTE_LABELS", False),
                    shuffle_labels=shuffle_labels,
                    in_batch_sampling=in_batch_sampling and (dataset_type == "train"),
                    world_size=world_size,
                    rank=rank,
                ),
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=(dataset_type == "train"),
                sampler=sequence_sampler,
            )
            loaders[dataset_type].append(loader)
    return loaders
