import torch
from typing import List, Tuple

try:
    from torch_geometric.data import Batch as PyGBatch
except ImportError:
    PyGBatch = None


def collate_structure_batch(
    batch: List[dict],
    label_sample_size=None,
    distribute_labels=False,
    shuffle_labels=False,
    in_batch_sampling=False,
    grid_sampler=False,
    world_size=1,
    rank=0,
):
    """
    Collate for structural encoder: list of dicts with structure_batch (PyG Data), label_multihots, etc.
    Returns dict with structure_batch (PyG Batch), sequence_ids, label_multihots, label_embeddings, label_token_counts.
    """
    if PyGBatch is None:
        raise ImportError("collate_structure_batch requires torch_geometric: pip install torch-geometric")
    graphs = [item["structure_batch"] for item in batch]
    structure_batch = PyGBatch.from_data_list(graphs)
    sequence_ids = [item["sequence_id"] for item in batch]
    label_multihots = torch.stack([item["label_multihots"] for item in batch])
    label_embeddings = batch[0]["label_embeddings"]
    label_token_counts = batch[0]["label_token_counts"]

    if label_sample_size and not grid_sampler and not in_batch_sampling:
        num_labels = label_multihots.shape[1]
        if distribute_labels:
            labels_per_partition = num_labels // world_size
            start_idx = rank * labels_per_partition
            end_idx = start_idx + labels_per_partition
            sampled_label_indices = torch.arange(start_idx, end_idx)[: label_sample_size // world_size]
        else:
            sampled_label_indices = (
                torch.randperm(num_labels)[:label_sample_size]
                if shuffle_labels
                else torch.arange(min(label_sample_size, num_labels))
            )
        label_embeddings = label_embeddings[sampled_label_indices]
        label_multihots = label_multihots[:, sampled_label_indices]
        label_token_counts = label_token_counts[sampled_label_indices]
    elif in_batch_sampling:
        sampled_label_indices = torch.where(label_multihots.sum(dim=0) > 0)[0]
        label_embeddings = label_embeddings[sampled_label_indices]
        label_multihots = label_multihots[:, sampled_label_indices]
        label_token_counts = label_token_counts[sampled_label_indices]

    return {
        "structure_batch": structure_batch,
        "sequence_ids": sequence_ids,
        "label_multihots": label_multihots,
        "label_embeddings": label_embeddings,
        "label_token_counts": label_token_counts,
    }


def collate_variable_sequence_length(
    batch: List[Tuple],
    label_sample_size=None,
    distribute_labels=False,
    shuffle_labels=False,
    in_batch_sampling=False,
    grid_sampler=False,
    return_label_multihots=True,
    world_size=1,
    rank=0,
):
    """
    Collates a batch of data with variable sequence lengths. Pads sequences to the maximum length within the batch to handle the variable
    lengths.

    Args:
        batch (List[Tuple]): A list of tuples, where each tuple represents one data point.
                             Each tuple contains a dictionary with keys like 'sequence_onehots',
                             'sequence_length', 'label_multihots', etc.
        label_sample_size (int, optional): The number of labels to sample for training.
                                           Used with grid_sampler or in_batch_sampling.
        distribute_labels (bool, optional): Whether to distribute labels across different GPUs.
        return_label_multihots (bool, optional): Whether to batched multihot labels.
        shuffle_labels (bool, optional): Whether to shuffle labels during sampling.
        in_batch_sampling (bool, optional): If True, samples labels that are present within the batch.
        grid_sampler (bool, optional): If True, uses a grid sampling strategy for labels.
        world_size (int, optional): The total number of distributed processes or GPUs.
        rank (int, optional): The rank of the current process in distributed training.

    Returns:
        Dict: A dictionary containing the processed batch data. Keys include:
              - 'sequence_onehots': Tensor, padded one-hot encoded sequences.
              - 'sequence_ids': List, sequence IDs.
              - 'sequence_lengths': Tensor, lengths of sequences.
              - 'label_multihots': Tensor, multihot encoded labels (possibly sampled).
              - 'label_embeddings': Tensor, label embeddings if provided. Otherwise None.
              - 'label_token_counts': Tensor, token counts for each label.

    """

    # Determine the maximum sequence length in the batch
    max_length = max(item["sequence_length"] for item in batch)

    # Initialize lists to store the processed values
    processed_sequence_onehots = []
    processed_sequence_ids = []
    processed_sequence_lengths = []
    processed_label_multihots = []
    processed_label_token_counts = []
    processed_label_embeddings = None

    if grid_sampler:
        assert (
            label_sample_size is not None
        ), "Must provide label_sample_size if using grid sampler"
        assert not in_batch_sampling, "Can't use in batch sampling with grid sampler"

    else:
        assert not (
            in_batch_sampling and (label_sample_size is not None)
        ), "Cant use both in_batch_sampling with lable_sample_size"

    sampled_label_indices = None
    num_labels = batch[0]["label_multihots"].shape[0]

    if label_sample_size:
        if grid_sampler:
            sampled_label_indices = batch[0]["label_idxs"]
        else:
            if not distribute_labels:
                # If not distributing labels, sample from entire dataset
                sampled_label_indices = (
                    torch.randperm(num_labels)[:label_sample_size]
                    if shuffle_labels
                    else torch.arange(label_sample_size)
                )
            else:
                # Otherwise, sample from the labels on this GPU
                labels_per_partition = num_labels // world_size
                start_idx = rank * labels_per_partition
                end_idx = start_idx + labels_per_partition
                partition_indices = torch.arange(start_idx, end_idx)
                sampled_label_indices = partition_indices[
                    torch.randperm(len(partition_indices))[
                        : label_sample_size // world_size
                    ]
                ]
                # if rank < 2:
                #     print("GPU {}. Sampling range: {} to {}. Sampled {} labels".format(rank, start_idx, end_idx, sampled_label_indices[:10]))

    elif in_batch_sampling:
        sampled_label_indices = torch.where(
            sum(i["label_multihots"] for i in batch) > 0
        )[0]

    # Apply the sampled labels to the label embeddings
    # We only use the first sequence in the batch to get the label embeddings to minimize complexity
    label_embeddings = batch[0]["label_embeddings"]

    # Likewise, we only use the first sequence in the batch to get the label token counts
    label_token_counts = batch[0]["label_token_counts"]

    if sampled_label_indices is not None:
        # Create a new tensor of embeddings with only the sampled labels
        processed_label_embeddings = label_embeddings[sampled_label_indices]
    # Otherwise, use the original label embeddings
    else:
        processed_label_embeddings = label_embeddings

    # Loop through the batch
    for row in batch:
        # Get the sequence onehots, sequence length, sequence id, and label multihots
        sequence_onehots = row["sequence_onehots"]
        sequence_id = row["sequence_id"]
        sequence_length = row["sequence_length"]
        label_multihots = row["label_multihots"]

        # Set padding
        padding_length = max_length - sequence_length

        # Get the sequence dimension (e.g., 20 for amino acids)
        sequence_dim = sequence_onehots.shape[0]

        # Pad the sequence to the max_length and append to the processed_sequences list
        processed_sequence_onehots.append(
            torch.cat(
                (sequence_onehots, torch.zeros((sequence_dim, padding_length))), dim=1
            )
        )

        # Use the sampled labels for each element in the batch.
        if sampled_label_indices is not None:
            label_multihots = label_multihots[sampled_label_indices]

        # Append the other values to the processed lists
        processed_sequence_ids.append(sequence_id)
        processed_sequence_lengths.append(sequence_length)
        processed_label_multihots.append(label_multihots)

    processed_batch = {
        "sequence_onehots": torch.stack(processed_sequence_onehots),
        "sequence_ids": processed_sequence_ids,
        "sequence_lengths": torch.stack(processed_sequence_lengths),
        "label_embeddings": processed_label_embeddings,
        "label_token_counts": label_token_counts,
    }

    if return_label_multihots:
        processed_batch["label_multihots"] = torch.stack(processed_label_multihots)

    return processed_batch
