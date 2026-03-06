import random
from itertools import product
import numpy as np
from torch.utils.data import BatchSampler
from torch.utils.data import Sampler, WeightedRandomSampler
from typing import Optional
import math
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import math
from torch.utils.data import Dataset


class GeneralDistributedSampler(DistributedSampler):

    """
    Class to use distributed sampler with any sampler!
    """

    def __init__(
        self,
        sampler: Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
    ):
        # Same as normal DistributedSampler with shuffle = False
        super().__init__(
            dataset=sampler,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=drop_last,
        )

        assert len(sampler) > num_replicas, "Total samples must be > num replicas"

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch + self.seed)
        indices = list(self.dataset)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class DistributedWeightedSampler(Sampler):
    def __init__(self, weights, world_size=None, rank=None, replacement=True):
        # Get the world size and rank if not provided
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.weights = weights
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement

        # Ensure weights is a tensor
        if not isinstance(self.weights, torch.Tensor):
            self.weights = torch.tensor(self.weights, dtype=torch.double)

        # Determine the number of samples for each GPU, rounding down to ensure it is evenly divisible
        self.num_samples = int(math.floor(len(self.weights) * 1.0 / self.world_size))

        # Determine the total number of samples
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        # Shuffle based on the epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # Create a weighted sample for the entire dataset
        if self.replacement:
            indices = torch.multinomial(
                self.weights, self.total_size, replacement=True, generator=g
            )
        else:
            assert (
                len(self.weights) > self.total_size
            ), "When sampling without replacement, number of samples to draw must be less than the number of elements in the dataset"
            indices = torch.multinomial(
                self.weights, self.total_size, replacement=False, generator=g
            )

        # Subsample for the current process
        indices_for_one_gpu = indices[self.rank : self.total_size : self.world_size]

        # Shuffle each epoch
        indices_for_one_gpu = indices_for_one_gpu[
            torch.randperm(len(indices_for_one_gpu), generator=g)
        ].tolist()

        return iter(indices_for_one_gpu)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _batch_count_for_atom_budget(
    indices, atom_counts, max_atoms_per_batch, drop_last, max_elements_per_batch=None
):
    """Count batches for a given index list (same logic as DynamicBatchSampler.__iter__)."""
    count, batch_atoms, batch_size = 0, 0, 0
    for idx in indices:
        n = atom_counts[idx]
        if batch_atoms > 0 and (
            batch_atoms + n > max_atoms_per_batch
            or (max_elements_per_batch is not None and batch_size >= max_elements_per_batch)
        ):
            count += 1
            batch_atoms = 0
            batch_size = 0
        batch_atoms += n
        batch_size += 1
    if batch_atoms > 0 and not drop_last:
        count += 1
    return max(1, count)


PADDING_SEQUENCE_ID = "__PADDING__"


class DynamicBatchSampler(BatchSampler):
    """Groups indices into variable-size batches targeting a max atom budget.

    A single protein always forms its own batch (never split), even if it
    exceeds the budget. Wraps any existing element sampler (Distributed,
    Weighted, etc.).

    Optional max_elements_per_batch: cap on number of samples per batch (e.g. 8).
    A batch is emitted when either the atom budget or this element cap is reached.
    """

    def __init__(
        self,
        element_sampler,
        atom_counts,
        max_atoms_per_batch,
        drop_last=False,
        world_size=1,
        rank=0,
        max_elements_per_batch=None,
    ):
        # Intentionally skip BatchSampler.__init__ — we manage state ourselves
        self.atom_counts = atom_counts
        self.max_atoms_per_batch = max_atoms_per_batch
        self.drop_last = drop_last
        self.max_elements_per_batch = max_elements_per_batch
        self.element_sampler = element_sampler

    def _batch_count_for_indices(self, indices):
        """Count batches for a given index list (same logic as __iter__)."""
        return _batch_count_for_atom_budget(
            indices,
            self.atom_counts,
            self.max_atoms_per_batch,
            self.drop_last,
            self.max_elements_per_batch,
        )

    def __iter__(self):
        batch, batch_atoms = [], 0
        for idx in self.element_sampler:
            n = self.atom_counts[idx]
            if batch and (
                batch_atoms + n > self.max_atoms_per_batch
                or (
                    self.max_elements_per_batch is not None
                    and len(batch) >= self.max_elements_per_batch
                )
            ):
                yield batch
                batch, batch_atoms = [], 0
            batch.append(idx)
            batch_atoms += n
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        # Exact count so DataLoader/ranks stay in sync (avoids one rank waiting at dist.reduce)
        indices = list(self.element_sampler)
        return self._batch_count_for_indices(indices)

    def set_epoch(self, epoch):
        if hasattr(self.element_sampler, "set_epoch"):
            self.element_sampler.set_epoch(epoch)


class PaddedDynamicBatchSampler(BatchSampler):
    """
    Wrap a DynamicBatchSampler so that all ranks see the same number of batches.

    - local_batches: len(base_batch_sampler) on this rank
    - global_batches: max(local_batches) across ranks (via dist.all_reduce at init / set_epoch)
    - For ranks with fewer real batches, we emit extra "padding" batches whose indices
      are >= len(dataset), which triggers _get_padding_sample() in the Dataset.
    """

    def __init__(self, base_batch_sampler, dataset_len: int, world_size: int = 1, rank: int = 0):
        # Base sampler is something like DynamicBatchSampler
        self.base_batch_sampler = base_batch_sampler
        self.dataset_len = int(dataset_len)
        self.world_size = world_size
        self.rank = rank
        self._sync_batch_counts()

    def _sync_batch_counts(self):
        # Local number of batches on this rank
        self.local_batches = len(self.base_batch_sampler)
        self.global_batches = self.local_batches

        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            # Use current CUDA device for NCCL; assumes torch.cuda.set_device(rank) has been called.
            if torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                # Fallback for non-CUDA backends (unlikely in this project)
                device = torch.device("cpu")
            t = torch.tensor(self.local_batches, device=device, dtype=torch.int64)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            self.global_batches = int(t.item())

    def __len__(self):
        return self.global_batches

    def __iter__(self):
        # Yield real batches first
        count = 0
        for batch in self.base_batch_sampler:
            yield batch
            count += 1

        # Then pad up to global_batches with dummy batches that force padding samples
        pad_batches = self.global_batches - count
        if pad_batches <= 0:
            return

        # Each padding batch is a single index >= dataset_len; Dataset.__getitem__
        # will return _get_padding_sample(), and collate_* will mark is_padding=True.
        padding_index = self.dataset_len
        for _ in range(pad_batches):
            yield [padding_index]

    def set_epoch(self, epoch):
        if hasattr(self.base_batch_sampler, "set_epoch"):
            self.base_batch_sampler.set_epoch(epoch)
        # After shuffling, batch counts might change; resync
        self._sync_batch_counts()


class GridBatchSampler(BatchSampler):
    def __init__(
        self,
        observation_sampler,
        observations_batch_size,
        drop_last_observation_batch,
        num_labels,
        labels_batch_size,
        shuffle_grid=True,
    ):
        self.observation_sampler = observation_sampler
        self.observations_batch_size = observations_batch_size
        self.drop_last_observation_batch = drop_last_observation_batch

        self.num_labels = num_labels
        self.labels_batch_size = labels_batch_size
        self.shuffle_grid = shuffle_grid
        self.labels_idxs = list(range(num_labels))
        self.calculate_num_batches()

    def __iter__(self):
        random.shuffle(self.labels_idxs)
        print("Getting label batches...")
        observation_batches = self.get_observation_batches()
        print("Done...")

        print("Getting observation batches...")
        label_batches = self.get_label_batches()
        print("Done...")

        print("Getting combinations...")
        obs_labels_batch_combinations = list(
            product(observation_batches, label_batches)
        )

        print("Done...")

        if self.shuffle_grid:
            print("Shuffling...")
            random.shuffle(obs_labels_batch_combinations)
        print("Done...")
        for observation_batch, label_batch in obs_labels_batch_combinations:
            yield list(
                product(observation_batch, [label_batch])
            )  # [observation_batch,label_batch]

    def calculate_num_batches(self):
        num_label_batches = np.ceil(self.num_labels / self.labels_batch_size)
        num_observation_batches = (
            np.ceil(len(self.observation_sampler) / self.observations_batch_size)
            if not self.drop_last_observation_batch
            else len(self.observation_sampler) // self.observations_batch_size
        )
        print("Done...")

        self.total_num_batches = int(num_label_batches * num_observation_batches)
        print(
            f"num label batches = {num_label_batches}, num observation batches = {num_observation_batches}"
        )
        print(f"total batches = {self.total_num_batches}")

    def __len__(self):
        return self.total_num_batches

    def get_label_batches(self):
        # n_chunks = int(np.ceil(self.num_labels/self.labels_batch_size))
        return [
            self.labels_idxs[i : i + self.labels_batch_size]
            for i in range(0, self.num_labels, self.labels_batch_size)
        ]

    def get_observation_batches(self):
        batches = []

        if self.drop_last_observation_batch:
            observation_sampler_iter = iter(self.observation_sampler)
            while True:
                try:
                    batch = [
                        next(observation_sampler_iter)
                        for _ in range(self.observations_batch_size)
                    ]
                    batches.append(batch)
                except StopIteration:
                    break
        else:
            batch = [0] * self.observations_batch_size
            idx_in_batch = 0
            for idx in self.observation_sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.observations_batch_size:
                    batches.append(batch)
                    idx_in_batch = 0
                    batch = [0] * self.observations_batch_size
            if idx_in_batch > 0:
                batches.append(batch[:idx_in_batch])
        return batches


def observation_sampler_factory(
    distribute_labels: bool,
    weighted_sampling: bool,
    shuffle: bool,
    dataset: Dataset = None,
    world_size: int = 1,
    rank: int = 0,
    sequence_weights: torch.Tensor = None,
):
    if distribute_labels and not weighted_sampling:
        print("WARNING: No Sampler used for distribute labels")
        sampler = None
    elif not distribute_labels and world_size == 1 and weighted_sampling:
        # If NOT distributing labels, and not training on multiple GPU's, create a non-distributed weighted sampler with replacement
        assert sequence_weights is not None, "Weighted RandomSampler requires weights"

        sampler = WeightedRandomSampler(
            sequence_weights, len(sequence_weights), replacement=True
        )
    elif not distribute_labels and world_size > 1 and weighted_sampling:
        # If distributing sequences across multiple GPUs with a weighted sampler, create custom DistributedWeightedSampler
        sampler = DistributedWeightedSampler(
            sequence_weights,
            world_size=world_size,
            rank=rank,
            replacement=True,
        )
    elif not distribute_labels and not weighted_sampling:
        # If simply distributing sequences across GPU's without weighted sampling, use a distributed sampler

        assert dataset is not None, "DistributeSampler requires dataset"

        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
    else:
        # Raise error
        raise ValueError(
            "Invalid combination of WEIGHTED_SAMPLING, WORLD_SIZE, and DISTRIBUTE_LABELS parameters"
        )

    return sampler
