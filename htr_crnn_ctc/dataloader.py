from typing import Tuple, Union

from torch import Tensor, device as Device
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torch
import numpy as np

from htr_crnn_ctc.dataset import CTCDataset, CTCConcatDataset
from htr_crnn_ctc.types import Sample

__all__ = [
    "CTCDataLoader"
]

class CTCDataLoader:
    def __init__(self, dataset: Union[CTCDataset, CTCConcatDataset], train_batch_size=16, validation_batch_size=16,
                 validation_split=0.2, shuffle=True, seed=42, device=Device("cpu")) -> None:
        assert isinstance(dataset, (CTCDataset, CTCConcatDataset))
        assert isinstance(train_batch_size, int)
        assert isinstance(validation_batch_size, int)
        assert isinstance(validation_split, float)
        assert isinstance(shuffle, bool)
        assert isinstance(seed, int)
        assert isinstance(device, Device)

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.seed = seed
        self.device = device

    def  __call__(self) -> tuple[DataLoader, DataLoader]:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))

        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(
            self.dataset,
            batch_size=self.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.collate_fn
        )

        validation_loader = DataLoader(
            self.dataset,
            batch_size=self.validation_batch_size,
            sampler=valid_sampler,
            collate_fn=self.collate_fn
        )

        return (train_loader, validation_loader)

    def collate_fn(self, batch: list[Sample]) -> Tuple[Tensor, Tensor, Tensor]:
        images: list[Tensor] = []
        texts: list[str] = []
        lengths: list[int] = []
        sum_length = 0

        for sample in batch:
            assert isinstance(sample.image, Tensor)

            images.append(sample.image)
            texts.append(sample.text)
            length = len(sample.text)
            lengths.append(length)
            sum_length += length

        targets = torch.zeros(sum_length).long()

        for idx, text in enumerate(texts):
            start = sum(lengths[:idx])
            end = lengths[idx]
            targets[start:start + end] = torch.tensor([self.dataset.char_dict[letter] for letter in text]).long()

        return (
            torch.stack(images, 0).to(self.device),
            targets.to(self.device),
            torch.tensor(lengths).to(self.device)
        )
