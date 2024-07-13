from typing import Iterable, Optional, Union

from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose

from htr_crnn_ctc.datasource import DataSource
from htr_crnn_ctc.types import CharDict, Sample

__all__ = [
    "CTCConcatDataset",
    "CTCDataset"
]

class CTCConcatDataset(ConcatDataset):
    char_dict: CharDict

    def __init__(self, datasets: Iterable["CTCDataset"]) -> None:
        super().__init__(datasets)
        self.char_dict_re_merge()

    def __add__(self, other: Union["CTCDataset", "CTCConcatDataset"]) -> "CTCConcatDataset":
        if isinstance(other, CTCConcatDataset):
            self.datasets.extend(other.datasets)
        else:
            self.datasets.append(other)

        self.cumulative_sizes = self.cumsum(self.datasets)
        self.char_dict_re_merge()

        return self

    def char_dict_re_merge(self) -> None:
        chars: set[str] = set()
        for dataset in self.datasets:
            assert isinstance(dataset, CTCDataset)
            chars.update(dataset.char_dict.keys())

        self.char_dict = {char: idx for idx, char in enumerate(sorted(list(chars)), 1)}

class CTCDataset(Dataset):
    def __init__(self, data_source: DataSource, transform: Optional[Compose] = None,
                       char_dict: Optional[CharDict] = None) -> None:
        super().__init__()

        self.data_source = data_source
        self.transform = transform

        if char_dict is None:
            chars: set[str] = set()
            for text in data_source.text_iter():
                chars.update(text)
            self.char_dict = {char: idx for idx, char in enumerate(sorted(list(chars)), 1)}
        else:
            self.char_dict = char_dict
            self.chars = set(self.char_dict.keys())

    def __getitem__(self, index: int) -> Sample:
        return self.transform(self.data_source[index]) if self.transform else self.data_source[index]

    def __len__(self) -> int:
        return self.data_source.__len__()

    def __add__(self, other: "CTCDataset") -> CTCConcatDataset:
        return CTCConcatDataset((self, other))
