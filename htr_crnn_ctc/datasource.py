from random import shuffle

from io import BytesIO
from os import path
from typing import Generator, Iterable, Optional, Literal, Tuple, Dict, Any

import pickle

from pandas import read_parquet, read_csv
from skimage import io

from htr_crnn_ctc.utils import copy_sample, simulate_english_line_from_word
from htr_crnn_ctc.types import Sample, SampleImage

__all__ = [
    "DataSource",
    "IAM_CSVDataSource",
    "ParquetDataSource",
    "InMemoryDataSource",
    "HotSwapDataSource",
    "SimulateLineDataSource"
]

class DataSource:
    def __getitem__ (self, index: int) -> Sample: ...
    def __setitem__ (self, index: int, value: Sample) -> None: ...
    def __iter__(self) -> Iterable[Sample]: ...
    def __len__ (self) -> int: ...
    def __del__(self) -> None: ...

    def image(self, index: int) -> SampleImage: ...
    def text(self, index: int) -> str: ...

    def image_iter(self) -> Iterable[SampleImage]: ...
    def text_iter(self) -> Iterable[str]: ...

class IAM_CSVDataSource(DataSource):
    def __init__(self, file: str, root_path: str, map_columns: Optional[Dict[Literal["file", "text"], str]] = None) -> None:
        self.file = file
        self.root_path = root_path
        self.file_path = path.join(self.root_path, file)
        self.map_columns = map_columns
        self.columns = ["file", "text"]

        if map_columns is not None:
            for idx, key in enumerate(self.columns):
                if key in map_columns:
                    self.columns[idx] = map_columns[key]

        self.data_frame = read_csv(filepath_or_buffer=self.file_path, usecols=self.columns)

    def __getitem__(self, index: int) -> Sample:
        image, text = self.data_frame.iloc[index]
        return Sample(
            index=index,
            image=io.imread(self.get_image_relative_path(image)),
            text=text
        )

    def __setitem__(self, index: int, value: Sample) -> None:
        raise NotImplementedError("")

    def __iter__(self) -> Generator[Sample, Any, None]:
        for idx, image, text in self.data_frame.itertuples(index=True):
            yield Sample(
                index=idx,
                image=io.imread(self.get_image_relative_path(image)),
                text=text
            )

    def __len__(self) -> int:
        return self.data_frame.__len__()

    def image(self, index: int) -> SampleImage:
        return io.imread(self.get_image_relative_path(self.data_frame.iloc[index].iloc[0]))

    def text(self, index: int) -> str:
        return self.data_frame.iloc[index].iloc[1]

    def image_iter(self) -> Iterable[SampleImage]:
        return iter(self.data_frame.iloc[:, 0])

    def text_iter(self) -> Iterable[str]:
        return iter(self.data_frame.iloc[:, 1])

    def index_filename(self, index: int) -> str:
        return self.data_frame.iloc[index].iloc[0]

    def get_image_relative_path(self, name: str) -> str:
        name_split = name.split("-")
        return path.join(self.root_path, name_split[0], "-".join(name_split[:2]), name)

    @staticmethod
    def get_image_dirname(name: str) -> str:
        name_split = name.split("-")
        return path.join(name_split[0], "-".join(name_split[:2]))

class ParquetDataSource(DataSource):
    def __init__(self, file: str, map_columns: Optional[Dict[Literal["image", "text"], str]] = None) -> None:
        self.file = file
        self.map_columns = map_columns
        self.columns = ["image", "text"]

        if map_columns is not None:
            for idx, key in enumerate(self.columns):
                if key in map_columns:
                    self.columns[idx] = map_columns[key]

        self.data_frame = read_parquet(path=self.file, columns=self.columns)

    def __getitem__(self, index: int) -> Sample:
        image, text = self.data_frame.iloc[index]
        return Sample(
            index=index,
            image=io.imread(BytesIO(image["bytes"])),
            text=text
        )

    def __setitem__(self, index: int, value: Sample) -> None:
        raise NotImplementedError("")

    def __iter__(self) -> Generator[Sample, Any, None]:
        for idx, image, text in self.data_frame.itertuples(index=True):
            yield Sample(
                index=idx,
                image=io.imread(BytesIO(image["bytes"])),
                text=text
            )

    def __len__(self) -> int:
        return self.data_frame.__len__()

    def image(self, index: int) -> SampleImage:
        return io.imread(BytesIO(self.data_frame.iloc[index].iloc[0]["bytes"]))

    def text(self, index: int) -> str:
        return self.data_frame.iloc[index].iloc[1]

    def image_iter(self) -> Iterable[SampleImage]:
        return iter(self.data_frame.iloc[:, 0])

    def text_iter(self) -> Iterable[str]:
        return iter(self.data_frame.iloc[:, 1])

class InMemoryDataSource(DataSource):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data: list[Sample] = []

    def __getitem__(self, index: int) -> Sample:
        return copy_sample(self.data[index])

    def __setitem__(self, index: int, value: Sample) -> None:
        self.data[index] = value

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return self.data.__len__()

    def from_datasource(self, datasource: DataSource) -> None:
        self.data = list(iter(datasource))

    def image(self, index: int) -> SampleImage:
        return self.data[index].image

    def text(self, index: int) -> str:
        return self.data[index].text

    def image_iter(self) -> Iterable[SampleImage]:
        return (obj.image for obj in self.data)

    def text_iter(self) -> Iterable[str]:
        return (obj.text for obj in self.data)

    def load(self) -> None:
        with open(self.file_path, "rb") as file:
            self.data = pickle.load(file=file)

    def dump(self) -> None:
        with open(self.file_path, "wb") as file:
            pickle.dump(self.data, file=file, protocol=pickle.HIGHEST_PROTOCOL)

class SimulateLineDataSource(DataSource):
    def __init__(self, datasource: DataSource, words: Tuple[int, int], word_separate_size: Tuple[int, int], onetime = False) -> None:
        self.onetime = onetime
        self.data: list[Sample] = []

        le = len(datasource)
        li = [x for x in range(le)]
        i = 0
        c = 0
        shuffle(li)
        while i < le:
            for j in range(words[0], words[1] + 1):
                b = False
                data = []
                for _ in range(j):
                    if i < le:
                        data.append(datasource[li[i]])
                        i += 1
                    else:
                        b = True
                        break

                if len(data) != 0:
                    _ = simulate_english_line_from_word(data, word_separate_size)
                    _.index = c
                    self.data.append(_)
                    c += 1

                if b:
                    break

            print(f"\r{i}/{le}", end="")

        print()

    def __getitem__(self, index: int) -> Sample:
        return self.data[index] if self.onetime else copy_sample(self.data[index])

    def __setitem__(self, index: int, value: Sample) -> None:
        raise NotImplementedError("")

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return self.data.__len__()

    def image(self, index: int) -> SampleImage:
        return self.data[index].image

    def text(self, index: int) -> str:
        return self.data[index].text

    def image_iter(self) -> Iterable[SampleImage]:
        return (obj.image for obj in self.data)

    def text_iter(self) -> Iterable[str]:
        return (obj.text for obj in self.data)

class HotSwapDataSource(DataSource):
    def __init__(self, datasource: DataSource) -> None:
        super().__init__()
        self.datasource = datasource

    def __getitem__(self, index: int) -> Sample:
        return self.datasource.__getitem__(index)

    def __setitem__(self, index: int, value: Sample) -> None:
        return self.datasource.__setitem__(index, value)

    def __iter__(self):
        return self.datasource.__iter__()

    def __len__(self) -> int:
        return self.datasource.__len__()

    def image(self, index: int) -> SampleImage:
        return self.datasource.image(index)

    def text(self, index: int) -> str:
        return self.datasource.text(index)

    def image_iter(self) -> Iterable[SampleImage]:
        return self.datasource.image_iter()

    def text_iter(self) -> Iterable[str]:
        return self.datasource.text_iter()

    def swap(self, datasource: DataSource) -> None:
        self.datasource = datasource
