from os import listdir
from itertools import chain
from typing import Optional

from torch import Tensor
from numpy.typing import NDArray

import numpy as np

from htr_crnn_ctc.types import Sample, SampleImage, CharDict, DecodeMap

__all__ = [
    "copy_sample",
    "get_decode_map",
    "get_char_dict",
    "get_best_leven_value_model",
    "simulate_english_line_from_word"
]

def copy_sample(sample: Sample) -> Sample:
    image: SampleImage

    if isinstance(sample.image, np.ndarray):
        image = sample.image.copy()
    elif isinstance(sample.image, Tensor):
        image = sample.image.detach().clone()
    else:
        raise TypeError("only np.ndarray or Tensor")

    return Sample(
        index=sample.index,
        image=image,
        text=sample.text
    )

def get_decode_map(char_dict: CharDict) -> DecodeMap:
    return {idx: char for char, idx in char_dict.items()}

def get_best_leven_value_model(path: Optional[str] = None) -> str:
    l = [f for f in listdir(path) if f.endswith("_model.pth")]
    assert len(l) != 0

    best = (10e9, "")
    for file in l:
        leven = float(".".join(file.split("_", 2)[:2]))
        if leven < best[0]:
            best = (leven, file)

    return best[1]

def get_char_dict(data_source: object, add: Optional[list[str]] = None) -> CharDict:
    assert hasattr(data_source, "text_iter")
    chars: set[str] = set()
    for text in chain(data_source.text_iter(), add if add else []): # type: ignore
        chars.update(text)
    return {char: idx for idx, char in enumerate(sorted(list(chars)), 1)}

def get_center_of_word(image: NDArray, threshold_color: Optional[int] = None, threshold_value=0.5) -> int:
    image = image.copy()

    if threshold_color:
        image[image > threshold_color] = 255

    image = 255 - image
    sum_y_axis: NDArray = image.sum(axis=1, dtype=np.int32)
    high_value_indices = np.where(sum_y_axis > int(sum_y_axis.max() * threshold_value))[0]

    if len(high_value_indices) > 0:
        middle_index = high_value_indices[len(high_value_indices) // 2]
    else:
        middle_index = image.shape[0] // 2

    return middle_index

def simulate_english_line_from_word(samples: list[Sample], word_separate_size: tuple[int, int],
                                    center_word_threshold_color: Optional[int] = 200,
                                    center_word_threshold_value: float = 0.5) -> Sample:
    centers = [get_center_of_word(sample.image, center_word_threshold_color, center_word_threshold_value) for sample in samples] # type: ignore
    text = " ".join([sample.text for sample in samples])
    seqs: list[int] = [0] + np.random.randint(*word_separate_size, size=len(samples)).tolist()
    top = max(centers)
    bottom = max(sample.image.shape[0] - center for sample, center in zip(samples, centers))
    height = top + bottom
    width = sum(sample.image.shape[1] for sample in samples) + sum(seqs)
    canvas = np.ones((height, width), dtype=np.uint8) * 255
    x = 0
    for sample, x_seq, center in zip(samples, seqs, centers):
        x += x_seq
        y = top - center
        canvas[y:y + sample.image.shape[0], x:(x := x + sample.image.shape[1])] = sample.image

    return Sample(
        index=0,
        image=canvas,
        text=text
    )
