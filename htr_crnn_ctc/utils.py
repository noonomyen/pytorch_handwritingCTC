from os import listdir
from typing import Optional

from torch import Tensor

import numpy as np

from htr_crnn_ctc.types import Sample, SampleImage, CharDict, DecodeMap

__all__ = [
    "copy_sample",
    "get_decode_map",
    "get_best_leven_value_model"
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
