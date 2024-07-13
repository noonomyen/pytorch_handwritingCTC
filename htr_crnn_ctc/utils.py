from torch import Tensor

import numpy as np

from htr_crnn_ctc.types import Sample, SampleImage, CharDict, DecodeMap

__all__ = [
    "copy_sample",
    "get_decode_map"
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
