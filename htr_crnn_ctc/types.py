from typing import Union, Dict, Any
from dataclasses import dataclass

from torchvision.transforms import Normalize
from numpy.typing import NDArray
from torch import Tensor

__all__ = [
    "Sample",
    "SampleImage",
    "CharDict",
    "DecodeMap"
]

SampleImage = Union[NDArray[Any], Normalize, Tensor]
CharDict = Dict[str, int]
DecodeMap = Dict[int, str]

@dataclass
class Sample:
    index: int
    image: SampleImage
    text: str
