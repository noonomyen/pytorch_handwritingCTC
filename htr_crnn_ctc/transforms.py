from typing import Literal, Optional, Tuple, Dict, Any

from skimage import transform, color
from torchvision.transforms import Normalize as TorchNormalize
from deslant_img import deslant_img

import torch
import numpy as np

from htr_crnn_ctc.types import Sample

__all__ = [
    "Rescale",
    "Deslant",
    "ToRGB",
    "ToGray",
    "ToTensor",
    "Normalise"
]

class Rescale(object):
    def __init__(self, output_size: Tuple[int, int], random_pad=False, border_pad: Tuple[int, int] = (0, 0),
                 random_rotation = 0.0, random_stretch=1.0, fill_space=False, fill_threshold=200) -> None:
        assert isinstance(output_size, tuple)
        assert isinstance(random_pad, bool)
        assert isinstance(border_pad, tuple)
        assert isinstance(random_rotation, (float, int))
        assert isinstance(random_stretch, float)
        assert isinstance(fill_space, bool)
        assert isinstance(fill_threshold, int) and 0 <= fill_threshold < 255

        self.output_size = output_size
        self.random_pad = random_pad
        self.border_pad = border_pad
        self.rotation = random_rotation
        self.random_stretch = random_stretch
        self.fill_space = fill_space
        self.fill_threshold = fill_threshold

    def __call__(self, sample: Sample):
        if self.fill_space:
            assert isinstance(sample.image, np.ndarray)
            sample.image[sample.image < self.fill_threshold] = 255

        if self.border_pad[0] > 0 or self.border_pad[1] > 0:
            resize = (self.output_size[0] - self.border_pad[0], self.output_size[1] - self.border_pad[1])
        else:
            resize = self.output_size

        h, w = sample.image.shape[:2]
        fx = w / resize[1]
        fy = h / resize[0]

        f = max(fx, fy)

        new_size = (max(min(resize[0], int(h / f)), 1), max(min(resize[1], int(w / f * self.random_stretch)), 1))

        sample.image = transform.resize(sample.image, new_size, preserve_range=True, mode="constant", cval=255)
        if self.rotation != 0:
            rot = np.random.choice(np.arange(-self.rotation, self.rotation), 1)[0]
            sample.image = transform.rotate(sample.image, rot, mode="constant", cval=255, preserve_range=True)

        canvas = np.ones(self.output_size, dtype=np.uint8) * 255

        if self.random_pad:
            v_pad_max = self.output_size[0] - new_size[0]
            h_pad_max = self.output_size[1] - new_size[1]

            v_pad = int(np.random.choice(np.arange(0, v_pad_max + 1), 1)[0])
            h_pad = int(np.random.choice(np.arange(0, h_pad_max + 1), 1)[0])

            canvas[v_pad:v_pad + new_size[0], h_pad:h_pad + new_size[1]] = sample.image
        else:
            canvas[0:new_size[0], 0:new_size[1]] = sample.image

        sample.image = transform.rotate(canvas, -90, resize=True)[:, :-1]

        return sample

class Deslant:
    def __init__(self, optim_algo: Optional[Literal["grid", "powell"]] = None,
                       lower_bound: Optional[float] = None,
                       upper_bound: Optional[float] = None,
                       num_steps: Optional[int] = None,
                       bg_color: Optional[int] = None) -> None:

        self.kwargs: Dict[str, Any] = {}

        if optim_algo:
            self.kwargs["optim_algo"] = optim_algo
        if lower_bound:
            self.kwargs["lower_bound"] = lower_bound
        if upper_bound:
            self.kwargs["upper_bound"] = upper_bound
        if num_steps:
            self.kwargs["num_steps"] = num_steps
        if bg_color:
            self.kwargs["bg_color"] = bg_color

    def __call__(self, sample: Sample) -> Sample:
        assert isinstance(sample.image, np.ndarray)
        sample.image = deslant_img(sample.image, **self.kwargs).img

        return sample

class ToRGB:
    def __call__(self, sample: Sample) -> Sample:
        sample.image = color.gray2rgb(sample.image)

        return sample

class ToGray:
    def __call__(self, sample: Sample) -> Sample:
        sample.image = color.rgb2gray(sample.image)

        return sample

class ToTensor:
    def __init__(self, rgb=True):
        assert isinstance(rgb, bool)
        self.rgb = rgb

    def __call__(self, sample: Sample):
        assert isinstance(sample.image, np.ndarray)

        if self.rgb:
            sample.image = torch.from_numpy(sample.image.transpose((2, 0, 1))).float()
        else:
            sample.image = torch.from_numpy(sample.image)[None, :, :].float()

        return sample

class Normalise:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std = torch.tensor(std, dtype=torch.float)
        self.norm = TorchNormalize(mean, std)

    def __call__(self, sample: Sample) -> Sample:
        sample.image = self.norm(sample.image)
        return sample
