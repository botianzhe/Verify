"""Image and mask I/O shared by model inference and standalone forgery."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def load_image(image, *, size=256, device="cpu", dtype=torch.float32, input_range="minus_one_one"):
    """Load RGB image files/PIL or explicitly normalized in-memory tensors.

    Tensors explicitly default to [-1,1], even for all-positive bright images.
    For [0,1] tensors pass input_range='zero_one'. PIL/files use [0,1] pixels.
    """
    if input_range not in {'minus_one_one', 'zero_one'}:
        raise ValueError("input_range must be minus_one_one or zero_one")
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.suffix.lower() in {'.pt','.npz'}:
            raise ValueError('Use image files or in-memory tensors, not PT/NPZ data')
        with Image.open(path) as opened:
            image = opened.convert("RGB")
    if isinstance(image, Image.Image):
        image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        input_range = 'zero_one'
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a path, PIL image, or torch.Tensor")
    x = image.float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[0] == 0 or x.shape[1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {tuple(x.shape)}")
    if not torch.isfinite(x).all() or x.min() < -1 or x.max() > 1:
        raise ValueError("Image tensor must contain finite RGB values in [0, 1] or [-1, 1]")
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, (size, size), mode="bilinear", align_corners=False)
    if input_range == 'zero_one':
        if x.min() < 0:
            raise ValueError('zero_one tensor contains negative values')
        x = x * 2 - 1
    return x.to(device=device, dtype=dtype)


def load_mask(path, *, size=256):
    """Load a binary foreground mask; white means forged, black preserved."""
    if Path(path).suffix.lower() in {'.pt','.npz'}:
        raise ValueError('Mask must be an image file, not PT/NPZ data')
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
    return torch.from_numpy((np.asarray(image).copy() > 127).astype(np.float32))[None, None]


def image_to_pil(tensor):
    values = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()[0]
    return Image.fromarray(values.permute(1, 2, 0).numpy())


def mask_to_pil(mask, *, binary=False):
    values = mask.detach().float().cpu().clamp(0, 1)[0, 0]
    if binary:
        values = (values > .5).float()
    return Image.fromarray((values * 255).round().byte().numpy())


def save_image(tensor, path):
    path = Path(path)
    if path.suffix.lower() != '.png':
        raise ValueError('Image output must be PNG')
    path.parent.mkdir(parents=True, exist_ok=True)
    image_to_pil(tensor).save(path)
