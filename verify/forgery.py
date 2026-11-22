"""Training-compatible random broad-stroke masks and evaluation blending.

Ported from faker/maskgen.py, faker/generate.py and models/attacklayer2.py
in the 2026-09-22T11-23-02 training snapshot. All images use the native
256x256 canvas. No checkpoint, VAE, landmarks or training modules are needed.
"""

import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch
from torchvision.transforms import functional as TF


def _broad_stroke_mask(rng):
    """Match gen_randmask's Python RNG draw order and raster operations."""
    size = 256
    regions = []
    for _ in range(rng.randint(1, 3)):
        area = rng.uniform(.05, .25) * size * size
        radius = int(math.sqrt(area / math.pi))
        for _ in range(100):
            cx, cy = rng.randint(radius, size - radius), rng.randint(radius, size - radius)
            if all(math.sqrt((cx - x) ** 2 + (cy - y) ** 2) >= radius + r + size * .15
                   for x, y, r in regions):
                break
        regions.append((cx, cy, radius))

    mask = np.zeros((size, size), dtype=np.float32)
    for cx, cy, radius in regions:
        canvas = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(canvas)
        previous_angle = 0.0
        for stroke in range(rng.randint(1, 2)):
            width = max(15, min(60, rng.randint(int(radius * .4), int(radius * .8))))
            length = rng.randint(int(radius * 1.2), int(radius * 2))
            angle = rng.uniform(0, 2 * math.pi)
            distance = rng.uniform(0, radius * .2)
            sx, sy = cx + distance * math.cos(angle), cy + distance * math.sin(angle)
            angle = (rng.uniform(0, 2 * math.pi) if stroke == 0 else
                     previous_angle + math.pi / 2 + rng.uniform(-math.pi / 4, math.pi / 4))
            ex, ey = sx + length * math.cos(angle), sy + length * math.sin(angle)
            sx, sy, ex, ey = [max(0, min(size - 1, v)) for v in (sx, sy, ex, ey)]
            previous_angle = angle
            segments = max(8, int(length / 20))
            points = []
            for i in range(segments + 1):
                t = i / segments
                x, y = sx + t * (ex - sx), sy + t * (ey - sy)
                if 0 < t < 1:
                    x += math.sin(t * math.pi) * rng.uniform(-.3 * 8, .3 * 8)
                    y += math.sin(t * math.pi) * rng.uniform(-.3 * 8, .3 * 8)
                points.append((x, y))
            for start, end in zip(points, points[1:]):
                draw.line([start, end], fill=255, width=int(width * (1 + rng.uniform(-.15, .15))))
            for x, y in (points[0], points[-1]):
                r = width * .6
                draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        region = np.asarray(canvas, dtype=np.float32) / 255
        kernel = np.ones((max(5, int(width * .15)),) * 2, dtype=np.uint8)
        mask = np.maximum(mask, cv2.dilate(region, kernel, iterations=1))
    mask = (ndimage.gaussian_filter(mask, sigma=1.2) > .4).astype(np.uint8) * 255
    return Image.fromarray(mask)


def generate_random_mask(seed=2025):
    """Return binary [1,1,256,256], including the dataset's blur/threshold.

    Uses local Python/Torch generators, leaving the caller's RNGs unchanged.
    This is the training dataset's random-mask branch; landmark masks are
    intentionally not synthesized from arbitrary face geometry.
    """
    mask = _broad_stroke_mask(random.Random(seed))
    generator = torch.Generator().manual_seed(seed)
    sigma = torch.empty(1).uniform_(.1, 2., generator=generator).item()
    blurred = TF.gaussian_blur(mask, [5, 5], [sigma, sigma])
    return (TF.to_tensor(blurred)[:1] > .5).float().unsqueeze(0)


def _color_transfer(target, source, mask):
    indices = np.where(mask != 0)
    result = source.copy()
    target_pixels = target[indices].astype(np.int32)
    source_pixels = source[indices].astype(np.int32)
    shifted = source_pixels - source_pixels.mean(axis=0) + target_pixels.mean(axis=0)
    result[indices] = shifted.clip(0, 255)
    return result


def _feather_blend(source, target, mask):
    indices = np.where(mask != 0)
    points = np.column_stack((indices[1], indices[0]))
    extent = np.max(np.max(points, axis=0) - np.min(points, axis=0))
    alpha = np.zeros(mask.shape, dtype=np.float32)
    # The original divides by zero for a single pixel. Its hull has no
    # interior; zero feather weight is the finite limiting result.
    if extent == 0:
        return target.copy(), alpha
    hull = cv2.convexHull(points)
    distances = np.array([cv2.pointPolygonTest(hull, (int(x), int(y)), True) for x, y in points])
    weights = np.clip(distances / (.2 * extent), 0, 1) * .8
    result = target.copy()
    result[indices] = weights[:, None] * source[indices] + (1 - weights[:, None]) * target[indices]
    alpha[indices] = weights
    return result.clip(0, 255), alpha


def _validate_image(image, name):
    if not isinstance(image, torch.Tensor) or image.shape != (1, 3, 256, 256):
        raise ValueError(f"{name} must be a normalized tensor with shape [1,3,256,256]")
    if not torch.isfinite(image).all() or image.min() < -1 or image.max() > 1:
        raise ValueError(f"{name} must contain finite values in [-1,1]")


@torch.inference_mode()
def blend_forgery(carrier, source=None, *, mask=None, seed=2025, method="random"):
    """Forge an embedded carrier using the training evaluation stage=3.

    Inputs are single normalized RGB tensors in [-1,1]. The random method
    chooses feather/hard blending equally after masked mean color transfer.
    As in evaluation, a further 20% branch quantizes to uint8 before the
    current main script's mask-only composition restores the float background.
    Stage 3's 256-to-256 resize is identity. Training-only stage 1
    rotation/crop augmentation is not part of this inference attack.

    Returns image, nominal binary mask, actual blending alpha, augmented
    target and metadata. Outputs follow the carrier device in FP32.
    """
    _validate_image(carrier, "carrier")
    self_source = source is None
    if self_source:
        source = carrier.flip(-1)
    _validate_image(source, "source")
    if method not in {"random", "hard", "feather"}:
        raise ValueError("method must be random, hard or feather")
    generated = mask is None
    if generated:
        mask = generate_random_mask(seed)
    if not isinstance(mask, torch.Tensor) or mask.shape != (1, 1, 256, 256):
        raise ValueError("mask must have shape [1,1,256,256]")
    if not torch.isfinite(mask).all() or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("mask must be binary (0 or 1)")
    rng = random.Random(seed)
    # Preserve the original evaluation attack's initial unused draws.
    rng.randint(1, 2)
    rng.randint(128, 256)
    rng.randint(128, 256)
    target = (carrier.detach().float().cpu() / 2 + .5).permute(0, 2, 3, 1).numpy()[0] * 255
    source_array = (source.detach().float().cpu() / 2 + .5).permute(0, 2, 3, 1).numpy()[0] * 255
    binary_mask = mask.detach().float().cpu().numpy()[0, 0]
    quantized = False
    if not binary_mask.any():
        blend, alpha, selected = target.copy(), binary_mask.copy(), "empty"
    else:
        transferred = _color_transfer(target, source_array, binary_mask)
        draw = rng.random()
        selected = ("feather" if draw > .5 else "hard") if method == "random" else method
        if selected == "feather":
            blend, alpha = _feather_blend(transferred, target, binary_mask)
        else:
            alpha = binary_mask.copy()
            blend = target * (1 - alpha[..., None]) + transferred * alpha[..., None]
        quantized = rng.random() > .8
        if quantized:
            blend, target = blend.astype(np.uint8), target.astype(np.uint8)

    def normalized(array):
        tensor = torch.from_numpy(np.array(array)).permute(2, 0, 1).unsqueeze(0) / 255
        return (tensor.clamp(0, 1) - .5).mul(2).to(carrier.device)

    nominal_mask = mask.detach().to(device=carrier.device, dtype=torch.float32)
    # Current SDSteganography.forward constrains even the quantization branch
    # to the provided mask. Preserve the original float background exactly.
    image = normalized(blend) * nominal_mask + carrier * (1 - nominal_mask)
    return {
        "image": image,
        "target": normalized(target),
        "mask": mask.detach().to(device=carrier.device, dtype=torch.float32),
        "alpha": torch.from_numpy(alpha.copy())[None, None].to(carrier.device),
        "metadata": {
            "attack": "blending", "stage": 3, "seed": seed,
            "source": "self_horizontal_flip" if self_source else "provided",
            "background_preserved": True,
            "method": selected, "quantized": quantized,
            "mask_source": "random_broad_strokes" if generated else "provided",
            "mask_fraction": float(binary_mask.mean()),
        },
    }
