"""Embed, blend, extract, localize and recover a single image in memory."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from .assets import DEFAULT_CHECKPOINT
from .cli import _bits
from .forgery import blend_forgery
from .image_io import load_image, load_mask, mask_to_pil, save_image
from .pipeline import VerifyPipeline


def psnr(image, target):
    mse = float(((image - target) / 2).square().mean())
    return -10 * math.log10(mse) if mse > 0 else None


def roi_psnr(image, target, mask):
    count = float(mask.sum()) * image.shape[1]
    if count == 0:
        return None
    mse = float((((image - target) / 2).square() * mask).sum()) / count
    return -10 * math.log10(mse) if mse > 0 else None


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(prog='verify-workflow', description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='donor; default: flipped original')
    parser.add_argument('--mask', type=Path, help='white pixels select the tampered area')
    parser.add_argument('--watermark', type=_bits, help='100 binary bits; default: seeded random bits')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-dir', type=Path, help='new output directory; default: logs/workflow_timestamp')
    args = parser.parse_args()
    if any(path and path.suffix.lower() in {'.pt', '.npz'} for path in (args.original, args.source, args.mask)):
        parser.error('Use image inputs, not PT/NPZ data')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    directory = args.output_dir or Path(__file__).resolve().parents[1]/'logs'/f'workflow_{stamp}'
    if directory.exists():
        parser.error(f'Output directory already exists: {directory}; choose a new path')
    torch.set_num_threads(4)
    pipe = VerifyPipeline(args.checkpoint, device=args.device)
    original = load_image(args.original, device=pipe.device)
    source = load_image(args.source, device=pipe.device) if args.source else original.flip(-1)
    bits = args.watermark
    if bits is None:
        bits = torch.randint(0, 2, (100,), generator=torch.Generator().manual_seed(args.seed)).tolist()
    carrier = pipe.embed(original, bits, return_tensor=True)
    attack = blend_forgery(carrier, source, mask=load_mask(args.mask) if args.mask else None, seed=args.seed)
    forged, target_mask = attack['image'], attack['mask']
    copyright = pipe.extract(forged)
    predicted_mask = pipe.locate(forged)
    recovered = pipe.recover(forged, return_tensor=True)['image']
    directory.mkdir(parents=True, exist_ok=False)
    for name, image in [('original', original), ('source', source), ('watermarked', carrier),
                        ('forged', forged), ('recovered', recovered)]:
        save_image(image, directory/f'{name}.png')
    mask_to_pil(target_mask, binary=True).save(directory/'mask.png')
    mask_to_pil(predicted_mask, binary=True).save(directory/'predicted_mask.png')
    extracted = ''.join(str(int(v)) for v in copyright['watermark'][0].tolist())
    expected = torch.tensor(bits, device=pipe.device)
    accuracy = float((copyright['watermark'][0] == expected).float().mean())
    (directory/'copyright.json').write_text(json.dumps(dict(expected=''.join(map(str, bits)),
        extracted=extracted,accuracy=accuracy),indent=2),encoding='utf-8')
    predicted, target = predicted_mask > .5, target_mask > .5
    before, after = roi_psnr(forged, original, target_mask), roi_psnr(recovered, original, target_mask)
    report = dict(copyright_accuracy=accuracy,
                  mask_iou=float((predicted & target).sum() / (predicted | target).sum().clamp_min(1)),
                  watermarked_psnr=psnr(carrier, original), forged_psnr=psnr(forged, original),
                  recovery_psnr=psnr(recovered, original), roi_baseline_psnr=before, roi_recovery_psnr=after,
                  roi_gain_db=after-before if before is not None and after is not None else None,
                  attack=attack['metadata'], metric_target='original_before_embedding',
                  intermediate_png_reload=False)
    (directory/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Results: {directory.resolve()}')


if __name__ == '__main__':
    main()
