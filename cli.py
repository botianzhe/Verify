"""Five independent watermark and image-recovery operations."""

import argparse
import json
from pathlib import Path

import torch

from .assets import DEFAULT_CHECKPOINT
from .image_io import load_image, load_mask, mask_to_pil, save_image


def _bits(value: str) -> list[int]:
    value = value.strip().replace(' ', '').replace(',', '')
    if len(value) != 100 or any(c not in '01' for c in value):
        raise argparse.ArgumentTypeError('watermark must contain exactly 100 binary bits')
    return [int(c) for c in value]


def main():
    parser = argparse.ArgumentParser(prog='verify', description=__doc__)
    parser.add_argument('--mode', choices=['embed', 'forge', 'extract', 'locate', 'recover'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='PNG image; JSON for extract')
    parser.add_argument('--watermark', type=_bits, help='100 binary bits, required for embed')
    parser.add_argument('--source', type=Path, help='donor for forge; default: flipped input')
    parser.add_argument('--mask', type=Path, help='mask for forge; white means tampered')
    parser.add_argument('--seed', type=int, default=2025, help='forgery random seed')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if any(path and path.suffix.lower() in {'.pt', '.npz'} for path in (args.input, args.source, args.mask)):
        parser.error('Use image files, not PT/NPZ data')
    if args.output.suffix.lower() != ('.json' if args.mode == 'extract' else '.png'):
        parser.error('extract requires JSON output; other operations require PNG output')
    if args.mode == 'embed' and args.watermark is None:
        parser.error('embed requires --watermark')
    if args.watermark is not None and args.mode != 'embed':
        parser.error('--watermark is only used by embed')
    if (args.source or args.mask) and args.mode != 'forge':
        parser.error('--source and --mask are only used by forge')
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'forge':
        from .forgery import blend_forgery
        result = blend_forgery(load_image(args.input), load_image(args.source) if args.source else None,
                              mask=load_mask(args.mask) if args.mask else None, seed=args.seed)
        save_image(result['image'], args.output)
        mask_to_pil(result['mask'], binary=True).save(args.output.with_suffix('.mask.png'))
    else:
        from .pipeline import VerifyPipeline
        pipe = VerifyPipeline(args.checkpoint, device=args.device,
                              vae=None if args.mode in {'extract', 'locate'} else 'bundled')
        if args.mode == 'embed':
            save_image(pipe.embed(args.input, args.watermark, return_tensor=True), args.output)
        elif args.mode == 'extract':
            result = pipe.extract(args.input)
            payload = {'watermark': ''.join(str(int(v)) for v in result['watermark'][0].tolist())}
            args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        elif args.mode == 'locate':
            mask_to_pil(pipe.locate(args.input), binary=True).save(args.output)
        else:
            save_image(pipe.recover(args.input, return_tensor=True)['image'], args.output)
    print(args.output)


if __name__ == '__main__':
    main()
