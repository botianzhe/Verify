"""Real-weight offline independent CLI checks with one user-supplied image."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    image = args.image.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    environment['PYTHONPATH'] = str(root)
    bits = '01' * 50

    def run(*arguments):
        subprocess.run([sys.executable, '-m', 'verify', *map(str, arguments)],
                       cwd=output, env=environment, check=True)

    run('--mode', 'embed', '--input', image, '--output', 'carrier.png', '--watermark', bits, '--device', args.device)
    run('--mode', 'forge', '--input', 'carrier.png', '--output', 'forged.png', '--seed', '2025')
    run('--mode', 'extract', '--input', 'forged.png', '--output', 'copyright.json', '--device', args.device)
    run('--mode', 'locate', '--input', 'forged.png', '--output', 'mask.png', '--device', args.device)
    run('--mode', 'recover', '--input', 'forged.png', '--output', 'recovered.png', '--device', args.device)
    copyright = json.loads((output / 'copyright.json').read_text())
    assert len(copyright['watermark'])==100 and set(copyright['watermark'])<={'0','1'}
    for name in ('carrier.png','forged.png','mask.png','recovered.png'):
        with Image.open(output/name) as image:
            assert image.size==(256,256)
            image.verify()
    accuracy=sum(a==b for a,b in zip(bits,copyright['watermark']))/100
    report = dict(status='pass', modes=['embed', 'forge', 'extract', 'locate', 'recover'],
                  offline=True, watermark_accuracy=accuracy,
                  scope='independent CLI PNG disk-path test; not in-memory validation')
    (output / 'cli_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
