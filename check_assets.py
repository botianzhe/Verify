"""Verify every bundled binary against its SHA-256 manifest, without downloads."""

import hashlib
import json

from .assets import WEIGHTS


def main():
    manifest_path = WEIGHTS / 'manifest.json'
    if not manifest_path.is_file():
        raise SystemExit(f'Missing asset manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (WEIGHTS / name).resolve()
        if not path.is_relative_to(WEIGHTS.resolve()):
            raise SystemExit('Invalid manifest path')
        if not path.is_file() or path.stat().st_size != expected['bytes']:
            raise SystemExit(f'Missing/incomplete asset (possibly a Git LFS pointer): {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected['sha256']:
            raise SystemExit(f'SHA-256 mismatch: {path}')
        print(f'OK {name} ({expected["bytes"]} bytes)')


if __name__ == '__main__':
    main()
