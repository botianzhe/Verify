"""Build an offline wheel and GitHub-ready archive, both INCLUDING real weights."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "dist"
    destination.mkdir(exist_ok=True)
    subprocess.run([sys.executable, '-m', 'verify.check_assets'], cwd=root, check=True)
    # Large real weights can exceed a container's small /tmp volume. Keep
    # ephemeral build files on the release filesystem, disable pip's cache.
    with tempfile.TemporaryDirectory(prefix='.wheel-tmp-', dir=destination) as temporary:
        subprocess.run([
            sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-cache-dir",
            "--no-build-isolation", "--wheel-dir", str(destination),
        ], cwd=root, env=dict(os.environ, TMPDIR=temporary), check=True)
    names = ["README.md", "pyproject.toml", "requirements.txt", "Dockerfile", ".gitignore", ".dockerignore", ".gitattributes", "THIRD_PARTY_NOTICES.md"]
    files = [root / name for name in names]
    for directory, pattern in (("verify", "*.py"), ("tools", "*.py"), ("tests", "*.py"),
                               (".github", "*.yml")):
        files.extend(path for path in (root / directory).rglob(pattern) if "__pycache__" not in path.parts)
    files.extend(path for path in (root / 'verify' / 'weights').rglob('*') if path.is_file())
    files.extend(path for path in (root/'examples').rglob('*') if path.is_file())
    files.append(root/'logs/README.md')
    archive_path = destination / "verify-2.0.0-offline.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, Path("Verify") / path.relative_to(root))
    print(f"Offline archive: {archive_path} ({len(files)} files)")


if __name__ == "__main__":
    main()
