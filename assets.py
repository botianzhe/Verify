"""Offline assets installed with the package (never relative to the caller's cwd)."""

from pathlib import Path

WEIGHTS = Path(__file__).resolve().parent / "weights"
DEFAULT_CHECKPOINT = WEIGHTS / "verify.pth"
DEFAULT_VAE = WEIGHTS / "vae"


def bundled_vae():
    if not (DEFAULT_VAE / "config.json").is_file():
        raise FileNotFoundError(
            f"Bundled SD2 VAE missing: {DEFAULT_VAE}. Install the weight-inclusive release "
            "or pass --vae /absolute/path/to/vae. No automatic download is performed."
        )
    return str(DEFAULT_VAE)
