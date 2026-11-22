Official implementation of **VeriFi**, a tamper-resilient versatile watermarking framework for
**high-fidelity face content recovery**, introduced in the ACM MM 2026 paper:

# High-Fidelity Face Content Recovery via Tamper-Resilient Versatile Watermarking


This toolkit provides five functions: watermark embedding, local forgery, copyright extraction, forgery localization, and content recovery. It ships with built-in model weights and a VAE. After installing dependencies, it supports fully offline inference.

## Installation

1. Download the model weights from Baidu Cloud Disk and replace the `weights/` folder inside the `verify/` package before running any commands.

https://pan.baidu.com/s/1Lp7is_LrCde3PBQuvcXh-g?pwd=dw5i

2. Requires Python ≥3.10. A CUDA-enabled PyTorch installation is recommended.

```bash
cd Verify
pip install -r requirements.txt
pip install --no-deps .
```

All commands below can be run directly from the `Verify/` directory without installing a CLI entry point. Replace `--device cuda` with `--device cpu` as needed; the default is automatic device selection.

## Five Functions

```bash
# 1. Watermark embedding: copyright is a 100-bit binary string
python -m verify --mode embed --input original.png --output logs/watermarked.png \
  --watermark 0101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101

# 2. Local forgery: donor content is blended into the region specified by mask
python -m verify --mode forge --input logs/watermarked.png --source source.png \
  --mask mask.png --output logs/forged.png

# 3. Copyright extraction: outputs a 100-bit copyright string
python -m verify --mode extract --input logs/forged.png --output logs/copyright.json

# 4. Forgery localization: outputs a black-and-white mask; white indicates predicted tampered regions
python -m verify --mode locate --input logs/forged.png --output logs/predicted_mask.png

# 5. Content recovery: only the image to recover needs to be provided
python -m verify --mode recover --input logs/forged.png --output logs/recovered.png
```

For the `forge` command: if `--source` is omitted, the horizontally flipped input image is used. If `--mask` is omitted, a random region is generated. `--seed` controls the random process. The forgery result also saves a `.mask.png` file containing the ground-truth mask. When forgery is not needed, extraction, localization, or recovery can be applied directly to an existing watermarked image.

## Full Workflow

```bash
python -m verify.workflow --original examples/01_val_0144/original.png \
  --source examples/01_val_0144/source.png --mask examples/01_val_0144/mask.png \
  --device cuda
```

This sequentially performs embedding, blending, copyright extraction, localization, and recovery. Results are saved to a newly created `logs/workflow_<timestamp>/` directory:

- `watermarked.png`, `forged.png`, `recovered.png`: watermarked image, forged image, and recovered image.
- `mask.png`, `predicted_mask.png`: ground-truth region and predicted localization.
- `copyright.json`: expected and extracted copyright bits with accuracy.
- `metrics.json`: recovery PSNR (full image / tampered region), localization IoU, and other performance metrics.

Use `--watermark` to specify a custom 100-bit copyright string; if omitted, one is generated based on the seed. Use `--output-dir` to specify a new, non-existent directory. Intermediate data is passed in memory only; saved PNG files are not reloaded for subsequent steps in the same run.

## Test Examples

```bash
python tools/test_examples.py --device cuda
```

The examples/d directory contains 6 sets of original images, donor images, masks, forged PNGs, and copyright labels.

The test reports two separate scenarios: (1) a complete re-embedding and forgery pipeline, and (2) direct inference on existing forged PNGs. Since the inputs differ, metrics should be examined separately. The curated examples do not represent overall dataset performance.

## Usage Notes

- Images are processed at 256×256 RGB; outputs are PNG. Cropped face images are recommended as input.
- Copyright information is a 100-bit binary string — not an image or logo.
- Localization and recovery do **not** require the original image or ground-truth mask; the ground-truth mask is only used for forgery generation and evaluation.
- Recovery relies on content features carried during embedding. It is a lossy estimate and cannot guarantee recovery for arbitrary non-watermarked images.
- In the full workflow, PSNR is computed between the in-memory recovered result and the pre-embedding original. Step-by-step reloading of saved PNGs introduces quantization effects.
- By default, the bundled weights are used. To replace them, pass `--checkpoint /path/model.pth`; the model architecture must be compatible.

## Deployment

```bash
docker build -t verify-watermark .
docker run --gpus all --rm -v /absolute/data:/data verify-watermark \
  --mode recover --input /data/forged.png --output /data/recovered.png
```

## Citation

```bash
@inproceedings{10.1145/3767308.3835208,
author = {Yu, Peipeng and Xie, Jinfeng and Ou, Chengfu and Zhou, Xiaoyu and Fei, Jianwei and Dai, Yunshu and Xia, Zhihua and Chang, Chip Hong},
title = {High-Fidelity Face Content Recovery via Tamper-Resilient Versatile Watermarking},
year = {2026},
isbn = {9798400722134},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3767308.3835208},
doi = {10.1145/3767308.3835208},
booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
pages = {13793–13802},
numpages = {10},
series = {MM '26}
}
```
