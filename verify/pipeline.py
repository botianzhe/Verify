"""Inference pipeline for Verify watermark embedding and image recovery.

The training checkpoint contains the watermark encoder/decoder, tamper localizer
and recovery transformer in one state dictionary.  This module intentionally
keeps all training-only code out of the public inference API.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL

from .image_io import load_image, image_to_pil, mask_to_pil
from .models.recovery import BitGuidedRecoveryTransformer
from .models.localizer import SwinUNetWatermarkBits
from .models.watermark import WatermarkModel
from .assets import DEFAULT_CHECKPOINT, bundled_vae


TensorLike = Union[torch.Tensor, Image.Image, str, Path]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifyConfig:
    image_size: int = 256
    watermark_bits: int = 100
    latent_bits: int = 1024
    wm_epsilon_px: float = 3.0

    def __post_init__(self):
        if self.image_size != 256 or self.latent_bits != 1024:
            raise ValueError("The trained architecture requires image_size=256 and latent_bits=1024")
        if self.watermark_bits < 1:
            raise ValueError("watermark_bits must be positive")
        if not 0 <= self.wm_epsilon_px <= 255:
            raise ValueError("wm_epsilon_px must be finite and in [0, 255]")


def _device(value: Optional[Union[str, torch.device]]) -> torch.device:
    if value is None or str(value) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


class VerifyPipeline:
    """Load and run the trained Verify model.

    Inputs and outputs use RGB tensors in ``[-1, 1]`` internally.  Public
    methods accept PIL images, paths, or explicitly normalized ``[-1,1]``
    tensors. Convert ``[0,1]`` tensors using load_image(input_range='zero_one').
    """

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        vae: Optional[Union[str, Path]] = "bundled",
        dtype: Optional[torch.dtype] = None,
        use_ema: Optional[bool] = None,
        config: Optional[VerifyConfig] = None,
        strict: bool = True,
    ) -> None:
        checkpoint = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Install the weight-inclusive release or pass --checkpoint.")
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        self.training_version = payload.get('training_version', 1) if isinstance(payload, dict) else 1
        if self.training_version not in (1, 2):
            raise ValueError(f"Unsupported training_version: {self.training_version}")
        saved_config = payload.get('training_config', {}) if isinstance(payload, dict) else {}
        saved_epsilon = saved_config.get('wm_epsilon_px')
        config = config or VerifyConfig(wm_epsilon_px=3.0 if saved_epsilon is None else saved_epsilon)
        if saved_epsilon is not None and config.wm_epsilon_px != saved_epsilon:
            raise ValueError(f"wm_epsilon_px={config.wm_epsilon_px} conflicts with checkpoint value {saved_epsilon}")
        self.strict = strict
        self.config = config
        self.device = _device(device)
        # The published checkpoint was trained and validated in FP32. Keep
        # inference in FP32 by default for matching reconstruction quality;
        # callers may still opt into a lower precision dtype explicitly.
        self.dtype = dtype or torch.float32
        self.use_ema = use_ema
        wm_cfg = {
            "wm_enc_config": {
                "image_size": config.image_size, "in_channels": 3, "channels": 3,
                "norm_type": "batch", "final_skip": True,
                "message_length": config.watermark_bits + config.latent_bits,
            },
            "wm_dec_config": {
                "image_size": config.image_size, "in_channels": 3,
                "norm_type": "batch",
                "message_length": config.watermark_bits + config.latent_bits,
            },
        }
        self.watermark = WatermarkModel("", **wm_cfg)
        self.localizer = SwinUNetWatermarkBits(
            img_size=config.image_size,
            bit_length=config.watermark_bits + config.latent_bits,
            window_size=8,
        )
        # Same parameter shapes do not imply identical semantics. Old snapshots
        # must retain their binary-latent/double-blend path unless retrained v2.
        self.recovery = BitGuidedRecoveryTransformer(wm_len=config.latent_bits, use_router=False,
                                                      legacy_mode=self.training_version < 2)
        self._load_checkpoint(Path(checkpoint))
        self.vae = None
        if vae:
            if str(vae) == "bundled":
                vae = bundled_vae()
            # Accept either a Diffusers directory or a model id.  For a local
            # directory, ``subfolder=vae`` is used only when that subfolder
            # exists; this makes copied SD2 VAE directories convenient.
            vae_path = Path(str(vae))
            if vae_path.is_dir() and (vae_path / "config.json").is_file():
                self.vae = AutoencoderKL.from_pretrained(str(vae_path))
            else:
                self.vae = AutoencoderKL.from_pretrained(str(vae), subfolder="vae")
            self.vae.eval().to(self.device, dtype=self.dtype)
            for parameter in self.vae.parameters():
                parameter.requires_grad_(False)
        self.watermark.eval().to(self.device, dtype=self.dtype)
        # WatermarkBitEncoder intentionally normalizes its input with
        # ``Tensor.float()``. Keep the localizer in FP32 so that this is
        # correct on CUDA as well as CPU (and avoids Float/Half matmul errors).
        self.localizer.eval().to(self.device, dtype=torch.float32)
        # The recovery network creates FP32 routing gates internally. Keep
        # its arithmetic in FP32, as in the training evaluator.
        self.recovery.eval().to(self.device, dtype=torch.float32)
        for module in (self.watermark, self.localizer, self.recovery):
            module.requires_grad_(False)

    def _load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. Restore verify/weights/verify.pth or pass --checkpoint."
            )
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise ValueError("Checkpoint must contain a PyTorch state dictionary")
        groups = {
            "CopyRightWMmodel.": self.watermark,
            "segmodel.": self.localizer,
            "recoverymodel.": self.recovery,
        }
        model_state = {
            prefix + key: value
            for prefix, module in groups.items()
            for key, value in module.state_dict().items()
        }
        compatible = {}
        skipped = {}
        for key, value in state.items():
            if key not in model_state:
                skipped[key] = "key not found in model"
            elif not isinstance(value, torch.Tensor):
                skipped[key] = "checkpoint value is not a tensor"
            elif value.shape != model_state[key].shape:
                skipped[key] = (
                    f"shape mismatch: checkpoint={tuple(value.shape)}, "
                    f"model={tuple(model_state[key].shape)}"
                )
            else:
                compatible[key] = value

        for prefix, module in groups.items():
            substate = {
                key[len(prefix):]: value
                for key, value in compatible.items()
                if key.startswith(prefix)
            }
            module.load_state_dict(substate, strict=False)

        # Keep unmatched model parameters at their current initialized values.
        # Retain full key names so callers can audit partial checkpoint loads.
        missing = sorted(set(model_state) - set(compatible))
        self.checkpoint_load_report = {
            "loaded_keys": sorted(compatible),
            "skipped_keys": skipped,
            "missing_keys": missing,
            "model_is_ema": payload.get("model_is_ema") if isinstance(payload, dict) else None,
            "ema_applied": False,
            "training_version": getattr(self, 'training_version', 1),
            "config": {
                "image_size": self.config.image_size,
                "watermark_bits": self.config.watermark_bits,
                "latent_bits": self.config.latent_bits,
                "wm_epsilon_px": self.config.wm_epsilon_px,
            },
            "checkpoint": str(path.resolve()),
            "complete_model_load": not missing,
        }
        log = logger.warning if skipped or missing else logger.info
        log(
            "Checkpoint %s: loaded=%d, skipped=%d, model keys not loaded=%d",
            path, len(compatible), len(skipped), len(missing),
        )
        for key, reason in skipped.items():
            logger.warning("Skipped checkpoint key %s: %s", key, reason)
        if getattr(self, 'strict', True) and (missing or skipped):
            raise ValueError(
                f"Incomplete/incompatible checkpoint: {len(missing)} missing, {len(skipped)} skipped keys. "
                "Refusing inference with randomly initialized parameters. "
                "strict=False is for diagnostics only."
            )

        # ``latest.pth`` is deliberately a raw training snapshot. The training
        # evaluator wraps inference in ``ema.average_parameters()``; reproduce
        # that behavior here when the checkpoint says its model is not EMA.
        # ``final.pth`` and ``best_locator.pth`` already store EMA weights in
        # ``model`` and must not be averaged a second time.
        self.ema_applied = False
        model_is_ema = payload.get("model_is_ema") if isinstance(payload, dict) else None
        ema_payload = payload.get("ema") if isinstance(payload, dict) else None
        apply_ema = (not model_is_ema) if self.use_ema is None else self.use_ema
        self.checkpoint_load_report["ema_reason"] = "disabled or model already contains EMA"
        if apply_ema and isinstance(ema_payload, dict):
            shadow = ema_payload.get("shadow_params")
            named_parameters = [
                (prefix + name, parameter)
                for prefix, module in groups.items()
                for name, parameter in module.named_parameters()
            ]
            parameter_names = {name for name, _ in named_parameters}
            # torch_ema stores an ordered list, not keyed tensors. Positional
            # loading is only valid for an entirely matching parameter layout.
            # A partially compatible checkpoint must never bypass the key
            # filtering above via a same-shaped EMA entry.
            layout_matches = (
                isinstance(shadow, list)
                and len(shadow) == len(named_parameters)
                and all(
                    name in compatible
                    and isinstance(averaged, torch.Tensor)
                    and averaged.shape == parameter.shape
                    for (name, parameter), averaged in zip(named_parameters, shadow)
                )
                and [key for key in state if key in parameter_names]
                == [name for name, _ in named_parameters]
            )
            if layout_matches:
                with torch.no_grad():
                    for (_, parameter), averaged in zip(named_parameters, shadow):
                        parameter.copy_(averaged.to(device=parameter.device, dtype=parameter.dtype))
                self.ema_applied = True
                self.checkpoint_load_report["ema_applied"] = True
                self.checkpoint_load_report["ema_reason"] = "complete ordered parameter layout matches"
                logger.info(
                    "Applied EMA shadow parameters from %s (%d parameters; model_is_ema=%s)",
                    path, len(named_parameters), model_is_ema,
                )
            else:
                self.checkpoint_load_report["ema_reason"] = "incomplete or incompatible parameter layout"
                logger.warning(
                    "EMA skipped: parameter layout does not fully match; shadow_params=%s, model_params=%d",
                    len(shadow) if isinstance(shadow, list) else type(shadow).__name__,
                    len(named_parameters),
                )
        elif apply_ema:
            self.checkpoint_load_report["ema_reason"] = "checkpoint has no EMA state"
            logger.warning("Checkpoint has no EMA state; using model weights as stored")
        if self.use_ema is True and not self.ema_applied:
            raise ValueError("Explicit EMA requested but no complete, ordered EMA state is available")

    @torch.inference_mode()
    def extract(self, image: TensorLike):
        """Copyright extraction only: no VAE or localizer forward pass.

        Confidence is a sigmoid score, not proof that an image has a watermark.
        """
        logits = self.watermark.decoder(self._image(image))
        probability = logits[:, :self.config.watermark_bits].sigmoid()
        return {"watermark": (probability > .5).to(torch.uint8), "confidence": probability}

    @torch.inference_mode()
    def locate(self, image: TensorLike):
        """Return [B,1,256,256] probability of tampering, without a VAE."""
        return self.decode(image, return_tensor=True)["mask"]

    def _image(self, image: TensorLike) -> torch.Tensor:
        return load_image(image, size=self.config.image_size, device=self.device, dtype=self.dtype)

    @staticmethod
    def _to_pil(x: torch.Tensor) -> Image.Image:
        return image_to_pil(x)

    @staticmethod
    def _mask_to_pil(mask: torch.Tensor, *, binary: bool = False) -> Image.Image:
        """Grayscale mask: white is forged and black is preserved background."""
        return mask_to_pil(mask, binary=binary)

    def _latent(self, x: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("This operation requires a VAE. Set vae to a local SD2 VAE path.")
        with torch.autocast(self.device.type, enabled=self.dtype == torch.float16):
            z = self.vae.encode(F.interpolate(x, (128, 128), mode="bilinear", align_corners=False)).latent_dist.mode()
        return z.float().flatten(1)

    @torch.inference_mode()
    def embed(self, image: TensorLike, watermark: Sequence[int], *, return_tensor: bool = False):
        """Embed config.watermark_bits binary values (100 by default)."""
        x = self._image(image)
        bits = torch.as_tensor(watermark, device=self.device, dtype=self.dtype).flatten()
        if bits.numel() != self.config.watermark_bits or not bool(((bits == 0) | (bits == 1)).all()):
            raise ValueError(f"watermark must contain exactly {self.config.watermark_bits} binary values")
        msg = torch.cat([bits.unsqueeze(0).expand(x.shape[0], -1), self._latent(x).to(self.dtype)], dim=1)
        encoded = self.watermark.encoder(x, msg)
        eps = 2 * self.config.wm_epsilon_px / 255.0
        out = (x + (encoded - x).clamp(-eps, eps)).clamp(-1, 1)
        return out if return_tensor else self._to_pil(out)

    @torch.inference_mode()
    def decode(self, image: TensorLike, *, return_tensor: bool = False):
        """Decode watermark bits and predict a tamper mask."""
        x = self._image(image)
        logits = self.watermark.decoder(x)
        bits = torch.sigmoid(logits[:, : self.config.watermark_bits])
        mask = self.localizer(x.float(), logits.float()).clamp(0, 1)
        result = {"watermark": (bits > 0.5).to(torch.uint8), "confidence": bits, "mask": mask}
        if return_tensor:
            return result
        result["mask_image"] = self._mask_to_pil(mask)
        result["binary_mask_image"] = self._mask_to_pil(mask, binary=True)
        return result

    @torch.inference_mode()
    def recover(self, image: TensorLike, *, return_tensor: bool = False):
        """Return the reference script's recovery output, plus diagnostic images."""
        x = self._image(image)
        logits = self.watermark.decoder(x)
        wm_bits = torch.sigmoid(logits[:, : self.config.watermark_bits])
        latent_bits = logits[:, self.config.watermark_bits :]
        mask = self.localizer(x.float(), logits.float()).clamp(0, 1)
        if self.vae is None:
            raise RuntimeError("Recovery requires a local or downloadable SD2 VAE")
        with torch.autocast(self.device.type, enabled=self.dtype == torch.float16):
            init = self.vae.decode(latent_bits.view(-1, 4, 16, 16)).sample
        # SD2's VAE decodes a 16x16 latent to 128x128.  The trained recovery
        # transformer operates on the 256x256 training canvas, so mirror the
        # training path's resize before concatenating the two image streams.
        init = F.interpolate(
            init,
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
        )
        # Match SDSteganography.forward exactly, including the network's own
        # blend against its already-masked input. Do not silently substitute
        # a different final compositing rule for the reference output.
        recovery_input = (x.float() * (1.0 - mask.float())
                          if self.training_version < 2 else x.float())
        components = self.recovery(
            recovery_input, init.float(), latent_bits.float(), mask.float(),
            return_components=True,
        )
        prediction = components["prediction"]
        recovered = components["training_image"]
        single_blend = (x.float() * (1.0 - mask) + prediction * mask).clamp(-1, 1)
        result = {
            "image": recovered if return_tensor else self._to_pil(recovered),
            "prediction": prediction if return_tensor else self._to_pil(prediction),
            "training_image": components["training_image"] if return_tensor else self._to_pil(components["training_image"]),
            "single_blend_image": single_blend if return_tensor else self._to_pil(single_blend),
            "recovery_input": components["recovery_input"],
            "initial_image": init if return_tensor else self._to_pil(init),
            "watermark": (wm_bits > .5).to(torch.uint8),
            "confidence": wm_bits,
            "latent": latent_bits,
            "mask": mask,
        }
        if not return_tensor:
            result["mask_image"] = self._mask_to_pil(mask)
            result["binary_mask_image"] = self._mask_to_pil(mask, binary=True)
        return result
