"""Verify watermark embedding, forgery localization and image recovery."""

__all__ = ["VerifyConfig", "VerifyPipeline", "blend_forgery", "generate_random_mask"]


def __getattr__(name):
    # Standalone forge does not import or initialize model/VAE dependencies.
    if name in {"VerifyConfig", "VerifyPipeline"}:
        from .pipeline import VerifyConfig, VerifyPipeline
        return {"VerifyConfig": VerifyConfig, "VerifyPipeline": VerifyPipeline}[name]
    if name in {"blend_forgery", "generate_random_mask"}:
        from .forgery import blend_forgery, generate_random_mask
        return {"blend_forgery": blend_forgery, "generate_random_mask": generate_random_mask}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
