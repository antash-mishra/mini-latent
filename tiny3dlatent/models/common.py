from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tiny3dlatent.utils.io import ensure_dir


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(
    path: Path, model: nn.Module, *, config: dict[str, Any], **extra: Any
) -> None:
    ensure_dir(path.parent)
    torch.save({"model_state": model.state_dict(), "config": config, **extra}, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def build_vae_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    """Instantiate the right VAE class (shape-only or color) from a checkpoint."""
    from tiny3dlatent.models.color_vae import ColorVAE
    from tiny3dlatent.models.vae import VAE

    config = checkpoint["config"]
    model_class = ColorVAE if config.get("model_type") == "color_vae" else VAE
    model = model_class(
        resolution=int(config["resolution"]),
        latent_dim=int(config["latent_dim"]),
        base_channels=int(config["base_channels"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
