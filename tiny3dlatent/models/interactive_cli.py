from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from tiny3dlatent.models.common import (
    build_vae_from_checkpoint,
    load_checkpoint,
    select_device,
)
from tiny3dlatent.models.flow import ConditionedLatentFlow
from tiny3dlatent.models.metrics import occupancy_from_logits
from tiny3dlatent.representation.cleanup import clean_mesh
from tiny3dlatent.representation.export import (
    export_glb_with_material,
    export_mesh,
)
from tiny3dlatent.representation.marching_cubes import extract_mesh_from_occupancy
from tiny3dlatent.text.parser import (
    ATTRIBUTE_SIZES,
    ATTRIBUTE_VOCABULARIES,
    attribute_indices,
    parse_prompt,
)
from tiny3dlatent.utils.io import ensure_dir, write_json


class PromptGenerator:
    """Loads the trained flow + VAE once and generates assets from prompts."""

    def __init__(self, checkpoint_path: Path, *, steps: int, guidance: float) -> None:
        self.steps = steps
        self.guidance = guidance

        flow_checkpoint = load_checkpoint(checkpoint_path)
        config = flow_checkpoint["config"]
        self.device = select_device(str(config["device"]))

        vae_checkpoint = load_checkpoint(Path(str(flow_checkpoint["vae_checkpoint"])))
        self.vae = build_vae_from_checkpoint(vae_checkpoint).to(self.device)
        self.has_material = hasattr(self.vae, "decode_full")
        vae_config = vae_checkpoint["config"]

        self.flow = ConditionedLatentFlow(
            latent_dim=int(vae_config["latent_dim"]),
            attribute_sizes=ATTRIBUTE_SIZES,
            attribute_dim=int(config["attribute_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            time_dim=int(config["time_dim"]),
            hidden_layers=int(config["hidden_layers"]),
        ).to(self.device)
        self.flow.load_state_dict(flow_checkpoint["model_state"])
        self.flow.eval()

        self.latent_mean = flow_checkpoint["latent_mean"].to(self.device)
        self.latent_std = flow_checkpoint["latent_std"].to(self.device)
        self.checkpoint_path = checkpoint_path

    def generate(self, prompt: str, *, seed: int) -> dict[str, object]:
        """Generate one shape for the prompt; returns mesh + material info."""
        attributes = parse_prompt(prompt)
        indices = torch.tensor([attribute_indices(attributes)], device=self.device)
        torch.manual_seed(seed)
        z = self.flow.sample(indices, steps=self.steps, guidance_scale=self.guidance)
        z = z * self.latent_std + self.latent_mean

        with torch.no_grad():
            if self.has_material:
                logits, rgb_grid, material = self.vae.decode_full(z)
            else:
                logits = self.vae.decode(z)
                rgb_grid = material = None

        occupancy = occupancy_from_logits(logits)[0, 0].cpu().numpy().astype(np.uint8)
        if occupancy.sum() == 0:
            raise RuntimeError("generated an empty grid; try a different seed")
        mesh = clean_mesh(extract_mesh_from_occupancy(occupancy))

        result: dict[str, object] = {
            "prompt": prompt,
            "parsed_attributes": attributes.to_metadata(),
            "seed": seed,
            "steps": self.steps,
            "guidance_scale": self.guidance,
            "filled_voxels": int(occupancy.sum()),
            "mesh": mesh,
        }
        if self.has_material:
            mask = occupancy.astype(bool)
            result["base_color"] = tuple(
                float(channel[mask].mean()) for channel in rgb_grid[0].cpu().numpy()
            )
            result["roughness"] = float(material[0, 0])
            result["metallic"] = float(material[0, 1])
        return result


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint or _latest_flow_checkpoint()
    print(f"loading checkpoint {checkpoint_path} ...")
    generator = PromptGenerator(
        checkpoint_path, steps=args.steps, guidance=args.guidance
    )
    run_dir = _make_run_dir(Path("outputs/runs"))
    print(f"assets will be saved to {run_dir}")
    _print_vocabulary()

    counter = 0
    seed = args.seed
    while True:
        try:
            prompt = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if prompt.lower() in ("help", "?"):
            _print_vocabulary()
            continue

        try:
            result = generator.generate(prompt, seed=seed)
        except ValueError as error:
            print(f"  invalid prompt: {error}")
            continue
        except RuntimeError as error:
            print(f"  generation failed: {error}")
            seed += 1
            continue

        slug = prompt.replace(" ", "_")
        mesh = result.pop("mesh")
        saved = []
        if args.format in ("glb", "both") and "base_color" in result:
            glb_path = run_dir / f"{counter:03d}_{slug}.glb"
            export_glb_with_material(
                mesh,
                glb_path,
                base_color=result["base_color"],  # type: ignore[arg-type]
                roughness=float(result["roughness"]),  # type: ignore[arg-type]
                metallic=float(result["metallic"]),  # type: ignore[arg-type]
                name=prompt,
            )
            saved.append(glb_path)
        if args.format in ("obj", "both") or "base_color" not in result:
            obj_path = run_dir / f"{counter:03d}_{slug}.obj"
            export_mesh(mesh, obj_path)
            saved.append(obj_path)

        metadata_path = run_dir / f"{counter:03d}_{slug}.json"
        write_json(
            metadata_path,
            {**result, "files": [path.as_posix() for path in saved]},
        )

        print(f"  voxels: {result['filled_voxels']}", end="")
        if "base_color" in result:
            color = tuple(round(v, 2) for v in result["base_color"])  # type: ignore[arg-type]
            print(
                f"  color: {color}  roughness: {result['roughness']:.2f}"
                f"  metallic: {result['metallic']:.2f}",
                end="",
            )
        print()
        for path in saved:
            print(f"  saved {path}")
        counter += 1
        seed += 1

    print(f"done: {counter} assets in {run_dir}")


def _print_vocabulary() -> None:
    print(
        "describe a shape using these words (any subset, e.g. 'red metallic sphere'):"
    )
    for attribute, vocabulary in ATTRIBUTE_VOCABULARIES.items():
        words = ", ".join(word.replace("_", " ") for word in vocabulary)
        print(f"  {attribute:11s} {words}")
    print("type 'quit' to exit, 'help' to reprint this list")


def _latest_flow_checkpoint() -> Path:
    for pattern in ("*-color-text-flow/text_flow.pt", "*-text-flow/text_flow.pt"):
        candidates = sorted(Path("outputs/runs").glob(pattern))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(
        "no text-flow checkpoint found under outputs/runs/; train one with "
        "python -m tiny3dlatent.models.train_text_flow --config "
        "configs/color_text_flow.json"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive prompt-to-3D shell: type prompts, get OBJ/GLB files."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a text_flow.pt checkpoint (default: latest color flow).",
    )
    parser.add_argument(
        "--format",
        choices=("glb", "obj", "both"),
        default="glb",
        help="Output format (default: glb with PBR material).",
    )
    parser.add_argument("--steps", type=int, default=50, help="Euler steps.")
    parser.add_argument(
        "--guidance", type=float, default=2.0, help="Classifier-free guidance scale."
    )
    parser.add_argument("--seed", type=int, default=0, help="Starting seed.")
    return parser.parse_args()


def _make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(output_root / f"{timestamp}-interactive")


if __name__ == "__main__":
    main()
