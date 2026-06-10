from __future__ import annotations

import numpy as np
import pytest
import torch

from tiny3dlatent.models.flow import ConditionedLatentFlow
from tiny3dlatent.models.text_generate_cli import _aspect_ratio
from tiny3dlatent.text.parser import (
    ATTRIBUTE_SIZES,
    PromptAttributes,
    attribute_indices,
    parse_prompt,
)


def test_parse_prompt_full() -> None:
    attributes = parse_prompt("red large tall cylinder")

    assert attributes.shape_type == "cylinder"
    assert attributes.color == "red"
    assert attributes.size == "large"
    assert attributes.descriptor == "tall"


def test_parse_prompt_partial_and_multiword() -> None:
    attributes = parse_prompt("green rounded box")

    assert attributes.shape_type == "rounded_box"
    assert attributes.color == "green"
    assert attributes.size is None
    assert attributes.descriptor is None


def test_parse_prompt_rejects_unknown_word() -> None:
    with pytest.raises(ValueError, match="unknown word 'dragon'"):
        parse_prompt("red dragon")


def test_parse_prompt_rejects_conflicts_and_empty() -> None:
    with pytest.raises(ValueError, match="conflicting color"):
        parse_prompt("red blue sphere")
    with pytest.raises(ValueError, match="empty prompt"):
        parse_prompt("   ")


def test_attribute_indices_uses_null_for_unspecified() -> None:
    indices = attribute_indices(PromptAttributes(shape_type="sphere"))

    assert indices[0] == 0  # sphere is first in SHAPE_TYPES
    assert indices[1:] == list(ATTRIBUTE_SIZES[1:])  # null index == vocab size


def test_conditioned_flow_shapes_and_guidance() -> None:
    model = ConditionedLatentFlow(
        latent_dim=16, attribute_sizes=ATTRIBUTE_SIZES, hidden_dim=32
    )
    attributes = model.null_attributes(5, torch.device("cpu"))

    velocity = model(torch.randn(5, 16), torch.rand(5), attributes)
    samples = model.sample(attributes, steps=4, guidance_scale=2.0)

    assert velocity.shape == (5, 16)
    assert samples.shape == (5, 16)
    assert torch.isfinite(samples).all()


def test_aspect_ratio_detects_tall_grids() -> None:
    tall = np.zeros((16, 16, 16), dtype=np.uint8)
    tall[6:10, 2:14, 6:10] = 1
    wide = np.zeros((16, 16, 16), dtype=np.uint8)
    wide[2:14, 6:10, 2:14] = 1

    assert _aspect_ratio(tall) > 1.5
    assert _aspect_ratio(wide) < 0.7
