from __future__ import annotations

from dataclasses import dataclass

from tiny3dlatent.data.labels import COLORS, DESCRIPTORS, SHAPE_TYPES, SIZES

ATTRIBUTE_ORDER = ("shape_type", "color", "size", "descriptor")

ATTRIBUTE_VOCABULARIES: dict[str, tuple[str, ...]] = {
    "shape_type": SHAPE_TYPES,
    "color": COLORS,
    "size": SIZES,
    "descriptor": DESCRIPTORS,
}

_WORD_TO_ATTRIBUTE: dict[str, tuple[str, str]] = {}
for _attribute, _vocabulary in ATTRIBUTE_VOCABULARIES.items():
    for _word in _vocabulary:
        _WORD_TO_ATTRIBUTE[_word] = (_attribute, _word)


@dataclass(frozen=True)
class PromptAttributes:
    """Parsed prompt attributes; `None` means the prompt did not specify one."""

    shape_type: str | None = None
    color: str | None = None
    size: str | None = None
    descriptor: str | None = None

    def get(self, attribute: str) -> str | None:
        return getattr(self, attribute)

    def to_metadata(self) -> dict[str, str | None]:
        return {attribute: self.get(attribute) for attribute in ATTRIBUTE_ORDER}


ATTRIBUTE_SIZES = tuple(
    len(ATTRIBUTE_VOCABULARIES[attribute]) for attribute in ATTRIBUTE_ORDER
)


def attribute_indices(attributes: PromptAttributes) -> list[int]:
    """Map parsed attributes to embedding indices; unspecified -> vocab size."""
    indices = []
    for attribute in ATTRIBUTE_ORDER:
        vocabulary = ATTRIBUTE_VOCABULARIES[attribute]
        value = attributes.get(attribute)
        indices.append(
            vocabulary.index(value) if value is not None else len(vocabulary)
        )
    return indices


def parse_prompt(prompt: str) -> PromptAttributes:
    """Parse a tiny-vocabulary prompt like "red tall cylinder".

    Raises `ValueError` on unknown words, repeated attribute categories, or an
    empty prompt. Multi-word shape names ("rounded box") are supported.
    """
    normalized = prompt.lower().replace("rounded box", "rounded_box")
    words = normalized.split()
    if not words:
        raise ValueError("empty prompt: provide words like 'red tall cylinder'")

    found: dict[str, str] = {}
    for word in words:
        if word not in _WORD_TO_ATTRIBUTE:
            known = ", ".join(sorted(_WORD_TO_ATTRIBUTE))
            raise ValueError(f"unknown word {word!r}; known words: {known}")
        attribute, value = _WORD_TO_ATTRIBUTE[word]
        if attribute in found and found[attribute] != value:
            raise ValueError(
                f"conflicting {attribute} words: {found[attribute]!r} and {value!r}"
            )
        found[attribute] = value

    return PromptAttributes(**found)
