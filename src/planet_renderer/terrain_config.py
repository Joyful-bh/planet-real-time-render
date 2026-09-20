"""Stable configuration envelope for selecting a terrain generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TerrainConfig:
    """Select a generator and carry its uninterpreted configuration payload.

    Validation of ``params`` belongs to the selected generator's strongly
    typed configuration class. This envelope deliberately assigns no terrain
    semantics to keys inside the mapping.
    """

    generator: str = "procedural_fbm_v1"
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.generator, str) or not self.generator.strip():
            raise ValueError("terrain generator must be a non-empty string")
        if not isinstance(self.params, Mapping):
            raise TypeError("terrain params must be a mapping")
        object.__setattr__(self, "params", dict(self.params))


__all__ = ["TerrainConfig"]
