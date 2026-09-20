"""Explicit registry and factory for terrain-generation algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .height import TerrainHeightModel
from .fbm_terrain import FbmTerrainConfig, FbmTerrainGenerator
from .landforms_terrain import LandformsTerrainConfig, LandformsTerrainGenerator
from .terrain_config import TerrainConfig


@dataclass(frozen=True)
class TerrainGeneratorRegistration:
    """Connect one stable generator ID to its private config and model types."""

    config_type: type[Any]
    generator_type: Callable[[Any], TerrainHeightModel]


_REGISTRY: dict[str, TerrainGeneratorRegistration] = {}


def register_terrain_generator(
    generator_id: str,
    config_type: type[Any],
    generator_type: Callable[[Any], TerrainHeightModel],
) -> None:
    """Register one generator explicitly and reject accidental replacement."""

    if not generator_id:
        raise ValueError("terrain generator ID must not be empty")
    if generator_id in _REGISTRY:
        raise ValueError(f"terrain generator is already registered: {generator_id!r}")
    _REGISTRY[generator_id] = TerrainGeneratorRegistration(
        config_type=config_type,
        generator_type=generator_type,
    )


def create_terrain_model(config: TerrainConfig) -> TerrainHeightModel:
    """Resolve an envelope, validate its payload and construct the model."""

    registration = _REGISTRY.get(config.generator)
    if registration is None:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown terrain generator {config.generator!r}; available: {available}"
        )

    try:
        algorithm_config = registration.config_type(**dict(config.params))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid parameters for terrain generator {config.generator!r}: {exc}"
        ) from exc
    return registration.generator_type(algorithm_config)


def registered_terrain_generators() -> tuple[str, ...]:
    """Return stable IDs for diagnostics and configuration tooling."""

    return tuple(sorted(_REGISTRY))


register_terrain_generator(
    "procedural_fbm_v1",
    FbmTerrainConfig,
    FbmTerrainGenerator,
)
register_terrain_generator(
    "procedural_landforms_v1",
    LandformsTerrainConfig,
    LandformsTerrainGenerator,
)


__all__ = [
    "TerrainGeneratorRegistration",
    "create_terrain_model",
    "register_terrain_generator",
    "registered_terrain_generators",
]
