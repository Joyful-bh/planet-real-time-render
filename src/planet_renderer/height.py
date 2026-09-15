"""统一高度提供器接口；程序高度可在 CPU 查询并由 GPU 实现同一算法。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np

from .planet import Vec3d, normalize


class HeightProvider(Protocol):
    seed: int
    gpu_kind: int

    def sample_height_m(self, direction_global: Vec3d) -> float: ...
    def gpu_parameters(self) -> tuple[int, float, float]: ...


def _hash3(x: int, y: int, z: int, seed: int) -> float:
    mask = 0xFFFFFFFF
    value = ((x * 0x1F123BB5) ^ (y * 0x05491333) ^ (z * 0x72E12A4D) ^ seed) & mask
    value = ((value ^ (value >> 15)) * 0x2C1B3C6D) & mask
    value = ((value ^ (value >> 12)) * 0x297A2D39) & mask
    value ^= value >> 15
    return (value & mask) / mask


def _value_noise3(position: Vec3d, seed: int) -> float:
    base = np.floor(position).astype(np.int64)
    f = position - base
    w = f * f * (3.0 - 2.0 * f)
    value = 0.0
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                corner = _hash3(int(base[0]+dx), int(base[1]+dy), int(base[2]+dz), seed)
                value += corner * (w[0] if dx else 1-w[0]) * (w[1] if dy else 1-w[1]) * (w[2] if dz else 1-w[2])
    return value


def _fbm(position: Vec3d, seed: int, octaves: int) -> float:
    value = total = 0.0
    amplitude = 0.5
    point = position.copy()
    for octave in range(octaves):
        value += _value_noise3(point, seed + octave * 1013) * amplitude
        total += amplitude
        point = point * 2.03 + np.array([7.1, -3.7, 5.3])
        amplitude *= 0.5
    return value / total


@dataclass(frozen=True)
class ProceduralHeightProvider:
    seed: int = 7
    continent_amplitude_m: float = 2800.0
    mountain_amplitude_m: float = 4200.0
    gpu_kind: int = 1

    def sample_height_m(self, direction_global: Vec3d) -> float:
        direction = normalize(np.asarray(direction_global, np.float64))
        warp = np.array([
            _fbm(direction*3.1+11.0, self.seed+17, 3),
            _fbm(direction*3.1-7.0, self.seed+31, 3),
            _fbm(direction*3.1+3.0, self.seed+47, 3),
        ]) - 0.5
        continent = (_fbm(direction*1.65+warp*0.7, self.seed, 5)-0.5) * self.continent_amplitude_m * 2.0
        ridge_noise = _fbm(direction*8.0+warp, self.seed+211, 5)
        ridge = (1.0-abs(ridge_noise*2.0-1.0))**3
        land = np.clip((continent+700.0)/1800.0, 0.0, 1.0)
        return float(np.clip(continent+ridge*self.mountain_amplitude_m*land, -5000.0, 8500.0))

    def gpu_parameters(self) -> tuple[int, float, float]:
        return self.seed, self.continent_amplitude_m, self.mountain_amplitude_m


@dataclass(frozen=True)
class DemHeightProvider:
    """DEM 扩展契约；tile pyramid/I/O/解压由未来实现提供。"""
    pyramid_uri: str
    seed: int = 0
    gpu_kind: int = 2

    def sample_height_m(self, direction_global: Vec3d) -> float:
        raise RuntimeError("DEM provider 尚未连接分级 tile pyramid")

    def gpu_parameters(self) -> tuple[int, float, float]:
        return self.seed, 0.0, 0.0
