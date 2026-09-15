"""地形门面：组合 LOD selector、tile manager 与 height provider。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING
import numpy as np

from .camera import PlanetCamera
from .height import DemHeightProvider, HeightProvider, ProceduralHeightProvider
from .planet import PlanetModel, Vec3d, normalize
from .terrain_lod import MixedLodSelector, cube_face_direction, direction_to_cube_face_uv
from .terrain_streaming import TerrainTileManager
from .terrain_types import PatchDescriptor, PatchKey, PatchState, TerrainDebugStats, TerrainFrame
if TYPE_CHECKING:
    from .renderer import PlanetRenderer

# 兼容原配置和调用名；实现已迁移到统一 provider 模块。
ProceduralHeightSource = ProceduralHeightProvider
HeightSource = HeightProvider


@dataclass(frozen=True)
class SurfaceDescriptor:
    cell_id:int; height_m:float; normal_global:tuple[float,float,float]
    material_weights:tuple[float,float,float,float]; seed:int


@dataclass(frozen=True)
class PatchResidencyEvent:
    added:tuple[PatchKey,...]; removed:tuple[PatchKey,...]


class SurfaceCoverageConsumer(Protocol):
    def on_patch_residency_changed(self,event:PatchResidencyEvent)->None: ...


@dataclass(frozen=True)
class TerrainSettings:
    patch_resolution:int=12
    max_level:int=16
    split_sse_pixels:float=64.0
    merge_sse_pixels:float=32.0
    max_desired_patches:int=180
    max_gpu_patches:int=256
    build_budget_per_frame:int=8
    upload_budget_per_frame:int=4
    lod_changes_per_update:int=8
    cache_capacity:int=1024
    selection_interval_s:float=.1
    canonical_surface_level:int=14

    def __post_init__(self)->None:
        if not 0<self.merge_sse_pixels<self.split_sse_pixels:raise ValueError("merge SSE 必须小于 split SSE")
        if not 0<=self.max_level<=20:raise ValueError("max_level 必须在 0..20")
        if min(self.max_desired_patches,self.max_gpu_patches,self.build_budget_per_frame,self.upload_budget_per_frame)<=0:raise ValueError("地形预算必须为正")


def surface_cell_id(direction:Vec3d,level:int=14)->int:
    face,u,v=direction_to_cube_face_uv(direction); side=1<<level
    x=min(max(int((u*.5+.5)*side),0),side-1); y=min(max(int((v*.5+.5)*side),0),side-1)
    return (face<<28)|(y<<14)|x


class CubeSphereTerrain:
    """兼容门面；不再生成或拼装 TerrainMeshBatch。"""
    def __init__(self,planet:PlanetModel,height_source:HeightProvider,settings:TerrainSettings):
        self.planet=planet; self.height_provider=height_source; self.height_source=height_source; self.settings=settings
        self.selector=MixedLodSelector(planet,settings.patch_resolution,settings.max_level,settings.split_sse_pixels,settings.merge_sse_pixels,settings.max_desired_patches,settings.lod_changes_per_update)
        self.tile_manager=TerrainTileManager(planet,self.selector,settings.max_gpu_patches,settings.cache_capacity,settings.build_budget_per_frame,settings.upload_budget_per_frame,settings.selection_interval_s)
        self._consumers:list[SurfaceCoverageConsumer]=[]; self._resident=frozenset()

    def add_coverage_consumer(self,consumer:SurfaceCoverageConsumer)->None:self._consumers.append(consumer)

    def update(self,camera:PlanetCamera,viewport_height:int,renderer:"PlanetRenderer",now:float|None=None)->TerrainFrame:
        frame=self.tile_manager.update(camera,viewport_height,renderer,now)
        added=tuple(sorted(frame.resident-self._resident)); removed=tuple(sorted(self._resident-frame.resident))
        if added or removed:
            event=PatchResidencyEvent(added,removed)
            for consumer in self._consumers: consumer.on_patch_residency_changed(event)
        self._resident=frame.resident
        return frame

    def select_patches(self,camera:PlanetCamera,viewport_height:int)->tuple[PatchKey,...]:
        return tuple(sorted(self.selector.select(camera,viewport_height)))

    def describe_surface(self,direction_global:Vec3d)->SurfaceDescriptor:
        direction=normalize(np.asarray(direction_global,np.float64)); helper=np.array([0.,1.,0.]) if abs(direction[1])<.9 else np.array([1.,0.,0.])
        tu=normalize(np.cross(helper,direction)); tv=normalize(np.cross(direction,tu)); eps=1e-5
        def position(d):
            d=normalize(d); return d*(self.planet.radius_m+self.height_provider.sample_height_m(d))
        normal=normalize(np.cross(position(direction+tu*eps)-position(direction-tu*eps),position(direction+tv*eps)-position(direction-tv*eps)))
        if np.dot(normal,direction)<0:normal=-normal
        height=self.height_provider.sample_height_m(direction); slope=1-max(float(np.dot(normal,direction)),0.)
        snow=np.clip((height-2600)/1800,0,1); rock=np.clip(slope*7,0,1)*(1-snow); sand=np.clip(1-abs(height)/500,0,1)*(1-rock)*(1-snow); weights=np.array([max(1-snow-rock-sand,0),rock,sand,snow]); weights/=weights.sum()
        return SurfaceDescriptor(surface_cell_id(direction,self.settings.canonical_surface_level),height,tuple(normal),tuple(weights),self.height_provider.seed)


__all__=["CubeSphereTerrain","TerrainSettings","PatchKey","PatchState","PatchDescriptor","TerrainDebugStats","TerrainFrame","ProceduralHeightSource","ProceduralHeightProvider","DemHeightProvider","cube_face_direction","direction_to_cube_face_uv","surface_cell_id"]
