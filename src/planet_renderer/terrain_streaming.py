"""Patch 生命周期、优先级请求、LRU 缓存和预算化 GPU 驻留。"""
from __future__ import annotations
from dataclasses import dataclass
import heapq, math, time
from typing import TYPE_CHECKING
import numpy as np

from .camera import PlanetCamera
from .planet import PlanetModel, normalize
from .terrain_lod import MixedLodSelector, patch_center_direction, patch_uv_bounds, cube_face_direction
from .terrain_types import PatchDescriptor, PatchKey, PatchState, TerrainDebugStats, TerrainFrame
if TYPE_CHECKING:
    from .renderer import PlanetRenderer


@dataclass
class PatchRecord:
    key: PatchKey
    state: PatchState = PatchState.UNLOADED
    slot: int | None = None
    last_used_frame: int = 0
    priority: float = 0.0
    request_version: int = 0


class TerrainTileManager:
    def __init__(self,planet:PlanetModel,selector:MixedLodSelector,max_slots:int=256,
                 cache_capacity:int=1024,build_budget:int=8,upload_budget:int=4,selection_interval_s:float=.1):
        self.planet=planet; self.selector=selector; self.max_slots=max_slots; self.cache_capacity=cache_capacity
        self.build_budget=build_budget; self.upload_budget=upload_budget; self.selection_interval_s=selection_interval_s
        self.queue_scan_budget=max(64,build_budget*8)
        self.records:dict[PatchKey,PatchRecord]={}; self.queue:list[tuple[float,int,PatchKey]]=[]; self.free_slots=list(range(max_slots-1,-1,-1))
        self.desired=frozenset(); self.resident:set[PatchKey]=set(); self.render_keys:set[PatchKey]=set()
        self.frame=0; self._version=0; self._last_selection=-1e9; self._last_position:np.ndarray|None=None; self._last_time=time.perf_counter()
        self.cache_hits=self.cache_misses=0; self.stats=TerrainDebugStats()
        self._bounds_cache: dict[
            PatchKey,
            tuple[np.ndarray, float, float]
        ] = {}
        self._balanced_input:frozenset[PatchKey]|None=None
        self._balanced_output:frozenset[PatchKey]=frozenset()
        self._needed_desired:frozenset[PatchKey]|None=None
        self._needed_cache:set[PatchKey]=set()
        self._boundary_signature:frozenset[PatchKey]|None=None
        self._boundary_cache:dict[PatchKey,tuple[int,int]]={}
        self._visibility_signature:tuple|None=None
        self._visible_cache:set[PatchKey]=set()

        # 当前程序地形范围是 -5000~8500 m，取 10 km 保守包围。
        self.max_terrain_relief_m = 10_000.0

    def _patch_bounds(
        self,
        key: PatchKey,
    ) -> tuple[np.ndarray, float, float]:
        """
        返回：
        center_world   世界空间包围球中心
        angular_radius 从星球中心看的角半径，用于 horizon culling
        world_radius   世界空间包围球半径，用于 frustum culling
        """
        cached = self._bounds_cache.get(key)
        if cached is not None:
            return cached

        u0, u1, v0, v1 = patch_uv_bounds(key)
        um = (u0 + u1) * 0.5
        vm = (v0 + v1) * 0.5

        center_dir = patch_center_direction(key)
        center_world = center_dir * self.planet.radius_m

        # 四角 + 四条边中点，避免只检查角点时低估范围。
        sample_uvs = (
            (u0, v0), (um, v0), (u1, v0),
            (u1, vm),
            (u1, v1), (um, v1), (u0, v1),
            (u0, vm),
        )

        sample_dirs = [
            cube_face_direction(key.face, u, v)
            for u, v in sample_uvs
        ]

        angular_radius = max(
            math.acos(float(np.clip(np.dot(center_dir, d), -1.0, 1.0)))
            for d in sample_dirs
        )

        surface_radius = max(
            float(np.linalg.norm(
                d * self.planet.radius_m - center_world
            ))
            for d in sample_dirs
        )

        # 把山峰、谷地也包进去。
        world_radius = surface_radius + self.max_terrain_relief_m

        result = (center_world, angular_radius, world_radius)
        self._bounds_cache[key] = result
        return result

    def _descriptor(
        self, key: PatchKey, camera: PlanetCamera, viewport_height: int,
        priority: float = 0.0, skirt_mask: int = 0, stitch_mask: int = 0,
    ) -> PatchDescriptor:
        direction = patch_center_direction(key)
        return PatchDescriptor(
            key=key,
            anchor_global=direction * self.planet.radius_m,
            sse=self.selector.sse(
                key,
                camera,
                viewport_height,
            ),
            priority=priority,
            skirt_mask=skirt_mask,
            stitch_mask=stitch_mask,
        )
    
    def _request(self,key:PatchKey,priority:float)->None:
        record=self.records.get(key)
        if record and record.state in (PatchState.READY,PatchState.GPU_RESIDENT): self.cache_hits+=1; record.last_used_frame=self.frame; return
        self.cache_misses+=1
        if record is None: record=PatchRecord(key); self.records[key]=record
        if record.state==PatchState.REQUESTED and priority<=record.priority: return
        self._version+=1; record.state=PatchState.REQUESTED; record.priority=priority; record.request_version=self._version
        heapq.heappush(self.queue,(-priority,self._version,key))

    @staticmethod
    def _ancestors(keys:frozenset[PatchKey])->set[PatchKey]:
        result=set(keys)
        for key in tuple(keys):
            parent=key.parent()
            while parent is not None: result.add(parent); parent=parent.parent()
        return result

    def _priority(self,key:PatchKey,camera:PlanetCamera,viewport_height:int,velocity_direction:np.ndarray|None,forward:np.ndarray)->float:
        center=self.selector.center(key)*self.planet.radius_m;delta=center-camera.position_global;distance=max(float(np.linalg.norm(delta)),1.)
        delta_direction=delta/distance
        center_bias=max(float(np.dot(delta_direction,forward)),0.0);motion_bias=0.0
        if velocity_direction is not None:motion_bias=max(float(np.dot(delta_direction,velocity_direction)),0.0)
        focal=viewport_height/(2*math.tan(math.radians(camera.vertical_fov_degrees)*.5))
        geometric_error=self.planet.radius_m*2.4/((1<<key.level)*self.selector.resolution)
        sse=geometric_error*focal/distance
        return sse*1000.0+center_bias*100.0+motion_bias*200.0+key.level-distance*1e-9

    def _select_if_due(self,camera:PlanetCamera,viewport_height:int,now:float)->tuple[np.ndarray,float]:
        position=camera.position_global.copy(); dt=max(now-self._last_time,1e-6)
        velocity=np.zeros(3) if self._last_position is None else (position-self._last_position)/dt
        moved=math.inf if self._last_position is None else float(np.linalg.norm(position-self._last_position))
        altitude=max(self.planet.altitude_m(position),1.); update_distance=max(10.,altitude*.01)
        selection_ms=0.0
        if now-self._last_selection>=self.selection_interval_s and (moved>=update_distance or self.selector.last_changes>0):
            self.desired=self.selector.select(camera,viewport_height); self._last_selection=now; self._last_position=position
            selection_ms=self.selector.last_ms
        self._last_time=now
        return velocity,selection_ms

    def update(self,camera:PlanetCamera,viewport_height:int,renderer:"PlanetRenderer",now:float|None=None)->TerrainFrame:
        started=time.perf_counter(); now=time.perf_counter() if now is None else now; self.frame+=1
        velocity,selection_ms=self._select_if_due(camera,viewport_height,now)
        camera_frame=camera.frame(self.planet);_,_,forward_local=camera.view_basis_local();forward=camera_frame.local_to_global_direction(forward_local)
        velocity_length=float(np.linalg.norm(velocity));velocity_direction=None if velocity_length<=1e-6 else velocity/velocity_length
        if self.desired!=self._needed_desired:
            self._needed_cache=self._ancestors(self.desired);self._needed_desired=self.desired
        needed=self._needed_cache
        # 粗层祖先必须先驻留，才能在细子块准备期间提供无洞 fallback。
        for key in needed:
            existing=self.records.get(key)
            if existing is not None and existing.state!=PatchState.UNLOADED:
                existing.last_used_frame=self.frame;self.cache_hits+=1;continue
            fallback_bias=(self.selector.max_level-key.level)*1.0e9
            self._request(key,self._priority(key,camera,viewport_height,velocity_direction,forward)+fallback_bias)
        # 运动方向上的下一层子块作为低优先级预取，不进入 desired 集合。
        prefetch:set[PatchKey]=set()
        if velocity_direction is not None:
            ranked=sorted(self.desired,key=lambda k:self._priority(k,camera,viewport_height,velocity_direction,forward),reverse=True)[:2]
            for key in ranked:
                if key.level<self.selector.max_level:
                    for child in key.children():
                        prefetch.add(child);existing=self.records.get(child)
                        if existing is None or existing.state==PatchState.UNLOADED:self._request(child,self._priority(child,camera,viewport_height,velocity_direction,forward)*.25)

        active_requests=needed|prefetch
        for record in self.records.values():
            if record.state==PatchState.REQUESTED and record.key not in active_requests:
                record.state=PatchState.UNLOADED;record.request_version+=1
        if len(self.queue)>self.cache_capacity*2:self._compact_request_queue()

        build_start=time.perf_counter();built=0;scanned=0
        while self.queue and built<self.build_budget and scanned<self.queue_scan_budget:
            scanned+=1
            _,version,key=heapq.heappop(self.queue); record=self.records.get(key)
            if record is None or record.state!=PatchState.REQUESTED or record.request_version!=version: continue
            record.state=PatchState.READY; record.last_used_frame=self.frame; built+=1
        build_ms=(time.perf_counter()-build_start)*1000

        upload_start=time.perf_counter(); uploaded=0
        ready=sorted((r for r in self.records.values() if r.state==PatchState.READY and r.key in active_requests),key=lambda r:r.priority,reverse=True)
        for record in ready:
            if uploaded>=self.upload_budget: break
            if not self.free_slots and not self._evict_one(renderer,needed): break
            slot=self.free_slots.pop(); descriptor=self._descriptor(record.key,camera,viewport_height,record.priority)
            renderer.upload_patch(slot,descriptor); record.slot=slot; record.state=PatchState.GPU_RESIDENT; record.last_used_frame=self.frame
            self.resident.add(record.key); uploaded+=1
        upload_ms=(time.perf_counter()-upload_start)*1000

        coverage=set()
        for face in range(6): self._resolve(PatchKey(face,0,0,0),coverage,needed)
        balance_input=frozenset(coverage)
        if balance_input==self._balanced_input:
            coverage=set(self._balanced_output)
        else:
            self._balance_render_coverage(coverage)
            self._balanced_input=balance_input;self._balanced_output=frozenset(coverage)
        coverage_signature=frozenset(coverage)
        position_key=tuple(float(value) for value in camera.position_global)
        visibility_signature=(coverage_signature,position_key,camera.yaw_degrees,camera.pitch_degrees,camera.vertical_fov_degrees,renderer.width,renderer.height)
        if visibility_signature!=self._visibility_signature:
            visibility_context=self._visibility_context(camera,renderer.width,renderer.height,camera_frame)
            self._visible_cache={key for key in coverage if self._visible(key,visibility_context)}
            self._visibility_signature=visibility_signature
        visible=set(self._visible_cache)
        visible_signature=frozenset(visible)
        if visible_signature!=self._boundary_signature:
            self._boundary_cache={key:self.selector.boundary_masks(key,visible) for key in visible}
            self._boundary_signature=visible_signature
        descriptors=[]
        for key in visible:
            record=self.records[key]; record.last_used_frame=self.frame
            skirt_mask,stitch_mask=self._boundary_cache[key]
            descriptors.append(
                self._descriptor(
                    key,
                    camera,
                    viewport_height,
                    record.priority,
                    skirt_mask,
                    stitch_mask,
                )
            )
        self.render_keys=visible; renderer.set_render_patches(descriptors,{k:self.records[k].slot for k in visible})
        self._trim_record_cache(needed)
        levels=[k.level for k in visible] or [0]; hits=self.cache_hits+self.cache_misses
        self.stats=TerrainDebugStats(len(self.desired),len(self.resident),len(visible),sum(r.state==PatchState.REQUESTED for r in self.records.values()),sum(r.state==PatchState.READY for r in self.records.values()),min(levels),max(levels),selection_ms,build_ms,upload_ms,self.cache_hits/max(hits,1))
        return TerrainFrame(self.desired,frozenset(self.resident),tuple(descriptors),self.stats)

    def _resolve(self,key:PatchKey,out:set[PatchKey],needed:set[PatchKey])->bool:
        if key in self.desired:
            if key in self.resident: out.add(key); return True
            return False
        descendants=key in needed
        if not descendants:
            if key in self.resident: out.add(key); return True
            return False
        child_sets=[]; all_ready=True
        for child in key.children():
            values=set(); ready=self._resolve(child,values,needed); child_sets.append(values); all_ready&=ready
        if all_ready:
            for values in child_sets: out.update(values)
            return True
        if key in self.resident: out.add(key); return True
        return False

    def _visibility_context(self,camera:PlanetCamera,viewport_width:int,viewport_height:int,frame):
        camera_position=camera.position_global;camera_radius=float(np.linalg.norm(camera_position));camera_dir=normalize(camera_position)
        right_local,up_local,forward_local=camera.view_basis_local()
        right=frame.local_to_global_direction(right_local);view_up=frame.local_to_global_direction(up_local);forward=frame.local_to_global_direction(forward_local)
        tan_y=math.tan(math.radians(camera.vertical_fov_degrees)*.5);tan_x=tan_y*viewport_width/viewport_height
        return camera_position,camera_radius,camera_dir,right,view_up,forward,tan_x,tan_y,math.sqrt(1.+tan_x*tan_x),math.sqrt(1.+tan_y*tan_y)

    def _visible(self,key:PatchKey,context)->bool:
        center_world, angular_radius, world_radius = self._patch_bounds(key)
        camera_position,camera_radius,camera_dir,right,view_up,forward,tan_x,tan_y,horizontal_scale,vertical_scale=context
        planet_radius = self.planet.radius_m
        center_dir = center_world / self.planet.radius_m

        # ---------------------------------------------------------
        # 1. Horizon culling
        # ---------------------------------------------------------

        # 相机位于基准球外时才使用解析地平线。
        if camera_radius > planet_radius + 1.0:
            horizon_angle = math.acos(
                np.clip(planet_radius / camera_radius, -1.0, 1.0)
            )

            # 高山可以越过基准球的理论地平线，因此增加保守余量。
            relief_angle = math.acos(
                planet_radius /
                (planet_radius + self.max_terrain_relief_m)
            )
            separation = math.acos(
                float(np.clip(
                    np.dot(center_dir, camera_dir),
                    -1.0,
                    1.0,
                ))
            )
            if separation > (
                horizon_angle
                + angular_radius
                + relief_angle
            ):
                return False
        # ---------------------------------------------------------
        # 2. World-space bounding sphere -> camera view coordinates
        # ---------------------------------------------------------
        delta = center_world - camera_position
        x = float(np.dot(delta, right))
        y = float(np.dot(delta, view_up))
        z = float(np.dot(delta, forward))
        r = world_radius
        # 包围球整个都在相机后方。
        if z + r <= 0.0:
            return False

        # ---------------------------------------------------------
        # 3. 正确构造 horizontal / vertical FOV
        # ---------------------------------------------------------
        # 点到四个视锥平面的未归一化距离需要乘平面法线长度。
        # Right
        if x - z * tan_x > r * horizontal_scale:
            return False
        # Left
        if -x - z * tan_x > r * horizontal_scale:
            return False
        # Top
        if y - z * tan_y > r * vertical_scale:
            return False
        # Bottom
        if -y - z * tan_y > r * vertical_scale:
            return False
        return True

    def _balance_render_coverage(self,coverage:set[PatchKey])->None:
        """异步 fallback 期间也保证相邻实际渲染叶最多相差一级。"""
        while True:
            replacements:set[PatchKey]=set()
            snapshot=tuple(coverage)
            for key in snapshot:
                for edge in range(4):
                    neighbor=self.selector._neighbor(coverage,key,edge)
                    if neighbor is not None and key.level-neighbor.level>1:
                        parent=key.parent()
                        if parent is not None and parent in self.resident:replacements.add(parent)
                        break
            if not replacements:return
            # Apply independent coarsenings together.  If two candidates are
            # nested, keeping only the coarser ancestor avoids redundant work.
            selected:set[PatchKey]=set()
            for parent in sorted(replacements,key=lambda value:value.level):
                if not any(existing.is_ancestor_of(parent) for existing in selected):selected.add(parent)
            for parent in selected:
                coverage.difference_update(key for key in tuple(coverage) if parent.is_ancestor_of(key))
                coverage.add(parent)

    def _evict_one(self,renderer:"PlanetRenderer",needed:set[PatchKey])->bool:
        candidates=[r for r in self.records.values() if r.state==PatchState.GPU_RESIDENT and r.key not in needed and r.key not in self.render_keys]
        if not candidates: return False
        victim=min(candidates,key=lambda r:r.last_used_frame); renderer.release_patch(victim.slot); self.free_slots.append(victim.slot)
        self.resident.discard(victim.key); victim.slot=None; victim.state=PatchState.READY; return True

    def _trim_record_cache(self,needed:set[PatchKey])->None:
        excess=len(self.records)-self.cache_capacity
        if excess<=0:return
        candidates=sorted((r for r in self.records.values() if r.key not in needed and r.state!=PatchState.GPU_RESIDENT),key=lambda r:r.last_used_frame)
        for record in candidates[:excess]: self.records.pop(record.key,None)
        if len(self.queue)>self.cache_capacity*2:self._compact_request_queue()

    def _compact_request_queue(self)->None:
        self.queue=[(-record.priority,record.request_version,record.key) for record in self.records.values() if record.state==PatchState.REQUESTED]
        heapq.heapify(self.queue)
