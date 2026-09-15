"""持久 Mixed-LOD 选择器：SSE、滞回与相邻层级平衡。"""
from __future__ import annotations
import math
import time
import numpy as np

from .camera import PlanetCamera
from .planet import PlanetModel, Vec3d, normalize
from .terrain_types import PatchKey


def cube_face_direction(face: int, u: float, v: float) -> Vec3d:
    return normalize(cube_face_vector(face,u,v))


def cube_face_vector(face:int,u:float,v:float)->Vec3d:
    mappings = ((1.,v,-u),(-1.,v,u),(u,1.,-v),(u,-1.,v),(u,v,1.),(-u,v,-1.))
    return np.asarray(mappings[face],np.float64)


def direction_to_cube_face_uv(direction: Vec3d) -> tuple[int,float,float]:
    x,y,z=direction;ax,ay,az=abs(x),abs(y),abs(z)
    if max(ax,ay,az)<=1e-30:raise ValueError("direction must be non-zero")
    if ax>=ay and ax>=az: return (0,-z/ax,y/ax) if x>=0 else (1,z/ax,y/ax)
    if ay>=ax and ay>=az: return (2,x/ay,-z/ay) if y>=0 else (3,x/ay,z/ay)
    return (4,x/az,y/az) if z>=0 else (5,-x/az,y/az)


def patch_uv_bounds(key: PatchKey) -> tuple[float,float,float,float]:
    size=2.0/(1<<key.level)
    return -1+key.x*size,-1+(key.x+1)*size,-1+key.y*size,-1+(key.y+1)*size


def patch_center_direction(key: PatchKey) -> Vec3d:
    u0,u1,v0,v1=patch_uv_bounds(key)
    return cube_face_direction(key.face,(u0+u1)*.5,(v0+v1)*.5)


class MixedLodSelector:
    def __init__(self, planet: PlanetModel, resolution: int, max_level: int=16,
                 split_sse: float=64., merge_sse: float=32., max_leaves: int=180,max_changes:int=8):
        if merge_sse >= split_sse: raise ValueError("merge_sse 必须小于 split_sse")
        self.planet=planet; self.resolution=resolution; self.max_level=max_level
        self.split_sse=split_sse; self.merge_sse=merge_sse; self.max_leaves=max_leaves
        self.max_changes=max_changes
        self.leaves=frozenset(PatchKey(face,0,0,0) for face in range(6)); self.last_ms=0.
        self.last_changes=max_changes
        self._center_cache:dict[PatchKey,Vec3d]={}

    def center(self,key:PatchKey)->Vec3d:
        value=self._center_cache.get(key)
        if value is None:value=patch_center_direction(key);self._center_cache[key]=value
        return value

    def sse(self,key:PatchKey,camera:PlanetCamera,viewport_height:int)->float:
        center=self.center(key)*self.planet.radius_m
        distance=max(float(np.linalg.norm(center-camera.position_global)),1.)
        focal=viewport_height/(2*math.tan(math.radians(camera.vertical_fov_degrees)*.5))
        geometric_error=self.planet.radius_m*2.4/((1<<key.level)*self.resolution)
        return geometric_error*focal/distance

    def select(self,camera:PlanetCamera,viewport_height:int)->frozenset[PatchKey]:
        started=time.perf_counter();leaves=set(self.leaves);changes=0
        parents={key.parent() for key in leaves if key.level>0};merge=[]
        for parent in parents:
            if parent is not None and all(child in leaves for child in parent.children()):
                error=self.sse(parent,camera,viewport_height)
                if error<self.merge_sse:merge.append((error,parent))
        for _,parent in sorted(merge):
            if changes>=self.max_changes:break
            if all(child in leaves for child in parent.children()):leaves.difference_update(parent.children());leaves.add(parent);changes+=1
        split=sorted(((self.sse(key,camera,viewport_height),key) for key in leaves if key.level<self.max_level),reverse=True)
        for error,key in split:
            if changes>=self.max_changes or error<=self.split_sse or len(leaves)+3>self.max_leaves:break
            if key in leaves:leaves.remove(key);leaves.update(key.children());changes+=1
        self._balance_neighbors(leaves,camera,viewport_height)
        self.leaves=frozenset(leaves);self.last_changes=changes;self.last_ms=(time.perf_counter()-started)*1000
        return self.leaves

    def _leaf_at(self,leaves:set[PatchKey],direction:Vec3d)->PatchKey|None:
        face,u,v=direction_to_cube_face_uv(direction)
        for level in range(self.max_level+1):
            side=1<<level;x=min(max(int((u*.5+.5)*side),0),side-1);y=min(max(int((v*.5+.5)*side),0),side-1);key=PatchKey(face,level,x,y)
            if key in leaves:return key
        return None

    def _neighbor(self,leaves:set[PatchKey],key:PatchKey,edge:int)->PatchKey|None:
        u0,u1,v0,v1=patch_uv_bounds(key); eps=2e-8
        uv=(((u0+u1)*.5,v0-eps),(u1+eps,(v0+v1)*.5),((u0+u1)*.5,v1+eps),(u0-eps,(v0+v1)*.5))[edge]
        return self._leaf_at(leaves,cube_face_vector(key.face,*uv))

    def _balance_neighbors(self,leaves:set[PatchKey],camera:PlanetCamera,viewport_height:int)->None:
        changed=True
        while changed:
            changed=False
            for key in tuple(leaves):
                for edge in range(4):
                    neighbor=self._neighbor(leaves,key,edge)
                    if neighbor is not None and key.level-neighbor.level>1:
                        if len(leaves)+3<=self.max_leaves:
                            leaves.remove(neighbor);leaves.update(neighbor.children())
                        else:
                            parent=key.parent()
                            if parent is None:continue
                            leaves.difference_update(leaf for leaf in tuple(leaves) if parent.is_ancestor_of(leaf));leaves.add(parent)
                        changed=True;break
                if changed: break

    def stitch_mask(
        self, key: PatchKey, render_keys: set[PatchKey],
    ) -> int:
        return self.boundary_masks(key,render_keys)[1]


    def skirt_mask(
        self, key: PatchKey, render_keys: set[PatchKey],
    ) -> int:
        return self.boundary_masks(key,render_keys)[0]

    def boundary_masks(self,key:PatchKey,render_keys:set[PatchKey])->tuple[int,int]:
        skirt=0;stitch=0
        for edge in range(4):
            neighbor=self._neighbor(render_keys,key,edge)
            if neighbor is None:skirt|=1<<edge
            elif neighbor.level==key.level-1:stitch|=1<<edge
        return skirt,stitch
