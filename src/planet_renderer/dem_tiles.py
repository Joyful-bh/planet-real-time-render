"""未来 DEM tile pyramid 的异步 I/O 与固定容量 GPU cache 契约。"""
from __future__ import annotations
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


@dataclass(frozen=True,order=True)
class DemTileKey:
    level:int; x:int; y:int


class DemTileState(IntEnum):
    UNLOADED=0; IO_PENDING=1; DECOMPRESSING=2; READY=3; GPU_RESIDENT=4


@dataclass
class DemTile:
    key:DemTileKey; state:DemTileState=DemTileState.UNLOADED
    height_samples:np.ndarray|None=None; gpu_slot:int|None=None


class DemGpuTileCache:
    """只定义预算化提交与 LRU 生命周期；具体格式/I/O 在 DEM 里程碑接入。"""
    def __init__(self,capacity:int=128,uploads_per_frame:int=2):
        self.capacity=capacity;self.uploads_per_frame=uploads_per_frame
        self.tiles:dict[DemTileKey,DemTile]={};self.ready:deque[DemTileKey]=deque();self.lru:OrderedDict[DemTileKey,None]=OrderedDict()

    def mark_ready(self,key:DemTileKey,samples:np.ndarray)->None:
        tile=self.tiles.setdefault(key,DemTile(key));tile.height_samples=np.asarray(samples,np.float32);tile.state=DemTileState.READY;self.ready.append(key)

    def drain_upload_budget(self,upload_callback)->int:
        uploaded=0
        while self.ready and uploaded<self.uploads_per_frame:
            key=self.ready.popleft();tile=self.tiles[key]
            if tile.state!=DemTileState.READY:continue
            while len(self.lru)>=self.capacity:
                victim,_=self.lru.popitem(last=False);self.tiles[victim].state=DemTileState.READY;self.tiles[victim].gpu_slot=None
            tile.gpu_slot=upload_callback(tile);tile.state=DemTileState.GPU_RESIDENT;self.lru[key]=None;uploaded+=1
        return uploaded
