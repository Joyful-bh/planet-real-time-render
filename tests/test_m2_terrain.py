import numpy as np
import taichi as ti
import time

from planet_renderer.camera import PlanetCamera
from planet_renderer.planet import PlanetModel
from planet_renderer.terrain import (
    CubeSphereTerrain, PatchKey, ProceduralHeightSource, TerrainSettings,
    cube_face_direction, direction_to_cube_face_uv, surface_cell_id,
)
from planet_renderer.renderer import PlanetRenderer
from planet_renderer.lighting import LightingState


def _camera(planet: PlanetModel, altitude: float = 1000.0) -> PlanetCamera:
    return PlanetCamera(planet.surface_position(np.array([0.0, 0.0, 1.0]), altitude), 0.0, -10.0, 60.0)


def test_cube_mapping_round_trip_and_cross_face_height_continuity():
    source = ProceduralHeightSource(seed=19)
    for face in range(6):
        direction = cube_face_direction(face, 0.21, -0.37)
        mapped_face, u, v = direction_to_cube_face_uv(direction)
        assert mapped_face == face
        np.testing.assert_allclose(cube_face_direction(mapped_face, u, v), direction, atol=1e-12)
    a = cube_face_direction(0, -1.0, 0.3)
    b = cube_face_direction(4, 1.0, 0.3)
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert source.sample_height_m(a) == source.sample_height_m(b)


def test_surface_cell_id_is_render_lod_independent():
    direction = cube_face_direction(4, 0.123, -0.456)
    assert surface_cell_id(direction) == surface_cell_id(direction.copy())


def test_mesh_semantics_and_residency_are_stable():
    planet = PlanetModel(6_360_000.0)
    terrain = CubeSphereTerrain(planet, ProceduralHeightSource(seed=3), TerrainSettings(patch_resolution=3,max_level=6,split_sse_pixels=80,merge_sse_pixels=40,max_desired_patches=24,max_gpu_patches=32,build_budget_per_frame=8,upload_budget_per_frame=4))
    class RendererStub:
        def __init__(self): self.uploaded=[]; self.released=[]; self.width=320; self.height=180
        def upload_patch(self,slot,descriptor): self.uploaded.append((slot,descriptor.key))
        def release_patch(self,slot): self.released.append(slot)
        def set_render_patches(self,descriptors,slots): self.render=tuple(d.key for d in descriptors)
    renderer=RendererStub()
    events = []
    class Consumer:
        def on_patch_residency_changed(self, event): events.append(event)
    terrain.add_coverage_consumer(Consumer())
    first = terrain.update(_camera(planet), 180, renderer, now=1.0)
    uploaded_after_first=len(renderer.uploaded)
    descriptor = terrain.describe_surface(cube_face_direction(4, 0.1, -0.2))
    second = terrain.update(_camera(planet), 180, renderer, now=1.01)
    uploaded_after_second=len(renderer.uploaded)
    rotated = _camera(planet); rotated.yaw_degrees = -120.0
    rotated_frame = terrain.update(rotated, 180, renderer, now=1.02)
    assert first.desired == second.desired == rotated_frame.desired
    assert first.desired != first.resident and first.resident != first.render
    assert renderer.uploaded and uploaded_after_first <= 4 and uploaded_after_second-uploaded_after_first <= 4 and len(renderer.uploaded)-uploaded_after_second <= 4 and events
    assert set(renderer.render) <= rotated_frame.resident
    render_set=set(renderer.render)
    for key in render_set:
        for edge in range(4):
            neighbor=terrain.selector._neighbor(render_set,key,edge)
            if neighbor is not None: assert abs(key.level-neighbor.level)<=1
    assert descriptor.cell_id >= 0 and abs(sum(descriptor.material_weights) - 1.0) < 1e-9
    assert all(abs(a.level-b.level)<=1 for a in first.desired for b in first.desired if a.face==b.face and a!=b and (a.x==b.x or a.y==b.y))


def test_patch_children_cover_parent_without_overlap_in_key_space():
    children = PatchKey(2, 3, 4, 5).children()
    assert children == (PatchKey(2, 4, 8, 10), PatchKey(2, 4, 9, 10), PatchKey(2, 4, 8, 11), PatchKey(2, 4, 9, 11))


def test_mixed_lod_is_incremental_and_reaches_high_local_levels():
    planet=PlanetModel(6_360_000.0);camera=_camera(planet,2.0)
    terrain=CubeSphereTerrain(planet,ProceduralHeightSource(),TerrainSettings(max_level=16,max_desired_patches=180,lod_changes_per_update=8))
    previous=terrain.selector.leaves
    for _ in range(30):
        current=terrain.selector.select(camera,720)
        assert len(current.symmetric_difference(previous)) <= 40
        previous=current
    levels={key.level for key in current}
    assert len(levels)>1 and max(levels)>=15
    current_set=set(current)
    for key in current_set:
        for edge in range(4):
            neighbor=terrain.selector._neighbor(current_set,key,edge)
            if neighbor is not None: assert abs(key.level-neighbor.level)<=1


def test_high_speed_cross_scale_streaming_remains_bounded():
    planet=PlanetModel(6_360_000.0)
    camera=_camera(planet,2_000_000.0)
    settings=TerrainSettings(patch_resolution=3,max_level=12,max_desired_patches=60,max_gpu_patches=64,build_budget_per_frame=4,upload_budget_per_frame=2,lod_changes_per_update=6,cache_capacity=128,selection_interval_s=.01)
    terrain=CubeSphereTerrain(planet,ProceduralHeightSource(seed=11),settings)
    class RendererStub:
        width=320;height=180
        def upload_patch(self,slot,descriptor):pass
        def release_patch(self,slot):pass
        def set_render_patches(self,descriptors,slots):pass
    renderer=RendererStub();now=1.0;started=time.perf_counter()
    for frame in range(180):
        now+=.02
        if frame<80:camera.move_local(planet,0.0,-24_000.0,0.0)
        else:camera.move_local(planet,18_000.0,0.0,9_000.0)
        terrain.update(camera,180,renderer,now=now)
        assert len(terrain.tile_manager.queue)<=settings.cache_capacity*2
        leaves=set(terrain.selector.leaves)
        for key in leaves:
            for edge in range(4):
                neighbor=terrain.selector._neighbor(leaves,key,edge)
                if neighbor is not None:assert abs(key.level-neighbor.level)<=1
    assert time.perf_counter()-started<15.0


def test_small_cpu_render_has_finite_gbuffer():
    ti.init(arch=ti.cpu, offline_cache=False)
    planet = PlanetModel(6_360_000.0)
    settings = TerrainSettings(patch_resolution=2,max_level=2,split_sse_pixels=100,merge_sse_pixels=50,max_desired_patches=12,max_gpu_patches=16,build_budget_per_frame=16,upload_budget_per_frame=16)
    terrain = CubeSphereTerrain(planet, ProceduralHeightSource(seed=5), settings)
    direction = np.array([0.0, 0.0, 1.0])
    terrain_height = terrain.describe_surface(direction).height_m
    camera = PlanetCamera(planet.surface_position(direction, terrain_height + 1000.0), 0.0, -30.0, 60.0)
    renderer = PlanetRenderer(64, 48, 16, 2, terrain.height_provider)
    terrain.update(camera, 48, renderer, now=1.0)
    light = LightingState(np.array([0.2, 0.8, 0.4]), 0.266, (4.0, 3.9, 3.7), (80.0, 74.0, 62.0))
    renderer.render(planet, camera, light, (0.16, 0.2, 0.12), 0.0)
    valid=renderer.valid.to_numpy()!=0
    projected=renderer.screen.to_numpy()[valid,:,:2]
    assert np.isfinite(projected).all()
    assert projected[:,:,0].min()>=-1.0/256.0 and projected[:,:,0].max()<=renderer.width+1.0/256.0
    assert projected[:,:,1].min()>=-1.0/256.0 and projected[:,:,1].max()<=renderer.height+1.0/256.0
    render_key = next(iter(renderer._render_slots))
    slot = renderer._render_slots[render_key]
    center_direction = cube_face_direction(render_key.face,
        -1.0 + (render_key.x + 0.5) * 2.0 / (1 << render_key.level),
        -1.0 + (render_key.y + 0.5) * 2.0 / (1 << render_key.level))
    gpu_center_height = float(renderer.height_m.to_numpy()[slot, 4])
    assert abs(gpu_center_height - terrain.height_provider.sample_height_m(center_direction)) < 80.0
    assert np.isfinite(renderer.hdr_numpy()).all()
    assert int(renderer.tile_overflow[None])==0
    weld_count=int(renderer.weld_count[None])
    assert weld_count>0
    weld_dst=renderer.weld_dst.to_numpy()[:weld_count]
    weld_src=renderer.weld_src.to_numpy()[:weld_count]
    view=renderer.view.to_numpy()
    for dst,src in zip(weld_dst,weld_src):
        np.testing.assert_array_equal(view[dst[0],dst[1]],view[src[0],src[1]])
    stitch_count=int(renderer.stitch_count[None])
    stitch=renderer.stitch_vertex.to_numpy()[:stitch_count]
    for slot,vertex,a,b in stitch:
        np.testing.assert_allclose(view[slot,vertex],(view[slot,a]+view[slot,b])*.5,rtol=0,atol=1e-5)
    hits = renderer.gbuffer_surface_id.to_numpy() >= 0
    assert hits.any()
    albedo = renderer.gbuffer_albedo.to_numpy()[hits]
    assert np.isfinite(albedo).all() and np.ptp(albedo, axis=0).max() > 0.01
