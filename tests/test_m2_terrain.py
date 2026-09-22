import time

import numpy as np
import pytest
import taichi as ti

from planet_renderer.atmosphere import AtmosphereConfig, AtmosphereDiagnosticView
from planet_renderer.camera import PlanetCamera
from planet_renderer.lighting import LightingState
from planet_renderer.ocean import OCEAN_SURFACE_ID
from planet_renderer.planet import PlanetModel
from planet_renderer.postprocess import PostprocessConfig
from planet_renderer.renderer import PlanetRenderer
from planet_renderer.terrain import (CubeSphereTerrain, PatchKey,
                                     TerrainSettings, cube_face_direction,
                                     direction_to_cube_face_uv,
                                     surface_cell_id)
from planet_renderer.terrain_config import TerrainConfig
from planet_renderer.terrain_factory import create_terrain_model


def _camera(planet: PlanetModel, altitude: float = 1000.0) -> PlanetCamera:
    return PlanetCamera(
        planet.surface_position(np.array([0.0, 0.0, 1.0]), altitude), 0.0, -10.0, 60.0
    )


def _terrain_model(seed: int = 7):
    return create_terrain_model(
        TerrainConfig("procedural_fbm_v1", {"seed": seed})
    )


def test_terrain_config_envelope_is_resolved_by_registry():
    model = create_terrain_model(
        TerrainConfig(
            generator="procedural_fbm_v1",
            params={"seed": 23, "ridge_power": 4.0},
        )
    )
    assert model.config.seed == 23
    assert model.config.ridge_power == 4.0

    with pytest.raises(ValueError, match="unknown terrain generator"):
        create_terrain_model(TerrainConfig("missing_generator", {}))
    with pytest.raises(ValueError, match="invalid parameters"):
        create_terrain_model(
            TerrainConfig("procedural_fbm_v1", {"unsupported_parameter": 1})
        )


def test_landforms_generator_exposes_normalized_regions_and_error_estimate():
    model = create_terrain_model(
        TerrainConfig("procedural_landforms_v1", {"seed": 17})
    )
    samples = [
        model.sample_terrain_m(cube_face_direction(face, 0.23, -0.41))
        for face in range(6)
    ]
    assert all(len(sample) == 7 for sample in samples)
    assert all(abs(sum(sample[1:]) - 1.0) < 1.0e-12 for sample in samples)
    assert max(sample[0] for sample in samples) > min(sample[0] for sample in samples)

    direction = cube_face_direction(0, 0.17, -0.29)
    coarse = model.estimate_error_m(direction, 2, 24, 50_000.0)
    fine = model.estimate_error_m(direction, 10, 24, 50_000.0)
    assert coarse > fine > 0.0


def test_cube_mapping_round_trip_and_cross_face_height_continuity():
    source = _terrain_model(seed=19)
    for face in range(6):
        direction = cube_face_direction(face, 0.21, -0.37)
        mapped_face, u, v = direction_to_cube_face_uv(direction)
        assert mapped_face == face
        np.testing.assert_allclose(
            cube_face_direction(mapped_face, u, v), direction, atol=1e-12
        )
    a = cube_face_direction(0, -1.0, 0.3)
    b = cube_face_direction(4, 1.0, 0.3)
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert source.sample_height_m(a) == source.sample_height_m(b)


def test_surface_cell_id_is_render_lod_independent():
    direction = cube_face_direction(4, 0.123, -0.456)
    assert surface_cell_id(direction) == surface_cell_id(direction.copy())


def test_terrain_frame_is_renderer_independent():
    planet = PlanetModel(6_360_000.0)
    terrain = CubeSphereTerrain(
        planet,
        _terrain_model(seed=8),
        TerrainSettings(
            patch_resolution=2,
            max_level=2,
            max_desired_patches=12,
            max_gpu_patches=16,
            build_budget_per_frame=4,
            upload_budget_per_frame=2,
        ),
    )

    frame = terrain.update(_camera(planet), 320, 180, now=1.0)

    assert frame.desired
    assert frame.uploads
    assert all(request.slot >= 0 for request in frame.uploads)
    assert all(request.descriptor.face in range(6) for request in frame.uploads)
    render_slot_map = dict(frame.render_slots)
    assert all(descriptor.key in render_slot_map for descriptor in frame.render)
    assert all(slot >= 0 for slot in render_slot_map.values())


def test_patch_children_cover_parent_without_overlap_in_key_space():
    children = PatchKey(2, 3, 4, 5).children()
    assert children == (
        PatchKey(2, 4, 8, 10),
        PatchKey(2, 4, 9, 10),
        PatchKey(2, 4, 8, 11),
        PatchKey(2, 4, 9, 11),
    )


def test_mixed_lod_is_incremental_and_reaches_high_local_levels():
    planet = PlanetModel(6_360_000.0)
    camera = _camera(planet, 2.0)
    terrain = CubeSphereTerrain(
        planet,
        _terrain_model(),
        TerrainSettings(
            max_level=16, max_desired_patches=180, lod_changes_per_update=8
        ),
    )
    previous = terrain.selector.leaves
    for _ in range(30):
        current = terrain.selector.select(camera, 720)
        assert len(current.symmetric_difference(previous)) <= 40
        previous = current
    levels = {key.level for key in current}
    assert len(levels) > 1 and max(levels) >= 15
    current_set = set(current)
    leaf_index = terrain.selector._build_leaf_index(current_set)
    for key in current_set:
        for edge in range(4):
            neighbor = terrain.selector._neighbor(current_set, key, edge)
            assert (
                terrain.selector._neighbor_indexed(leaf_index, key, edge)
                == neighbor
            )
            if neighbor is not None:
                assert abs(key.level - neighbor.level) <= 1


def test_high_speed_cross_scale_streaming_remains_bounded():
    planet = PlanetModel(6_360_000.0)
    camera = _camera(planet, 2_000_000.0)
    settings = TerrainSettings(
        patch_resolution=3,
        max_level=12,
        max_desired_patches=60,
        max_gpu_patches=64,
        build_budget_per_frame=4,
        upload_budget_per_frame=2,
        lod_changes_per_update=6,
        cache_capacity=128,
        selection_interval_s=0.01,
    )
    terrain = CubeSphereTerrain(planet, _terrain_model(seed=11), settings)

    now = 1.0
    started = time.perf_counter()
    for frame in range(180):
        now += 0.02
        if frame < 80:
            camera.move_local(planet, 0.0, -24_000.0, 0.0)
        else:
            camera.move_local(planet, 18_000.0, 0.0, 9_000.0)
        frame_result = terrain.update(camera, 320, 180, now=now)
        assert frame_result.render
        assert len(terrain.tile_manager.queue) <= settings.cache_capacity * 2
        leaves = set(terrain.selector.leaves)
        for key in leaves:
            for edge in range(4):
                neighbor = terrain.selector._neighbor(leaves, key, edge)
                if neighbor is not None:
                    assert abs(key.level - neighbor.level) <= 1
    assert time.perf_counter() - started < 15.0


def test_small_cpu_render_has_finite_gbuffer():
    ti.init(arch=ti.cpu, offline_cache=False)
    planet = PlanetModel(6_360_000.0)
    settings = TerrainSettings(
        patch_resolution=2,
        max_level=2,
        split_sse_pixels=100,
        merge_sse_pixels=50,
        max_desired_patches=12,
        max_gpu_patches=16,
        build_budget_per_frame=16,
        upload_budget_per_frame=16,
    )
    terrain = CubeSphereTerrain(planet, _terrain_model(seed=5), settings)
    direction = np.array([0.0, 0.0, 1.0])
    terrain_height = terrain.describe_surface(direction).height_m
    camera = PlanetCamera(
        planet.surface_position(direction, terrain_height + 1000.0), 0.0, -30.0, 60.0
    )
    renderer = PlanetRenderer(
        64,
        48,
        terrain.height_model,
        AtmosphereConfig(
            transmittance_lut_width=16,
            transmittance_lut_height=16,
            multi_scattering_lut_width=8,
            multi_scattering_lut_height=8,
            sky_view_lut_width=16,
            sky_view_lut_height=16,
            aerial_lut_width=4,
            aerial_lut_height=4,
            aerial_lut_depth=8,
            transmittance_steps=4,
            multi_scattering_directions=4,
            multi_scattering_steps=4,
            sky_view_steps=4,
            aerial_steps_per_slice=1,
        ),
        PostprocessConfig(bloom_passes=1),
        16,
        2,
    )
    terrain_frame = terrain.update(camera, 64, 48, now=1.0)
    renderer.apply_terrain_frame(terrain_frame)
    resident_normals = renderer.terrain_renderer.normal.to_numpy().copy()
    resident_heights = renderer.terrain_renderer.height_m.to_numpy().copy()
    resident_materials = renderer.terrain_renderer.material.to_numpy().copy()
    resident_cells = renderer.terrain_renderer.cell.to_numpy().copy()
    light = LightingState(
        np.array([0.2, 0.8, 0.4]), 0.266, (4.0, 3.9, 3.7)
    )
    renderer.warmup_raster_paths(
        planet,
        camera,
        light,
        (0.16, 0.2, 0.12),
        0.0,
    )
    renderer.render(planet, camera, light, (0.16, 0.2, 0.12), 0.0)
    transmittance = renderer.atmosphere_renderer.transmittance_lut.to_numpy()
    sky_view = renderer.atmosphere_renderer.sky_view_lut.to_numpy()
    multiple_scattering = (
        renderer.atmosphere_renderer.multi_scattering_lut.to_numpy()
    )
    aerial_scattering = (
        renderer.atmosphere_renderer.aerial_scattering_lut.to_numpy()
    )
    assert np.isfinite(transmittance).all()
    assert np.all((0.0 <= transmittance) & (transmittance <= 1.0))
    assert np.isfinite(sky_view).all()
    assert np.all(sky_view >= 0.0)
    assert float(np.max(sky_view)) > 0.0
    assert np.isfinite(multiple_scattering).all()
    assert np.all(multiple_scattering >= 0.0)
    assert float(np.max(multiple_scattering)) > 0.0
    assert np.isfinite(aerial_scattering).all()
    assert np.all(aerial_scattering >= 0.0)
    assert float(np.max(aerial_scattering)) > 0.0
    assert np.isfinite(renderer.hdr.to_numpy()).all()
    clipped_count = min(int(renderer.clipped_count[None]), renderer.raster_capacity)
    projected = renderer.screen.to_numpy()[:clipped_count, :, :2]
    assert np.isfinite(projected).all()
    assert (
        projected[:, :, 0].min() >= -1.0 / 256.0
        and projected[:, :, 0].max() <= renderer.width + 1.0 / 256.0
    )
    assert (
        projected[:, :, 1].min() >= -1.0 / 256.0
        and projected[:, :, 1].max() <= renderer.height + 1.0 / 256.0
    )
    render_key = next(iter(renderer._render_slots))
    slot = renderer._render_slots[render_key]
    center_direction = cube_face_direction(
        render_key.face,
        -1.0 + (render_key.x + 0.5) * 2.0 / (1 << render_key.level),
        -1.0 + (render_key.y + 0.5) * 2.0 / (1 << render_key.level),
    )
    gpu_center_height = float(renderer.terrain_renderer.height_m.to_numpy()[slot, 4])
    assert (
        abs(
            gpu_center_height
            - terrain.height_model.sample_height_m(center_direction)
        )
        < 80.0
    )
    assert np.isfinite(renderer.hdr_numpy()).all()
    assert int(renderer.tile_overflow[None]) == 0
    weld_count = int(renderer.weld_count[None])
    assert weld_count > 0
    edge_operations = renderer.edge_operations.to_numpy()
    weld_dst = edge_operations[:weld_count, :2]
    weld_src = edge_operations[:weld_count, 2:]
    view = renderer.view.to_numpy()
    frame_normals = renderer.frame_normal.to_numpy()
    for dst, src in zip(weld_dst, weld_src):
        np.testing.assert_array_equal(view[dst[0], dst[1]], view[src[0], src[1]])
        np.testing.assert_allclose(
            frame_normals[dst[0], dst[1]],
            frame_normals[src[0], src[1]],
            rtol=0,
            atol=1e-6,
        )
    # Welding and stitching are render-frame operations.  They must not
    # corrupt the generated attributes kept in resident patch slots.
    np.testing.assert_array_equal(
        renderer.terrain_renderer.normal.to_numpy(), resident_normals
    )
    np.testing.assert_array_equal(
        renderer.terrain_renderer.height_m.to_numpy(), resident_heights
    )
    np.testing.assert_array_equal(
        renderer.terrain_renderer.material.to_numpy(), resident_materials
    )
    np.testing.assert_array_equal(
        renderer.terrain_renderer.cell.to_numpy(), resident_cells
    )
    stitch_count = int(renderer.stitch_count[None])
    stitch = edge_operations[weld_count : weld_count + stitch_count]
    for slot, vertex, a, b in stitch:
        np.testing.assert_allclose(
            view[slot, vertex], (view[slot, a] + view[slot, b]) * 0.5, rtol=0, atol=1e-5
        )
    surface_ids = renderer.gbuffer_surface_id.to_numpy()
    hits = surface_ids >= 0
    assert hits.any()
    albedo = renderer.gbuffer_albedo.to_numpy()[hits]
    assert np.isfinite(albedo).all()
    ocean_hits = surface_ids == OCEAN_SURFACE_ID
    assert ocean_hits.any()
    ocean_heights = renderer.gbuffer_height_m.to_numpy()[ocean_hits]
    assert np.isfinite(ocean_heights).all()
    assert np.all(
        np.abs(ocean_heights)
        <= renderer.ocean_renderer.max_geometry_displacement_m + 0.05
    )
    ocean_depth = renderer.gbuffer_water_depth_m.to_numpy()[ocean_hits]
    assert np.isfinite(ocean_depth).all()
    assert np.all(ocean_depth >= 0.0)
    assert np.all(ocean_depth <= renderer.ocean_renderer.config.max_visible_depth_m)
    ocean_variance = renderer.gbuffer_ocean_slope_variance.to_numpy()[ocean_hits]
    assert np.isfinite(ocean_variance).all()
    assert np.all(ocean_variance >= 0.0)
    seabed_normals = renderer.gbuffer_seabed_normal.to_numpy()[ocean_hits]
    assert np.isfinite(seabed_normals).all()
    assert np.allclose(np.linalg.norm(seabed_normals, axis=1), 1.0, atol=2.0e-3)
    terrain_hits = hits & ~ocean_hits
    if terrain_hits.any():
        terrain_albedo = renderer.gbuffer_albedo.to_numpy()[terrain_hits]
        assert np.ptp(terrain_albedo, axis=0).max() > 0.01

    atmosphere = renderer.atmosphere_renderer
    atmosphere.set_luts_frozen(True)
    frozen_rebuilds = (
        atmosphere.sky_view_rebuilds,
        atmosphere.aerial_rebuilds,
    )
    # Two metres is well below the removed top_altitude / 512 bucket while
    # remaining representable at Earth radius in f32.  Frozen resources must
    # retain their snapshot, then immediately rebuild once resumed.
    camera.position_global += (
        camera.position_global / np.linalg.norm(camera.position_global) * 2.0
    )
    for diagnostic_view in AtmosphereDiagnosticView:
        renderer.atmosphere_diagnostic_view = diagnostic_view
        renderer.render(planet, camera, light, (0.16, 0.2, 0.12), 0.0)
        diagnostic = renderer.display.to_numpy()
        assert np.isfinite(diagnostic).all()
        assert np.all((0.0 <= diagnostic) & (diagnostic <= 1.0))
    assert frozen_rebuilds == (
        atmosphere.sky_view_rebuilds,
        atmosphere.aerial_rebuilds,
    )
    surface_mask = renderer.display.to_numpy()
    assert set(np.unique(surface_mask)).issubset({0.0, 1.0})

    atmosphere.set_luts_frozen(False)
    renderer.atmosphere_diagnostic_view = AtmosphereDiagnosticView.COMPOSITE
    renderer.render(planet, camera, light, (0.16, 0.2, 0.12), 0.0)
    assert atmosphere.sky_view_rebuilds > frozen_rebuilds[0]
    assert atmosphere.aerial_rebuilds > frozen_rebuilds[1]

    live_sky_rebuilds = atmosphere.sky_view_rebuilds
    camera.position_global += (
        camera.position_global / np.linalg.norm(camera.position_global) * 2.0
    )
    renderer.render(planet, camera, light, (0.16, 0.2, 0.12), 0.0)
    assert atmosphere.sky_view_rebuilds == live_sky_rebuilds + 1
