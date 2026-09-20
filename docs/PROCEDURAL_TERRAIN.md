# Procedural terrain authoring

The default hierarchical landform implementation lives in
`src/planet_renderer/landforms_terrain.py`. The smaller reference FBM model
remains in `src/planet_renderer/fbm_terrain.py`. Terrain selection, residency
and rasterization must not contain their own copies of either algorithm.

## Data contract

The terrain model accepts a normalized direction in global planet space and
returns radial height in metres relative to the planet reference radius. It
does not receive camera-relative positions, patch IDs or renderer state. This
keeps the generated world deterministic across LOD levels and floating-origin
revisions.

Each model also reports a conservative `height_range_m`. This is output
metadata required for safe patch and horizon bounds, not an algorithm
parameter schema; the streaming system never guesses a generator's relief.

`TerrainHeightModel` in `height.py` is the renderer-independent interface.
`LandformsTerrainGenerator` and `FbmTerrainGenerator` are registered
implementations. `DemTerrainModel` is only a future data-source boundary and
deliberately cannot enter the GPU renderer until its sampling path exists.

In addition to height, a procedural model returns normalized semantic weights.
`procedural_landforms_v1` uses ocean, plain, mountain, plateau, basin and
canyon weights. It also reports an unresolved local-relief estimate for each
LOD level; this is consumed by screen-space error selection and patch request
priority, not by the rasterizer.

## Configuration envelope

Files and application entry points use a thin `TerrainConfig`. The factory is
the only code that interprets `generator`; it passes `params` unchanged to the
selected algorithm's private, strongly typed configuration class.

```python
terrain_config = TerrainConfig(
    generator="procedural_landforms_v1",
    params={
        "seed": 17,
        "ocean_threshold": 0.52,
        "mountain_amplitude_m": 3500.0,
        "canyon_depth_m": 750.0,
    },
)
height_model = create_terrain_model(terrain_config)
```

The envelope does not define noise layers, continents, mountains, craters or
any other terrain concept. A new algorithm supplies its own config class and
generator, then adds one explicit Registry entry.

## Configuration migration

The main renderer configuration is version 3. The former version-1 field:

```json
"terrain": {"seed": 7}
```

must become:

```json
"config_version": 3,
"terrain": {
  "generator": "procedural_fbm_v1",
  "params": {"seed": 7}
}
```

Old terrain fields are not interpreted as an implicit compatibility path.
Version 3 also removes the independently configurable
`lighting.sun_disk_radiance` and moves `rendering.exposure_ev` into the
`postprocess` group. Solar-disk radiance is derived from irradiance and angular
radius by `LightingState`.

## Editing a terrain algorithm

For ordinary landform art-direction changes, edit `LandformsTerrainConfig` or
change the envelope's `params`. The factory performs the typed conversion:

```python
algorithm_config = LandformsTerrainConfig(
    seed=17,
    ocean_threshold=0.52,
    mountain_amplitude_m=3500.0,
    canyon_depth_m=750.0,
)
height_model = LandformsTerrainGenerator(algorithm_config)
```

For formula changes, edit the adjacent CPU/GPU methods on that generator. An
unrelated algorithm belongs in a separate generator class with its own config:

- `sample_height_m()` is the CPU reference used by queries and tests.
- `sample_terrain_m()` additionally produces semantic landform weights.
- `estimate_error_m()` reports unresolved relief to terrain-aware SSE.
- `sample_height_gpu()` is the Taichi implementation used to generate rendered
  patch vertices.
- `sample_terrain_gpu()` additionally produces weights consumed by surface
  material generation.

Taichi device functions cannot call arbitrary Python code, so the two small
entry points remain separate. They share one config and must preserve the same
mathematical result. Helper functions should stay inside this class, using
matching `_..._cpu` and `_..._gpu` names when both execution paths need them.

The renderer receives the model explicitly and calls only its GPU sampling
contract; it owns no procedural noise parameters or equations. The formal
surface view maps semantic weights plus slope and height to material weights.
The height color-band view is a separate debug mode and cannot affect formal
surface shading.

`tools/render_landform_checks.py` renders the standard space, 20 km, 2 km and
near-surface viewpoints in one Taichi process after bounded LOD settling.
