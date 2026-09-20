# Atmosphere diagnostic baseline

The preview exposes diagnostic views that isolate the inputs to final
atmosphere composition.  They do not change atmosphere parameters or rebuild
LUTs with a different algorithm.

## Views

- `Atmo: Composite`: normal surface, sky, sun and aerial-perspective result.
- `Atmo: Sky-view`: reconstructed Sky-View radiance over the entire camera
  view.  Opaque geometry is intentionally ignored so the LUT can be inspected
  behind the terrain silhouette.
- `Atmo: Camera T`: RGB transmittance from the camera along each view ray.
  Camera altitude is represented independently from the planet radius and a
  single stable shell interval supplies the ground-hit decision. Isolated
  black pixels or repeated black arcs in the sky are therefore correctness
  failures, not expected integration noise.
- `Atmo: Transmittance LUT`: raw 2D texture.  Horizontal position reconstructs
  path distance to the atmosphere top; vertical position is normalized radial
  distance from the planet-radius cylinder. Ground-blocked rays are rejected
  analytically and are therefore not stored in the texture.
- `Atmo: Multi-scattering LUT`: raw 2D higher-order approximation.  It uses the
  horizon-centred solar-direction and square-root-altitude axes; it does not
  share the Transmittance LUT parameterization.
- `Atmo: Aerial scattering`: in-scattered radiance sampled at each visible
  surface depth. It shows the hybrid result: coarse froxel scattering in smooth
  regions and direct integration in the configured limb band. Background pixels
  are black.
- `Atmo: Aerial transmittance`: full-resolution surface-to-camera RGB
  transmittance reconstructed from the actual G-buffer endpoint with the 2-D
  Transmittance LUT. It no longer displays a 3-D aerial-volume channel;
  background pixels are black.
- `Debug: Surface mask`: white for a G-buffer surface hit and black for the
  background.  This is the authoritative view for deciding whether a moving
  horizon comes from terrain geometry or atmosphere composition.

Radiance views pass through the current exposure and display transform.
Transmittance, raw Transmittance LUT and the binary mask are displayed directly
in `[0, 1]`, so exposure cannot hide their boundaries.

## Freeze controls

- `Freeze atmosphere LUTs` retains the current Sky-View and
  Aerial-Perspective LUTs while camera motion and composition continue.  Static
  Transmittance and Multi-Scattering LUTs remain unchanged as usual.  Resuming
  explicitly invalidates Sky-View before the next render.
- `Freeze terrain LOD` stops submitting new terrain selection/streaming work.
  An update already running when the button is pressed is allowed to finish
  and is applied once; dropping that incremental result would desynchronize GPU
  Patch slots from the tile manager.
- `Freeze camera input` ignores mouse-look and movement keys.  Altitude preset
  buttons remain deliberate controls and can still reposition the camera.

The panel reports live/frozen state, the altitude captured by the Sky-View LUT,
camera yaw/pitch and LUT rebuild counters.

## Reproducible checks

### Whole-horizon jumps

1. Select `Debug: Surface mask` and freeze terrain LOD.
2. Move vertically through the altitude that showed the jump.
3. If the white/black boundary is continuous, the geometry is not responsible.
4. Switch to `Atmo: Sky-view`, first live and then frozen.  A jump present only
   with live LUTs belongs to Sky-View invalidation or parameterization.

### Horizontal dark bands

1. Freeze terrain LOD and use `Atmo: Camera T` while moving toward and away
   from the sun.
2. Inspect the raw Transmittance LUT for a matching sharp boundary.
3. Compare Sky-View with Multi-Scattering LUT.  A discontinuity already in
   transmittance is a shadow-boundary mapping problem; a smooth transmittance
   field with a band only in Sky-View points to view-ray integration.

### Dark daytime sky

Compare `Sky-view`, `Multi-scattering LUT` and `Composite` without changing
exposure.  Save the exact altitude, yaw/pitch, exposure and LUT rebuild counts
shown by the panel with each screenshot.

### Terminator blocks in aerial scattering

If `Atmo: Aerial transmittance` is smooth but `Atmo: Aerial scattering` shows
segmented light near the dawn/dusk limb, the camera path is valid and the
remaining error belongs to the scattering source quadrature. The direct limb
integrator keeps a modest altitude-warped base count and locally uses
`aerial_terminator_substeps` around the solar-shadow crossing. Raising
`aerial_horizon_raymarch_steps` globally is useful only as an isolation test:
blocks becoming smaller without disappearing identifies deterministic
quadrature aliasing and is not the production fix.

### Near-ground black arcs and dawn/dusk sky bands

Use altitudes `0.1, 0.25, 0.5, 1, 2, 5, 10` metres with fixed yaw/pitch. In
`Atmo: Camera T`, the sky side of the analytic horizon must contain no isolated
zeros and the boundary must move monotonically. Background atmosphere in final
Composite and `Atmo: Sky-view` samples the same Sky-View radiance. If a band is
present only in Composite, isolate the remaining Composite-only inputs in this
order: surface aerial perspective, camera transmittance, finite sun disk and
Bloom. Do not add a second fixed-step sky integrator to hide a smooth LUT.
