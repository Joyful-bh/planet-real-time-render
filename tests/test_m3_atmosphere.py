"""M3.1 spherical-atmosphere reference mathematics tests."""

import math

import numpy as np

from planet_renderer.atmosphere import AtmosphereConfig, AtmosphereModel
from planet_renderer.atmosphere.model import ray_sphere_roots


def _model() -> AtmosphereModel:
    return AtmosphereModel(
        50_000.0,
        AtmosphereConfig(
            top_altitude_m=15_000.0,
            rayleigh_scale_height_m=3_000.0,
            mie_scale_height_m=800.0,
            absorption_peak_altitude_m=7_500.0,
            absorption_half_width_m=5_000.0,
        ),
    )


def test_ray_sphere_roots_from_outside_and_inside() -> None:
    outside = ray_sphere_roots(
        np.array([0.0, 70_000.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        65_000.0,
    )
    assert outside is not None
    np.testing.assert_allclose(outside, (5_000.0, 135_000.0), atol=1.0e-8)

    inside = ray_sphere_roots(
        np.array([0.0, 55_000.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        65_000.0,
    )
    assert inside is not None
    np.testing.assert_allclose(inside, (-120_000.0, 10_000.0), atol=1.0e-8)


def test_shell_interval_stops_at_opaque_ground() -> None:
    model = _model()
    interval = model.atmosphere_interval(
        np.array([0.0, 51_000.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    )
    assert interval is not None
    assert interval.ends_at_ground
    assert math.isclose(interval.end_m, 1_000.0, abs_tol=1.0e-8)

    outward = model.atmosphere_interval(
        np.array([0.0, 50_000.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    assert outward is not None
    assert not outward.ends_at_ground
    assert math.isclose(outward.end_m, 15_000.0, abs_tol=1.0e-8)


def test_density_and_transmittance_ranges() -> None:
    model = _model()
    sea_level = model.density(0.0)
    high = model.density(12_000.0)
    assert sea_level[0] > high[0] >= 0.0
    assert sea_level[1] > high[1] >= 0.0

    origin = np.array([0.0, 50_001.0, 0.0])
    transmission = model.transmittance(
        origin,
        np.array([0.0, 1.0, 0.0]),
        steps=64,
    )
    assert np.isfinite(transmission).all()
    assert np.all((0.0 <= transmission) & (transmission <= 1.0))
    assert transmission[2] < transmission[0]

    blocked = model.transmittance(
        origin,
        np.array([0.0, -1.0, 0.0]),
        steps=16,
        ground_is_opaque=True,
    )
    np.testing.assert_array_equal(blocked, np.zeros(3))


def test_single_scattering_is_finite_and_non_negative() -> None:
    model = _model()
    radiance = model.single_scattering(
        np.array([0.0, 50_010.0, 0.0]),
        np.array([0.0, 0.2, 1.0]),
        np.array([0.0, 0.7, 0.7]),
        (4.0, 3.9, 3.7),
        steps=12,
        sun_steps=12,
    )
    assert np.isfinite(radiance).all()
    assert np.all(radiance >= 0.0)
    assert float(np.max(radiance)) > 0.0
