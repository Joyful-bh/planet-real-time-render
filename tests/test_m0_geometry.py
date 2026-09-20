"""M0 行星坐标和精度边界测试。"""

import math
import unittest

import numpy as np

from planet_renderer.camera import PlanetCamera
from planet_renderer.lighting import LightingState, StaticLightingProvider
from planet_renderer.planet import FloatingOrigin, PlanetModel


class PlanetGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planet = PlanetModel(6_360_000.0)

    def test_altitude_and_horizon(self) -> None:
        position = self.planet.surface_position(np.array([0.0, 0.0, 1.0]), 2_000.0)
        self.assertAlmostEqual(self.planet.altitude_m(position), 2_000.0, places=6)
        expected = math.sqrt(2_000.0 * (2.0 * self.planet.radius_m + 2_000.0))
        self.assertAlmostEqual(
            self.planet.horizon_distance_m(2_000.0), expected, places=6
        )

    def test_local_frame_round_trip(self) -> None:
        position = self.planet.surface_position(np.array([0.3, 0.5, 0.8]), 10.0)
        frame = self.planet.local_frame(position)
        local = np.array([0.2, -0.4, 0.7])
        restored = frame.global_to_local_direction(
            frame.local_to_global_direction(local)
        )
        np.testing.assert_allclose(restored, local, atol=1.0e-12)

    def test_floating_origin_preserves_small_offsets(self) -> None:
        camera = np.array([2.0e6, 4.0e6, 5.0e6], dtype=np.float64)
        origin = FloatingOrigin(camera.copy(), 1_000.0)
        point = camera + np.array([0.125, -0.25, 0.5])
        np.testing.assert_allclose(
            origin.relative_f32(point), [0.125, -0.25, 0.5], atol=1.0e-7
        )
        self.assertTrue(origin.update(camera + np.array([2_000.0, 0.0, 0.0])))
        self.assertEqual(origin.revision, 1)

    def test_camera_never_enters_planet(self) -> None:
        position = self.planet.surface_position(np.array([0.0, 0.0, 1.0]), 2.0)
        camera = PlanetCamera(position)
        camera.move_local(self.planet, 0.0, -100.0, 0.0)
        self.assertGreaterEqual(self.planet.altitude_m(camera.position_global), 0.49)

    def test_camera_tangent_motion_stays_on_one_great_circle(self) -> None:
        position = self.planet.surface_position(np.array([0.0, 0.0, 1.0]), 1000.0)
        camera = PlanetCamera(position, yaw_degrees=45.0, pitch_degrees=0.0)
        initial_radius = np.linalg.norm(camera.position_global)
        frame = self.planet.local_frame(camera.position_global)
        _, _, forward_local = camera.view_basis_local()
        tangent = frame.local_to_global_direction(
            np.array([forward_local[0], 0.0, forward_local[2]])
        )
        orbit_normal = np.cross(
            camera.position_global / initial_radius,
            tangent / np.linalg.norm(tangent),
        )

        for _ in range(200):
            yaw = math.radians(camera.yaw_degrees)
            camera.move_local(
                self.planet,
                1000.0 * math.sin(yaw),
                0.0,
                1000.0 * math.cos(yaw),
            )

        final_radius = np.linalg.norm(camera.position_global)
        self.assertAlmostEqual(final_radius, initial_radius, places=6)
        self.assertAlmostEqual(
            float(np.dot(orbit_normal, camera.position_global / final_radius)),
            0.0,
            places=10,
        )

    def test_static_lighting_snapshot_identity(self) -> None:
        state = LightingState(
            np.array([1.0, 2.0, 3.0]), 0.266, (1.0, 1.0, 1.0)
        )
        self.assertIs(StaticLightingProvider(state).snapshot(), state)

    def test_solar_disk_radiance_integrates_to_irradiance(self) -> None:
        state = LightingState(
            np.array([0.0, 1.0, 0.0]),
            0.266,
            (4.0, 3.9, 3.7),
        )
        integrated = (
            np.asarray(state.sun_disk_radiance)
            * state.sun_projected_solid_angle_sr
        )
        np.testing.assert_allclose(integrated, state.solar_irradiance, rtol=1.0e-12)

        smaller_disk = LightingState(
            np.array([0.0, 1.0, 0.0]),
            0.133,
            state.solar_irradiance,
        )
        self.assertGreater(
            smaller_disk.sun_disk_radiance[0],
            state.sun_disk_radiance[0],
        )


if __name__ == "__main__":
    unittest.main()
