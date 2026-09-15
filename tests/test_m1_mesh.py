"""M1 立方体球拓扑的纯 CPU 测试。"""

import unittest

import numpy as np

from planet_renderer.renderer import create_cube_sphere


class CubeSphereTests(unittest.TestCase):
    def test_counts_and_unit_radius(self) -> None:
        subdivisions = 8
        vertices, triangles = create_cube_sphere(subdivisions)
        self.assertEqual(vertices.shape, (6 * (subdivisions + 1) ** 2, 3))
        self.assertEqual(triangles.shape, (6 * 2 * subdivisions**2, 3))
        np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1.0, atol=1.0e-6)

    def test_indices_are_in_bounds_and_triangles_non_degenerate(self) -> None:
        vertices, triangles = create_cube_sphere(6)
        self.assertGreaterEqual(int(triangles.min()), 0)
        self.assertLess(int(triangles.max()), len(vertices))
        p0, p1, p2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
        area = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        self.assertTrue(np.all(area > 1.0e-8))


if __name__ == "__main__":
    unittest.main()
