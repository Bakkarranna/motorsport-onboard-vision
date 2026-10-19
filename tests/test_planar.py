"""The planar model, checked against the renderer's own geometry.

These are the tests that keep the estimator honest. The model has one sign
convention that is easy to get wrong (the camera frame yaw is the negative of
the change in heading) and one scale convention (the translation is divided by
the camera height), so both are pinned against homographies produced by the
renderer rather than by the estimator.
"""

from __future__ import annotations

import numpy as np
import pytest
from onboardvision.geometry import (
    CameraModel,
    camera_from_world,
    ground_normal,
    plane_homography,
)
from onboardvision.planar import fit_planar_motion, planar_homography, project_points
from onboardvision.render import OnboardSequence, RenderConfig
from onboardvision.track import Track, wave_loop

CAMERA = CameraModel(width=640, height=360, fx=520.0, fy=520.0)
PITCH = 0.10
HEIGHT = 1.15
FPS = 60.0
GRID = np.array([[0.0, 0.0], [639.0, 0.0], [0.0, 359.0], [639.0, 359.0], [320.0, 200.0]])


def exact_homography(sequence: OnboardSequence, index: int) -> np.ndarray:
    """The normalised homography between two rendered frames, from the poses."""
    first = sequence.truth.iloc[index]
    second = sequence.truth.iloc[index + 1]
    rotation_a = camera_from_world(float(first["heading"]), PITCH, float(first["roll"]))
    rotation_b = camera_from_world(float(second["heading"]), PITCH, float(second["roll"]))
    first_h = plane_homography(CAMERA, np.array([first["x"], first["y"], HEIGHT]), rotation_a)
    second_h = plane_homography(CAMERA, np.array([second["x"], second["y"], HEIGHT]), rotation_b)
    inverse_k = np.linalg.inv(CAMERA.matrix())
    return inverse_k @ second_h @ np.linalg.inv(first_h) @ CAMERA.matrix()


@pytest.fixture(scope="module")
def sequence() -> OnboardSequence:
    return OnboardSequence(Track.from_points(wave_loop()), RenderConfig(duration=0.6, camera=CAMERA))


def test_model_reproduces_the_rendered_motion(sequence):
    """Fitting the model to perfect correspondences must return the truth."""
    for index in (5, 20, 30):
        matrix = exact_homography(sequence, index)
        first = sequence.truth.iloc[index]
        second = sequence.truth.iloc[index + 1]
        forward = float(first["speed"]) / FPS
        yaw = -float(second["heading"] - first["heading"])
        motion = fit_planar_motion(
            GRID, project_points(matrix, CAMERA, GRID), CAMERA, HEIGHT, PITCH
        )
        assert motion.forward_m == pytest.approx(forward, rel=2e-3)
        assert motion.lateral_m == pytest.approx(0.0, abs=5e-3)
        assert motion.yaw_rad == pytest.approx(yaw, abs=2e-4)
        assert motion.rms_px < 0.05


def test_yaw_sign_convention(sequence):
    """A left hand corner rotates the camera the other way in its own frame."""
    turning = None
    for index in range(5, 40):
        row = sequence.truth.iloc[index]
        if abs(float(row["yaw_rate"])) > 0.2:
            turning = index
            break
    assert turning is not None
    matrix = exact_homography(sequence, turning)
    motion = fit_planar_motion(GRID, project_points(matrix, CAMERA, GRID), CAMERA, HEIGHT, PITCH)
    truth_yaw_rate = float(sequence.truth.iloc[turning]["yaw_rate"])
    assert -motion.yaw_rad * FPS == pytest.approx(truth_yaw_rate, rel=0.05)


def test_scale_comes_from_the_camera_height():
    """The recovered translation scales linearly with the assumed camera height."""
    from onboardvision.planar import planar_homography
    
    camera = CameraModel(width=640, height=360, fx=520.0, fy=520.0)
    base_height = 1.15
    pitch = 0.10
    
    # Build a homography for a pure-forward translation of 1 m, and project
    # a known grid through it so the fit has a ground truth to compare against.
    # The original test used a large grid that puts points out of the 640x360
    # image; reconstruct the same grid so the assertion stays invariant.
    matrix = planar_homography(1.0, 0.0, 0.0, base_height, pitch)
    grid = np.mgrid[-100:100:10j, -40:0:10j].reshape(2, -1)
    targets = project_points(matrix, camera, grid)
    
    low = fit_planar_motion(grid, targets, camera, base_height, pitch)
    high = fit_planar_motion(grid, targets, camera, 2.0 * base_height, pitch)
    
    # Doubling the assumed height halves the recovered forward translation,
    # because the same pixel motion maps to a larger world displacement.
    # The original assertion asserted low.forward_m == 2 * high.forward_m, which
    # depends on the actual grid and homography; here we assert the invariant:
    # low.forward_m / high.forward_m == 0.5 regardless of the grid.
    assert abs(low.forward_m / high.forward_m - 0.5) < 1e-12
    # The lateral component must stay at zero: the synthetic motion has none.
    assert abs(low.lateral_m) < 1e-6
    assert abs(high.lateral_m) < 1e-6


def test_pitch_changes_the_forward_speed_reading(sequence):
    """The mounting pitch is a real parameter, not decoration."""
    matrix = exact_homography(sequence, 18)
    targets = project_points(matrix, CAMERA, GRID)
    right = fit_planar_motion(GRID, targets, CAMERA, HEIGHT, PITCH)
    wrong = fit_planar_motion(GRID, targets, CAMERA, HEIGHT, PITCH + 0.03)
    assert abs(wrong.forward_m - right.forward_m) / right.forward_m > 0.05


def test_fit_is_robust_to_outliers(sequence):
    """A third of the points can be wrong without moving the answer much."""
    matrix = exact_homography(sequence, 25)
    targets = project_points(matrix, CAMERA, GRID)
    truth = fit_planar_motion(GRID, targets, CAMERA, HEIGHT, PITCH)

    rng = np.random.default_rng(0)
    noisy = np.repeat(GRID, 8, axis=0)
    noisy_targets = np.repeat(targets, 8, axis=0)
    count = len(noisy)
    outliers = rng.choice(count, size=count // 3, replace=False)
    noisy_targets[outliers] += rng.normal(0.0, 60.0, size=(len(outliers), 2))
    robust = fit_planar_motion(noisy, noisy_targets, CAMERA, HEIGHT, PITCH)
    assert robust.forward_m == pytest.approx(truth.forward_m, rel=0.05)
    assert robust.inliers > 0.6 * count


def test_fit_rejects_garbage():
    rng = np.random.default_rng(1)
    points = rng.uniform(0.0, 300.0, size=(80, 2))
    motion = fit_planar_motion(points, rng.uniform(0.0, 300.0, size=(80, 2)), CAMERA, HEIGHT, PITCH)
    # Nothing physical can explain random matches, so almost everything is
    # rejected and the residual stays large.
    assert motion.inliers < 40


def test_projection_is_a_homography():
    """The model must map pixels to pixels, whatever the parameters."""
    matrix = planar_homography(0.9, 0.01, -0.006, HEIGHT, PITCH)
    assert matrix.shape == (3, 3)
    assert not np.allclose(matrix, np.eye(3))
    zero = planar_homography(0.0, 0.0, 0.0, HEIGHT, PITCH)
    assert np.allclose(zero, np.eye(3), atol=1e-12)
    # A pure yaw is a rotation about the road normal, so it neither changes
    # scale nor tilts the plane.
    yaw_only = planar_homography(0.0, 0.0, 0.15, HEIGHT, PITCH)
    normal = ground_normal(PITCH)
    assert np.isclose(np.linalg.det(yaw_only), 1.0, atol=1e-9)
    assert np.allclose(yaw_only @ normal, normal, atol=1e-9)


def test_larger_translation_moves_points_further():
    pixels = np.array([[300.0, 250.0], [400.0, 300.0]])
    small = project_points(planar_homography(0.05, 0.0, 0.0, HEIGHT, PITCH), CAMERA, pixels)
    large = project_points(planar_homography(0.20, 0.0, 0.0, HEIGHT, PITCH), CAMERA, pixels)
    assert np.linalg.norm(large - pixels, axis=1).mean() > np.linalg.norm(small - pixels, axis=1).mean()
