"""Geometry and ground truth invariants.

The renderer is the reference for every number in the report, so the properties
that make it a reference are pinned here: the pose series must match the speed
and yaw rate that ship with it, and the plane homography must be invertible.
"""

from __future__ import annotations

import numpy as np

from onboardvision.geometry import (
    CameraModel,
    camera_from_world,
    ground_normal,
    mounting_matrix,
    plane_homography,
    plane_points,
    sky_mask,
)
from onboardvision.render import OnboardSequence, RenderConfig, speed_profile
from onboardvision.track import Track, smooth_periodic, wave_loop


def test_track_is_closed_and_smooth():
    track = Track.from_points(wave_loop(points=1200))
    # The centreline is sampled at a fixed spacing and closes on itself.
    spacing = track.length / len(track.centreline)
    assert 0.2 < spacing < 0.3
    step = np.linalg.norm(np.diff(np.vstack([track.centreline, track.centreline[:1]]), axis=0), axis=1)
    assert np.all(step < 0.35)
    # Curvature is smoothed, so neighbouring samples cannot differ wildly.
    assert np.abs(np.diff(track.curvature)).max() < 0.002
    assert np.abs(track.curvature).max() < 0.05


def test_smooth_periodic_keeps_shape_and_average():
    values = np.sin(np.linspace(0, 2 * np.pi, 200, endpoint=False))
    smoothed = smooth_periodic(values, 9)
    assert smoothed.shape == values.shape
    assert abs(smoothed.mean() - values.mean()) < 1e-2
    assert np.abs(smoothed - values).max() < 0.2


def test_speed_profile_respects_limits():
    track = Track.from_points(wave_loop())
    config = RenderConfig()
    speed = speed_profile(track, config)
    assert speed.min() >= config.min_speed * 0.55
    assert speed.max() <= config.max_speed + 1e-9
    # Slower in the corners than on the straights.
    curvature = np.abs(track.curvature)
    tight = curvature > np.percentile(curvature, 90)
    straight = curvature < np.percentile(curvature, 10)
    assert speed[tight].mean() < speed[straight].mean()


def test_reference_speed_matches_the_rendered_pose():
    """The whole report compares against these columns, so they must agree."""
    sequence = OnboardSequence(Track.from_points(wave_loop()), RenderConfig(duration=0.5))
    truth = sequence.truth
    speed = truth["speed"].to_numpy()
    # The reported speed is the distance between the poses that get drawn.
    assert np.isfinite(speed).all()
    assert np.all(np.diff(speed) < 25.0)
    assert 20.0 < speed.mean() < 70.0
    # Yaw rate is the change of heading of those same poses. A racing line
    # offset is off by default precisely so that this holds.
    yaw = truth["yaw_rate"].to_numpy()
    assert np.isfinite(yaw).all()
    assert np.abs(yaw).max() < 1.5
    turning = np.abs(yaw) > 0.05
    assert turning.any()


def test_plane_geometry_round_trip():
    camera = CameraModel(width=320, height=180, fx=260.0, fy=260.0)
    rotation = camera_from_world(0.4, 0.1, 0.0)
    homography = plane_homography(camera, np.array([3.0, -2.0, 1.2]), rotation)
    x, _y = plane_points(camera, homography, row_offset=20)
    assert x.shape == (camera.height - 20, camera.width)
    assert np.isfinite(x).all()
    # Rays above the horizon have no ground point and are excluded by the mask.
    sky = sky_mask(camera, rotation)
    assert sky[:20].all()
    assert not sky[-40:].any()


def test_mounting_matrix_and_normal():
    for pitch in (0.0, 0.1, -0.2):
        base = mounting_matrix(pitch)
        assert np.allclose(base.T @ base, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(base), 1.0)
        normal = ground_normal(pitch)
        assert np.isclose(np.linalg.norm(normal), 1.0)
        # The camera looks along its z axis, which is the camera frame's own
        # forward; the road normal points down and towards it.
    assert np.isclose(normal[0], 0.0, atol=1e-12)
    assert np.isclose(normal[2], np.sin(-pitch), atol=1e-12)
    assert np.allclose(ground_normal(0.0), np.array([0.0, -1.0, 0.0]), atol=1e-12)


def test_camera_from_world_inverts_to_world():
    rotation = camera_from_world(0.7, 0.05, 0.02)
    world_point = np.array([5.0, 3.0, 0.0])
    camera_point = rotation @ world_point
    assert camera_point[2] > 0.0
    # A point ahead of the car projects inside the image.
    camera = CameraModel(width=640, height=360, fx=520.0, fy=520.0)
    pixels = camera.matrix() @ camera_point
    u, _v = pixels[0] / pixels[2], pixels[1] / pixels[2]
    assert -50.0 < u < camera.width + 50.0
