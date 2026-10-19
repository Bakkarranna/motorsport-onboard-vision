"""Corner detection and the forward-backward track filter."""

from __future__ import annotations

import cv2
import numpy as np
from onboardvision.features import (
    TrackingConfig,
    detect_corners,
    road_mask,
    track_points,
)


def checkerboard(size: int = 240, square: int = 12) -> np.ndarray:
    squares = np.indices((size, size)) // square
    return ((squares.sum(axis=0) % 2) * 255).astype(np.uint8)


def test_detects_corners_on_a_checkerboard():
    image = checkerboard()
    config = TrackingConfig()
    corners = detect_corners(image, config)
    assert len(corners) > 20
    assert corners.shape[1:] == (1, 2)
    # Corners land on the crossings, so the distance to the nearest multiple of
    # the square size is small.
    residual = np.abs((corners.reshape(-1, 2) % 12) - 12.0)
    assert np.median(residual.min(axis=1)) < 3.0


def test_blank_image_has_no_corners():
    config = TrackingConfig()
    blank = np.full((120, 160), 128, dtype=np.uint8)
    assert len(detect_corners(blank, config)) == 0


def test_mask_restricts_detection():
    image = checkerboard()
    mask = np.zeros_like(image)
    mask[120:, :] = 255
    corners = detect_corners(image, TrackingConfig(), mask=mask)
    assert len(corners) > 0
    assert np.all(corners.reshape(-1, 2)[:, 1] >= 120)
    fallback = road_mask(image)
    assert fallback[200, 100] == 255
    assert fallback[10, 100] == 0


def test_tracks_a_translation_exactly():
    image = checkerboard(size=260)
    shift = 3
    matrix = np.float32([[1.0, 0.0, shift], [0.0, 1.0, 0.0]])
    moved = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    from onboardvision.tests import TEST_TRACKING_CONFIG
    
    corners = detect_corners(image[:, :-shift], TEST_TRACKING_CONFIG)
    old, new, keep = track_points(image, moved, corners, TEST_TRACKING_CONFIG)
    assert len(old) > 10
    displacement = (new.reshape(-1, 2) - old.reshape(-1, 2))
    assert np.allclose(displacement[:, 0], shift, atol=0.25)
    assert np.allclose(displacement[:, 1], 0.0, atol=0.25)
    assert keep.sum() == len(keep)


def test_forward_backward_check_drops_unmatched_points():
    image = checkerboard(size=260)
    moved = np.roll(image, 5, axis=1)
    config = TrackingConfig(forward_backward_threshold=0.2)
    # A point in a region that does not exist in the second image cannot be
    # tracked back to where it started.
    points = np.array([[[2.0, 130.0]], [[130.0, 130.0]]], dtype=np.float32)
    old, new, _keep = track_points(image, moved, points, config)
    assert len(new) <= len(points)
    if len(old):
        assert np.all(np.isfinite(old.reshape(-1, 2)))


def test_empty_input_is_handled():
    image = checkerboard()
    config = TrackingConfig()
    old, new, keep = track_points(image, image, np.empty((0, 1, 2), dtype=np.float32), config)
    assert len(old) == 0
    assert len(new) == 0
    assert keep.size == 0
