"""Corner detection and sparse tracking.

Everything here is standard practice for monocular motion estimation, with one
extra guard: every track is verified by running the optical flow backwards and
throwing away points that do not return close to where they started. On a
vibration-heavy onboard camera that check removes most of the bad matches
before the homography is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TrackingConfig:
    """Feature detection, tracking and the gates on the motion fit.

    The values are the ones selected by the sweep in ``scripts/run_experiments.py
    --sweep``: many weak corners with a short optical flow window beat fewer
    strong corners with a long one, because the accuracy of the fit is set by how
    many independent measurements there are, not by how good any single one is.
    """

    max_corners: int = 900
    quality: float = 0.002
    min_distance: float = 3.0
    block_size: int = 5
    window: int = 11
    pyramid_levels: int = 4
    forward_backward_threshold: float = 1.5
    ransac_threshold: float = 2.0
    min_inliers: int = 15
    # Robust loss and rejection gates of the planar fit.
    huber_px: float = 1.5
    inlier_px: float = 3.0
    residual_threshold: float = 2.5
    max_speed: float = 120.0


def detect_corners(gray: np.ndarray, config: TrackingConfig, mask: np.ndarray | None = None):
    """Shi-Tomasi corners, restricted to a mask when one is given."""
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=config.max_corners,
        qualityLevel=config.quality,
        minDistance=config.min_distance,
        blockSize=config.block_size,
        mask=mask,
    )
    if corners is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return corners.astype(np.float32)


def track_points(
    previous: np.ndarray,
    current: np.ndarray,
    points: np.ndarray,
    config: TrackingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track points forward, then check them by tracking back again."""
    if len(points) == 0:
        return points, points, np.zeros(0, dtype=bool)

    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(config.window, config.window),
        maxLevel=config.pyramid_levels,
    )
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        forward,
        None,
        winSize=(config.window, config.window),
        maxLevel=config.pyramid_levels,
    )

    error = np.linalg.norm(points - backward, axis=2).reshape(-1)
    keep = (
        status.reshape(-1).astype(bool)
        & back_status.reshape(-1).astype(bool)
        & (error < config.forward_backward_threshold)
    )
    return points[keep], forward[keep], keep


def road_mask(gray: np.ndarray) -> np.ndarray:
    """Rough mask that keeps the lower two thirds of the image.

    On a real onboard clip this would come from a horizon estimate. The
    renderer knows its own horizon, so the synthetic runs use the exact mask and
    this fallback is only for supplied footage.
    """
    height, width = gray.shape
    mask = np.zeros_like(gray)
    mask[int(height * 0.35) :, :] = 255
    return mask
