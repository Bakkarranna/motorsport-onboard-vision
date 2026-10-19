"""Monocular motion estimation from frame to frame.

The camera looks at a plane, so two views of that plane are related by a
homography. The classical route is to fit a general homography with RANSAC and
decompose it into a rotation and a scaled translation, but eight free degrees of
freedom need a lot of well spread support and the resulting speed is noisy. This
module uses the decomposition only as an initial guess and then fits the three
parameter planar model of :mod:`onboardvision.planar` directly, which is both
smoother and cheaper.

The scale of the translation is recovered from the known height of the camera
above the road. That single assumption is what turns a scale free homography
into a speed in metres per second, and it is also the pipeline's largest source
of systematic error: see the ``height_wrong`` experiment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .features import TrackingConfig, detect_corners, track_points
from .geometry import CameraModel, ground_normal, mounting_matrix
from .planar import fit_planar_motion


@dataclass
class MotionEstimate:
    """One frame to frame estimate."""

    ok: bool
    forward_m: float = 0.0
    lateral_m: float = 0.0
    speed_mps: float = 0.0
    yaw_rate: float = 0.0
    inliers: int = 0
    tracks: int = 0
    rms_px: float = float("nan")
    reason: str = ""


class HomographyEstimator:
    """Frame to frame speed and yaw rate from visual features.

    Parameters
    ----------
    camera:
        Intrinsics, in the OpenCV convention. The focal length has to be known;
        on this renderer it is exact, on real footage it comes from a
        calibration on a target.
    height:
        Camera height above the road. This sets the scale of the recovered
        translation.
    dt:
        Time between frames in seconds.
    pitch:
        Camera pitch below horizontal in radians, needed to build the expected
        ground plane normal and to turn an image plane rotation into a world
        yaw change.
    """

    def __init__(
        self,
        camera: CameraModel,
        height: float,
        dt: float,
        pitch: float,
        config: TrackingConfig | None = None,
    ) -> None:
        self.camera = camera
        self.height = height
        self.dt = dt
        self.pitch = pitch
        self.config = config or TrackingConfig()
        self.matrix = camera.matrix()
        self.inverse_matrix = np.linalg.inv(self.matrix)
        self._base = mounting_matrix(pitch)
        self._expected_normal = ground_normal(pitch)

    def seed(self, homography: np.ndarray) -> tuple[float, float, float]:
        """Decompose a free homography into an initial planar motion.

        The decomposition returns four candidate solutions. The road plane has
        a known normal, so the candidates with a normal pointing away from the
        road are rejected, and the one best aligned with the expected normal is
        kept.
        """
        normalised = self.inverse_matrix @ homography @ self.matrix
        _, rotations, _, normals = cv2.decomposeHomographyMat(normalised, np.eye(3))
        best = None
        best_score = -np.inf
        for rotation, normal in zip(rotations, normals, strict=False):
            rotation = np.asarray(rotation).reshape(3, 3)
            normal = np.asarray(normal).reshape(3)
            similarity = float(normal @ self._expected_normal)
            if similarity < 0.5:
                continue
            translation = (normalised - rotation) @ normal / float(normal @ normal)
            if translation[2] <= 0.0:
                continue
            if similarity > best_score:
                best_score = similarity
                best = (rotation, translation)
        if best is None:
            return (0.0, 0.0, 0.0)

        rotation, translation_over_distance = best
        metres = translation_over_distance * self.height
        displacement = self._base @ metres
        # The camera frame rotation between the two views is a yaw about the
        # vertical, which is exactly the model parameter.
        turn = self._base @ rotation @ self._base.T
        yaw = float(np.arctan2(turn[1, 0], turn[0, 0]))
        return (float(displacement[0]), float(displacement[1]), yaw)

    def estimate(self, previous: np.ndarray, current: np.ndarray) -> MotionEstimate:
        points = detect_corners(previous, self.config)
        old_points, new_points, _ = track_points(previous, current, points, self.config)
        tracks = len(old_points)
        if tracks < self.config.min_inliers:
            return MotionEstimate(ok=False, tracks=tracks, reason="too few tracks")

        old_pixels = old_points.reshape(-1, 2)
        new_pixels = new_points.reshape(-1, 2)
        homography, mask = cv2.findHomography(
            old_pixels, new_pixels, cv2.RANSAC, self.config.ransac_threshold
        )
        if homography is None:
            return MotionEstimate(ok=False, tracks=tracks, reason="no homography")
        ransac_inliers = int(mask.sum()) if mask is not None else 0
        if ransac_inliers < self.config.min_inliers:
            return MotionEstimate(
                ok=False, inliers=ransac_inliers, tracks=tracks, reason="too few inliers"
            )

        initial = self.seed(homography)
        motion = fit_planar_motion(
            old_pixels,
            new_pixels,
            self.camera,
            self.height,
            self.pitch,
            initial=initial,
            huber_px=self.config.huber_px,
            inlier_px=self.config.inlier_px,
        )
        if motion.inliers < self.config.min_inliers:
            return MotionEstimate(
                ok=False,
                inliers=motion.inliers,
                tracks=tracks,
                rms_px=motion.rms_px,
                reason="fit rejected too many tracks",
            )
        if not np.isfinite(motion.rms_px) or motion.rms_px > self.config.residual_threshold:
            return MotionEstimate(
                ok=False,
                inliers=motion.inliers,
                tracks=tracks,
                rms_px=motion.rms_px,
                reason="residual too high",
            )

        distance = motion.distance_m
        speed = distance / self.dt
        # A race car drives forwards. An estimate that says otherwise is a
        # failed fit, not a reverse gear, and it is better to report no estimate
        # at all than to hand a deceleration spike to the next stage.
        if motion.forward_m <= 0.0 or speed > self.config.max_speed:
            return MotionEstimate(
                ok=False,
                inliers=motion.inliers,
                tracks=tracks,
                rms_px=motion.rms_px,
                reason="implausible motion",
            )
        if abs(motion.lateral_m) > 0.5 * distance:
            return MotionEstimate(
                ok=False,
                inliers=motion.inliers,
                tracks=tracks,
                rms_px=motion.rms_px,
                reason="implausible side slip",
            )

        return MotionEstimate(
            ok=True,
            forward_m=motion.forward_m,
            lateral_m=motion.lateral_m,
            speed_mps=speed,
            # The model parameter is the camera frame yaw, which is the negative
            # of the change in the car's heading.
            yaw_rate=-motion.yaw_rad / self.dt,
            inliers=motion.inliers,
            tracks=tracks,
            rms_px=motion.rms_px,
        )


def prepare(sequence) -> list[np.ndarray]:
    """Render a sequence to grayscale frames.

    Rendering is the expensive part, so it is separated from estimation: a
    parameter sweep can render once and replay the same frames through several
    configurations.
    """
    return [cv2.cvtColor(sequence.frame(index), cv2.COLOR_RGB2GRAY) for index in range(sequence.frames)]


def estimate_frames(frames: list[np.ndarray], estimator: HomographyEstimator, truth=None, stride: int = 1):
    """Run the estimator over prepared frames and return per frame values."""
    import pandas as pd

    rows = []
    previous = frames[0]
    count = len(frames) if truth is None else min(len(frames), len(truth))
    for index in range(1, count):
        current = frames[index]
        if index % stride == 0:
            started = time.perf_counter()
            estimate = estimator.estimate(previous, current)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            rows.append(
                {
                    "ms": elapsed_ms,
                    "frame": index,
                    "time": float(truth["time"].iloc[index]) if truth is not None else float(index) * estimator.dt,
                    "truth_speed": float(truth["speed"].iloc[index]) if truth is not None else float("nan"),
                    "truth_yaw_rate": (
                        float(truth["yaw_rate"].iloc[index]) if truth is not None else float("nan")
                    ),
                    "speed": estimate.speed_mps,
                    "yaw_rate": estimate.yaw_rate,
                    "forward_m": estimate.forward_m,
                    "lateral_m": estimate.lateral_m,
                    "rms_px": estimate.rms_px,
                    "inliers": estimate.inliers,
                    "tracks": estimate.tracks,
                    "ok": estimate.ok,
                    "reason": estimate.reason,
                }
            )
        previous = current
    return pd.DataFrame(rows)


def run_sequence(sequence, estimator: HomographyEstimator, stride: int = 1):
    """Run the estimator over a rendered sequence and return per frame values."""
    return estimate_frames(prepare(sequence), estimator, sequence.truth, stride)
