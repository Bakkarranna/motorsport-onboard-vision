"""Scratch probe: how accurate are the tracked correspondences? Not part of the package."""

from __future__ import annotations

import cv2
import numpy as np

from onboardvision.features import detect_corners, track_points
from onboardvision.geometry import CameraModel, camera_from_world, plane_homography
from onboardvision.render import OnboardSequence, RenderConfig
from onboardvision.track import Track, wave_loop

CAM = CameraModel(640, 360, 520, 520)
PITCH = 0.10
HEIGHT = 1.15


def exact_matrix(sequence, index) -> np.ndarray:
    row_a = sequence.truth.iloc[index]
    row_b = sequence.truth.iloc[index + 1]
    rot_a = camera_from_world(float(row_a["heading"]), PITCH, float(row_a["roll"]))
    rot_b = camera_from_world(float(row_b["heading"]), PITCH, float(row_b["roll"]))
    homography_a = plane_homography(CAM, np.array([row_a["x"], row_a["y"], HEIGHT]), rot_a)
    homography_b = plane_homography(CAM, np.array([row_b["x"], row_b["y"], HEIGHT]), rot_b)
    inverse_k = np.linalg.inv(CAM.matrix())
    return inverse_k @ homography_b @ np.linalg.inv(homography_a) @ CAM.matrix()


def project(matrix, pixels):
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    matrix_k = CAM.matrix()
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(matrix_k).T
    image = (rays @ np.asarray(matrix).T) @ matrix_k.T
    return image[:, :2] / image[:, 2:3]


def main() -> None:
    track = Track.from_points(wave_loop())
    sequence = OnboardSequence(track, RenderConfig(duration=1.5, camera=CAM))
    frames = []
    for index in range(20, 60):
        frames.append(
            (
                cv2.cvtColor(sequence.frame(index), cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(sequence.frame(index + 1), cv2.COLOR_RGB2GRAY),
                index,
            )
        )

    class Config:
        pass

    for window, levels, quality in ((7, 4, 0.01), (11, 4, 0.01), (21, 4, 0.01), (21, 3, 0.01), (15, 5, 0.005)):
        config = Config()
        config.max_corners = 900
        config.quality = quality
        config.min_distance = 6.0
        config.block_size = 7
        config.window = window
        config.pyramid_levels = levels
        config.forward_backward_threshold = 1.5
        config.ransac_threshold = 2.0
        config.min_inliers = 15
        errors = []
        outliers = []
        tracked = []
        detected = []
        for gray_a, gray_b, index in frames:
            corners = detect_corners(gray_a, config)
            old, new, _ = track_points(gray_a, gray_b, corners, config)
            detected.append(len(corners))
            if len(old) < 15:
                continue
            matrix = exact_matrix(sequence, index)
            # Map the current points back through the inverse, which stays well
            # conditioned for points that leave the image.
            back = project(np.linalg.inv(matrix), new)
            error = np.linalg.norm(back - old.reshape(-1, 2), axis=1)
            errors.append(error)
            outliers.append(np.mean(error > 1.5))
            tracked.append(len(old))
        flat = np.concatenate(errors)
        print(
            f"window {window:2d} levels {levels} quality {quality:<6}: detected {np.mean(detected):5.0f}"
            f"  kept {np.mean(tracked):5.0f}"
            f"  median {np.median(flat):5.2f}px  p90 {np.percentile(flat, 90):5.2f}px"
            f"  outlier share {np.mean(outliers):.2f}"
        )


if __name__ == "__main__":
    main()
