"""Scratch probe: where does the low texture speed bias come from?"""

from __future__ import annotations

import cv2
import numpy as np

import scratch_lk as L
from onboardvision.features import TrackingConfig, detect_corners, track_points
from onboardvision.motion import HomographyEstimator
from onboardvision.planar import fit_planar_motion, project_points
from onboardvision.render import OnboardSequence, RenderConfig
from onboardvision.track import Track, wave_loop

HEIGHT = 1.15
FPS = 60.0


def analyse(texture: float, frames: int = 50) -> None:
    track = Track.from_points(wave_loop())
    sequence = OnboardSequence(
        track, RenderConfig(duration=frames / FPS, camera=L.CAM, texture_scale=texture)
    )
    config = TrackingConfig()
    estimator = HomographyEstimator(L.CAM, HEIGHT, 1.0 / FPS, L.PITCH, config)
    rows = []
    detection = []
    for index in range(10, frames - 1):
        gray_a = cv2.cvtColor(sequence.frame(index), cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(sequence.frame(index + 1), cv2.COLOR_RGB2GRAY)
        corners = detect_corners(gray_a, config)
        old, new, _ = track_points(gray_a, gray_b, corners, config)
        detection.append(len(corners))
        if len(old) < 15:
            continue
        matrix = L.exact_matrix(sequence, index)
        back = project_points(np.linalg.inv(matrix), L.CAM, new.reshape(-1, 2))
        truth_error = np.linalg.norm(back - old.reshape(-1, 2), axis=1)
        motion = fit_planar_motion(
            old.reshape(-1, 2), new.reshape(-1, 2), L.CAM, HEIGHT, L.PITCH
        )
        truth_forward = float(sequence.truth.speed.iloc[index]) / FPS
        rows.append(
            (
                np.median(truth_error),
                np.mean(truth_error > 1.5),
                motion.forward_m,
                truth_forward,
                motion.rms_px,
                motion.inliers,
            )
        )
    data = np.array(rows)
    print(
        f"texture {texture:4.2f}: corners {np.mean(detection):5.0f}"
        f"  track median {np.median(data[:, 0]):5.2f}px  outlier share {np.mean(data[:, 1]):.2f}"
        f"  fit residual {np.median(data[:, 4]):5.2f}px  inliers {np.median(data[:, 5]):4.0f}"
    )
    print(
        f"              forward fit {np.mean(data[:, 2]):+.4f} m"
        f"  truth {np.mean(data[:, 3]):+.4f} m"
        f"  bias {np.mean((data[:, 2] - data[:, 3]) / data[:, 3]) * 100.0:+6.1f}%"
    )


def main() -> None:
    for texture in (1.0, 0.35, 0.05):
        analyse(texture)


if __name__ == "__main__":
    main()
