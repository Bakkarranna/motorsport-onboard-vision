"""Scratch sweep: tune tracking parameters on cached frames. Not part of the package."""

from __future__ import annotations

import os
import tempfile
import time

import cv2
import numpy as np

from onboardvision.features import TrackingConfig
from onboardvision.geometry import CameraModel
from onboardvision.motion import HomographyEstimator
from onboardvision.render import OnboardSequence, RenderConfig
from onboardvision.track import Track, wave_loop

CAM = CameraModel(640, 360, 520, 520)
PITCH = 0.10
HEIGHT = 1.15
FPS = 60.0
CACHE = os.path.join(tempfile.gettempdir(), "onboard_frames.npz")


def build_cache(frames: int = 120) -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(CACHE):
        data = np.load(CACHE)
        return data["gray"], data["truth"]
    sequence = OnboardSequence(
        Track.from_points(wave_loop()), RenderConfig(duration=(frames + 1) / FPS, camera=CAM)
    )
    gray = np.stack([cv2.cvtColor(sequence.frame(i), cv2.COLOR_RGB2GRAY) for i in range(frames + 1)])
    truth = sequence.truth[["speed", "yaw_rate", "curvature"]].to_numpy()
    np.savez_compressed(CACHE, gray=gray, truth=truth)
    return gray, truth


def score(gray: np.ndarray, truth: np.ndarray, config: TrackingConfig):
    estimator = HomographyEstimator(CAM, HEIGHT, 1.0 / FPS, PITCH, config)
    speeds, yaws, ok = [], [], []
    start = time.time()
    for index in range(30, len(gray) - 1):
        estimate = estimator.estimate(gray[index], gray[index + 1])
        ok.append(estimate.ok)
        if estimate.ok:
            speeds.append(estimate.speed_mps)
            yaws.append(estimate.yaw_rate)
    ok = np.array(ok)
    speeds = np.array(speeds)
    yaws = np.array(yaws)
    truth_speed = truth[31 : len(gray), 0]
    truth_yaw = truth[31 : len(gray), 1]
    if len(speeds) == 0:
        return None
    return {
        "coverage": float(ok.mean()),
        "bias": float(np.mean(speeds - truth_speed[ok])),
        "mae": float(np.mean(np.abs(speeds - truth_speed[ok]))),
        "mae_pct": float(np.mean(np.abs(speeds - truth_speed[ok]) / truth_speed[ok]) * 100.0),
        "yaw_mae": float(np.mean(np.abs(yaws - truth_yaw[ok]))),
        "per_frame_ms": (time.time() - start) / len(ok) * 1000.0,
    }


def main() -> None:
    gray, truth = build_cache()
    print(f"cached {gray.shape[0]} frames")
    grid = [
        (0.0020, 3.0, 5, 9, 4),
        (0.0020, 3.0, 5, 11, 4),
        (0.0020, 3.0, 5, 13, 4),
        (0.0015, 3.0, 5, 11, 4),
        (0.0030, 4.0, 5, 11, 4),
        (0.0020, 3.0, 7, 11, 4),
        (0.0020, 2.0, 5, 11, 4),
        (0.0020, 3.0, 5, 11, 3),
    ]
    print(f"{'quality':>8s} {'dist':>5s} {'block':>5s} {'window':>6s} {'levels':>6s}"
          f" {'cov':>5s} {'bias':>7s} {'mae':>7s} {'mae%':>6s} {'yawmae':>7s} {'ms':>6s}")
    for quality, distance, block, window, levels in grid:
        config = TrackingConfig()
        config.quality = quality
        config.min_distance = distance
        config.block_size = block
        config.window = window
        config.pyramid_levels = levels
        result = score(gray, truth, config)
        if result is None:
            print(f"{quality:8.4f} {distance:5.0f} {block:5d} {window:6d} {levels:6d}   no estimates")
            continue
        print(
            f"{quality:8.4f} {distance:5.0f} {block:5d} {window:6d} {levels:6d}"
            f" {result['coverage']:5.2f} {result['bias']:+7.2f} {result['mae']:7.2f}"
            f" {result['mae_pct']:6.2f} {result['yaw_mae']:7.3f} {result['per_frame_ms']:6.1f}"
        )


if __name__ == "__main__":
    main()
