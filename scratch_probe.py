"""Scratch probe: verify the 3 parameter planar motion model. Not part of the package."""

from __future__ import annotations

import time

import cv2
import numpy as np

from onboardvision.features import TrackingConfig, detect_corners, track_points
from onboardvision.geometry import CameraModel, camera_from_world, plane_homography, rotation_z
from onboardvision.render import OnboardSequence, RenderConfig
from onboardvision.track import Track, wave_loop

CAM = CameraModel(640, 360, 520, 520)
PITCH = 0.10
HEIGHT = 1.15
DT = 1.0 / 60.0


def mounting(pitch: float) -> np.ndarray:
    c, s = np.cos(pitch), np.sin(pitch)
    forward = np.array([c, 0.0, -s])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(forward, right)
    return np.column_stack([right, down, forward])


B = mounting(PITCH)
N_CAM = B.T @ np.array([0.0, 0.0, 1.0])


def model_matrix(forward: float, lateral: float, yaw: float, height: float) -> np.ndarray:
    displacement = B.T @ np.array([forward, lateral, 0.0])
    rotation = B.T @ rotation_z(yaw) @ B
    return rotation @ (np.eye(3) + np.outer(displacement, N_CAM) / height)


def project(matrix: np.ndarray, camera: CameraModel, pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    matrix_k = camera.matrix()
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(matrix_k).T
    moved = rays @ np.asarray(matrix).T
    image = moved @ matrix_k.T
    return image[:, :2] / image[:, 2:3]


def fit(pixels_old, pixels_new, height, initial, iterations=25):
    pixels_old = np.asarray(pixels_old, dtype=float).reshape(-1, 2)
    pixels_new = np.asarray(pixels_new, dtype=float).reshape(-1, 2)
    params = np.array(initial, dtype=float)
    for _ in range(iterations):
        residual = project(model_matrix(*params, height), CAM, pixels_old) - pixels_new
        r = residual.ravel()
        norm = np.linalg.norm(residual, axis=1)
        scale = max(1.0, float(np.median(norm)) * 1.4826)
        weight = np.where(norm <= 2.0 * scale, 1.0, 2.0 * scale / np.maximum(norm, 1e-9))
        jac = np.zeros((r.size, 3))
        for index in range(3):
            step = 1e-5 * max(1.0, abs(params[index]))
            shifted = params.copy()
            shifted[index] += step
            jac[:, index] = (
                (project(model_matrix(*shifted, height), CAM, pixels_old) - pixels_new).ravel() - r
            ) / step
        w = np.repeat(weight, 2)
        update, *_ = np.linalg.lstsq(jac * w[:, None], -r * w, rcond=None)
        params = params + update
        if np.linalg.norm(update) < 1e-9:
            break
    residual = project(model_matrix(*params, height), CAM, pixels_old) - pixels_new
    return params, float(np.sqrt(np.mean(residual**2)))


def exact_matrix(sequence, index) -> np.ndarray:
    row_a = sequence.truth.iloc[index]
    row_b = sequence.truth.iloc[index + 1]
    rot_a = camera_from_world(float(row_a["heading"]), PITCH, float(row_a["roll"]))
    rot_b = camera_from_world(float(row_b["heading"]), PITCH, float(row_b["roll"]))
    homography_a = plane_homography(CAM, np.array([row_a["x"], row_a["y"], HEIGHT]), rot_a)
    homography_b = plane_homography(CAM, np.array([row_b["x"], row_b["y"], HEIGHT]), rot_b)
    inverse_k = np.linalg.inv(CAM.matrix())
    return inverse_k @ homography_b @ np.linalg.inv(homography_a) @ CAM.matrix()


def main() -> None:
    track = Track.from_points(wave_loop())
    sequence = OnboardSequence(track, RenderConfig(duration=2.0, camera=CAM))
    grid = np.array([[0.0, 0.0], [640.0, 0.0], [0.0, 360.0], [640.0, 360.0], [320.0, 200.0]])

    print("model check against exact renderer geometry")
    for index in (10, 40, 70):
        row_a = sequence.truth.iloc[index]
        row_b = sequence.truth.iloc[index + 1]
        exact = exact_matrix(sequence, index)
        # The model yaw is the camera-1-frame rotation, which is Rz(-delta psi).
        truth = (row_a["speed"] * DT, 0.0, -(row_b["heading"] - row_a["heading"]))
        fitted, rms = fit(grid, project(exact, CAM, grid), HEIGHT, truth)
        print(
            f"  frame {index}: truth {truth[0]:+.4f} {truth[1]:+.4f} {truth[2]:+.6f}"
            f"  fit {fitted[0]:+.4f} {fitted[1]:+.4f} {fitted[2]:+.6f}  rms {rms:.4f} px"
        )

    print("\nrendered pose consistency (speed from poses versus commanded speed)")
    for index in (10, 11, 12, 40, 70):
        row = sequence.truth.iloc[index]
        print(
            f"  frame {index}: pose speed {row['speed']:7.3f}  commanded {row['commanded_speed']:7.3f}"
            f"  yaw {row['yaw_rate']:+.5f}"
        )

    print("\ntracked points: three parameter fit versus homography decomposition")
    inverse_k = np.linalg.inv(CAM.matrix())
    config = TrackingConfig()
    rows = []
    start = time.time()
    for index in range(30, 110):
        gray_a = cv2.cvtColor(sequence.frame(index), cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(sequence.frame(index + 1), cv2.COLOR_RGB2GRAY)
        corners = detect_corners(gray_a, config)
        old, new, _ = track_points(gray_a, gray_b, corners, config)
        if len(old) < 15:
            continue
        homography, _ = cv2.findHomography(
            old.reshape(-1, 2), new.reshape(-1, 2), cv2.RANSAC, config.ransac_threshold
        )
        normalised = inverse_k @ homography @ CAM.matrix()
        _, rotations, translations, normals = cv2.decomposeHomographyMat(normalised, np.eye(3))
        best = None
        best_score = -np.inf
        for rotation, normal in zip(rotations, normals, strict=False):
            rotation = np.asarray(rotation).reshape(3, 3)
            normal = np.asarray(normal).reshape(3)
            similarity = float(normal @ N_CAM)
            if similarity < 0.5:
                continue
            translation = (normalised - rotation) @ normal / float(normal @ normal)
            if translation[2] <= 0:
                continue
            if similarity > best_score:
                best_score = similarity
                best = (rotation, translation)
        row_a = sequence.truth.iloc[index]
        row_b = sequence.truth.iloc[index + 1]
        truth_forward = float(row_a["speed"]) * DT
        truth_yaw = -float(row_b["heading"] - row_a["heading"])
        baseline = float(np.linalg.norm(best[1])) * HEIGHT / DT if best else float("nan")
        if best is not None:
            seed_translation = B @ (best[1] * HEIGHT)
            world_rotation = B @ best[0] @ B.T
            seed = (
                float(seed_translation[0]),
                float(seed_translation[1]),
                float(np.arctan2(world_rotation[1, 0], world_rotation[0, 0])),
            )
        else:
            seed = (0.3, 0.0, 0.0)
        fitted, rms = fit(old, new, HEIGHT, seed)
        refined_speed = float(np.hypot(fitted[0], fitted[1])) / DT
        rows.append(
            (index, baseline, refined_speed, truth_forward / DT, fitted[2], truth_yaw, rms)
        )
        if index % 20 == 0:
            print(
                f"  frame {index}: tracks {len(old):3d}  speed truth {truth_forward / DT:6.2f}"
                f"  8dof {baseline:6.2f}  3param {refined_speed:6.2f}  rms {rms:5.2f}px"
                f"  yaw {fitted[2]:+.5f}/{truth_yaw:+.5f}"
            )
    print(f"elapsed {time.time() - start:.0f}s")
    truth = np.array([row[3] for row in rows])
    base = np.array([row[1] for row in rows])
    refined = np.array([row[2] for row in rows])
    print(f"8dof      bias {np.mean(base - truth):+.2f} m/s  mae {np.mean(np.abs(base - truth)):.2f}")
    print(f"3parameter bias {np.mean(refined - truth):+.2f} m/s  mae {np.mean(np.abs(refined - truth)):.2f}")


if __name__ == "__main__":
    main()
