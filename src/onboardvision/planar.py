"""Planar motion model: three unknowns instead of eight.

A free homography has eight degrees of freedom, and every one of them has to be
supported by the tracked points before the decomposition can say anything about
speed. On a race car the motion is far more constrained than that:

* the camera looks at the road plane, whose orientation is known from the
  mounting pitch, so the plane normal is a constant;
* the camera height above that plane is known, which fixes the scale of the
  translation, and that is the whole reason a single camera can measure speed;
* the car moves along the road, so the translation has no component normal to
  the plane;
* the rotation between two frames is essentially a yaw about the vertical.

Fitting those three unknowns (forward, lateral, yaw) directly to the tracked
points is what :func:`fit_planar_motion` does. It replaces the "fit a general
homography, then decompose it" route, which is kept as the initial guess.

The measured errors on rendered footage, in metres per frame, are roughly a
third of the free homography route, because the model cannot absorb motion
noise into the degrees of freedom it does not have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CameraModel, ground_normal, mounting_matrix, rotation_z


@dataclass(frozen=True)
class PlanarMotion:
    """Result of a planar fit, in the camera frame of the previous image."""

    forward_m: float
    lateral_m: float
    yaw_rad: float
    rms_px: float
    inliers: int

    @property
    def distance_m(self) -> float:
        return float(np.hypot(self.forward_m, self.lateral_m))


def planar_homography(
    forward_m: float,
    lateral_m: float,
    yaw_rad: float,
    height: float,
    pitch: float,
) -> np.ndarray:
    """Homography of a ground plane seen by a camera moving over it.

    ``forward_m`` and ``lateral_m`` are the translation of the camera expressed
    in the axes of the previous camera frame projected onto the road, ``yaw_rad``
    rotates the camera about the vertical, ``height`` is the distance to the
    plane along its normal and ``pitch`` is the mounting pitch angle. The result
    is in normalised camera coordinates, so it maps rays rather than pixels.
    """
    base = mounting_matrix(pitch)
    displacement = base.T @ np.array([forward_m, lateral_m, 0.0])
    rotation = base.T @ rotation_z(yaw_rad) @ base
    normal = ground_normal(pitch)
    return rotation @ (np.eye(3) + np.outer(displacement, normal) / height)


def project_points(matrix: np.ndarray, camera: CameraModel, pixels: np.ndarray) -> np.ndarray:
    """Map pixels through a normalised homography back to pixels."""
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    inverse = np.linalg.inv(camera.matrix())
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ inverse.T
    moved = (rays @ np.asarray(matrix).T) @ camera.matrix().T
    return moved[:, :2] / moved[:, 2:3]


def residuals(
    parameters: np.ndarray,
    camera: CameraModel,
    height: float,
    pitch: float,
    old_pixels: np.ndarray,
    new_pixels: np.ndarray,
) -> np.ndarray:
    """Per point pixel error of the planar model."""
    matrix = planar_homography(*parameters, height, pitch)
    return project_points(matrix, camera, old_pixels) - new_pixels


def fit_planar_motion(
    old_pixels: np.ndarray,
    new_pixels: np.ndarray,
    camera: CameraModel,
    height: float,
    pitch: float,
    initial: tuple[float, float, float] = (0.0, 0.0, 0.0),
    iterations: int = 30,
    huber_px: float = 1.5,
    inlier_px: float = 3.0,
) -> PlanarMotion:
    """Gauss-Newton fit of the three parameter motion, robust to bad tracks.

    The loss is Huber on the per point pixel error, which keeps the handful of
    tracks that land on a moving object or on a broken down edge from dragging
    the estimate. A step that does not reduce the robust cost is halved, so the
    iteration cannot diverge even from a poor initial guess.
    """
    old_pixels = np.asarray(old_pixels, dtype=float).reshape(-1, 2)
    new_pixels = np.asarray(new_pixels, dtype=float).reshape(-1, 2)
    if len(old_pixels) < 3:
        raise ValueError("A planar fit needs at least three point pairs")

    parameters = np.array(initial, dtype=float)

    def cost(values: np.ndarray) -> float:
        error = np.linalg.norm(
            residuals(values, camera, height, pitch, old_pixels, new_pixels), axis=1
        )
        inner = np.minimum(error, huber_px)
        return float(np.sum(inner**2 + 2.0 * huber_px * (error - inner).clip(min=0.0)))

    for _ in range(iterations):
        error = residuals(parameters, camera, height, pitch, old_pixels, new_pixels)
        norm = np.linalg.norm(error, axis=1)
        weight = np.where(norm <= huber_px, 1.0, huber_px / np.maximum(norm, 1e-9))

        jacobian = np.zeros((error.size, 3))
        for index in range(3):
            step = 1e-6 * max(1.0, abs(parameters[index]))
            shifted = parameters.copy()
            shifted[index] += step
            jacobian[:, index] = (
                residuals(shifted, camera, height, pitch, old_pixels, new_pixels).ravel()
                - error.ravel()
            ) / step

        weights = np.repeat(weight, 2)
        try:
            update, *_ = np.linalg.lstsq(jacobian * weights[:, None], -error.ravel() * weights, rcond=None)
        except np.linalg.LinAlgError:
            break

        # Backtracking keeps the robust cost decreasing.
        current = cost(parameters)
        scale = 1.0
        for _ in range(8):
            candidate = parameters + scale * update
            if cost(candidate) < current:
                parameters = candidate
                break
            scale *= 0.5
        else:
            break
        if scale * float(np.linalg.norm(update)) < 1e-9:
            break

    final = np.linalg.norm(
        residuals(parameters, camera, height, pitch, old_pixels, new_pixels), axis=1
    )
    inliers = int(np.sum(final <= inlier_px))
    rms = float(np.sqrt(np.mean(final[final <= inlier_px] ** 2))) if inliers else float("inf")
    return PlanarMotion(
        forward_m=float(parameters[0]),
        lateral_m=float(parameters[1]),
        yaw_rad=float(parameters[2]),
        rms_px=rms,
        inliers=inliers,
    )
