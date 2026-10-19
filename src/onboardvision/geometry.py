"""Camera geometry for a monocular camera looking at a ground plane.

World frame: x and y are on the ground, z points up. The camera follows the
OpenCV convention, x to the right of the image, y downwards in the image, z
along the optical axis.

The renderer and the estimator both use the plane induced homography. For a
plane at z = 0 the map from plane coordinates to image coordinates is

    H = K [r1 r2 t]      with t = -R C

where R and C are the camera rotation and position in the world. Because the
inverse of a homography is another homography, the same expression runs
backwards: every pixel can be mapped to the point on the road where its ray
lands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    width: int = 640
    height: int = 360
    fx: float = 520.0
    fy: float = 520.0
    cx: float | None = None
    cy: float | None = None

    @property
    def principal_point(self) -> tuple[float, float]:
        return (
            self.cx if self.cx is not None else (self.width - 1) / 2.0,
            self.cy if self.cy is not None else (self.height - 1) / 2.0,
        )

    def matrix(self) -> np.ndarray:
        px, py = self.principal_point
        return np.array([[self.fx, 0.0, px], [0.0, self.fy, py], [0.0, 0.0, 1.0]])


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def mounting_matrix(pitch: float) -> np.ndarray:
    """Camera axes expressed in world coordinates, with yaw and roll at zero.

    Columns are the camera x, y and z axes. The camera looks along its z axis,
    which is tilted ``pitch`` radians below horizontal.
    """
    c, s = np.cos(pitch), np.sin(pitch)
    forward = np.array([c, 0.0, -s])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(forward, right)
    return np.column_stack([right, down, forward])


def ground_normal(pitch: float) -> np.ndarray:
    """Unit normal of the road plane, in the camera frame.

    Yaw rotates the camera about the vertical axis, which leaves the road
    normal unchanged, so the normal only depends on the mounting pitch. This is
    what makes the single camera case work: the plane orientation is known even
    though the heading is not.
    """
    return mounting_matrix(pitch).T @ np.array([0.0, 0.0, 1.0])


def camera_from_world(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """Rotation that takes world vectors into the camera frame.

    ``pitch`` is the angle the optical axis is tilted below horizontal, in
    radians. ``roll`` rotates the image about the optical axis.
    """
    base = mounting_matrix(pitch)
    rot_camera_to_world = rotation_z(yaw) @ base @ rotation_z(roll)
    return rot_camera_to_world.T


def plane_homography(
    camera: CameraModel,
    position: np.ndarray,
    rotation: np.ndarray,
    height: float | None = None,
) -> np.ndarray:
    """Homography from ground plane coordinates to image pixels."""
    matrix = camera.matrix()
    translation = -rotation @ position
    columns = np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    homography = matrix @ columns
    if height is not None:
        homography = homography / height
    return homography


def plane_points(
    camera: CameraModel, homography: np.ndarray, row_offset: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Map image pixels to the ground point their ray hits.

    ``row_offset`` skips the rows above the horizon, which are sky and never
    reach the road. The returned arrays then have ``camera.height - row_offset``
    rows.
    """
    width, height = camera.width, camera.height
    v = np.arange(row_offset, height, dtype=float)
    u = np.arange(width, dtype=float)
    uu, vv = np.meshgrid(u, v)
    pixels = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)])
    inverse = np.linalg.inv(homography)
    ground = inverse @ pixels
    ground = ground[:2] / ground[2]
    return ground[0].reshape(len(v), width), ground[1].reshape(len(v), width)


def sky_mask(camera: CameraModel, rotation: np.ndarray) -> np.ndarray:
    """True where a pixel ray points above the horizon and never hits the road."""
    width, height = camera.width, camera.height
    u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    pixels = np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    rays_camera = np.linalg.inv(camera.matrix()) @ pixels
    rotation_camera_to_world = rotation.T
    rays_world = rotation_camera_to_world @ rays_camera
    return (rays_world[2] >= 0.0).reshape(height, width)


def pose_series(
    positions: np.ndarray,
    yaw: np.ndarray,
    height: float,
    pitch: float,
    roll: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Turn a position and yaw trace into camera poses."""
    poses = []
    for index in range(len(yaw)):
        roll_angle = 0.0 if roll is None else float(roll[index])
        rotation = camera_from_world(float(yaw[index]), pitch, roll_angle)
        position = np.array([positions[index, 0], positions[index, 1], height])
        poses.append((position, rotation))
    return poses
