"""Procedural onboard camera footage with exact ground truth.

There is no legal way to ship broadcast onboard video, so the pipeline is
developed and validated against a renderer. The renderer is not a video game:
every pixel is the exact intersection of a camera ray with the road plane, and
the camera pose, speed and yaw rate are known to machine precision. That makes
it a fair test bed for a monocular estimator and it keeps the ground truth
honest.

The road is drawn from the geometry in :mod:`onboardvision.track`, so markings
and kerbs stay in the right place when the car turns, and the visual features a
tracker can lock onto are the same kind of features it would see on a real
circuit: seam patches, kerb stripes and painted lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .geometry import (
    CameraModel,
    camera_from_world,
    plane_homography,
    plane_points,
    sky_mask,
)
from .track import Track

ASPHALT = np.array([96.0, 96.0, 99.0])
LINE = np.array([238.0, 238.0, 236.0])
KERB_RED = np.array([190.0, 60.0, 55.0])
KERB_WHITE = np.array([232.0, 232.0, 228.0])
GRASS = np.array([86.0, 112.0, 66.0])
SKY = np.array([150.0, 168.0, 196.0])


def _hash_noise(x: np.ndarray, y: np.ndarray, scale: float) -> np.ndarray:
    """Deterministic value noise in [0, 1) from quantised world coordinates."""
    u = np.floor(x * scale)
    v = np.floor(y * scale)
    value = np.sin(u * 127.1 + v * 311.7) * 43758.5453
    return value - np.floor(value)


@dataclass
class RenderConfig:
    camera: CameraModel = field(default_factory=CameraModel)
    height: float = 1.15
    pitch: float = 0.10
    fps: float = 60.0
    duration: float = 8.0
    lateral_accel_limit: float = 22.0
    max_speed: float = 62.0
    min_speed: float = 26.0
    # Offsetting the car towards the inside of a corner is available, but the
    # offset path has a yaw rate of its own, so the pose series stops matching
    # curvature times speed. The default keeps the car on the centreline, where
    # the geometry of the ground truth is exact.
    racing_line: float = 0.0
    noise_std: float = 3.0
    texture_scale: float = 1.0
    roll_gain: float = 0.0
    seed: int = 5


def speed_profile(track: Track, config: RenderConfig) -> np.ndarray:
    """Corner limited speed at every centreline sample (a simple lap model)."""
    curvature = np.abs(track.curvature)
    limit = np.sqrt(config.lateral_accel_limit / np.maximum(curvature, 1e-6))
    speed = np.clip(limit, config.min_speed, config.max_speed)

    # Braking and traction limits propagate forwards and backwards along the lap.
    for _ in range(12):
        back = np.roll(speed, 1)
        allowed_back = np.sqrt(np.maximum(back**2 - 2 * 12.0 * (track.length / len(speed)), 0.0))
        speed = np.minimum(speed, np.maximum(allowed_back, config.min_speed * 0.6))
        ahead = np.roll(speed, -1)
        allowed_ahead = np.sqrt(ahead**2 + 2 * 8.0 * (track.length / len(speed)))
        speed = np.minimum(speed, allowed_ahead)
    return speed


class OnboardSequence:
    """Frames of an onboard camera driving one lap segment."""

    def __init__(self, track: Track, config: RenderConfig | None = None) -> None:
        self.track = track
        self.config = config or RenderConfig()
        self.lookup = track.centreline_lookup()
        self.speed = speed_profile(track, self.config)
        self.frames = int(self.config.duration * self.config.fps)
        self.dt = 1.0 / self.config.fps
        self._build_truth()

    def _sample(self, values: np.ndarray, arc: np.ndarray) -> np.ndarray:
        """Linear interpolation of a periodic centreline quantity."""
        track = self.track
        count = len(track.centreline)
        spacing = track.length / count
        position = (np.asarray(arc) % track.length) / spacing
        low = np.floor(position).astype(int) % count
        high = (low + 1) % count
        weight = position - np.floor(position)
        return values[low] * (1.0 - weight) + values[high] * weight

    def _heading(self, arc: np.ndarray) -> np.ndarray:
        """Circular interpolation of the centreline heading."""
        track = self.track
        count = len(track.centreline)
        spacing = track.length / count
        position = (np.asarray(arc) % track.length) / spacing
        low = np.floor(position).astype(int) % count
        high = (low + 1) % count
        weight = position - np.floor(position)
        vector = (1.0 - weight) * np.exp(1j * track.headings[low]) + weight * np.exp(
            1j * track.headings[high]
        )
        return np.angle(vector)

    def _offset(self, arc: np.ndarray) -> np.ndarray:
        """Lateral offset of the racing line from the centreline."""
        curvature = self._sample(self.track.curvature, arc)
        return -self.config.racing_line * np.tanh(curvature / 0.005)

    def _point(self, arc: np.ndarray) -> np.ndarray:
        centre = np.column_stack(
            [
                self._sample(self.track.centreline[:, 0], arc),
                self._sample(self.track.centreline[:, 1], arc),
            ]
        )
        heading = self._heading(arc)
        normal = np.column_stack([-np.sin(heading), np.cos(heading)])
        return centre + normal * self._offset(arc)[:, None]

    def _build_truth(self) -> None:
        """Integrate the racing line and report the pose that was rendered.

        The earlier version indexed the centreline directly, so the rendered
        pose jumped by a sample spacing every few frames. Speed and yaw rate are
        now differentiated from the poses that the renderer actually draws,
        which keeps ground truth and footage consistent by construction.
        """
        track = self.track
        step = 0.5
        arc = 0.0
        arcs = []
        for _ in range(self.frames + 1):
            arcs.append(arc)
            speed = float(self._sample(self.speed, np.array([arc]))[0])
            arc += speed * self.dt
        arc_values = np.array(arcs)

        points = self._point(arc_values)
        headings = self._heading(arc_values)
        curved = self._sample(track.curvature, arc_values)
        speeds = self._sample(self.speed, arc_values)

        # The heading follows the racing line, not the centreline, so the
        # camera looks where the car is actually going. A central difference
        # keeps that direction smooth.
        ahead = self._point(arc_values + step)
        behind = self._point(arc_values - step)
        tangent = ahead - behind
        headings = np.arctan2(tangent[:, 1], tangent[:, 0])

        displacement = np.linalg.norm(np.diff(points, axis=0), axis=1)
        yaw_change = np.diff(np.unwrap(headings))

        truth = pd.DataFrame(
            {
                "frame": np.arange(self.frames),
                "time": np.arange(self.frames) * self.dt,
                "speed": displacement[: self.frames] / self.dt,
                "curvature": curved[: self.frames],
                "heading": headings[: self.frames],
                "yaw_rate": yaw_change[: self.frames] / self.dt,
                "x": points[: self.frames, 0],
                "y": points[: self.frames, 1],
                "offset": self._offset(arc_values)[: self.frames],
                "commanded_speed": speeds[: self.frames],
            }
        )
        truth["accel"] = np.gradient(truth["speed"], self.dt)
        truth["roll"] = (
            self.config.roll_gain * truth["curvature"] * truth["speed"] ** 2 / 9.81
        )
        self.truth = truth

    def frame(self, index: int, noise: bool = True) -> np.ndarray:
        row = self.truth.iloc[index]
        camera = self.config.camera
        # Small camera shake keeps the estimator honest about vertical motion.
        shake = 0.004 * np.sin(index / 3.0)
        rotation = camera_from_world(
            float(row["heading"]), self.config.pitch + shake, float(row["roll"])
        )
        position = np.array([row["x"], row["y"], self.config.height])
        homography = plane_homography(camera, position, rotation)

        # Rays above the horizon never reach the road, so the plane geometry,
        # the track lookup and the texture are only evaluated below the first
        # road row. That is about a third of the image here.
        sky = sky_mask(camera, rotation)
        top = 0
        if sky.any():
            road_rows = np.flatnonzero(~sky.any(axis=1))
            if road_rows.size:
                top = max(0, int(road_rows[0]) - 2)

        x, y = plane_points(camera, homography, top)
        self._world = (x, y)
        arc, offset = self.lookup.coordinates(x.ravel(), y.ravel())
        road = self._paint(arc.reshape(x.shape), offset.reshape(x.shape), sky[top:])

        image = np.empty((camera.height, camera.width, 3), dtype=float)
        if top:
            image[:top] = SKY[None, None, :]
        image[top:] = road
        if noise and self.config.noise_std > 0:
            rng = np.random.default_rng(self.config.seed + index)
            image = image + rng.normal(0.0, self.config.noise_std, image.shape)
        return np.clip(image, 0, 255).astype(np.uint8)

    def _paint(self, arc: np.ndarray, offset: np.ndarray, sky: np.ndarray) -> np.ndarray:
        config = self.config
        track = self.track
        shape = arc.shape + (3,)
        image = np.empty(shape, dtype=float)

        distance = np.abs(offset)
        on_road = distance <= track.half_width
        on_kerb = (distance > track.half_width) & (distance <= track.half_width + track.kerb_width)

        # Asphalt with three layers of texture. The fine speckle is a hash of
        # the world position, so it is fixed to the road and moves with it,
        # which is exactly what a tracker needs.
        world = self._world
        grain = np.sin(arc * 1.7) * np.cos(offset * 2.3) * 4.0 * config.texture_scale
        speckle = _hash_noise(world[0], world[1], 3.0) * 26.0 * config.texture_scale
        coarse = _hash_noise(world[0], world[1], 0.5) * 16.0 * config.texture_scale
        seam = (_hash_noise(world[0] * 0.2, world[1] * 0.2, 0.35) > 0.93) * -18.0
        asphalt = ASPHALT[None, None, :] + (grain + speckle + coarse + seam)[:, :, None]
        image[...] = asphalt

        # Centre line, dashed every 8 m of arc length.
        centre = (distance < 0.14) & (((arc % 9.0) < 4.5))
        # Solid edge lines.
        edge = (distance > track.half_width - 0.45) & (distance < track.half_width - 0.10)
        line = (centre | edge)[:, :, None]
        image = np.where(line, LINE[None, None, :], image)

        stripe = (np.floor(arc / 2.0) % 2.0) < 1.0
        kerb = np.where(
            stripe[:, :, None], KERB_RED[None, None, :], KERB_WHITE[None, None, :]
        )
        image = np.where(on_kerb[:, :, None], kerb, image)

        grass = GRASS[None, None, :] + (
            np.sin(arc * 2.9) * np.cos(offset * 1.9) * 9.0
        )[:, :, None]
        image = np.where((~on_road & ~on_kerb)[:, :, None], grass, image)

        # Rays above the horizon never reach the road. There is no sky or
        # scenery in this renderer, so they get a flat colour.
        if sky.any():
            image = np.where(sky[:, :, None], SKY[None, None, :], image)
        return image
