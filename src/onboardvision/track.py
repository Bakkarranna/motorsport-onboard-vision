"""Track geometry and a lookup from world position to track coordinates.

A track is defined by a closed centreline and a half width. The lookup grid
turns a world point into a distance along the centreline (s) and an offset from
it (n), which is what the procedural renderer needs in order to draw asphalt,
kerbs and markings in the right place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CENTRELINE_SPACING = 0.25
GRID_CELL = 1.0
LOOKUP_OFFSET = 8
# Curvature is a second derivative of a sampled polyline, so raw differencing
# amplifies the sampling noise into the racing line and the speed profile. Both
# are smoothed over a few metres, which is the scale at which a real circuit
# changes shape.
CENTRELINE_SMOOTHING = 9
CURVATURE_SMOOTHING = 41


def smooth_periodic(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average along the first axis of a periodic signal."""
    if window <= 1 or window >= len(values):
        return values
    kernel = np.ones(window) / window
    half = window // 2
    padded = np.concatenate([values[-half:], values, values[: window - 1 - half]], axis=0)
    smoothed = np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="valid"), 0, padded)
    return smoothed


def wave_loop(points: int = 4000, radius: float = 190.0, wobble: float = 25.0) -> np.ndarray:
    """A closed curve with a mix of fast and slow corners.

    The minimum radius is about 38 m and the curvature stays under 0.027 per
    metre, which is in the range of a real circuit rather than a hairpin.
    """
    t = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    x = radius * np.cos(t) + wobble * np.cos(4.0 * t)
    y = radius * np.sin(t) + wobble * np.sin(4.0 * t)
    return np.column_stack([x, y])


@dataclass
class Track:
    centreline: np.ndarray
    headings: np.ndarray
    curvature: np.ndarray
    arc_length: np.ndarray
    half_width: float = 6.0
    kerb_width: float = 3.0
    _lookup: TrackLookup | None = field(default=None, init=False, repr=False)

    @property
    def length(self) -> float:
        return float(self.arc_length[-1])

    def centreline_lookup(self) -> TrackLookup:
        """Build the world to track coordinate lookup once and reuse it."""
        if self._lookup is None:
            self._lookup = TrackLookup(self)
        return self._lookup

    @classmethod
    def from_points(
        cls, points: np.ndarray, half_width: float = 6.0, kerb_width: float = 3.0
    ) -> Track:
        closed = np.vstack([points, points[:1]])
        segments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        arc_length = np.concatenate([[0.0], np.cumsum(segments)])
        total = arc_length[-1]
        count = int(total / CENTRELINE_SPACING)
        sample = np.linspace(0.0, total, count, endpoint=False)

        dense_x = np.interp(sample, arc_length, closed[:, 0])
        dense_y = np.interp(sample, arc_length, closed[:, 1])
        centreline = smooth_periodic(
            np.column_stack([dense_x, dense_y]), CENTRELINE_SMOOTHING
        )

        forward = np.roll(centreline, -1, axis=0) - np.roll(centreline, 1, axis=0)
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        headings = np.arctan2(forward[:, 1], forward[:, 0])

        unwrapped = np.unwrap(headings)
        spacing = sample[1] - sample[0]
        curvature = smooth_periodic(np.gradient(unwrapped, spacing), CURVATURE_SMOOTHING)
        return cls(
            centreline=centreline,
            headings=headings,
            curvature=curvature,
            arc_length=np.arange(count, dtype=float) * spacing,
            half_width=half_width,
            kerb_width=kerb_width,
        )

    def lookup_grid(self) -> TrackLookup:
        return TrackLookup(self)


class TrackLookup:
    """Nearest centreline index for every cell of a coarse grid."""

    def __init__(self, track: Track) -> None:
        self.track = track
        centreline = track.centreline
        lower = centreline.min(axis=0) - 20.0
        upper = centreline.max(axis=0) + 20.0
        self.lower = lower
        self.origin = lower
        shape = np.ceil((upper - lower) / GRID_CELL).astype(int) + 1
        self.shape = tuple(shape)
        self.index = np.full(shape, -1, dtype=np.int32)

        reach = track.half_width + track.kerb_width + 4.0
        steps = int(np.ceil(reach / GRID_CELL))
        offsets = np.arange(-steps, steps + 1) * GRID_CELL
        offsets = np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)

        # One centreline point every metre is enough for a one metre grid.
        for idx, point in enumerate(centreline[::4]):
            idx = idx * 4
            locations = point + offsets - self.origin
            cells = np.floor(locations / GRID_CELL).astype(int)
            inside = (
                (cells[:, 0] >= 0)
                & (cells[:, 0] < shape[0])
                & (cells[:, 1] >= 0)
                & (cells[:, 1] < shape[1])
            )
            if not inside.any():
                continue
            rows = cells[inside, 0]
            columns = cells[inside, 1]
            centres = self.origin + (cells[inside] + 0.5) * GRID_CELL
            new_distance = np.linalg.norm(centres - point, axis=1)
            stored = self.index[rows, columns]
            old_distance = np.full(len(rows), np.inf)
            valid = stored >= 0
            if valid.any():
                old_distance[valid] = np.linalg.norm(
                    centres[valid] - centreline[stored[valid]], axis=1
                )
            better = new_distance < old_distance
            self.index[rows[better], columns[better]] = idx

    def coordinates(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return arc length and lateral offset for arrays of world points."""
        track = self.track
        centreline = track.centreline.astype(np.float32)
        count = len(centreline)

        cells = np.stack(
            [
                ((x - self.origin[0]) / GRID_CELL).astype(int),
                ((y - self.origin[1]) / GRID_CELL).astype(int),
            ]
        )
        cells[0] = np.clip(cells[0], 0, self.shape[0] - 1)
        cells[1] = np.clip(cells[1], 0, self.shape[1] - 1)
        seed = self.index[cells[0], cells[1]]
        # Cells that never came close to the centreline are not on the track at
        # all. They get a sentinel so the renderer can draw them as scenery.
        valid = seed >= 0
        seed = np.where(valid, seed, 0)

        # Refine the coarse index with a small local search along the centreline.
        # Walking the candidates one at a time keeps this to a few arrays the
        # size of the image instead of one array per candidate, which matters
        # because the renderer calls it for every frame.
        x32 = x.astype(np.float32)
        y32 = y.astype(np.float32)
        best = np.zeros(len(x), dtype=np.int64)
        best_distance = np.full(len(x), np.inf, dtype=np.float32)
        for offset in range(-LOOKUP_OFFSET, LOOKUP_OFFSET + 1):
            indices = (seed + offset) % count
            dx = centreline[indices, 0] - x32
            dy = centreline[indices, 1] - y32
            distance = dx * dx + dy * dy
            better = distance < best_distance
            best_distance = np.where(better, distance, best_distance)
            best = np.where(better, indices, best)
        indices = best

        point = centreline[indices]
        heading = track.headings[indices]
        dx = x - point[:, 0]
        dy = y - point[:, 1]
        # Signed lateral offset, positive to the left of the direction of travel.
        offset = np.cos(heading) * dy - np.sin(heading) * dx
        arc = track.arc_length[indices]
        return np.where(valid, arc, -1.0), np.where(valid, offset, 999.0)
