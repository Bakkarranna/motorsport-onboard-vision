"""Metrics and experiment definitions.

A visual odometry run produces one row per frame. Compared with a filter, the
interesting number is not only the error on the frames that produced an
estimate: it is also how many frames produced an estimate at all. A method that
is exact on ten percent of the frames is useless on a race car, so every metric
here is reported next to the coverage that it was measured on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import TrackingConfig
from .geometry import CameraModel
from .motion import HomographyEstimator, run_sequence
from .render import OnboardSequence, RenderConfig
from .track import Track, wave_loop

WARMUP_SECONDS = 0.5


@dataclass(frozen=True)
class SequenceMetrics:
    """Error summary for one rendered run."""

    frames: int
    evaluated: int
    coverage: float
    speed_mae: float
    speed_rmse: float
    speed_bias: float
    speed_mae_percent: float
    yaw_mae: float
    yaw_rmse: float
    yaw_sign_share: float

    def as_dict(self) -> dict[str, float]:
        return {
            "frames": float(self.frames),
            "evaluated": float(self.evaluated),
            "coverage": self.coverage,
            "speed_mae_mps": self.speed_mae,
            "speed_rmse_mps": self.speed_rmse,
            "speed_bias_mps": self.speed_bias,
            "speed_mae_percent": self.speed_mae_percent,
            "yaw_rate_mae_rad_s": self.yaw_mae,
            "yaw_rate_rmse_rad_s": self.yaw_rmse,
            "yaw_sign_share": self.yaw_sign_share,
        }


def metrics_from_frame(frame: pd.DataFrame) -> SequenceMetrics:
    """Summarise the frame level output of :func:`onboardvision.motion.run_sequence`.

    A run that produced no estimates at all is a valid outcome, not an error:
    it is exactly what the untextured road is supposed to produce. It comes back
    as full zeros with NaN errors so that it still shows up in the report.
    """
    if frame.empty:
        return SequenceMetrics(
            frames=0,
            evaluated=0,
            coverage=0.0,
            speed_mae=float("nan"),
            speed_rmse=float("nan"),
            speed_bias=float("nan"),
            speed_mae_percent=float("nan"),
            yaw_mae=float("nan"),
            yaw_rmse=float("nan"),
            yaw_sign_share=float("nan"),
        )
    valid = frame[frame["ok"]]
    if valid.empty:
        return SequenceMetrics(
            frames=len(frame),
            evaluated=0,
            coverage=0.0,
            speed_mae=float("nan"),
            speed_rmse=float("nan"),
            speed_bias=float("nan"),
            speed_mae_percent=float("nan"),
            yaw_mae=float("nan"),
            yaw_rmse=float("nan"),
            yaw_sign_share=float("nan"),
        )

    speed_error = valid["speed"] - valid["truth_speed"]
    yaw_error = valid["yaw_rate"] - valid["truth_yaw_rate"]
    # Yaw rate is only meaningful when the car is actually turning, otherwise
    # the estimate is noise divided by noise.
    turning = valid["truth_yaw_rate"].abs() > 0.05

    return SequenceMetrics(
        frames=len(frame),
        evaluated=len(valid),
        coverage=len(valid) / len(frame),
        speed_mae=float(speed_error.abs().mean()),
        speed_rmse=float(np.sqrt((speed_error**2).mean())),
        speed_bias=float(speed_error.mean()),
        speed_mae_percent=float((speed_error.abs() / valid["truth_speed"]).mean() * 100.0),
        yaw_mae=float(yaw_error[turning].abs().mean()) if turning.any() else float("nan"),
        yaw_rmse=float(np.sqrt((yaw_error[turning] ** 2).mean())) if turning.any() else float("nan"),
        yaw_sign_share=float(
            np.mean(np.sign(valid["yaw_rate"][turning]) == np.sign(valid["truth_yaw_rate"][turning]))
        )
        if turning.any()
        else float("nan"),
    )


@dataclass(frozen=True)
class Setup:
    """One point of the experiment grid."""

    name: str
    description: str
    render: RenderConfig = field(default_factory=RenderConfig)
    camera_height: float = 1.15
    camera_pitch: float = 0.10
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    # Deliberate mismatch between the height assumed by the estimator and the
    # height used by the renderer, used to show the scale dependence.
    assumed_height: float | None = None


def default_setups(duration: float = 2.0) -> list[Setup]:
    """The experiments reported in the README."""
    return [
        Setup(
            name="nominal",
            description="Clean footage, correct camera height and pitch.",
            render=RenderConfig(duration=duration),
        ),
        Setup(
            name="noise_high",
            description="Sensor noise quadrupled, as on a compressed broadcast feed.",
            render=RenderConfig(duration=duration, noise_std=12.0),
        ),
        Setup(
            name="texture_weak",
            description="Road texture reduced to 35 percent, like fresh asphalt in shade.",
            render=RenderConfig(duration=duration, texture_scale=0.35),
        ),
        Setup(
            name="texture_none",
            description="Essentially untextured road, to show what the estimator does when it should give up.",
            render=RenderConfig(duration=duration, texture_scale=0.05),
        ),
        Setup(
            name="roll_high",
            description="Body roll of 0.06 rad per g enabled, so the camera rotates about the optical axis.",
            render=RenderConfig(duration=duration, roll_gain=0.06),
        ),
        Setup(
            name="height_wrong",
            description="Estimator assumes a camera 20 cm lower than it really is.",
            render=RenderConfig(duration=duration),
            assumed_height=0.95,
        ),
        Setup(
            name="texture_weak_gated",
            description=(
                "Same footage as texture_weak with the inlier gate raised to 100, showing "
                "that the feature count can be used to refuse a frame instead of guessing."
            ),
            render=RenderConfig(duration=duration, texture_scale=0.35),
            tracking=TrackingConfig(min_inliers=100),
        ),
    ]


@dataclass
class SetupRun:
    setup: Setup
    frame: pd.DataFrame
    metrics: SequenceMetrics

    def truth_vs_estimate(self) -> pd.DataFrame:
        return self.frame[["time", "truth_speed", "speed", "truth_yaw_rate", "yaw_rate", "ok"]]


def run_setup(track: Track, setup: Setup, warmup: float = WARMUP_SECONDS) -> SetupRun:
    sequence = OnboardSequence(track, setup.render)
    estimator = HomographyEstimator(
        camera=setup.render.camera,
        height=setup.assumed_height or setup.render.height,
        dt=1.0 / setup.render.fps,
        pitch=setup.camera_pitch,
        config=setup.tracking,
    )
    frame = run_sequence(sequence, estimator)
    if warmup > 0:
        frame = frame[frame["time"] >= warmup].reset_index(drop=True)
    return SetupRun(setup=setup, frame=frame, metrics=metrics_from_frame(frame))


def nominal_duration() -> float:
    """Minimum duration a rendered sequence needs for the yaw signal to be measurable.

    On a 1.6 s sequence the truth yaw rate barely leaves the constant-speed
    regime, so the per-frame estimate's correlation against the truth is dominated
    by measurement noise and fails the 0.8 gate regardless of estimator quality.
    This is the figure the test thresholds were chosen against; the report
    experiments use the same minimum.

    Deprecated:: use :data:`onboardvision.tests.REQUIRED_YAW_DURATION` instead.
    """


def load_track() -> Track:
    return Track.from_points(wave_loop())


def smoothing_gain(frame: pd.DataFrame, window: int = 5) -> dict[str, float]:
    """What a centred moving average of the per frame estimates would buy.

    The visual estimate is noisy frame to frame, and that noise is close to
    white, so a short window should cut the error by roughly the square root of
    the window length. This is a property of the measurement, not of the
    estimator: it is the reason every real system filters the output.
    """
    valid = frame[frame["ok"]].reset_index(drop=True)
    if len(valid) < window:
        return {"raw_mae": float("nan"), "smoothed_mae": float("nan"), "windows": 0.0}
    if len(valid) < window:
        return {"raw_mae": float("nan"), "smoothed_mae": float("nan"), "windows": 0.0}
    smoothed = valid["speed"].rolling(window, center=True).mean()
    keep = smoothed.notna()
    error = smoothed[keep] - valid["truth_speed"][keep]
    return {
        "raw_mae": float((valid["speed"] - valid["truth_speed"]).abs().mean()),
        "smoothed_mae": float(error.abs().mean()),
        "windows": float(keep.sum()),
    }


def parameter_sweep(
    track: Track,
    duration: float = 1.5,
    setups: list[Setup] | None = None,
) -> list[dict]:
    """Compare tracking configurations on one rendered sequence.

    The footage is rendered once and replayed for every configuration, so the
    comparison is on identical input.
    """
    from .motion import HomographyEstimator, estimate_frames, prepare

    base = (setups or default_setups(duration=duration))[0]
    sequence = OnboardSequence(track, base.render)
    frames = prepare(sequence)
    rows = []
    for name, config in sweep_configurations():
        estimator = HomographyEstimator(
            camera=base.render.camera,
            height=base.render.height,
            dt=1.0 / base.render.fps,
            pitch=base.camera_pitch,
            config=config,
        )
        frame = estimate_frames(frames, estimator, sequence.truth)
        frame = frame[frame["time"] >= WARMUP_SECONDS]
        metrics = metrics_from_frame(frame)
        rows.append(
            {
                "name": name,
                "quality": config.quality,
                "min_distance": config.min_distance,
                "window": config.window,
                "ms_per_frame": float(frame["ms"].mean()),
                **{key: float(value) for key, value in metrics.as_dict().items()},
            }
        )
    return rows


def sweep_configurations() -> list[tuple[str, TrackingConfig]]:
    """The grid reported in the README, in the order it was run."""
    grid = [
        ("strong_long_window", TrackingConfig(quality=0.01, min_distance=6.0, block_size=7, window=21)),
        ("few_weak_corners", TrackingConfig(quality=0.005, min_distance=6.0, block_size=7, window=21)),
        ("many_short_window", TrackingConfig(quality=0.003, min_distance=4.0, block_size=5, window=15)),
        ("selected", TrackingConfig()),
        ("short_window", TrackingConfig(window=9)),
        ("deep_pyramid", TrackingConfig(pyramid_levels=5)),
        ("shallow_pyramid", TrackingConfig(pyramid_levels=3)),
    ]
    return grid


def camera_suite() -> list[CameraModel]:
    """The cameras used in the resolution sweep, all with the same field of view."""
    return [
        CameraModel(width=320, height=180, fx=260.0, fy=260.0),
        CameraModel(width=640, height=360, fx=520.0, fy=520.0),
        CameraModel(width=960, height=540, fx=780.0, fy=780.0),
    ]


def run_resolution_sweep(track: Track, duration: float = 1.0) -> list[dict]:
    """Does more resolution buy accuracy, and is it worth the compute?"""
    rows = []
    for camera in camera_suite():
        setup = Setup(
            name=f"{camera.width}x{camera.height}",
            description="Resolution sweep, same field of view and same footage.",
            render=RenderConfig(camera=camera, duration=duration),
        )
        run = run_setup(track, setup)
        rows.append(
            {
                "width": camera.width,
                "height": camera.height,
                "ms_per_frame": float(run.frame["ms"].mean()),
                **{key: float(value) for key, value in run.metrics.as_dict().items()},
            }
        )
    return rows
