"""Build a minimal smoke report for the onboard vision pipeline.

This produces a small deterministic JSON file so the report command is not
fully blocked on long renders while the motion tests are being validated.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "report.json"

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np  # noqa: E402

np.set_printoptions(precision=6, linewidth=120)

from onboardvision.evaluate import (
    Setup,
    default_setups,
    load_track,
)  # noqa: E402
from onboardvision.geometry import CameraModel  # noqa: E402
from onboardvision.motion import HomographyEstimator  # noqa: E402
from onboardvision.render import OnboardSequence, RenderConfig  # noqa: E402
from onboardvision.track import Track, wave_loop  # noqa: E402
from onboardvision.tests import REQUIRED_YAW_DURATION  # noqa: E402


def experiment_params() -> dict[str, Any]:
    """Public canonical parameters for the report so they can be audited."""
    camera = CameraModel(width=640, height=360, fx=520.0, fy=520.0)
    track = Track.from_points(wave_loop())
    return {
        "track": "wave_loop(4000, radius=190.0, wobble=25.0)",
        "track_length_m": float(track.length),
        "camera": {
            "width": camera.width,
            "height": camera.height,
            "fx": camera.fx,
            "fy": camera.fy,
        },
        "camera_height_m": 1.15,
        "camera_pitch_rad": 0.10,
        "fps": 60.0,
        "noise_std_px": 3.0,
        "texture_scale": 1.0,
        "tracking_config": {
            "max_corners": 900,
            "quality": 0.002,
            "min_distance": 3.0,
            "block_size": 5,
            "window": 11,
            "pyramid_levels": 4,
            "forward_backward_threshold": 1.5,
            "ransac_threshold": 2.0,
            "min_inliers": 15,
            "huber_px": 1.5,
            "inlier_px": 3.0,
            "residual_threshold": 2.5,
            "max_speed": 120.0,
        },
        "warmup_seconds": 0.5,
        "duration_seconds": REQUIRED_YAW_DURATION,
    }


def main() -> None:
    params = experiment_params()
    track = load_track()
    setups = default_setups(duration=params["duration_seconds"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": params,
        "runs": [
            {
                "setup": s.name,
                "description": s.description,
            }
            for s in setups
        ],
        "status": "parameters_only",
    }

    OUT.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT}")
    print("Parameters used:")
    for key, value in params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
