"""Test-time helpers: durations, configs, and fixture discovery.

This module exists so that other parts of the package can answer two questions
without importing the test modules themselves:

* how long must a rendered sequence be for the standard end-to-end tests to be
  meaningful, and
* what is the default tracking configuration used by the test suite.

The yaw rate test is the binding constraint: on the particular track and speed
the renderer uses, a 1.6 s sequence does not contain enough variation in the
car's heading for the estimator's per-frame output to be distinguished from
measurement noise at the 0.8 correlation ceiling the test asserts. The numbers
below were measured against that test, not chosen heuristically.
"""

from __future__ import annotations

from dataclasses import replace

from onboardvision.features import TrackingConfig

# Minimum duration that makes the yaw-rate correlation assertion meaningful.
#
# Measured on a nominal render (640x360, height 1.15 m, pitch 0.10 rad,
# noise 3 px) with TrackingConfig() on the wave_loop() track at full speed:
#   duration | truth_yaw_std | valid_frames | raw_corr | smoothed_5_corr
#   1.6 s    | 0.0031       | 96           | 0.091    | 0.092
#   3.0 s    | 0.0063       | 179          | 0.309    | 0.531
#   4.5 s    | 0.3960       | 265          | 0.995    | 0.998
#
# The step between 3.0 s and 4.5 s corresponds to the segment crossing a real
# corner: before that the car is in a near-constant-speed straighter section and
# the truth yaw barely moves. The 4.5 s minimum is what the standard tests use.
REQUIRED_YAW_DURATION = 4.5

# Default tracking configuration shared by the test suite. This is the "many
# weak corners, short window" config that the report selected as the best
# compromise on the rendered footage; it is NOT the same as the one you would
# tune for a different camera or a different track.
TEST_TRACKING_CONFIG = replace(
    TrackingConfig(),
    pyramid_levels=2,
)


def default_test_duration() -> float:
    """Return REQUIRED_YAW_DURATION.

    Deprecated:: use the constant directly. Kept for one call site inside
    ``scripts/build_report.py`` that still uses it as a function.
    """
    return REQUIRED_YAW_DURATION
