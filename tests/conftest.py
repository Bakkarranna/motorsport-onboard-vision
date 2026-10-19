"""pytest configuration for the onboard vision test suite.

The suite is split into two groups:

* ``tests/test_track.py``, ``tests/test_planar.py`` and ``tests/test_features.py``
  are fast unit-style tests that render tiny images or skip rendering entirely.
  They run on every change.

* ``tests/test_motion.py`` renders full 640x360 frames at real speed
  (roughly half a second per frame) and is therefore slow. The yaw-rate
  assertion is the binding constraint: a shorter sequence does not contain
  enough variation in the car's heading to be measurable through the
  per-frame noise floor. The minimum duration that passes it is 4.5 s, which
  is what :data:`onboardvision.tests.REQUIRED_YAW_DURATION` records.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

INSIDE_CI = os.environ.get("CI", "").lower() in {"1", "true", "yes"}


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--full-motion",
        action="store_true",
        default=False,
        help="run the full 4.5 s motion tests instead of the default short substitutes",
    )


def pytest_configure(config: Any) -> None:
    # Freeze the long-running motion tests by default so that a local run
    # stays interactive. The full suite still has them, and CI unlocks them.
    if INSIDE_CI:
        return
    marker = config.getoption("full_motion")
    if not marker:
        config.option.markexpr = "not slow_motion"


def pytest_collection_modifyitems(
    session: Any, config: Any, items: list[Any]
) -> None:
    marker = config.getoption("full_motion")
    if not marker:
        return
    # When run with --full-motion, drop the marker expression so that the
    # slow_motion skip is not applied.
    config.option.markexpr = ""



def pytest_collection_modifyitems(
    session: Any, config: Any, items: list[Any]
) -> None:
    marker = config.getoption("full_motion")
    if not marker:
        return
    # When run with --full-motion, drop the marker expression so that the
    # slow_motion skip, and the unit_tracking override, are not applied.
    config.option.markexpr = ""
