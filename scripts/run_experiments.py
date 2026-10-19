"""Run the whole experiment grid and write the report.

    python scripts/run_experiments.py --duration 2.0

Rendering is the slow part: about half a second per frame at 640 by 360, and
every number in the README comes from this script. Use ``--quick`` to check the
wiring without waiting for the full grid.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onboardvision.evaluate import (  # noqa: E402
    default_setups,
    load_track,
    parameter_sweep,
    run_resolution_sweep,
    run_setup,
    smoothing_gain,
)
from onboardvision.plots import (  # noqa: E402
    plot_errors,
    plot_frames,
    plot_resolution,
    plot_setups,
    plot_speed,
    plot_trust,
    plot_yaw,
)
from onboardvision.render import OnboardSequence  # noqa: E402
from onboardvision.report import write_html, write_json, write_markdown  # noqa: E402
from onboardvision.track import Track  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the onboard vision experiments")
    parser.add_argument("--duration", type=float, default=2.0, help="seconds of footage per setup")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--quick", action="store_true", help="short runs, for wiring checks only")
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--skip-parameter-sweep", action="store_true")
    args = parser.parse_args()

    duration = 0.6 if args.quick else args.duration
    track: Track = load_track()
    output = Path(args.output)
    figures = output / "figures"

    runs = []
    started = time.time()
    for setup in default_setups(duration=duration):
        run = run_setup(track, setup)
        runs.append(run)
        metrics = run.metrics
        print(
            f"{setup.name:16s} coverage {metrics.coverage * 100.0:5.1f}%"
            f"  speed MAE {metrics.speed_mae:6.2f} m/s ({metrics.speed_mae_percent:5.2f}%)"
            f"  bias {metrics.speed_bias:+6.2f}"
            f"  yaw MAE {metrics.yaw_mae:6.3f}"
            f"  {run.frame['ms'].mean():5.0f} ms/frame"
        )

    resolution = [] if args.skip_sweeps else run_resolution_sweep(track, duration=duration)
    if resolution:
        for row in resolution:
            print(
                f"resolution {row['width']}x{row['height']:<4d} coverage {row['coverage'] * 100.0:5.1f}%"
                f"  speed MAE {row['speed_mae_mps']:6.2f} m/s ({row['speed_mae_percent']:5.2f}%)"
                f"  {row['ms_per_frame']:5.0f} ms/frame"
            )

    sequence = OnboardSequence(track, default_setups(duration=1.0)[0].render)
    figure_paths = [
        plot_frames(sequence, [8, 40, 80, 120], figures / "frames.png"),
        plot_speed(runs[0], figures / "speed.png"),
        plot_yaw(runs[0], figures / "yaw_rate.png"),
        plot_setups(runs, figures / "setups.png"),
        plot_errors(runs[:3], figures / "error_distribution.png"),
        plot_trust([runs[0], runs[3]], figures / "trust.png"),
    ]
    if resolution:
        figure_paths.append(plot_resolution(resolution, figures / "resolution.png"))

    sweep = []
    if not args.skip_parameter_sweep:
        print("\ntracking configuration sweep (one rendered sequence, replayed for each config)")
        sweep = parameter_sweep(track, duration=duration)
        for row in sweep:
            print(
                f"{row['name']:20s} coverage {row['coverage'] * 100.0:5.1f}%"
                f"  speed MAE {row['speed_mae_mps']:6.2f} m/s ({row['speed_mae_percent']:5.2f}%)"
                f"  {row['ms_per_frame']:5.0f} ms/frame"
            )

    smoothing = smoothing_gain(runs[0].frame) if runs else None
    if smoothing:
        print(
            f"\ncentred five frame average of the per frame speed estimate: MAE "
            f"{smoothing['raw_mae']:.2f} to {smoothing['smoothed_mae']:.2f} m/s"
        )

    write_json(runs, resolution, output / "results.json")
    write_markdown(runs, resolution, output / "RESULTS.md", smoothing=smoothing, sweep=sweep)
    write_html(runs, resolution, figure_paths, output / "preview.html")

    print(f"\nwrote {output / 'RESULTS.md'}, {output / 'results.json'} and {len(figure_paths)} figures")
    print(f"total {time.time() - started:.0f} s")
    assert all(np.isfinite(run.metrics.speed_mae) or run.metrics.evaluated == 0 for run in runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
