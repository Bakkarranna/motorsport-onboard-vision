"""Command line interface.

Examples
--------
    onboardvision evaluate
    onboardvision evaluate --duration 1.5 --output reports
    onboardvision render --frames 3 --output reports/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evaluate import default_setups, load_track, run_resolution_sweep, run_setup
from .plots import plot_frames, plot_setups, plot_speed, plot_yaw
from .render import OnboardSequence, RenderConfig
from .report import write_html, write_json, write_markdown


def _evaluate(args: argparse.Namespace) -> int:
    track = load_track()
    setups = default_setups(duration=args.duration)
    if args.setups:
        wanted = {name.strip() for name in args.setups.split(",")}
        setups = [setup for setup in setups if setup.name in wanted]

    print(f"Track length {track.length:.0f} m, {len(setups)} setups, {args.duration:.1f} s each")
    print(f"{'setup':16s} {'cover%':>7s} {'MAE m/s':>8s} {'MAE %':>7s} {'bias':>7s} {'yaw MAE':>8s} {'ms':>6s}")
    runs = []
    for setup in setups:
        run = run_setup(track, setup)
        runs.append(run)
        metrics = run.metrics
        print(
            f"{setup.name:16s} {metrics.coverage * 100.0:7.1f} {metrics.speed_mae:8.2f}"
            f" {metrics.speed_mae_percent:7.2f} {metrics.speed_bias:+7.2f}"
            f" {metrics.yaw_mae:8.3f} {run.frame['ms'].mean():6.0f}"
        )

    resolution: list[dict] = []
    if not args.no_sweeps:
        print("\nresolution sweep")
        resolution = run_resolution_sweep(track, duration=args.duration / 2.0)
        print(f"{'camera':10s} {'cover%':>7s} {'MAE m/s':>8s} {'MAE %':>7s} {'ms':>6s}")
        for row in resolution:
            print(
                f"{row['width']}x{row['height']:<4d} {row['coverage'] * 100.0:7.1f}"
                f" {row['speed_mae_mps']:8.2f} {row['speed_mae_percent']:7.2f}"
                f" {row['ms_per_frame']:6.0f}"
            )

    output = Path(args.output)
    figures: list[Path] = []
    if not args.no_figures and runs:
        figures_dir = output / "figures"
        nominal = runs[0]
        sequence = OnboardSequence(track, nominal.setup.render)
        figures.append(plot_frames(sequence, [8, 40, 80, 120], figures_dir / "frames.png"))
        figures.append(plot_speed(nominal, figures_dir / "speed.png"))
        figures.append(plot_yaw(nominal, figures_dir / "yaw_rate.png"))
        figures.append(plot_setups(runs, figures_dir / "setups.png"))
        print(f"\nwrote {len(figures)} figures to {figures_dir}")

    write_json(runs, resolution, output / "results.json")
    write_markdown(runs, resolution, output / "RESULTS.md")
    if figures:
        write_html(runs, resolution, figures, output / "preview.html")
        print(f"wrote {output / 'preview.html'}")
    return 0


def _render(args: argparse.Namespace) -> int:
    track = load_track()
    sequence = OnboardSequence(track, RenderConfig(duration=max(0.05, args.frames / 60.0)))
    indices = [int(value) for value in args.indices.split(",")]
    output = Path(args.output)
    path = plot_frames(sequence, indices, output / "frames.png")
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onboardvision", description="Monocular onboard motion")
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate", help="run the experiment grid and write the report")
    evaluate.add_argument("--duration", type=float, default=2.0, help="seconds of footage per setup")
    evaluate.add_argument("--output", default="reports", help="directory for the report")
    evaluate.add_argument("--setups", default=None, help="comma separated subset of setup names")
    evaluate.add_argument("--no-figures", action="store_true")
    evaluate.add_argument("--no-sweeps", action="store_true", help="skip the resolution sweep")
    evaluate.set_defaults(func=_evaluate)

    render = sub.add_parser("render", help="draw sample frames")
    render.add_argument("--frames", type=int, default=4)
    render.add_argument("--indices", default="8,40,80,120")
    render.add_argument("--output", default="reports/figures")
    render.set_defaults(func=_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
