"""Figures for the onboard vision report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .evaluate import SetupRun

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    }
)

ACCENT = "#1d3557"
TRUTH = "#e63946"
MUTED = "#457b9d"
WARM = "#f4a261"


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_frames(sequence, indices: list[int], path: str | Path) -> Path:
    """A row of rendered frames, so the reader can see the test bed."""
    last = len(sequence.truth) - 1
    wanted = sorted({min(int(index), last) for index in indices})
    indices = wanted
    fig, axes = plt.subplots(1, len(indices), figsize=(3.1 * len(indices), 2.1))
    for axis, index in zip(np.atleast_1d(axes), indices, strict=False):
        axis.imshow(sequence.frame(index))
        axis.set_title(f"frame {index}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    return _save(fig, path)


def plot_speed(run: SetupRun, path: str | Path) -> Path:
    frame = run.frame
    ok = frame[frame["ok"]]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(frame["time"], frame["truth_speed"], color=TRUTH, lw=1.3, label="reference")
    axes[0].plot(ok["time"], ok["speed"], color=ACCENT, lw=1.0, label="visual odometry")
    if len(frame) > len(ok):
        rejected = frame[~frame["ok"]]
        axes[0].scatter(rejected["time"], rejected["truth_speed"], s=9, color=MUTED, zorder=3, label="no estimate")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title(f"Speed from a single camera, {run.setup.name.replace('_', ' ')} footage")
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="lower right")
    axes[1].plot(ok["time"], ok["speed"] - ok["truth_speed"], color=ACCENT, lw=1.0)
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set_ylabel("Error (m/s)")
    axes[1].set_xlabel("Time (s)")
    return _save(fig, path)


def plot_yaw(run: SetupRun, path: str | Path) -> Path:
    frame = run.frame
    ok = frame[frame["ok"]]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(frame["time"], frame["truth_yaw_rate"], color=TRUTH, lw=1.3, label="reference")
    axes[0].plot(ok["time"], ok["yaw_rate"], color=MUTED, lw=1.0, label="visual odometry")
    axes[0].set_ylabel("Yaw rate (rad/s)")
    axes[0].set_title("Yaw rate, the second quantity the road plane gives away")
    axes[0].legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    axes[1].plot(ok["time"], ok["yaw_rate"] - ok["truth_yaw_rate"], color=MUTED, lw=1.0)
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set_ylabel("Error (rad/s)")
    axes[1].set_xlabel("Time (s)")
    return _save(fig, path)


def plot_setups(runs: list[SetupRun], path: str | Path) -> Path:
    names = [run.setup.name.replace("_", " ") for run in runs]
    positions = np.arange(len(names))
    speed = [run.metrics.speed_mae_percent for run in runs]
    yaw = [run.metrics.yaw_mae for run in runs]
    coverage = [run.metrics.coverage * 100.0 for run in runs]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].bar(positions, speed, color=ACCENT)
    axes[0].set_ylabel("Speed error (percent)")
    axes[0].set_title("Speed error on frames that produced an estimate")
    axes[1].bar(positions, yaw, color=MUTED)
    axes[1].set_ylabel("Yaw rate MAE (rad/s)")
    axes[1].set_title("Yaw rate error while cornering")
    axes[2].bar(positions, coverage, color=WARM)
    axes[2].set_ylabel("Frames with an estimate (percent)")
    axes[2].set_ylim(0, 105)
    axes[2].set_title("Coverage")
    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(names, rotation=28, ha="right", fontsize=8)
    return _save(fig, path)


def plot_trust(runs: list[SetupRun], path: str | Path) -> Path:
    """Residual and inlier history for the clean and the untextured footage."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.0), sharex=True)
    for run, colour in zip(runs, (ACCENT, WARM)):
        frame = run.frame
        axes[0].plot(
            frame["time"], frame["rms_px"], color=colour, lw=0.9, label=run.setup.name.replace("_", " ")
        )
        axes[1].plot(
            frame["time"], frame["inliers"], color=colour, lw=0.9, label=run.setup.name.replace("_", " ")
        )
    axes[0].axhline(2.5, color="black", lw=0.9, ls="--", label="rejection threshold")
    axes[0].set_ylabel("Fit residual (px)")
    axes[0].set_title("Does the estimator know when it is wrong?")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].axhline(15, color="black", lw=0.9, ls="--", label="minimum inliers")
    axes[1].set_ylabel("Inliers per frame")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    return _save(fig, path)


def plot_resolution(rows: list[dict], path: str | Path) -> Path:
    width = [row["width"] for row in rows]
    speed = [row["speed_mae_percent"] for row in rows]
    coverage = [row["coverage"] * 100.0 for row in rows]
    fig, axis = plt.subplots(figsize=(6.0, 3.2))
    axis.plot(width, speed, marker="o", color=ACCENT, label="speed error")
    axis.set_xlabel("Image width (pixels)")
    axis.set_ylabel("Speed error (percent)")
    axis.set_title("Resolution against accuracy")
    second = axis.twinx()
    second.plot(width, coverage, marker="s", color=WARM, label="coverage")
    second.set_ylabel("Coverage (percent)")
    second.set_ylim(0, 105)
    second.grid(False)
    handles = axis.get_lines() + second.get_lines()
    axis.legend(handles, [line.get_label() for line in handles], frameon=False, fontsize=8)
    return _save(fig, path)


def plot_errors(runs: list[SetupRun], path: str | Path) -> Path:
    """Distribution of the per frame speed error across the setups."""
    fig, axis = plt.subplots(figsize=(7.0, 3.4))
    bins = np.linspace(-10.0, 6.0, 41)
    for run, colour in zip(runs, (ACCENT, MUTED, WARM)):
        frame = run.frame
        ok = frame[frame["ok"]]
        if ok.empty:
            continue
        axis.hist(
            ok["speed"] - ok["truth_speed"],
            bins=bins,
            histtype="step",
            lw=1.4,
            color=colour,
            label=run.setup.name.replace("_", " "),
        )
    axis.axvline(0.0, color="black", lw=0.8)
    axis.set_xlabel("Per frame speed error (m/s)")
    axis.set_ylabel("Frames")
    axis.set_title("Per frame error, before any temporal filtering")
    axis.legend(frameon=False, fontsize=8)
    return _save(fig, path)
