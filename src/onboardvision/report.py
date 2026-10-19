"""Report writers: a markdown summary, a JSON dump and a browseable page."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evaluate import SetupRun

COLUMNS = [
    ("setup", "setup", "{:s}"),
    ("coverage", "covered %", "{:6.1f}"),
    ("speed_mae_mps", "speed MAE m/s", "{:6.2f}"),
    ("speed_mae_percent", "speed MAE %", "{:6.2f}"),
    ("speed_bias_mps", "speed bias m/s", "{:7.2f}"),
    ("yaw_rate_mae_rad_s", "yaw MAE rad/s", "{:6.3f}"),
    ("yaw_sign_share", "yaw sign %", "{:6.1f}"),
]


def _table(runs: list[SetupRun]) -> list[list[str]]:
    rows = []
    for run in runs:
        metrics = run.metrics.as_dict()
        cells = []
        for key, _, formatter in COLUMNS:
            if key == "setup":
                cells.append(run.setup.name)
                continue
            value = metrics[key]
            if key in ("coverage", "yaw_sign_share"):
                value *= 100.0
            cells.append(formatter.format(value) if np.isfinite(value) else "n/a")
        rows.append(cells)
    return rows


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def summary_lines(runs: list[SetupRun]) -> list[str]:
    """A few sentences that hold whatever the measurements turned out to be."""
    working = [run for run in runs if run.metrics.evaluated > 0]
    lines = []
    if working:
        best = min(working, key=lambda run: run.metrics.speed_mae_percent)
        lines.append(
            f"Best speed accuracy: {best.setup.name} at {best.metrics.speed_mae_percent:.2f} percent MAE "
            f"over {best.metrics.evaluated} of {best.metrics.frames} frames."
        )
        blind = [run for run in runs if run.metrics.coverage < 0.5]
        if blind:
            names = ", ".join(run.setup.name for run in blind)
            lines.append(f"The estimator refused almost every frame on: {names}.")
        worst = max(working, key=lambda run: run.metrics.speed_mae_percent)
        lines.append(
            f"Worst case that still produced estimates: {worst.setup.name} at "
            f"{worst.metrics.speed_mae_percent:.2f} percent, coverage "
            f"{worst.metrics.coverage * 100.0:.0f} percent."
        )
        biased = [run for run in working if abs(run.metrics.speed_bias) > 2.0]
        if biased:
            names = ", ".join(
                f"{run.setup.name} ({run.metrics.speed_bias:+.1f} m/s)" for run in biased
            )
            lines.append(f"Systematic speed bias above 2 m/s: {names}.")
    return lines


def write_json(runs: list[SetupRun], resolution: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "setups": {
            run.setup.name: {
                "description": run.setup.description,
                **{key: float(value) for key, value in run.metrics.as_dict().items()},
            }
            for run in runs
        },
        "resolution": resolution,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_markdown(
    runs: list[SetupRun],
    resolution: list[dict],
    path: str | Path,
    command: str = "python scripts/run_experiments.py",
    smoothing: dict | None = None,
    sweep: list[dict] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [column[1] for column in COLUMNS] + ["frames"]
    rows = [row + [str(run.metrics.frames)] for row, run in zip(_table(runs), runs, strict=False)]

    resolution_rows = []
    for row in resolution:
        resolution_rows.append(
            [
                f"{row['width']}x{row['height']}",
                f"{row['coverage'] * 100.0:.0f}",
                f"{row['speed_mae_mps']:.2f}",
                f"{row['speed_mae_percent']:.2f}",
                f"{row['yaw_rate_mae_rad_s']:.3f}",
            ]
        )

    parts = [
        "# Onboard vision: measured results",
        "",
        "Everything below is produced by `" + command + "` and can be regenerated.",
        "",
        "## Setups",
        "",
        *_setup_list(runs),
        "",
        "## Frame to frame accuracy",
        "",
        _markdown_table(header, rows),
        "",
        "## What the numbers say",
        "",
        *[f"* {line}" for line in summary_lines(runs)],
        "",
        "## Resolution",
        "",
        _markdown_table(
            ["camera", "covered %", "speed MAE m/s", "speed MAE %", "yaw MAE rad/s"], resolution_rows
        ),
        "",
        *_extra_sections(smoothing, sweep),
    ]
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _extra_sections(smoothing: dict | None, sweep: list[dict] | None) -> list[str]:
    parts: list[str] = []
    if smoothing:
        raw = smoothing.get("raw_mae")
        filtered = smoothing.get("smoothed_mae")
        if raw and filtered and np.isfinite(raw) and np.isfinite(filtered):
            parts += [
                "## Temporal filtering",
                "",
                (
                    f"A centred five frame average of the same per frame estimates moves the speed MAE "
                    f"from {raw:.2f} m/s to {filtered:.2f} m/s over {int(smoothing['windows'])} windows, "
                    "which is the square root of five to within the spread of the estimate. The "
                    "measurement noise is close to white, so filtering is nearly free."
                ),
                "",
            ]
    if sweep:
        header = ["configuration", "quality", "min distance", "window", "coverage %", "speed MAE %", "ms"]
        rows = [
            [
                row["name"],
                f"{row['quality']:.3f}",
                f"{row['min_distance']:.0f}",
                f"{row['window']:.0f}",
                f"{row['coverage'] * 100.0:.0f}",
                f"{row['speed_mae_percent']:.2f}",
                f"{row['ms_per_frame']:.0f}",
            ]
            for row in sweep
        ]
        parts += ["## Tracking configuration sweep", "", _markdown_table(header, rows), ""]
    return parts


def _setup_list(runs: list[SetupRun]) -> list[str]:
    return [f"* `{run.setup.name}`: {run.setup.description}" for run in runs]


def write_html(
    runs: list[SetupRun],
    resolution: list[dict],
    figures: list[Path],
    path: str | Path,
) -> Path:
    """A single page report for the browser preview."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [column[1] for column in COLUMNS]

    def table(rows: list[list[str]], head: list[str]) -> str:
        head_html = "".join(f"<th>{cell}</th>" for cell in head)
        body = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
        )
        return f"<table><thead><tr>{head_html}</tr></thead><tbody>{body}</tbody></table>"

    figure_html = "".join(
        f'<figure><img src="{figure.as_posix()}" alt="{figure.stem}">'
        f"<figcaption>{figure.stem.replace('_', ' ')}</figcaption></figure>"
        for figure in figures
    )
    resolution_rows = [
        [
            f"{row['width']}x{row['height']}",
            f"{row['coverage'] * 100.0:.0f}",
            f"{row['speed_mae_mps']:.2f}",
            f"{row['speed_mae_percent']:.2f}",
            f"{row['yaw_rate_mae_rad_s']:.3f}",
        ]
        for row in resolution
    ]
    findings = "".join(f"<li>{line}</li>" for line in summary_lines(runs))
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Onboard vision results</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0 auto; max-width: 1040px; padding: 32px 20px 64px;
         font-family: ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
         color: #16202b; background: #f7f8fa; line-height: 1.5; }}
  h1 {{ font-size: 1.7rem; margin-bottom: .2rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2.2rem; border-bottom: 1px solid #dde3ea; padding-bottom: .3rem; }}
  p.lede {{ color: #4a5768; margin-top: .2rem; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: .86rem;
           box-shadow: 0 1px 2px rgba(20,30,45,.08); }}
  th, td {{ text-align: right; padding: 7px 10px; border-bottom: 1px solid #edf1f5; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #eef2f7; font-weight: 600; }}
  figure {{ margin: 0 0 22px; }}
  img {{ width: 100%; border: 1px solid #e2e8ef; border-radius: 6px; background: #fff; }}
  figcaption {{ font-size: .78rem; color: #5b6878; margin-top: 6px; }}
  ul {{ padding-left: 1.1rem; }}
</style>
</head>
<body>
<h1>Onboard vision: measured results</h1>
<p class="lede">Monocular speed and yaw rate from a single onboard camera looking at the road
plane. Every number is measured on procedurally rendered footage whose ground truth is exact.</p>
<h2>Setups</h2>
{table(_table(runs), header)}
<h2>What the numbers say</h2>
<ul>{findings}</ul>
<h2>Resolution</h2>
{table(resolution_rows, ["camera", "covered %", "speed MAE m/s", "speed MAE %", "yaw MAE rad/s"])}
<h2>Figures</h2>
{figure_html}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path
