#!/usr/bin/env python3
"""Aggregate PA3 session results and generate summary tables/plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "matplotlib-codex"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

DISPLAY_AVAILABLE = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

import matplotlib
if not DISPLAY_AVAILABLE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare


DEFAULT_METRICS = [
    "pairwise_frechet_m",
    "path_length_ratio_mean",
    "jerk_mean",
    "gp_sigma_mean_m",
    "demos_to_convergence",
    "cumulative_demo_time_to_convergence_s",
    "completion_time_s",
    "tlx_overall",
]

SCALAR_PLOT_TYPES = {
    "pairwise_frechet_m": "box",
    "path_length_ratio_mean": "box",
    "jerk_mean": "box",
    "tlx_overall": "box",
    "gp_sigma_mean_m": "bar",
    "demos_to_convergence": "bar",
    "cumulative_demo_time_to_convergence_s": "bar",
    "completion_time_s": "bar",
}

SERIES_PLOT_SPECS = {
    "gp_sigma_by_demo_m": {
        "ylabel": "GP sigma (m)",
        "title": "GP sigma vs demonstration number",
    },
    "path_length_ratio_by_demo": {
        "ylabel": "Path length ratio",
        "title": "Path length ratio vs demonstration number",
    },
    "jerk_by_demo": {
        "ylabel": "Jerk",
        "title": "Jerk vs demonstration number",
    },
    "gp_path_length_by_demo_m": {
        "ylabel": "GP path length (m)",
        "title": "GP path length vs demonstration number",
    },
}


def _coerce_csv_value(value: str):
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    if text[0] in "[{" and text[-1] in "]}":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_rows(results_dir: Path) -> list[dict]:
    csv_rows: list[dict] = []
    for csv_path in sorted(results_dir.rglob("metrics.csv")):
        if csv_path.parent.name == "analysis":
            continue
        if csv_path.parent == results_dir / "analysis":
            continue
        if csv_path.parent.name.startswith("participant_"):
            continue
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for idx, record in enumerate(reader, start=1):
                row = {key: _coerce_csv_value(value) for key, value in record.items()}
                row["session_folder"] = csv_path.parent.name
                row["trial_in_file"] = idx
                if csv_path.parent.name.startswith("participant_"):
                    row["participant_folder"] = csv_path.parent.name
                if "validation_run_" in str(csv_path):
                    row["validation_run"] = next(
                        (part for part in csv_path.parts if part.startswith("validation_run_")),
                        None,
                    )
                csv_rows.append(row)
    if csv_rows:
        return csv_rows

    rows: list[dict] = []
    for summary_path in sorted(results_dir.rglob("summary.json")):
        session_dir = summary_path.parent
        metrics_path = session_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        payload = json.loads(metrics_path.read_text())
        if isinstance(payload, dict):
            payload = [payload]

        for idx, record in enumerate(payload, start=1):
            row = dict(record)
            tlx = row.pop("tlx", None)
            if isinstance(tlx, dict):
                for key, value in tlx.items():
                    row[f"tlx_{key}"] = value

            row.setdefault("participant_number", summary.get("participant_number"))
            row.setdefault("condition", summary.get("condition"))
            row.setdefault("condition_label", summary.get("condition_label"))
            row.setdefault("input_mode", summary.get("input_mode"))
            row.setdefault("feedback_mode", summary.get("feedback_mode"))
            row.setdefault("tube", summary.get("tube"))
            row["session_folder"] = session_dir.name
            row["trial_in_file"] = idx
            if session_dir.parent.name.startswith("participant_"):
                row["participant_folder"] = session_dir.parent.name
            if "validation_run_" in str(session_dir):
                row["validation_run"] = next(
                    (part for part in session_dir.parts if part.startswith("validation_run_")),
                    None,
                )
            rows.append(row)
    return rows


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def condition_labels_in_order(rows: list[dict]) -> list[str]:
    label_to_order: dict[str, tuple[int, str]] = {}
    for row in rows:
        label = row.get("condition_label", "unknown")
        condition_id = row.get("condition_id")
        order_key = int(condition_id) if isinstance(condition_id, int) else 10**9
        if label not in label_to_order or order_key < label_to_order[label][0]:
            label_to_order[label] = (order_key, label)
    return [label for _, label in sorted(label_to_order.values(), key=lambda item: (item[0], item[1]))]


def numeric_sequence(value) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    seq = []
    for item in value:
        if isinstance(item, bool):
            seq.append(float(item))
        elif is_number(item):
            seq.append(float(item))
        else:
            return None
    return seq


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise_by_condition(rows: list[dict], metrics: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("condition_label", "unknown"), []).append(row)

    summaries: list[dict] = []
    for condition, items in grouped.items():
        for metric in metrics:
            values = [float(row[metric]) for row in items if is_number(row.get(metric)) or isinstance(row.get(metric), bool)]
            if not values:
                continue
            summaries.append({
                "condition_label": condition,
                "metric": metric,
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
    return summaries


def plot_metric(rows: list[dict], metric: str, out_path: Path, show_plot: bool = False) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if isinstance(value, bool):
            value = float(value)
        if is_number(value):
            grouped.setdefault(row.get("condition_label", "unknown"), []).append(float(value))

    if not grouped:
        return

    labels = [label for label in condition_labels_in_order(rows) if label in grouped]
    values = [grouped[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_type = SCALAR_PLOT_TYPES.get(metric, "box")
    if plot_type == "bar":
        means = [float(np.mean(ys)) for ys in values]
        stds = [float(np.std(ys, ddof=1)) if len(ys) > 1 else 0.0 for ys in values]
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color="#5b8ff9", alpha=0.8)
        ax.set_xticks(x, labels)
        for idx, ys in enumerate(values):
            xs = np.random.normal(idx, 0.04, size=len(ys))
            ax.scatter(xs, ys, s=22, alpha=0.7, color="#1f2a44")
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{mean:.3g}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax.boxplot(values, tick_labels=labels, patch_artist=True)
        for i, ys in enumerate(values, start=1):
            xs = np.random.normal(i, 0.04, size=len(ys))
            ax.scatter(xs, ys, s=20, alpha=0.7)
    ax.tick_params(axis="x", rotation=20)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    if not show_plot:
        plt.close(fig)


def plot_metric_series(rows: list[dict], metric: str, out_path: Path, show_plot: bool = False) -> None:
    labels = condition_labels_in_order(rows)
    series_by_condition: dict[str, list[list[float]]] = {}
    for row in rows:
        label = row.get("condition_label", "unknown")
        seq = numeric_sequence(row.get(metric))
        if seq:
            series_by_condition.setdefault(label, []).append(seq)

    if not series_by_condition:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(labels))))
    plotted = False

    for color, label in zip(colors, labels):
        sequences = series_by_condition.get(label)
        if not sequences:
            continue
        max_len = max(len(seq) for seq in sequences)
        xs = np.arange(1, max_len + 1)
        means = []
        stds = []
        valid_xs = []
        for idx in range(max_len):
            vals = [seq[idx] for seq in sequences if idx < len(seq) and math.isfinite(seq[idx])]
            if not vals:
                continue
            valid_xs.append(idx + 1)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        if not valid_xs:
            continue
        means_arr = np.array(means, dtype=float)
        stds_arr = np.array(stds, dtype=float)
        xs_arr = np.array(valid_xs, dtype=int)
        ax.plot(xs_arr, means_arr, marker="o", linewidth=2, color=color, label=label)
        if np.any(stds_arr > 0):
            ax.fill_between(xs_arr, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.15)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    spec = SERIES_PLOT_SPECS.get(metric, {})
    ax.set_xlabel("Demonstration number")
    ax.set_ylabel(spec.get("ylabel", metric))
    ax.set_title(spec.get("title", metric))
    ax.set_xticks(sorted({x for sequences in series_by_condition.values() for seq in sequences for x in range(1, len(seq) + 1)}))
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    if not show_plot:
        plt.close(fig)


def friedman_report(rows: list[dict], metrics: list[str]) -> list[dict]:
    results = []
    participant_rows = [row for row in rows if row.get("participant_number") is not None]
    if not participant_rows:
        return results

    conditions = condition_labels_in_order(participant_rows)
    for metric in metrics:
        per_participant = {}
        for row in participant_rows:
            value = row.get(metric)
            if isinstance(value, bool):
                value = float(value)
            if not is_number(value):
                continue
            pid = row["participant_number"]
            cond = row.get("condition_label", "unknown")
            per_participant.setdefault(pid, {}).setdefault(cond, []).append(float(value))

        complete = []
        for pid, cond_map in per_participant.items():
            if all(cond in cond_map for cond in conditions):
                complete.append((pid, [float(np.mean(cond_map[cond])) for cond in conditions]))

        if len(complete) < 2 or len(conditions) < 2:
            continue

        arrays = list(zip(*[vals for _, vals in complete]))
        with np.errstate(all="ignore"):
            stat, p_value = friedmanchisquare(*arrays)
        if not (math.isfinite(stat) and math.isfinite(p_value)):
            continue
        results.append({
            "metric": metric,
            "participants": len(complete),
            "conditions": conditions,
            "friedman_statistic": float(stat),
            "p_value": float(p_value),
        })
    return results


def run_analysis(
    results_dir: str | Path = "results",
    out_dir: str | Path = "analysis",
    show_plots: bool | None = None,
) -> Path:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if show_plots is None:
        show_plots = DISPLAY_AVAILABLE
    elif show_plots and not DISPLAY_AVAILABLE:
        show_plots = False

    rows = load_rows(results_dir)
    if not rows:
        raise SystemExit(f"No metrics.json files found under {results_dir}")

    metrics = [metric for metric in DEFAULT_METRICS if any(metric in row for row in rows)]
    write_csv(out_dir / "aggregate_metrics.csv", rows)
    write_csv(out_dir / "condition_summary.csv", summarise_by_condition(rows, metrics))

    friedman = friedman_report(rows, metrics)
    with (out_dir / "friedman_tests.json").open("w") as f:
        json.dump(friedman, f, indent=2)

    for metric in metrics:
        plot_metric(rows, metric, plots_dir / f"{metric}.png", show_plot=show_plots)

    for metric in SERIES_PLOT_SPECS:
        if any(numeric_sequence(row.get(metric)) for row in rows):
            plot_metric_series(rows, metric, plots_dir / f"{metric}.png", show_plot=show_plots)

    if show_plots and plt.get_fignums():
        plt.show()

    print(f"Analysis written to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Folder containing session_* and/or validation_run_* subfolders",
    )
    parser.add_argument("--out-dir", default="analysis", help="Where to write CSV files and plots")
    parser.add_argument("--no-show", action="store_true", help="Save plots only, without opening plot windows")
    args = parser.parse_args()
    run_analysis(results_dir=args.results_dir, out_dir=args.out_dir, show_plots=not args.no_show)


if __name__ == "__main__":
    main()
