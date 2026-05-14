#!/usr/bin/env python3
"""Plot accuracy vs CoT length from evaluation logs with target-based sampling."""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_target_results(path: Path) -> List[Tuple[int, float]]:
    """Parse per-target results from an evaluation log.

    Expects lines like:
      target=1024: n=100, min=1008, max=1040, mean=1024.2, median=1024.0, solved=95/100, acc=95.00%

    Returns list of (target, accuracy) tuples.
    """
    pattern = re.compile(
        r"target=(?P<target>\d+):.*acc=(?P<acc>[\d.]+)%"
    )
    results = []
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            target = int(match.group("target"))
            acc = float(match.group("acc")) / 100.0
            results.append((target, acc))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy vs CoT length from eval logs")
    parser.add_argument("logs", nargs="+", help="Evaluation log files to plot")
    parser.add_argument("--output", "-o", default="eval_accuracy.png", help="Output plot file")
    parser.add_argument("--labels", "-l", nargs="+", default=None, help="Labels for each log file")
    parser.add_argument("--xlim", nargs=2, type=float, default=None, help="X-axis limits")
    parser.add_argument("--ylim", nargs=2, type=float, default=[0, 1], help="Y-axis limits")
    parser.add_argument("--logx", action="store_true", help="Use log scale for x-axis")
    parser.add_argument("--groups", "-g", nargs="+", default=None,
                        help="Group name for each log file. Logs in the same group get colors in a progression.")
    parser.add_argument("--cmap", default=None,
                        help="Override colormap for a single-group plot (e.g. Blues, Greens).")
    parser.add_argument("--legend-loc", default="best",
                        help="Matplotlib legend location, e.g. 'lower left'.")
    return parser.parse_args()


# Base colormaps for different groups (dark to light progressions)
GROUP_CMAPS = ["Blues", "Oranges", "Greens", "Reds", "Purples", "Greys"]


def assign_group_colors(groups: List[str], cmap_override: str = None) -> List[tuple]:
    """Assign colors to each item based on group membership.

    Items in the same group get colors from a sequential colormap (dark to light).
    """
    # Count items per group and track order
    group_items: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        group_items[g].append(i)

    # Assign a colormap to each unique group
    unique_groups = list(group_items.keys())

    colors = [None] * len(groups)
    for group_idx, group_name in enumerate(unique_groups):
        cmap_name = cmap_override if (cmap_override and len(unique_groups) == 1) else GROUP_CMAPS[group_idx % len(GROUP_CMAPS)]
        cmap = plt.colormaps[cmap_name]
        indices = group_items[group_name]
        n = len(indices)
        # Use range 0.3-0.9 to avoid too light/dark colors
        for j, item_idx in enumerate(indices):
            t = 0.3 + 0.6 * (j / max(1, n - 1)) if n > 1 else 0.6
            colors[item_idx] = cmap(t)

    return colors


def main():
    args = parse_args()

    if args.labels and len(args.labels) != len(args.logs):
        raise SystemExit(f"Number of labels ({len(args.labels)}) must match number of logs ({len(args.logs)})")
    if args.groups and len(args.groups) != len(args.logs):
        raise SystemExit(f"Number of groups ({len(args.groups)}) must match number of logs ({len(args.logs)})")

    # Assign colors based on groups
    colors = assign_group_colors(args.groups, args.cmap) if args.groups else [None] * len(args.logs)

    fig, ax = plt.subplots(figsize=(10, 6))

    all_targets = set()
    for i, log_path in enumerate(args.logs):
        path = Path(log_path)
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping")
            continue

        results = parse_target_results(path)
        if not results:
            print(f"Warning: no target results found in {path}, skipping")
            continue

        # Sort by CoT length
        results.sort(key=lambda x: x[0])
        x = np.array([r[0] for r in results])
        y = np.array([r[1] for r in results])
        all_targets.update(x.tolist())

        label = args.labels[i] if args.labels else path.stem
        plot_kwargs = {"marker": "o", "markersize": 6, "linewidth": 2, "label": label}
        if colors[i] is not None:
            plot_kwargs["color"] = colors[i]
        ax.plot(x, y, **plot_kwargs)

    ax.set_xlabel("Correct CoT Length", fontsize=24)
    ax.set_ylabel("Accuracy", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18)
    if args.xlim:
        ax.set_xlim(args.xlim)
    if args.ylim:
        ax.set_ylim(args.ylim)
    if args.logx:
        ax.set_xscale("log")
        # Set ticks at all targets, but only label powers of 2
        all_targets = sorted(all_targets)
        # Find powers of 2 within range
        powers_of_2 = [2**n for n in range(32) if min(all_targets) <= 2**n <= max(all_targets)]
        ax.set_xticks(powers_of_2, minor=False)
        ax.set_xticks(all_targets, minor=True)
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([f'$2^{{{int(np.log2(p))}}}$' for p in powers_of_2]))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(axis='x', which='minor', length=4)
        ax.tick_params(axis='x', which='major', length=8)
    ax.legend(fontsize=18, loc=args.legend_loc)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
