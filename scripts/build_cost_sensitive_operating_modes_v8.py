"""Build the cost-sensitive SABER-PID operating-mode decision rule.

The analysis uses only already-scored Set-B TP/FP/FN counts.  With
``C_FP = 1`` and ``r = C_FN / C_FP``, each mode has loss

    L(r) = r * FN + FP.

The script computes the exact lower envelope, source-bootstrap probabilities
of being optimal, a machine-readable decision table, and vector/raster paper
figures.  No model inference and no reference-aware prediction construction
are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("set_intersection", "Intersection", "precision-first", "#CC79A7"),
    ("paddleocr_geometry", "OCR joined", "low-false-positive", "#009E73"),
    ("ocr_if_nonempty_else_qwen", "OCR-first", "balanced fallback", "#0072B2"),
    ("set_union", "Union", "recall-first", "#D55E00"),
)
RNG_SEED = 20260812


@dataclass(frozen=True)
class CostLine:
    method_id: str
    label: str
    role: str
    color: str
    fn: int
    fp: int
    tp: int

    def loss(self, ratio: float | np.ndarray) -> float | np.ndarray:
        return self.fn * ratio + self.fp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path, dataset: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["dataset"] == dataset]


def aggregate_lines(rows: Iterable[dict[str, str]]) -> tuple[list[CostLine], list[str], np.ndarray, np.ndarray]:
    rows = list(rows)
    by_method: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["source_id"]] = row
    required = [method_id for method_id, _label, _role, _color in METHODS]
    missing = [method_id for method_id in required if method_id not in by_method]
    if missing:
        raise ValueError(f"Missing operating methods: {missing}")
    source_sets = [set(by_method[method_id]) for method_id in required]
    if not source_sets[0] or any(sources != source_sets[0] for sources in source_sets[1:]):
        raise ValueError("Operating methods do not share identical non-empty source membership")
    sources = sorted(source_sets[0])
    fn_by_source = np.zeros((len(sources), len(required)), dtype=np.int64)
    fp_by_source = np.zeros_like(fn_by_source)
    lines: list[CostLine] = []
    for method_index, (method_id, label, role, color) in enumerate(METHODS):
        method_rows = by_method[method_id]
        for source_index, source_id in enumerate(sources):
            row = method_rows[source_id]
            fn_by_source[source_index, method_index] = int(row["fn"])
            fp_by_source[source_index, method_index] = int(row["fp"])
        lines.append(
            CostLine(
                method_id=method_id,
                label=label,
                role=role,
                color=color,
                fn=int(fn_by_source[:, method_index].sum()),
                fp=int(fp_by_source[:, method_index].sum()),
                tp=sum(int(method_rows[source_id]["tp"]) for source_id in sources),
            )
        )
    return lines, sources, fn_by_source, fp_by_source


def exact_lower_envelope(lines: list[CostLine]) -> list[dict[str, Any]]:
    """Return merged positive-ratio intervals on the exact lower envelope."""

    candidates = {0.0}
    for index, left in enumerate(lines):
        for right in lines[index + 1 :]:
            denominator = left.fn - right.fn
            if denominator == 0:
                continue
            crossing = (right.fp - left.fp) / denominator
            if crossing > 0 and math.isfinite(crossing):
                candidates.add(float(crossing))
    boundaries = sorted(candidates)
    intervals: list[dict[str, Any]] = []
    for lower, upper in zip(boundaries, boundaries[1:] + [math.inf]):
        if math.isinf(upper):
            probe = max(1.0, lower * 2.0)
        elif lower == 0:
            probe = upper / 2.0
        else:
            probe = math.sqrt(lower * upper)
        losses = [float(line.loss(probe)) for line in lines]
        winner = int(np.argmin(losses))
        if intervals and intervals[-1]["method_id"] == lines[winner].method_id:
            intervals[-1]["upper_ratio"] = upper
        else:
            intervals.append(
                {
                    "lower_ratio": lower,
                    "upper_ratio": upper,
                    "method_id": lines[winner].method_id,
                    "label": lines[winner].label,
                }
            )
    # Pairwise crossings that do not change the winning line have been merged;
    # restrict interval endpoints to actual winner changes.
    return intervals


def bootstrap_optimal_probabilities(
    fn_by_source: np.ndarray,
    fp_by_source: np.ndarray,
    ratios: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if reps <= 0:
        raise ValueError("Bootstrap replicate count must be positive")
    rng = np.random.default_rng(seed)
    source_count, method_count = fn_by_source.shape
    sampled = rng.integers(0, source_count, size=(reps, source_count))
    boot_fn = fn_by_source[sampled].sum(axis=1)
    boot_fp = fp_by_source[sampled].sum(axis=1)
    losses = boot_fn[:, :, None] * ratios[None, None, :] + boot_fp[:, :, None]
    minimum = losses.min(axis=1, keepdims=True)
    ties = np.isclose(losses, minimum, rtol=0.0, atol=1e-12)
    # Split tie mass equally, avoiding an arbitrary method-order preference at
    # exact crossing ratios.
    probabilities = (ties / ties.sum(axis=1, keepdims=True)).mean(axis=0)
    diagnostics = {
        "replicates": reps,
        "seed": seed,
        "source_count": source_count,
        "method_count": method_count,
        "probability_column_sum_min": float(probabilities.sum(axis=0).min()),
        "probability_column_sum_max": float(probabilities.sum(axis=0).max()),
    }
    return probabilities, diagnostics


def pairwise_switch_bootstrap(
    fn_by_source: np.ndarray,
    fp_by_source: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    source_count = fn_by_source.shape[0]
    sampled = rng.integers(0, source_count, size=(reps, source_count))
    boot_fn = fn_by_source[sampled].sum(axis=1)
    boot_fp = fp_by_source[sampled].sum(axis=1)
    output = []
    for left, right in pairs:
        numerator = boot_fp[:, right] - boot_fp[:, left]
        denominator = boot_fn[:, left] - boot_fn[:, right]
        valid = (denominator != 0) & (numerator / np.where(denominator == 0, 1, denominator) > 0)
        ratios = numerator[valid] / denominator[valid]
        output.append(
            {
                "left_method_index": left,
                "right_method_index": right,
                "valid_positive_replicates": int(valid.sum()),
                "valid_fraction": float(valid.mean()),
                "median": float(np.median(ratios)) if len(ratios) else None,
                "ci95": [float(x) for x in np.percentile(ratios, [2.5, 97.5])]
                if len(ratios)
                else None,
                "interpretation": "descriptive pairwise switch-ratio bootstrap; not a universal cost estimate",
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render_figure(
    output_dir: Path,
    lines: list[CostLine],
    source_count: int,
    ratios: np.ndarray,
    probabilities: np.ndarray,
    envelope: list[dict[str, Any]],
) -> list[Path]:
    plt.rcParams.update({"font.size": 8.0, "axes.titlesize": 9.0, "axes.labelsize": 8.0})
    fig, axes = plt.subplots(2, 1, figsize=(7.15, 6.1), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]})
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.11, hspace=0.30)
    fig.suptitle(
        "Cost-sensitive selection converts operating modes into an engineering decision rule",
        fontsize=11.0,
        fontweight="bold",
    )
    fig.text(
        0.12,
        0.915,
        r"Set B; $C_{FP}=1$ and $r=C_{FN}/C_{FP}$. Lower loss is better; predictions are unchanged.",
        fontsize=7.1,
        color="#5C5C5C",
    )

    for line in lines:
        losses = np.asarray(line.loss(ratios), dtype=float) / source_count
        axes[0].plot(ratios, losses, color=line.color, linewidth=1.1, alpha=0.52)
    line_by_id = {line.method_id: line for line in lines}
    for interval in envelope:
        lower = max(float(interval["lower_ratio"]), float(ratios[0]))
        upper = float(interval["upper_ratio"])
        upper = float(ratios[-1]) if math.isinf(upper) else min(upper, float(ratios[-1]))
        if upper <= lower:
            continue
        mask = (ratios >= lower) & (ratios <= upper)
        line = line_by_id[interval["method_id"]]
        axes[0].plot(
            ratios[mask],
            np.asarray(line.loss(ratios[mask])) / source_count,
            color=line.color,
            linewidth=4.0,
            solid_capstyle="round",
            label=line.label,
        )
        if interval["lower_ratio"] == 0:
            range_label = f"r < {interval['upper_ratio']:.3f}"
        elif math.isinf(interval["upper_ratio"]):
            range_label = f"r >= {interval['lower_ratio']:.3f}"
        else:
            range_label = f"r: {interval['lower_ratio']:.3f}-{interval['upper_ratio']:.3f}"
        midpoint = math.sqrt(max(lower, float(ratios[0])) * min(upper, float(ratios[-1])))
        log_width = math.log10(upper / lower) if lower > 0 and upper > lower else 1.0
        annotation_height = 0.22 if log_width < 0.12 else 0.055
        axes[0].text(
            midpoint,
            annotation_height,
            f"{line.label}\n{range_label}",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.0,
            fontweight="bold",
            color=line.color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": line.color, "alpha": 0.92, "linewidth": 0.7},
        )
    for interval in envelope[:-1]:
        boundary = float(interval["upper_ratio"])
        axes[0].axvline(boundary, color="#888888", linewidth=0.75, linestyle="--", zorder=0)
        axes[1].axvline(boundary, color="#888888", linewidth=0.75, linestyle="--", zorder=0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Expected error cost per drawing")
    axes[0].set_title("A. Deterministic loss curves and exact lower envelope", loc="left", fontweight="bold")
    axes[0].grid(True, which="both", color="#E8E8E8", linewidth=0.65)
    handles, labels = axes[0].get_legend_handles_labels()
    # Merging removed all duplicate envelope segments, but preserve this guard.
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), frameon=False, ncol=4, loc="upper left", fontsize=7.0)

    for index, line in enumerate(lines):
        axes[1].plot(ratios, probabilities[index], color=line.color, linewidth=2.0, label=line.label)
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel(r"Relative miss cost, $C_{FN}/C_{FP}$")
    axes[1].set_ylabel("Bootstrap probability of\nminimum loss")
    axes[1].set_title("B. Source-bootstrap decision stability (10,000 resamples)", loc="left", fontweight="bold")
    axes[1].grid(True, which="both", color="#E8E8E8", linewidth=0.65)
    axes[1].legend(frameon=False, ncol=4, loc="upper center", fontsize=7.0)
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "figure_4_cost_sensitive_operating_modes_v8.pdf",
        output_dir / "figure_4_cost_sensitive_operating_modes_v8.png",
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=500, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--input", default="reports/generated/rineng_revision_per_source_v6.csv")
    parser.add_argument("--dataset", default="set_b")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--output-json", default="reports/generated/rineng_cost_sensitive_operating_modes_v8.json")
    parser.add_argument("--grid-csv", default="reports/generated/rineng_cost_sensitive_operating_modes_v8.csv")
    parser.add_argument("--decision-csv", default="reports/generated/rineng_cost_sensitive_decision_rule_v8.csv")
    parser.add_argument("--figure-dir", default="paper/figures")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    input_path = root / args.input
    rows = read_rows(input_path, args.dataset)
    lines, sources, fn_by_source, fp_by_source = aggregate_lines(rows)
    envelope = exact_lower_envelope(lines)
    ratios = np.logspace(math.log10(0.05), math.log10(20.0), 321)
    probabilities, bootstrap_diagnostics = bootstrap_optimal_probabilities(
        fn_by_source,
        fp_by_source,
        ratios,
        reps=args.bootstrap_reps,
        seed=RNG_SEED,
    )
    adjacent_pairs = [
        (
            next(index for index, line in enumerate(lines) if line.method_id == envelope[i]["method_id"]),
            next(index for index, line in enumerate(lines) if line.method_id == envelope[i + 1]["method_id"]),
        )
        for i in range(len(envelope) - 1)
    ]
    switch_bootstrap = pairwise_switch_bootstrap(
        fn_by_source,
        fp_by_source,
        adjacent_pairs,
        reps=args.bootstrap_reps,
        seed=RNG_SEED + 1,
    )
    for item, (left_index, right_index) in zip(switch_bootstrap, adjacent_pairs):
        item["left_method_id"] = lines[left_index].method_id
        item["right_method_id"] = lines[right_index].method_id

    decision_rows = []
    for interval in envelope:
        line = next(candidate for candidate in lines if candidate.method_id == interval["method_id"])
        precision = line.tp / (line.tp + line.fp) if line.tp + line.fp else 0.0
        recall = line.tp / (line.tp + line.fn) if line.tp + line.fn else 0.0
        decision_rows.append(
            {
                "lower_ratio_inclusive": interval["lower_ratio"],
                "upper_ratio_exclusive": "infinity"
                if math.isinf(interval["upper_ratio"])
                else interval["upper_ratio"],
                "recommended_mode": line.label,
                "method_id": line.method_id,
                "engineering_role": line.role,
                "loss_expression": f"{line.fn}*r+{line.fp}",
                "tp": line.tp,
                "fp": line.fp,
                "fn": line.fn,
                "precision": precision,
                "recall": recall,
            }
        )
    grid_rows = []
    for ratio_index, ratio in enumerate(ratios):
        deterministic_losses = [float(line.loss(float(ratio))) for line in lines]
        winner = int(np.argmin(deterministic_losses))
        for method_index, line in enumerate(lines):
            grid_rows.append(
                {
                    "cost_ratio_fn_over_fp": float(ratio),
                    "method_id": line.method_id,
                    "label": line.label,
                    "loss_total": deterministic_losses[method_index],
                    "loss_per_source": deterministic_losses[method_index] / len(sources),
                    "deterministic_optimal": int(method_index == winner),
                    "bootstrap_probability_optimal": float(probabilities[method_index, ratio_index]),
                }
            )

    figure_paths = render_figure(
        root / args.figure_dir,
        lines,
        len(sources),
        ratios,
        probabilities,
        envelope,
    )
    write_csv(root / args.grid_csv, grid_rows)
    write_csv(root / args.decision_csv, decision_rows)
    report = {
        "version": "rineng-cost-sensitive-operating-modes-v8",
        "status": "pass",
        "analysis_role": "scorer-only decision analysis; no new inference and no prediction uses references",
        "dataset": args.dataset,
        "source_count": len(sources),
        "input": args.input,
        "input_sha256": sha256(input_path),
        "loss_definition": "L = C_FN*FN + C_FP*FP; normalized with C_FP=1 and r=C_FN/C_FP",
        "methods": [
            {
                "method_id": line.method_id,
                "label": line.label,
                "role": line.role,
                "tp": line.tp,
                "fp": line.fp,
                "fn": line.fn,
            }
            for line in lines
        ],
        "exact_decision_intervals": decision_rows,
        "switch_ratio_bootstrap": switch_bootstrap,
        "bootstrap": bootstrap_diagnostics,
        "grid": {
            "minimum_ratio": float(ratios[0]),
            "maximum_ratio": float(ratios[-1]),
            "point_count": len(ratios),
            "csv": args.grid_csv,
        },
        "decision_csv": args.decision_csv,
        "figures": [str(path.relative_to(root).as_posix()) for path in figure_paths],
        "integrity_checks": {
            "identical_source_membership": True,
            "source_count": len(sources),
            "probabilities_sum_to_one": bool(
                np.allclose(probabilities.sum(axis=0), 1.0, rtol=0.0, atol=1e-12)
            ),
            "all_counts_nonnegative": bool(np.all(fn_by_source >= 0) and np.all(fp_by_source >= 0)),
            "references_used_for_prediction": False,
        },
    }
    output_path = root / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "source_count": len(sources),
                "decision_intervals": decision_rows,
                "report": args.output_json,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
