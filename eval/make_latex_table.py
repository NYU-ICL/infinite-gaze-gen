"""Render evaluation summaries as a publication-ready LaTeX table.

The script consumes one or more ``final_metrics.csv`` / ``final_metrics.json``
files produced by the evaluation scripts.  It also accepts directories and
searches them recursively for those filenames, making it convenient to point
at a ``results`` directory containing one subdirectory per run.  Metrics may
come from different directories: records with the same ``model`` value are
merged into one table row.

Run with no arguments after evaluations complete:

``python eval/make_latex_table.py``

It scans ``eval/results`` recursively (including a separate discrete-Fr\'echet
subdirectory), merges scores with the same model name, and writes
``eval/results/latex_table.tex``.

Method labels are LaTeX snippets deliberately (for example,
``'diffeye=DiffEye \\cite{karadiffeye}'``).  They are therefore not escaped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


METRICS = (
    ("levenshtein", "Levenshtein ($\\times 10^3$) $\\downarrow$", 1e-3, "min", 2),
    ("discrete_frechet", "Disc. Fr\\'echet ($\\times 10^2$) $\\downarrow$", 1e-2, "min", 2),
    ("dtw", "DTW ($\\times 10^4$) $\\downarrow$", 1e-4, "min", 2),
    ("temporal_correlation", "Max. Temp. Corr. $\\uparrow$", 1.0, "max", 3),
)
METRIC_BY_NAME = {item[0]: item for item in METRICS}
RESULT_NAMES = ("final_metrics.csv", "final_metrics.json", "final_metrics_scanpathevals.csv", "final_metrics_scanpathevals.json")
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
# Fixed paper-table rows. Evaluation wrappers use several repository-specific
# names for these same methods, so normalize them before combining directories.
PAPER_MODELS = ("deepgaze", "diffeye", "gazeformer", "hat", "chen", "tppgaze", "ours")
PAPER_LABELS = {
    "deepgaze": r"DeepGaze III \cite{deepgaze3}",
    "diffeye": r"DiffEye \cite{karadiffeye}",
    "gazeformer": r"GazeFormer \cite{gazeformer}",
    "hat": r"HAT \cite{yang2024unifying}",
    "chen": r"Chen et al. \cite{chen2021predicting}",
    "tppgaze": r"TPP-Gaze \cite{d2025tpp}",
    "ours": "Ours",
}
MODEL_ALIASES = {
    "deepgaze3": "deepgaze",
    "diffeye_video_modes": "diffeye",
    "gazeformer_isp": "gazeformer",
    "lstm": "chen",
    "lstm_isp": "chen",
    "tppgaze_auto": "tppgaze",
    "diem_final_model_90_45_clip": "ours",
    "unet_clip_history": "ours",
}
# When an older and a reproduced run both report a method, use the reproduced
# DeepGaze result requested for the paper table.
PREFERRED_MODEL_SOURCE_DIR = {"deepgaze": "all_baselines_reproduce_final"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results",
        nargs="*",
        type=Path,
        default=[DEFAULT_RESULTS],
        help="Summary files or directories containing them (default: eval/results).",
    )
    parser.add_argument("--dataset", default="DIEM", help="Dataset label placed in the first column (default: DIEM).")
    parser.add_argument(
        "--metric-results",
        action="append",
        default=[],
        metavar="METRIC=PATH",
        help="Read one metric from an additional summary file/directory; repeatable.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="MODEL=LATEX_LABEL",
        help="Rename a model. Repeatable; labels may contain LaTeX.",
    )
    parser.add_argument("--order", nargs="*", metavar="MODEL", help="Explicit model order (unlisted models follow alphabetically).")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS / "latex_table.tex", help="Output path (default: eval/results/latex_table.tex).")
    parser.add_argument("--no-colors", action="store_true", help="Do not emit row/cell color commands.")
    parser.add_argument("--no-rank", action="store_true", help="Do not bold best or underline second-best scores.")
    return parser.parse_args()


def method_labels(values: list[str]) -> dict[str, str]:
    labels = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--method must be MODEL=LATEX_LABEL, got {value!r}")
        model, label = value.split("=", 1)
        if not model or not label:
            raise ValueError(f"--method must have both a model and label, got {value!r}")
        labels[model] = label
    return labels


def metric_result_sources(values: list[str]) -> list[tuple[str, Path]]:
    sources = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--metric-results must be METRIC=PATH, got {value!r}")
        metric, raw_path = value.split("=", 1)
        if metric not in METRIC_BY_NAME or not raw_path:
            raise ValueError(f"--metric-results needs a supported metric and path, got {value!r}")
        sources.append((metric, Path(raw_path)))
    return sources


def result_files(paths: Iterable[Path]) -> list[Path]:
    candidates: set[Path] = set()
    for path in paths:
        if path.is_file():
            candidates.add(path)
        elif path.is_dir():
            candidates.update(candidate for name in RESULT_NAMES for candidate in path.rglob(name))
        else:
            raise FileNotFoundError(path)
    if not candidates:
        names = ", ".join(RESULT_NAMES)
        raise FileNotFoundError(f"No evaluation summary files found (expected: {names})")
    # Evaluators write identical CSV and JSON versions of each summary.  Prefer
    # CSV so a directory can be passed directly without creating duplicates.
    selected: dict[tuple[Path, str], Path] = {}
    for path in sorted(candidates):
        key = path.parent, path.stem
        if key not in selected or path.suffix == ".csv":
            selected[key] = path
    return sorted(selected.values())


def load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            raise ValueError(f"{path} must contain a JSON array of metric records")
        return data
    raise ValueError(f"Unsupported summary format: {path}")


def infer_model(path: Path) -> str:
    """Use the run directory when older summaries lack a model column."""
    return path.parent.name


def canonical_model(model: str) -> str:
    """Normalize known equivalent result-directory model names."""
    return MODEL_ALIASES.get(model.lower(), model)


def prefers_source(model: str, candidate: Path, existing: Path) -> bool:
    """Whether ``candidate`` is the configured authoritative source."""
    preferred_dir = PREFERRED_MODEL_SOURCE_DIR.get(model)
    return (
        preferred_dir is not None
        and preferred_dir in candidate.parts
        and preferred_dir not in existing.parts
    )


def collect_scores(sources: Iterable[tuple[Path, str | None]]) -> dict[str, dict[str, dict[str, float]]]:
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    origins: dict[tuple[str, str], Path] = {}
    for path, only_metric in sources:
        for row in load_rows(path):
            metric = str(row.get("metric", ""))
            if only_metric is not None and metric != only_metric:
                continue
            if metric not in METRIC_BY_NAME:
                continue
            model = canonical_model(str(row.get("model") or infer_model(path)))
            key = (model, metric)
            try:
                mean, best = float(row["mean_score"]), float(row["best_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: {model}/{metric} needs numeric mean_score and best_score") from exc
            if not math.isfinite(mean) or not math.isfinite(best):
                raise ValueError(f"{path}: {model}/{metric} contains a non-finite score")
            if key in origins:
                old = scores[model][metric]
                if old["mean"] == mean and old["best"] == best:
                    continue  # The same CSV/JSON data was supplied twice.
                if prefers_source(model, path, origins[key]):
                    scores[model][metric] = {"mean": mean, "best": best}
                    origins[key] = path
                    continue
                if prefers_source(model, origins[key], path):
                    continue
                raise ValueError(f"Conflicting scores for {model!r}, {metric!r}: {origins[key]} and {path}")
            scores[model][metric] = {"mean": mean, "best": best}
            origins[key] = path
    if not scores:
        raise ValueError("No supported trajectory metrics were found in the supplied summaries")
    return scores


def latex_escape(text: str) -> str:
    return re.sub(r"([&_#%])", r"\\\1", text)


def ranks(scores: dict[str, dict[str, dict[str, float]]]) -> dict[tuple[str, str, str], str]:
    """Return ``best`` / ``second`` decorations, preserving ties at a rank."""
    decorations: dict[tuple[str, str, str], str] = {}
    for metric, _, _, direction, _ in METRICS:
        for stat in ("mean", "best"):
            available = {model: values[metric][stat] for model, values in scores.items() if metric in values}
            if not available:
                continue
            ordered = sorted(set(available.values()), reverse=direction == "max")
            for model, value in available.items():
                if value == ordered[0]:
                    decorations[model, metric, stat] = "best"
                elif len(ordered) > 1 and value == ordered[1]:
                    decorations[model, metric, stat] = "second"
    return decorations


def format_value(value: float | None, scale: float, decimals: int, decoration: str | None, color: str, use_rank: bool, use_colors: bool) -> str:
    if value is None:
        cell = "--"
        return f"\\cellcolor{{{color}}}{cell}" if use_colors else cell
    cell = f"{value * scale:.{decimals}f}"
    if use_rank and decoration == "best":
        cell = f"\\textbf{{{cell}}}"
    elif use_rank and decoration == "second":
        cell = f"\\underline{{{cell}}}"
    if use_colors:
        cell = f"\\cellcolor{{{'bestcell' if decoration == 'best' and use_rank else color}}}{cell}"
    return cell


def render(scores: dict[str, dict[str, dict[str, float]]], dataset: str, labels: dict[str, str], order: list[str] | None, use_rank: bool, use_colors: bool) -> str:
    models = list(PAPER_MODELS)
    if order:
        unknown = set(order).difference(scores)
        if unknown:
            raise ValueError(f"--order names have no scores: {', '.join(sorted(unknown))}")
        models = list(dict.fromkeys(order)) + [model for model in models if model not in order]
    missing_models = set(models).difference(scores)
    if missing_models:
        raise ValueError(f"Missing results for expected paper-table models: {', '.join(sorted(missing_models))}")
    decorations = ranks(scores)
    lines = [
        "\\begin{tabular}{l l cccccccc}",
        "\\toprule",
        "\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Method}",
    ]
    for index, (_, title, *_) in enumerate(METRICS):
        suffix = " \\\\" if index == len(METRICS) - 1 else ""
        lines.append(f"& \\multicolumn{{2}}{{c}}{{{title}}}{suffix}")
    for index, _ in enumerate(METRICS):
        suffix = " \\\\" if index == len(METRICS) - 1 else ""
        lines.append(f"& Mean & Best{suffix}")
    lines.append("\\midrule")
    for index, model in enumerate(models):
        label = labels.get(model, PAPER_LABELS.get(model, latex_escape(model)))
        prefix = f"\\multirow{{{len(models)}}}{{*}}{{\\textbf{{{latex_escape(dataset)}}}}} & " if index == 0 else "& "
        color = "rowgray" if index % 2 == 0 else "rowgray2"
        lines.append(prefix + label)
        for metric_index, (metric, _, scale, _, decimals) in enumerate(METRICS):
            cells = []
            for stat in ("mean", "best"):
                value = scores[model].get(metric, {}).get(stat)
                cells.append(format_value(value, scale, decimals, decorations.get((model, metric, stat)), color, use_rank, use_colors))
            suffix = " \\\\" if metric_index == len(METRICS) - 1 else ""
            lines.append("& " + " & ".join(cells) + suffix)
    lines.extend(["\\midrule", "\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    try:
        sources: list[tuple[Path, str | None]] = [(path, None) for path in result_files(args.results)]
        for metric, root in metric_result_sources(args.metric_results):
            sources.extend((path, metric) for path in result_files([root]))
        table = render(collect_scores(sources), args.dataset, method_labels(args.method), args.order, not args.no_rank, not args.no_colors)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
