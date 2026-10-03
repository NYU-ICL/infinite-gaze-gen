"""Evaluate configurable baseline outputs against DIEM GT after frame 90.

Each --model value is NAME|KIND|PATH, where KIND is one of:
  unet          use its first 810 generated points;
   simulated_raw use the first 810 points of each *_simraw.csv (shorter runs
                 are compared over their available prefix);
   diffeye       evenly downsample each full prediction_*.csv to 810 points.

GT shorter than 900 valid points is retained: its available tail after the
90-frame history is compared with the equally truncated prediction prefix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:  # Supports both `python eval/run_baseline_metrics.py` and package imports.
    from .trajectory_metrics import (
        _score_pair,
        _video_size_from_name,
        align_trajectory_prefixes,
        accumulate_pairwise_comparisons,
        load_diem_ground_truth,
        load_prediction_csv,
        select_trajectory_metrics,
    )
except ImportError:
    from trajectory_metrics import (
        _score_pair,
        _video_size_from_name,
        align_trajectory_prefixes,
        accumulate_pairwise_comparisons,
        load_diem_ground_truth,
        load_prediction_csv,
        select_trajectory_metrics,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_POINTS = 810
GT_HISTORY_POINTS = 90
MODEL_KINDS = {"unet", "simulated_raw", "diffeye"}


def parse_model(value: str) -> tuple[str, str, Path]:
    try:
        name, kind, raw_path = value.split("|", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--model must be NAME|KIND|PATH") from exc
    if not name:
        raise argparse.ArgumentTypeError("Model name cannot be empty")
    if kind not in MODEL_KINDS:
        raise argparse.ArgumentTypeError(f"Unknown model kind {kind!r}; choose from {sorted(MODEL_KINDS)}")
    return name, kind, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        metavar="NAME|KIND|PATH",
        help="Repeat once per model to evaluate.",
    )
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        required=True,
        help="DIEM data directory containing <video>/event_data/*.txt.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        metavar="METRIC",
        help="Subset of: dtw, discrete_frechet, levenshtein, temporal_correlation.",
    )
    return parser.parse_args()


def prediction_paths(root: Path, kind: str) -> list[Path]:
    if kind == "unet":
        return sorted(root.rglob("scanpath.csv"))
    if kind == "simulated_raw":
        return sorted(root.rglob("*_simraw.csv"))
    paths = sorted(
        path
        for path in root.rglob("prediction_*.csv")
        if "simulated_raw" not in path.parts and "GT" not in path.parts and not path.name.endswith("_simraw.csv")
    )
    # DiffEye's autoregressive runnable stores predictions as
    # ``full_30s/pred_*/scanpath.csv`` rather than ``prediction_*.csv``.
    if not paths:
        paths = sorted(root.rglob("scanpath.csv"))
    return paths


def resolve_video_name(prediction_path: Path, root: Path, gt_names: set[str]) -> str:
    """Find the GT video directory encoded in an output's relative path."""
    for part in prediction_path.relative_to(root).parts:
        candidate = part.split("__", 1)[0]
        if candidate in gt_names:
            return candidate
        prefix_matches = [name for name in gt_names if name.startswith(candidate)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
    raise ValueError("could not map output path to a unique GT video directory")


def prepare_prediction(path: Path, kind: str) -> np.ndarray:
    points = load_prediction_csv(path)
    if kind == "unet":
        if len(points) < TARGET_POINTS:
            raise ValueError(f"needs at least {TARGET_POINTS} points; found {len(points)}")
        return points[:TARGET_POINTS]
    if kind == "simulated_raw":
        return points[:TARGET_POINTS]
    if len(points) < TARGET_POINTS:
        raise ValueError(f"needs at least {TARGET_POINTS} points for downsampling; found {len(points)}")
    indices = np.linspace(0, len(points) - 1, TARGET_POINTS).round().astype(int)
    return points[indices]


def save_equal_video_summary(per_gt: pd.DataFrame, output_dir: Path, metrics: dict) -> pd.DataFrame:
    """Average participant summaries within each video, then average videos."""
    video_rows: list[dict[str, object]] = []
    for (model, video), group in per_gt.groupby(["model", "video"], sort=True):
        item: dict[str, object] = {
            "model": model,
            "video": video,
            "num_gt_paths": len(group),
        }
        for metric in metrics:
            item[f"{metric}_mean"] = float(group[f"{metric}_mean"].mean())
            item[f"{metric}_best"] = float(group[f"{metric}_best"].mean())
        video_rows.append(item)
    per_video = pd.DataFrame(video_rows)
    per_video.to_csv(output_dir / "per_video_best_mean.csv", index=False)

    final_rows: list[dict[str, object]] = []
    for model, group in per_video.groupby("model", sort=True):
        for metric, (_, direction) in metrics.items():
            final_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "direction": direction,
                    "aggregation": "per_video",
                    "mean_score": float(group[f"{metric}_mean"].mean()),
                    "best_score": float(group[f"{metric}_best"].mean()),
                    "num_videos": int(len(group)),
                }
            )
    final = pd.DataFrame(final_rows)
    final.to_csv(output_dir / "final_metrics.csv", index=False)
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    return final


def main() -> None:
    args = parse_args()
    gt_root = args.ground_truth_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = select_trajectory_metrics(args.metrics)
    gt_dirs = {path.name: path for path in gt_root.iterdir() if path.is_dir()}
    if not gt_dirs:
        raise FileNotFoundError(f"No video directories found under {gt_root}")

    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for model_name, kind, raw_root in args.model:
        root = raw_root.expanduser().resolve()
        gt_history_points = GT_HISTORY_POINTS
        paths = prediction_paths(root, kind)
        if not paths:
            raise FileNotFoundError(f"No {kind} prediction files found under {root}")
        for prediction_path in tqdm(paths, desc=f"{model_name}: {kind}", unit="prediction"):
            try:
                video = resolve_video_name(prediction_path, root, set(gt_dirs))
                prediction = prepare_prediction(prediction_path, kind)
                width, height = _video_size_from_name(video)
                gt_paths = sorted((gt_dirs[video] / "event_data").glob("*.txt"))
                if not gt_paths:
                    raise FileNotFoundError(f"No GT event files for {video}")
                for gt_path in gt_paths:
                    try:
                        gt, valid_mask = load_diem_ground_truth(
                            gt_path,
                            width,
                            height,
                            max_samples=TARGET_POINTS,
                            history_points=gt_history_points,
                            allow_short_window=True,
                            raw_frame_window=True,
                            return_raw_frame_valid_mask=True,
                        )
                    except ValueError as exc:
                        skipped.append(f"{model_name}: {gt_path}: {exc}")
                        continue
                    # GT invalid samples are omitted by the loader. Apply the
                    # identical raw-frame mask to the matching prediction
                    # indices so subsequent points remain time-aligned.
                    prediction_for_gt = prediction[: len(valid_mask)][valid_mask]
                    gt_eval, prediction_eval = align_trajectory_prefixes(gt, prediction_for_gt)
                    score = _score_pair(gt_eval, prediction_eval, width, height, metrics)
                    rows.append(
                        {
                            "model": model_name,
                            "kind": kind,
                            "video": video,
                            "gt_id": gt_path.stem,
                            "gt_path": str(gt_path),
                            "prediction_id": prediction_path.stem,
                            "prediction_path": str(prediction_path),
                            "gt_points": len(gt),
                            "prediction_points": len(prediction_for_gt),
                            "evaluation_points": len(gt_eval),
                            "gt_history_points": gt_history_points,
                            **score,
                        }
                    )
            except (FileNotFoundError, ValueError) as exc:
                skipped.append(f"{model_name}: {prediction_path}: {exc}")

        pd.DataFrame(rows).to_csv(output_dir / "pairwise_comparisons.csv", index=False)

    if not rows:
        raise RuntimeError("No valid prediction/GT comparisons were produced. See skipped outputs above.")
    comparisons_csv = output_dir / "pairwise_comparisons.csv"
    # Always reduce as: predictions -> GT path -> video -> equal-weight videos.
    # This avoids participant-count differences changing a video's final weight.
    per_gt, _ = accumulate_pairwise_comparisons(
        comparisons_csv, output_dir, args.metrics, aggregation="per_gt"
    )
    final = save_equal_video_summary(per_gt, output_dir, metrics)
    print(f"Saved {len(rows):,} comparisons to {comparisons_csv}")
    print(f"Saved {len(per_gt):,} per-GT summaries to {output_dir}")
    print(f"Saved equal-weight per-video summaries to {output_dir / 'per_video_best_mean.csv'}")
    print(final.to_string(index=False))
    if skipped:
        print(f"Skipped {len(skipped):,} outputs:")
        for item in skipped[:20]:
            print(f"  {item}")
        if len(skipped) > 20:
            print("  ...")


if __name__ == "__main__":
    main()
