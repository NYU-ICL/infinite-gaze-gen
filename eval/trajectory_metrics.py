import numpy as np
import time
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from Levenshtein import distance as levenshtein
import os
from natsort import natsorted
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json
from collections.abc import Callable


try:
    from frechetdist import frdist as _frechetdist
except ImportError:
    _frechetdist = None

def _scanpath_to_grid_string(scanpath, height, width, xbins=12, ybins=8):
    """Convert (x, y) points to one character per spatial-grid cell."""
    points = np.asarray(scanpath, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("scanpath must have shape (N, 2)")
    if int(width) <= 0 or int(height) <= 0 or int(xbins) <= 0 or int(ybins) <= 0:
        raise ValueError("height, width, xbins, and ybins must be positive")
    xcells = np.clip(np.floor(points[:, 0] * int(xbins) / int(width)).astype(int), 0, int(xbins) - 1)
    ycells = np.clip(np.floor(points[:, 1] * int(ybins) / int(height)).astype(int), 0, int(ybins) - 1)
    return "".join(chr(0x100 + int(y * int(xbins) + x)) for x, y in zip(xcells, ycells))


def levenshtein_distance(seq1, seq2, height=720, width=1280, Xbins=12, Ybins=8, **kwargs):
    """Levenshtein distance between scanpaths quantized to a 12x8 spatial grid.

    ``height`` and ``width`` may be passed for the stimulus canvas; defaults keep
    existing evaluator calls compatible with the standard 1280x720 DIEM canvas.
    """
    seq1_str = _scanpath_to_grid_string(seq1, height, width, Xbins, Ybins)
    seq2_str = _scanpath_to_grid_string(seq2, height, width, Xbins, Ybins)
    return levenshtein(seq1_str, seq2_str)

def dynamic_time_warping(seq1, seq2):
    """
    Compute Dynamic Time Warping (DTW) distance.
    """
    distance, _ = fastdtw(seq1, seq2, dist=euclidean)
    return distance

def discrete_frechet_distance(p, q):
    """
    Compute Discrete Frechet Distance (DFD) between two 2D trajectories.
    """

    p = np.array(p, np.float64)
    q = np.array(q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:
        raise ValueError('Input curves are empty.')

    ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)

    for i in range(len_p):
        for j in range(len_q):
            if i == 0 and j == 0:
                ca[i, j] = np.linalg.norm(p[i] - q[j])
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], np.linalg.norm(p[i] - q[j]))
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], np.linalg.norm(p[i] - q[j]))
            elif i > 0 and j > 0:
                ca[i, j] = max(
                    min(
                        ca[i - 1, j],
                        ca[i - 1, j - 1],
                        ca[i, j - 1]
                    ),
                    np.linalg.norm(p[i] - q[j])
                )
            else:
                ca[i, j] = float('inf')

    dist = ca[len_p - 1, len_q - 1]
    return dist

def mean_absolute_error(seq1, seq2):
    """
    Compute Mean Absolute Error (MAE) between two sequences.
    """
    # Ensure sequences are the same length
    min_len = min(len(seq1), len(seq2))
    seq1, seq2 = seq1[:min_len], seq2[:min_len]
    return np.mean(np.abs(seq1 - seq2))

def root_mean_square_error(seq1, seq2):
    """
    Compute Root Mean Square Error (RMSE) between two sequences.
    """
    # Ensure sequences are the same length
    min_len = min(len(seq1), len(seq2))
    seq1, seq2 = seq1[:min_len], seq2[:min_len]
    return np.sqrt(np.mean((seq1 - seq2) ** 2))

import numpy as np



def maximum_temporal_correlation_trajectory(
    seq1,
    seq2,
    max_lag=30,
    use_abs=False,
    min_overlap_fraction=0.5,
    eps=1e-8,
):
    a = np.asarray(seq1, dtype=np.float64)
    b = np.asarray(seq2, dtype=np.float64)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs must have shape (N, 2).")

    if a.shape[1] < 2 or b.shape[1] < 2:
        raise ValueError("2D gaze trajectories must contain at least x and y coordinates.")

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Trajectories must contain at least 2 samples.")

    a = a[:, :2]
    b = b[:, :2]

    if not np.all(np.isfinite(a)):
        raise ValueError("seq1 contains NaN or Inf values.")

    if not np.all(np.isfinite(b)):
        raise ValueError("seq2 contains NaN or Inf values.")

    max_lag = max(0, int(max_lag))

    if not 0 < min_overlap_fraction <= 1:
        raise ValueError("min_overlap_fraction must be in the interval (0, 1].")

    shorter_length = min(len(a), len(b))
    min_overlap = max(
        2,
        int(np.ceil(shorter_length * min_overlap_fraction))
    )

    def safe_pearson(x, y):
        sx = np.std(x)
        sy = np.std(y)

        x_constant = sx < eps
        y_constant = sy < eps

        if x_constant and y_constant:
            if np.allclose(x, y, atol=eps, rtol=0):
                corr = 1.0
            else:
                corr = 0.0
        elif x_constant or y_constant:
            corr = 0.0
        else:
            corr = np.corrcoef(x, y)[0, 1]

            if not np.isfinite(corr):
                corr = 0.0

        if use_abs:
            corr = abs(corr)

        return float(corr)

    best_score = -np.inf
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[:lag]
            bb = b[-lag:]
        elif lag > 0:
            aa = a[lag:]
            bb = b[:-lag]
        else:
            aa = a
            bb = b

        n = min(len(aa), len(bb))

        if n < min_overlap:
            continue

        aa = aa[:n]
        bb = bb[:n]

        corr_x = safe_pearson(aa[:, 0], bb[:, 0])
        corr_y = safe_pearson(aa[:, 1], bb[:, 1])

        score = float((corr_x + corr_y) / 2.0)

        tolerance = 1e-12

        if (
            score > best_score + tolerance
            or (
                abs(score - best_score) <= tolerance
                and abs(lag) < abs(best_lag)
            )
        ):
            best_score = score
            best_lag = lag

    if not np.isfinite(best_score):
        return 0.0, 0

    return float(best_score), int(best_lag)


# The first three are distances (smaller is better); temporal correlation is a
# similarity (larger is better).  Keeping this beside the metric definitions
# prevents the "best of N samples" reduction from accidentally using the same
# direction for every metric.
TRAJECTORY_METRICS: dict[str, tuple[Callable, str]] = {
    "dtw": (dynamic_time_warping, "min"),
    "discrete_frechet": (discrete_frechet_distance, "min"),
    "levenshtein": (levenshtein_distance, "min"),
    "temporal_correlation": (maximum_temporal_correlation_trajectory, "max"),
}


def select_trajectory_metrics(metrics: list[str] | tuple[str, ...] | None = None) -> dict[str, tuple[Callable, str]]:
    """Validate a requested metric subset, accepting comma-separated names too."""
    if metrics is None:
        return TRAJECTORY_METRICS.copy()
    names = [name.strip() for value in metrics for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one metric must be selected")
    unknown = sorted(set(names).difference(TRAJECTORY_METRICS))
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Available: {sorted(TRAJECTORY_METRICS)}")
    return {name: TRAJECTORY_METRICS[name] for name in names}


def load_prediction_csv(path: str | Path) -> np.ndarray:
    """Load generated ``scanpath.csv`` coordinates in video-pixel space."""
    frame = pd.read_csv(path)
    required = {"x", "y"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain x and y columns; got {frame.columns.tolist()}")
    points = frame[["x", "y"]].to_numpy(dtype=np.float64)
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError(f"{path} does not contain at least two finite gaze points")
    return points


def align_trajectory_prefixes(gt: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the common temporal prefix of a GT/prediction trajectory pair.

    Pairwise trajectory metrics should compare the same time window.  This
    trims the longer sequence to the shorter one; it deliberately does not
    resample, normalize, or apply an evaluation sub-window.
    """
    n = min(len(gt), len(prediction))
    if n < 2:
        raise ValueError("GT and prediction need at least two overlapping points")
    return gt[:n], prediction[:n]


def load_diem_ground_truth(
    path: str | Path,
    movie_width: int,
    movie_height: int,
    max_samples: int,
    history_points: int = 0,
    allow_short_window: bool = False,
    raw_frame_window: bool = False,
    return_raw_frame_valid_mask: bool = False,
    screen_width: int = 1280,
    screen_height: int = 960,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Load the first generated-window worth of a DIEM event file.

    DIEM records screen coordinates (and may contain binocular or monocular
    rows).  This mirrors ``datasets.diem.DIEMDataset``: binocular positions
    are averaged, screen letterbox offsets are removed, and invalid zero rows
    are excluded.  ``max_samples`` is taken from the prediction length, so
    this evaluates exactly the generated 30-second window rather than a later
    portion of a longer event file.
    """
    rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if rows.shape[1] >= 9:
        x = (rows[:, 1] + rows[:, 5]) / 2.0
        y = (rows[:, 2] + rows[:, 6]) / 2.0
        invalid = (rows[:, 1] == 0) & (rows[:, 2] == 0) & (rows[:, 5] == 0) & (rows[:, 6] == 0)
    elif rows.shape[1] >= 4:
        x, y = rows[:, 1], rows[:, 2]
        invalid = (x == 0) & (y == 0)
    else:
        raise ValueError(f"Unsupported DIEM event-file format in {path}")

    x = x - (screen_width - movie_width) / 2.0
    y = y - (screen_height - movie_height) / 2.0
    points = np.column_stack((x, y))
    valid = (
        np.isfinite(points).all(axis=1)
        & ~invalid
        & (points[:, 0] >= 0)
        & (points[:, 0] < movie_width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < movie_height)
    )
    if raw_frame_window:
        # Keep one GT point per raw video frame so it lines up exactly with a
        # model sampled at every frame. Invalid rows are removed later with
        # their corresponding prediction rows; they are never center-filled.
        window_start = int(history_points)
        window_end = window_start + int(max_samples)
        if len(points) < window_end and not allow_short_window:
            raise ValueError(
                f"{path} has {len(points)} raw points; need {window_end} for "
                f"{history_points} history and {max_samples} evaluation points"
            )
        points = points[window_start:min(window_end, len(points))].copy()
        valid = valid[window_start:min(window_end, len(valid))]
        points = points[valid]
        if len(points) < 2:
            raise ValueError(f"{path} has fewer than two valid raw points after the history segment")
        if return_raw_frame_valid_mask:
            return points, valid
        return points

    points = points[valid]
    window_end = int(history_points) + int(max_samples)
    if len(points) < window_end and not allow_short_window:
        raise ValueError(
            f"{path} has {len(points)} valid points; need {window_end} for {history_points} history and {max_samples} evaluation points"
        )
    points = points[int(history_points):window_end]
    if len(points) < 2:
        raise ValueError(f"{path} has fewer than two valid points after the history segment")
    return points


def _video_size_from_name(video_name: str) -> tuple[int, int]:
    """Read DIEM's ``..._<width>x<height>`` suffix without opening a video."""
    import re

    match = re.search(r"_(\d+)x(\d+)$", video_name)
    if match is None:
        raise ValueError(f"Could not infer video dimensions from {video_name!r}")
    return int(match.group(1)), int(match.group(2))


def _score_pair(
    gt: np.ndarray, prediction: np.ndarray, width: int, height: int, metrics: dict[str, tuple[Callable, str]]
) -> dict[str, float | int]:
    scores: dict[str, float | int] = {}
    if "dtw" in metrics:
        scores["dtw"] = float(dynamic_time_warping(gt, prediction))
    if "discrete_frechet" in metrics:
        scores["discrete_frechet"] = float(discrete_frechet_distance(gt, prediction))
    if "levenshtein" in metrics:
        scores["levenshtein"] = float(levenshtein_distance(gt, prediction, width=width, height=height))
    if "temporal_correlation" in metrics:
        correlation, lag = maximum_temporal_correlation_trajectory(gt, prediction)
        scores["temporal_correlation"] = float(correlation)
        scores["temporal_correlation_lag"] = int(lag)
    return scores


def save_pairwise_comparisons(
    predictions_root: str | Path,
    ground_truth_root: str | Path,
    output_csv: str | Path,
    model_name: str | None = None,
    metrics: list[str] | tuple[str, ...] | None = None,
    history_points: int = 0,
    allow_short_ground_truth: bool = False,
) -> pd.DataFrame:
    """Evaluate every GT path against every generated sample and save rows.

    The output has one row per ``(video, GT event file, predicted sample)``.
    It is intentionally the primary artifact: aggregation is a separate,
    reproducible read of this CSV via :func:`accumulate_pairwise_comparisons`.
    """
    predictions_root = Path(predictions_root)
    ground_truth_root = Path(ground_truth_root)
    output_csv = Path(output_csv)
    model_name = model_name or predictions_root.name
    selected_metrics = select_trajectory_metrics(metrics)
    rows: list[dict[str, object]] = []
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted(path for path in predictions_root.iterdir() if path.is_dir())
    if not video_dirs:
        raise FileNotFoundError(f"No video directories found under {predictions_root}")
    for video_dir in tqdm(video_dirs, desc="Evaluating videos", unit="video"):
        width, height = _video_size_from_name(video_dir.name)
        prediction_paths = sorted(video_dir.glob("sample_*/scanpath.csv"))
        gt_paths = sorted((ground_truth_root / video_dir.name / "event_data").glob("*.txt"))
        if not prediction_paths:
            raise FileNotFoundError(f"No sample_*/scanpath.csv files found for {video_dir.name}")
        if not gt_paths:
            raise FileNotFoundError(f"No GT event files found for {video_dir.name}")
        predictions = [(path, load_prediction_csv(path)) for path in prediction_paths]
        # All generated samples have the same 30-second length; fail early if a
        # partial write slipped into the directory.
        lengths = {len(points) for _, points in predictions}
        if len(lengths) != 1:
            raise ValueError(f"Predictions for {video_dir.name} have inconsistent lengths: {sorted(lengths)}")
        max_samples = lengths.pop()

        for gt_path in gt_paths:
            try:
                loaded_gt = load_diem_ground_truth(
                    gt_path, width, height, max_samples, history_points=history_points,
                    allow_short_window=allow_short_ground_truth, raw_frame_window=True,
                    return_raw_frame_valid_mask=True,
                )
            except ValueError as exc:
                tqdm.write(f"Skipping GT {gt_path.name}: {exc}")
                continue
            for prediction_path, prediction in predictions:
                gt, valid_mask = loaded_gt
                # The mask addresses raw GT frames after the history. Apply it
                # to the same prediction indices before scoring.
                prediction = prediction[: len(valid_mask)][valid_mask]
                gt_eval, prediction_eval = align_trajectory_prefixes(gt, prediction)
                score = _score_pair(gt_eval, prediction_eval, width, height, selected_metrics)
                rows.append({
                    "model": model_name,
                    "video": video_dir.name,
                    "gt_id": gt_path.stem,
                    "gt_path": str(gt_path.resolve()),
                    "prediction_id": prediction_path.parent.name,
                    "prediction_path": str(prediction_path.resolve()),
                    "gt_points": len(gt),
                    "prediction_points": len(prediction),
                    "evaluation_points": len(gt_eval),
                    "gt_history_points": int(history_points),
                    "raw_frame_window": True,
                    "short_gt_window": len(gt) < len(prediction),
                    **score,
                })
        # Persist the raw rows before moving to the next video.  This leaves a
        # usable intermediate artifact even if a long evaluation is interrupted.
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    comparisons = pd.DataFrame(rows)
    return comparisons


def accumulate_pairwise_comparisons(
    comparisons_csv: str | Path,
    output_dir: str | Path,
    metrics: list[str] | tuple[str, ...] | None = None,
    aggregation: str = "per_gt",
    model_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-GT and final summaries from saved pairwise rows.

    ``per_gt`` gives every ``(video, GT)`` path equal final weight.  The
    ``scanpathevals`` mode mirrors ``summarize_autoreg_raw_eval.py``: its mean
    is over all pairwise rows, while its best first selects per-GT predictions,
    averages those within each video, then averages videos.
    """
    valid_aggregations = {"per_gt", "scanpathevals"}
    if aggregation not in valid_aggregations:
        raise ValueError(f"Unknown aggregation {aggregation!r}; choose from {sorted(valid_aggregations)}")
    comparisons_csv = Path(comparisons_csv)
    comparisons = pd.read_csv(comparisons_csv)
    # Accept legacy pairwise CSVs that do not include a model column.  Make
    # them usable for post-hoc accumulation with an explicit label when
    # provided, or infer one from the enclosing model directory.
    if "model" not in comparisons.columns:
        inferred_model_name = model_name or comparisons_csv.resolve().parent.parent.name
        comparisons.insert(0, "model", inferred_model_name)
    # When summarizing an existing partial run, default to every metric present
    # in that file.  An explicit --metrics value provides stricter validation.
    selected_metrics = (
        select_trajectory_metrics(metrics)
        if metrics is not None
        else {name: spec for name, spec in TRAJECTORY_METRICS.items() if name in comparisons.columns}
    )
    if not selected_metrics:
        raise ValueError("Comparison CSV does not contain any supported trajectory metric columns")
    required = {"model", "video", "gt_id", "prediction_id", *selected_metrics}
    missing = required.difference(comparisons.columns)
    if missing:
        raise ValueError(f"Comparison CSV is missing columns: {sorted(missing)}")
    group_columns = ["model", "video", "gt_id"]
    per_gt_rows: list[dict[str, object]] = []
    for keys, group in comparisons.groupby(group_columns, sort=True):
        item = dict(zip(group_columns, keys))
        item["num_predictions"] = len(group)
        for metric, (_, direction) in selected_metrics.items():
            values = group[metric]
            best_index = values.idxmin() if direction == "min" else values.idxmax()
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_best"] = float(comparisons.loc[best_index, metric])
            item[f"{metric}_best_prediction_id"] = comparisons.loc[best_index, "prediction_id"]
        per_gt_rows.append(item)
    per_gt = pd.DataFrame(per_gt_rows)
    if per_gt.empty:
        raise ValueError("No complete GT evaluation windows were available to aggregate")

    final_rows: list[dict[str, object]] = []
    for model, group in per_gt.groupby("model", sort=True):
        model_comparisons = comparisons[comparisons["model"] == model]
        for metric, (_, direction) in selected_metrics.items():
            if aggregation == "scanpathevals":
                # Match summarize_autoreg_raw_eval.py exactly: pairwise mean,
                # then equal video weighting of each video's per-GT best mean.
                metric_df = model_comparisons[["video", "gt_id", metric]].copy()
                metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
                metric_df = metric_df[np.isfinite(metric_df[metric].to_numpy(dtype=float))]
                mean_score = float(metric_df[metric].mean()) if not metric_df.empty else float("nan")
                if metric_df.empty:
                    best_score = float("nan")
                elif direction == "min":
                    best_per_video_gt = metric_df.groupby(["video", "gt_id"], dropna=False)[metric].min()
                    best_score = float(best_per_video_gt.groupby(level=0).mean().mean())
                else:
                    best_per_video_gt = metric_df.groupby(["video", "gt_id"], dropna=False)[metric].max()
                    best_score = float(best_per_video_gt.groupby(level=0).mean().mean())
            else:
                mean_score = float(group[f"{metric}_mean"].mean())
                best_score = float(group[f"{metric}_best"].mean())
            final_rows.append({
                "model": model,
                "metric": metric,
                "direction": direction,
                "aggregation": aggregation,
                "best_score": best_score,
                "mean_score": mean_score,
                "num_gt_paths": len(group),
                "num_videos": int(group["video"].nunique()),
                "num_comparisons": int(len(model_comparisons)),
            })
    final = pd.DataFrame(final_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if aggregation == "per_gt" else f"_{aggregation}"
    per_gt.to_csv(output_dir / f"per_gt_best_mean{suffix}.csv", index=False)
    final.to_csv(output_dir / f"final_metrics{suffix}.csv", index=False)
    (output_dir / f"final_metrics{suffix}.json").write_text(
        json.dumps(final.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    return per_gt, final
