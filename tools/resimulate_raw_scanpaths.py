"""Expand static scanpaths into raw trajectories at their declared duration.

Dry-run by default.  Use --apply to atomically replace existing *_simraw.csv
files with repeated original static points.  A duration encoded in the
baseline path (for example ``full_30s`` or ``autoreg15s``) controls that
job's frame count. TPP duration-based trajectories keep native timing and are
capped at that same per-job duration; shorter trajectories are extended with
evenly distributed repeats.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINES_ROOT = REPO_ROOT / "baselines"
DURATION_COMPONENT_PATTERN = re.compile(
    r"(?:"
    r"(?:^|[_-])(?P<separated>\d+(?:\.\d+)?)(?:s|sec(?:onds?)?)(?=$|[_-])"
    r"|(?:autoreg(?:ressive)?|full)[_-]?(?P<embedded>\d+(?:\.\d+)?)(?:s|sec(?:onds?)?)(?=$|[_-])"
    r")",
    re.IGNORECASE,
)
CHUNKED_PATTERN = re.compile(r"chunk", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat static scanpath points into per-frame raw simulations."
    )
    parser.add_argument("--baselines-root", type=Path, default=DEFAULT_BASELINES_ROOT)
    parser.add_argument(
        "--only-baseline",
        action="append",
        default=None,
        metavar="DIRECTORY",
        help=(
            "Restrict simulation to an immediate child directory of --baselines-root. "
            "Repeat to select multiple baselines."
        ),
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=27.0,
        help="Fallback duration when a baseline path does not declare one (default: 27).",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--include-static-scanpaths",
        action="store_true",
        help=(
            "Also create simulated_raw/scanpath_simraw.csv for pred_*/scanpath.csv "
            "files that do not already have a raw-output placeholder."
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=3.0,
        help="Normalize each duration-annotated prediction chunk to this duration (default: 3).",
    )
    parser.add_argument(
        "--tpp-timing",
        choices=["duration", "even"],
        default="duration",
        help=(
            "TPP raw-simulation timing: duration uses native duration_ms timing "
            "and caps at 810 frames; even ignores duration_ms and evenly repeats "
            "points to the requested frame count."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace existing simulated raw CSVs. Omit for a dry-run report.",
    )
    return parser.parse_args()


def duration_seconds_for_path(path: Path, root: Path) -> float | None:
    """Return the nearest duration token in a baseline-relative path."""
    relative_parts = path.relative_to(root).parts
    for component in reversed(relative_parts):
        if component.lower() == "simulated_raw":
            continue
        match = DURATION_COMPONENT_PATTERN.search(component)
        if match is not None:
            seconds = match.group("separated") or match.group("embedded")
            return float(seconds)
    return None


def is_timed_nonchunked(path: Path, root: Path) -> bool:
    relative_path = path.relative_to(root).as_posix()
    return bool(
        duration_seconds_for_path(path, root) is not None
        and not CHUNKED_PATTERN.search(relative_path)
    )


def static_path_for(simulated_path: Path) -> Path:
    """Map `simulated_raw/foo_simraw.csv` to its sibling static `foo.csv`."""
    if not simulated_path.name.endswith("_simraw.csv"):
        raise ValueError(f"Not a simulated raw CSV: {simulated_path}")
    return simulated_path.parent.parent / simulated_path.name.replace("_simraw.csv", ".csv")


def simulation_jobs(
    root: Path, include_static_scanpaths: bool = False
) -> list[tuple[Path, Path]]:
    """Return `(static_scanpath, raw_output)` jobs, including TPP auto outputs."""
    jobs: dict[Path, Path] = {}
    for raw_path in root.rglob("*_simraw.csv"):
        if is_timed_nonchunked(raw_path, root):
            jobs[raw_path] = static_path_for(raw_path)

    # Any static trajectory with explicit fixation durations can be simulated
    # even when no placeholder raw file exists yet. This covers the GazeFormer
    # and LSTM ISP exports, which provide `duration_s`.
    for static_path in root.rglob("scanpath.csv"):
        if "simulated_raw" in static_path.parts:
            continue
        with static_path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        if "duration_ms" in header or "duration_s" in header:
            raw_path = static_path.parent / "simulated_raw" / "scanpath_simraw.csv"
            jobs[raw_path] = static_path

    if include_static_scanpaths:
        # Static model exports such as DeepGaze use pred_###/scanpath.csv and
        # do not ship raw placeholders. Keep this opt-in so the default job
        # set remains limited to exports that explicitly request simulation.
        for static_path in root.rglob("scanpath.csv"):
            if "simulated_raw" in static_path.parts or not static_path.parent.name.startswith("pred_"):
                continue
            raw_path = static_path.parent / "simulated_raw" / "scanpath_simraw.csv"
            jobs[raw_path] = static_path

    # tppgaze_scanpaths_auto contains static TPP predictions but has no raw
    # files yet. Its duration_ms column is handled by the standard reader.
    tpp_auto_root = root / "tppgaze_scanpaths_auto"
    if tpp_auto_root.is_dir():
        for static_path in tpp_auto_root.rglob("scanpath.csv"):
            raw_path = static_path.parent / "simulated_raw" / "scanpath_simraw.csv"
            jobs[raw_path] = static_path
    return sorted(((static_path, raw_path) for raw_path, static_path in jobs.items()), key=lambda job: str(job[1]))


def read_scanpath(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("contains no points")
    try:
        points = np.asarray([(float(row["x"]), float(row["y"])) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("must have numeric x and y columns") from exc
    duration_column = "duration_ms" if "duration_ms" in rows[0] else "duration_s" if "duration_s" in rows[0] else None
    if duration_column is None:
        return points, None
    try:
        durations = np.asarray([float(row[duration_column]) for row in rows], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"has a non-numeric {duration_column} value") from exc
    if duration_column == "duration_s":
        durations *= 1_000.0
    if not np.isfinite(durations).all() or (durations < 0).any() or durations.sum() <= 0:
        raise ValueError("duration_ms values must be finite, non-negative, and sum to more than zero")
    return points, durations


def normalize_chunk_durations(
    durations_ms: np.ndarray, summary_path: Path, chunk_seconds: float
) -> np.ndarray:
    """Scale each exported prediction chunk to its fixed rollout duration."""
    if chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be positive")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        counts = [int(chunk["num_points"]) for chunk in summary]
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid chunk summary: {summary_path}") from exc
    if not counts or any(count < 1 for count in counts) or sum(counts) != len(durations_ms):
        raise ValueError(
            f"chunk counts in {summary_path} do not match {len(durations_ms)} scanpath points"
        )

    normalized: list[np.ndarray] = []
    start = 0
    target_ms = float(chunk_seconds) * 1_000.0
    for count in counts:
        chunk = durations_ms[start : start + count]
        total_ms = float(chunk.sum())
        if not np.isfinite(total_ms) or total_ms <= 0:
            raise ValueError(f"chunk starting at point {start} has a non-positive duration total")
        normalized.append(chunk * (target_ms / total_ms))
        start += count
    return np.concatenate(normalized)


def repeat_points_evenly(points: np.ndarray, frame_count: int) -> np.ndarray:
    """Map output frames to source points without creating intermediate values.

    Every output coordinate is an exact source coordinate. When there are fewer
    source points than frames, each is held for either floor(N/M) or ceil(N/M)
    frames. When there are more source points, they are evenly subsampled.
    """
    if frame_count < 1:
        raise ValueError("seconds * fps must produce at least one frame")
    if len(points) < 1:
        raise ValueError("scanpath contains no points")
    source_indices = np.floor(np.arange(frame_count) * len(points) / frame_count).astype(int)
    return points[source_indices]


def repeat_points_by_duration(points: np.ndarray, durations_ms: np.ndarray, max_frame_count: int, fps: float) -> tuple[np.ndarray, bool]:
    """Use native duration timing, holding the final fixation for a short run."""
    if len(points) != len(durations_ms):
        raise ValueError("points and duration_ms counts differ")
    total_duration = durations_ms.sum()
    native_frame_count = int(np.ceil(total_duration * float(fps) / 1_000.0))
    output_frame_count = min(max_frame_count, native_frame_count)
    # Assign each native 30-FPS sample to the duration interval containing its
    # start. No duration is stretched or compressed; only an overlong tail is
    # omitted once the requested frame cap is reached.
    frame_times = np.arange(output_frame_count, dtype=np.float64) * 1_000.0 / float(fps)
    source_indices = np.searchsorted(np.cumsum(durations_ms), frame_times, side="right")
    source_indices = np.clip(source_indices, 0, len(points) - 1)
    simulated = points[source_indices]
    if native_frame_count < max_frame_count:
        # Keep the declared timing intact, then hold the final fixation for the
        # remaining frames in the common evaluation window.
        extension = np.repeat(simulated[-1:], max_frame_count - native_frame_count, axis=0)
        return np.concatenate((simulated, extension), axis=0), True
    return simulated, False


def write_xy_atomically(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y"])
        writer.writerows(points.tolist())
    temporary_path.replace(path)


def render_progress(index: int, total: int) -> None:
    width = 30
    complete = int(width * index / total) if total else width
    bar = "#" * complete + "-" * (width - complete)
    print(f"\rResimulating: [{bar}] {index:,}/{total:,}", end="", flush=True)


def main() -> None:
    args = parse_args()
    root = args.baselines_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Baselines root does not exist or is not a directory: {root}")
    fallback_frame_count = round(float(args.seconds) * float(args.fps))
    jobs = simulation_jobs(root, include_static_scanpaths=args.include_static_scanpaths)
    if args.only_baseline:
        requested = set(args.only_baseline)
        available = {path.name for path in root.iterdir() if path.is_dir()}
        unknown = sorted(requested.difference(available))
        if unknown:
            raise ValueError(f"Unknown baseline directory/directories: {unknown}")
        jobs = [
            job for job in jobs
            if job[1].relative_to(root).parts[0] in requested
        ]
    updated = 0
    duration_weighted = 0
    duration_truncated = 0
    duration_extended = 0
    skipped: list[str] = []

    if fallback_frame_count < 1:
        raise ValueError("--seconds * --fps must produce at least one output frame")
    print(f"Fallback trajectory: {fallback_frame_count} points ({args.seconds:g}s at {args.fps:g} FPS)")
    print(f"Eligible simulated raw outputs: {len(jobs):,}")
    timed_jobs = 0
    for index, (static_path, simulation_path) in enumerate(jobs, start=1):
        if not static_path.is_file():
            skipped.append(f"missing static scanpath: {simulation_path.relative_to(root)}")
            render_progress(index, len(jobs))
            continue
        try:
            declared_seconds = duration_seconds_for_path(static_path, root)
            target_seconds = float(args.seconds) if declared_seconds is None else declared_seconds
            frame_count = round(target_seconds * float(args.fps))
            if frame_count < 1:
                raise ValueError(f"duration {target_seconds:g}s produces fewer than one frame")
            timed_jobs += int(declared_seconds is not None)
            points, durations_ms = read_scanpath(static_path)
            summary_path = static_path.parent / "chunk_prediction_summary.json"
            if durations_ms is not None and summary_path.is_file():
                durations_ms = normalize_chunk_durations(
                    durations_ms, summary_path, args.chunk_seconds
                )
            if durations_ms is None or args.tpp_timing == "even":
                resampled = repeat_points_evenly(points, frame_count)
            else:
                resampled, was_extended = repeat_points_by_duration(points, durations_ms, frame_count, args.fps)
                duration_weighted += 1
                duration_extended += int(was_extended)
                duration_truncated += int(
                    len(resampled) == frame_count
                    and durations_ms.sum() * args.fps / 1_000.0 > frame_count
                )
        except ValueError as exc:
            skipped.append(f"{static_path.relative_to(root)}: {exc}")
            render_progress(index, len(jobs))
            continue
        if args.apply:
            write_xy_atomically(simulation_path, resampled)
        updated += 1
        render_progress(index, len(jobs))
    if jobs:
        print()

    action = "Replaced" if args.apply else "Would replace"
    print(f"{action} {updated:,} simulated raw files.")
    if timed_jobs:
        print(f"Matched an encoded folder duration for {timed_jobs:,} files.")
    if duration_weighted:
        print(f"Used duration-aware point repeats for {duration_weighted:,} files.")
        print(f"Truncated the overlong tail for {duration_truncated:,} duration-based files.")
        print(f"Extended {duration_extended:,} short duration-based files by holding their final fixation.")
    if skipped:
        print(f"Skipped {len(skipped):,} files:")
        for message in skipped[:20]:
            print(f"  {message}")
        if len(skipped) > 20:
            print("  ...")
    if not args.apply:
        print("Dry run only. Re-run with --apply to replace the simulations.")


if __name__ == "__main__":
    main()
