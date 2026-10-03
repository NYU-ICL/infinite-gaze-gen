"""DIEM generator matching the original autoregressive_30s runnable."""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from common import get_video_frame_count, instantiate_from_config, load_checkpoint, load_yaml_config, merge_opts_to_config, seed_everything, unnormalize_points
from datasets.diem import DIEMDataset
from sample_video import (
    clip_normalized_history_to_movie,
    load_patch_sequence,
    sample_window,
)


def args():
    p = argparse.ArgumentParser(description="Generate DIEM scanpaths with original generator semantics.")
    p.add_argument(
        "--dataset-root",
        default=r"C:\Users\jk8659\NYU\research\Scanpath\diffeye\artifacts\datasets\DIEM",
    )
    p.add_argument("--split-json", default=None, help="Selects videos only; subjects are deliberately ignored.")
    p.add_argument(
        "--participants-manifest",
        default="baselines/unet_90_45_70epoch/participants_by_generation.csv",
        help="CSV or JSON mapping stimulus + pred_### to the participant used for history.",
    )
    p.add_argument(
        "--manifest-only",
        action="store_true",
        default=True,
        help="When using --participants-manifest, generate only stimuli named in that manifest.",
    )
    p.add_argument("--config", default="final_model_90_45/inference_config.yaml"); p.add_argument("--checkpoint", default="final_model_90_45/checkpoint_70.pth")
    p.add_argument("--output-dir", default="artifacts/diem_final_model_90_45_batched_reproduced_clip")
    p.add_argument("--num-predictions", "--num-samples", dest="num_predictions", type=int, default=10)
    p.add_argument("--chunk-seconds", type=int, default=3); p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-video-seconds", type=int, default=30)
    p.add_argument("--seed", type=int, default=12); p.add_argument("--seed-step", type=int, default=1)
    p.add_argument(
        "--seed-reset-every",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Restart the original per-process RNG state and video seed index every N selected videos. "
            "Use 4 to replay outputs that were generated as consecutive four-video batches. "
            "The default is 4. Use 0 for continuous single-run indexing."
        ),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    return p.parse_args()


def tracks_for(dataset, stim):
    out = {}
    for i, item in enumerate(dataset.stim_sub_list):
        if item.stim == stim:
            eye = dataset.eye_data_list[i]
            out[item.sub] = (np.stack([eye.m_eye_x_movie, eye.m_eye_y_movie], axis=1), np.asarray(dataset._valid_gaze_indices[i], dtype=np.int64))
    return dict(sorted(out.items()))


def load_manifest(path):
    """Return {stimulus: {prediction index: participant}} from exported metadata."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        rows = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("--participants-manifest must be a .csv or .json file.")
    mapping = {}
    for row in rows:
        try:
            stimulus = str(row["stimulus"])
            prediction_id = str(row["prediction_id"])
            participant = str(row["participant"])
            if not prediction_id.startswith("pred_"):
                raise ValueError
            index = int(prediction_id.removeprefix("pred_"))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid participant-manifest record: {row}") from exc
        existing = mapping.setdefault(stimulus, {}).get(index)
        if existing is not None and existing != participant:
            raise ValueError(f"Conflicting manifest participants for {stimulus}/pred_{index:03d}.")
        mapping[stimulus][index] = participant
    return mapping


def history(raw, valid, frame, length, movie_w, movie_h, stim_w, stim_h, device):
    """Original valid-event history: right aligned and zero padded."""
    out = np.zeros((length, 2), dtype=np.float32); indices = valid[valid < frame]
    if len(indices):
        take = min(length, len(indices)); points = raw[indices[-take:]].astype(np.float32, copy=True)
        points[:, 0] = (points[:, 0] * stim_w / max(movie_w, 1) - stim_w / 2) / max(stim_w / 2, 1e-6)
        points[:, 1] = (points[:, 1] * stim_h / max(movie_h, 1) - stim_h / 2) / max(stim_h / 2, 1e-6)
        out[-take:] = points
    return torch.from_numpy(out.T).unsqueeze(0).to(device)


def sample(model, scheduler, conditioning, hist, pred_len, bases, cfg_scale, eta, device):
    """DIEM adapter for the shared standalone-video diffusion sampler."""
    return sample_window(
        model=model,
        scheduler=scheduler,
        conditioning=conditioning,
        history=hist,
        pred_len=pred_len,
        num_samples=len(bases),
        cfg_scale=cfg_scale,
        eta=eta,
        bases=bases,
        device=device,
    )


def write_xy(folder, name, xy, start):
    folder.mkdir(parents=True, exist_ok=True)
    rows = [{"step": int(i), "x": float(x), "y": float(y)} for i, (x, y) in enumerate(np.asarray(xy), start=start)]
    with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "x", "y"]); w.writeheader(); w.writerows(rows)
    (folder / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def to_movie(points, movie_w, movie_h, stim_w, stim_h):
    points = unnormalize_points(points, stim_w, stim_h)
    points[:, 0] *= movie_w / stim_w
    points[:, 1] *= movie_h / stim_h
    points[:, 0] = np.clip(points[:, 0], 0.0, max(0.0, float(movie_w - 1)))
    points[:, 1] = np.clip(points[:, 1], 0.0, max(0.0, float(movie_h - 1)))
    return points


def capture_rng_state() -> dict[str, object]:
    """Capture the state present immediately before a fresh batch is sampled."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    """Restore a state captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def main():
    a = args()
    if not a.seed_step: raise ValueError("--seed-step must be non-zero.")
    if a.seed_reset_every < 0:
        raise ValueError("--seed-reset-every must be zero or positive.")
    seed_everything(a.seed); cfg = merge_opts_to_config(load_yaml_config(a.config), a.opts)
    dcfg = cfg["dataset"]; history_len, pred_len = int(dcfg["history_len"]), int(dcfg["pred_len"])
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = instantiate_from_config(cfg["model"]).to(device); load_checkpoint(Path(a.checkpoint), model); model.eval()
    scheduler = instantiate_from_config(cfg["diffusion"]["eval_scheduler"]); scheduler.set_timesteps(int(cfg["diffusion"]["eval_scheduler"]["num_inference_steps"]))
    dataset = DIEMDataset(root=str(Path(a.dataset_root).resolve()), stim_to_sub_pth=None, stim_w=int(dcfg["stim_w"]), stim_h=int(dcfg["stim_h"]), history_len=history_len, seq_len=pred_len, frame_stride=int(dcfg["frame_stride"]), saliency_patch_dir_name=str(dcfg["saliency_patch_dir_name"]), saliency_patch_feature_dim=int(dcfg["saliency_patch_feature_dim"]), saliency_patch_token_count=int(dcfg["saliency_patch_token_count"]))
    selected = sorted(dataset.stim_video_paths)
    if a.split_json:
        with Path(a.split_json).open(encoding="utf-8") as f: selected = sorted(set(json.load(f)) & set(selected))
    manifest = load_manifest(a.participants_manifest) if a.participants_manifest else None
    if a.manifest_only:
        if manifest is None:
            raise ValueError("--manifest-only requires --participants-manifest.")
        selected = [stim for stim in selected if stim in manifest]
        if not selected:
            raise ValueError("No dataset stimuli are named in --participants-manifest.")
    chunk, limit = a.chunk_seconds * a.fps, a.max_video_seconds * a.fps
    root = Path(a.output_dir) / f"checkpoint_{Path(a.checkpoint).stem}_autoregressive_30s"

    fresh_batch_rng_state = capture_rng_state() if a.seed_reset_every else None
    for stim_idx, stim in enumerate(selected):
        if a.seed_reset_every and stim_idx and stim_idx % a.seed_reset_every == 0:
            restore_rng_state(fresh_batch_rng_state)
        batch_stim_idx = stim_idx % a.seed_reset_every if a.seed_reset_every else stim_idx
        tracks = tracks_for(dataset, stim)
        if not tracks: continue
        total = min(get_video_frame_count(dataset.stim_video_paths[stim]), limit)
        starts = [x for x in range(0, total - chunk + 1, chunk) if x >= history_len]
        if not starts: continue
        start0, end_last = starts[0], starts[-1] + chunk
        target = len(starts) * chunk + history_len  # Intentional original-generator horizon.
        subjects = list(tracks)
        if manifest is None:
            rng = np.random.RandomState(a.seed + a.seed_step * batch_stim_idx * 100003)
            chosen = rng.choice(subjects, size=a.num_predictions, replace=True).tolist()
        else:
            entries = manifest.get(stim, {})
            missing = [f"pred_{i:03d}" for i in range(a.num_predictions) if i not in entries]
            if missing:
                raise ValueError(f"Participant manifest is missing {stim}: {', '.join(missing)}")
            chosen = [entries[i] for i in range(a.num_predictions)]
            unavailable = sorted(set(chosen) - set(tracks))
            if unavailable:
                raise ValueError(f"Participant manifest names unavailable subjects for {stim}: {', '.join(unavailable)}")
        movie_w, movie_h = dataset.stim_video_sizes[stim]
        hist = torch.cat([history(*tracks[s], start0, history_len, movie_w, movie_h, dataset.stim_w, dataset.stim_h, device) for s in chosen])
        bases = [a.seed + a.seed_step * (batch_stim_idx * 100000 + i * 1000) for i in range(a.num_predictions)]
        generated, made = [], 0
        for rollout in range(math.ceil(target / pred_len)):
            start = min(max(start0 + rollout * pred_len, start0), max(start0, end_last - 1)); end = min(end_last, start + pred_len)
            if end <= start: end = start + 1
            indices = list(range(start, end, dataset.frame_stride)) or [start]
            cond = load_patch_sequence(dataset.saliency_patch_dir / stim, indices, dataset.saliency_patch_feature_dim, dataset.saliency_patch_token_count, prefer_plain_names=True)
            keep = min(pred_len, target - made); step = sample(model, scheduler, cond, hist, pred_len, [base + rollout * a.seed_step for base in bases], float(cfg["train"]["cfg_scale"]), float(cfg["diffusion"]["eval_scheduler"]["eta"]), device)[:, :keep]
            generated.append(step); made += keep
            history_step = clip_normalized_history_to_movie(step, movie_w, movie_h)
            hist = torch.cat([hist, history_step.permute(0, 2, 1).to(device)], dim=2)[:, :, -history_len:]
        predicted = torch.cat(generated, dim=1).numpy(); full = root / stim / "full_30s"
        for i, (subject, points) in enumerate(zip(chosen, predicted)):
            folder = full / f"pred_{i:03d}"; write_xy(folder, "scanpath", to_movie(points, movie_w, movie_h, dataset.stim_w, dataset.stim_h), -1)
            (folder / "participant.txt").write_text(f"{subject}\n", encoding="utf-8")
        metadata = {"mode":"autoregressive_30s", "stimulus":stim, "generation_subject":"manifest" if manifest else "random_per_prediction", "participants_manifest":str(a.participants_manifest) if manifest else None, "manifest_only":a.manifest_only, "num_available_participants":len(subjects), "num_chunks":len(starts), "chunk_seconds":a.chunk_seconds, "fps":a.fps, "chunk_frames":chunk, "max_video_seconds":a.max_video_seconds, "max_video_frames":limit, "processed_frames":len(starts)*chunk, "num_predictions":a.num_predictions, "history_len":history_len, "clip_history_to_frame":True, "chunk_start_frames":starts, "seed":a.seed, "seed_step":a.seed_step, "seed_reset_every":a.seed_reset_every or None, "batch_stim_idx":batch_stim_idx}
        (root / stim).mkdir(parents=True, exist_ok=True); (root / stim / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"saved {stim}: {a.num_predictions} scanpaths x {target} points")


if __name__ == "__main__": main()
