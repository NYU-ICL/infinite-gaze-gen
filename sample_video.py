import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from common import (
    get_video_frame_count,
    get_video_fps,
    get_video_size,
    instantiate_from_config,
    load_checkpoint,
    load_yaml_config,
    merge_opts_to_config,
    normalize_saliency_patch_tensor,
    seed_everything,
    unnormalize_points,
)


# The bundled inference configuration is paired with the checkpoint below.
# Keeping these defaults together prevents accidentally loading this model with
# a stale training configuration.
DEFAULT_CONFIG = "final_model_90_45/inference_config.yaml"
DEFAULT_CHECKPOINT = "final_model_90_45/checkpoint_70.pth"
DEFAULT_ARTIFACT_ROOT = Path(os.environ.get("INFINITE_GAZE_ARTIFACT_ROOT", "artifacts"))
DEFAULT_OUTPUT_DIR = str(DEFAULT_ARTIFACT_ROOT / "video_samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scanpath samples for a single video.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--conditioning-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overlay-trail", type=int, default=20)
    parser.add_argument("--overlay-radius", type=int, default=6)
    parser.add_argument("--overlay-thickness", type=int, default=-1)
    parser.add_argument("--skip-video-overlay", action="store_true")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def resolve_conditioning_dir(video_path: Path, configured_name: str, override: str | None) -> Path:
    if override is not None:
        return Path(override)
    candidates = [
        video_path.parent / configured_name / video_path.stem,
        video_path.parent.parent / configured_name / video_path.stem,
        video_path.parent / video_path.stem,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find conditioning for {video_path.name}. "
        f"Pass --conditioning-dir or place frame features under a `{configured_name}/{video_path.stem}` folder."
    )


def resolve_patch_file(base_dir: Path, frame_idx: int, prefer_plain_names: bool) -> Path:
    index_candidates = [frame_idx + 1, frame_idx, max(0, frame_idx - 1)]
    candidates = []
    for idx in dict.fromkeys(index_candidates):
        idx_str = f"{idx:06d}"
        if prefer_plain_names:
            candidates.extend([base_dir / f"{idx_str}.pt", base_dir / f"{idx_str}.npy", base_dir / f"frame_{idx_str}.pt", base_dir / f"frame_{idx_str}.npy"])
        else:
            candidates.extend([base_dir / f"frame_{idx_str}.pt", base_dir / f"frame_{idx_str}.npy", base_dir / f"{idx_str}.pt", base_dir / f"{idx_str}.npy"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing conditioning frame {frame_idx} in {base_dir}")


def load_patch_sequence(
    conditioning_dir: Path,
    frame_indices: list[int],
    feature_dim: int,
    token_count: int,
    prefer_plain_names: bool,
) -> torch.Tensor:
    patches = []
    for frame_idx in frame_indices:
        patch_path = resolve_patch_file(conditioning_dir, int(frame_idx), prefer_plain_names)
        if patch_path.suffix.lower() == ".npy":
            patch = torch.from_numpy(np.load(patch_path))
        else:
            patch = torch.load(patch_path, map_location="cpu", weights_only=False)
        patch = normalize_saliency_patch_tensor(
            patch,
            has_time_dim=False,
            target_feature_dim=feature_dim,
            target_num_patches=token_count,
        )
        patches.append(patch)
    return torch.stack(patches, dim=0)


def sample_window(
    model,
    scheduler,
    conditioning: torch.Tensor,
    history: torch.Tensor,
    pred_len: int,
    num_samples: int,
    cfg_scale: float,
    eta: float,
    seed: int,
    seed_step: int,
    device: torch.device,
) -> torch.Tensor:
    conditioning = conditioning.to(device)
    if conditioning.dim() == 3:
        conditioning = conditioning.unsqueeze(0)
    conditioning = conditioning.expand(num_samples, *conditioning.shape[1:]).contiguous()
    history = history.to(device)
    if history.shape[0] == 1:
        history = history.expand(num_samples, -1, -1).contiguous()

    noises = []
    for idx in range(num_samples):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + idx * int(seed_step))
        noises.append(torch.randn((1, 2, pred_len), generator=generator, device=device))
    generated = torch.cat(noises, dim=0)

    for timestep in scheduler.timesteps:
        model_input = torch.cat([history, generated], dim=2) if history.shape[-1] > 0 else generated
        t_tensor = torch.full((num_samples,), int(timestep), device=device, dtype=torch.long)
        noise_with_cond, _ = model(model_input, t_tensor, conditioning)
        noise_without_cond, _ = model(model_input, t_tensor, torch.zeros_like(conditioning))
        noise_pred = (1.0 - cfg_scale) * noise_without_cond + cfg_scale * noise_with_cond
        noise_pred = noise_pred[:, :, -pred_len:]
        generated = scheduler.step(noise_pred, timestep, generated, eta=float(eta)).prev_sample

    return generated.detach().cpu().permute(0, 2, 1).contiguous()


def save_sample(output_dir: Path, sample_idx: int, normalized_xy: np.ndarray, pixel_xy: np.ndarray) -> None:
    sample_dir = output_dir / f"sample_{sample_idx:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    with (sample_dir / "scanpath.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "x", "y", "x_normalized", "y_normalized"])
        writer.writeheader()
        for step, (pix, norm) in enumerate(zip(pixel_xy.tolist(), normalized_xy.tolist())):
            writer.writerow(
                {
                    "step": int(step),
                    "x": float(pix[0]),
                    "y": float(pix[1]),
                    "x_normalized": float(norm[0]),
                    "y_normalized": float(norm[1]),
                }
            )

    payload = [
        {
            "step": int(step),
            "x": float(pix[0]),
            "y": float(pix[1]),
            "x_normalized": float(norm[0]),
            "y_normalized": float(norm[1]),
        }
        for step, (pix, norm) in enumerate(zip(pixel_xy.tolist(), normalized_xy.tolist()))
    ]
    with (sample_dir / "scanpath.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def render_overlay_video(
    video_path: Path,
    output_path: Path,
    pixel_xy: np.ndarray,
    fps: float,
    trail: int,
    radius: int,
    thickness: int,
) -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Video overlay requires OpenCV (`cv2`) in the runtime used to execute sample_video.py."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for overlay: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video size for overlay: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create overlay video: {output_path}")

    colors = [
        (0, 255, 255),
        (0, 220, 0),
        (255, 180, 0),
        (0, 128, 255),
        (255, 0, 180),
    ]
    point_count = int(pixel_xy.shape[0])
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx >= point_count:
            writer.write(frame)
            frame_idx += 1
            continue

        start_idx = max(0, frame_idx - max(0, int(trail)) + 1)
        history_points = pixel_xy[start_idx : frame_idx + 1]
        for hist_idx, point in enumerate(history_points):
            alpha = float(hist_idx + 1) / float(len(history_points))
            color = colors[(frame_idx + hist_idx) % len(colors)]
            draw_color = tuple(int(alpha * c) for c in color)
            x = int(np.clip(round(float(point[0])), 0, width - 1))
            y = int(np.clip(round(float(point[1])), 0, height - 1))
            draw_radius = max(2, int(round(radius * (0.45 + 0.55 * alpha))))
            cv2.circle(frame, (x, y), draw_radius, draw_color, thickness)

        if len(history_points) >= 2:
            poly = np.asarray(history_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [poly], False, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        curr_x = int(np.clip(round(float(pixel_xy[frame_idx, 0])), 0, width - 1))
        curr_y = int(np.clip(round(float(pixel_xy[frame_idx, 1])), 0, height - 1))
        cv2.circle(frame, (curr_x, curr_y), max(radius + 2, 4), (255, 255, 255), 2)
        cv2.circle(frame, (curr_x, curr_y), max(radius, 2), (0, 0, 255), thickness)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(f"Inference configuration not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    cfg = merge_opts_to_config(load_yaml_config(config_path), args.opts)
    seed_everything(int(args.seed))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = instantiate_from_config(cfg["model"]).to(device)
    scheduler = instantiate_from_config(cfg["diffusion"]["eval_scheduler"])
    scheduler.set_timesteps(int(cfg["diffusion"]["eval_scheduler"]["num_inference_steps"]))
    load_checkpoint(checkpoint_path, model)
    model.eval()

    video_path = Path(args.video_path).resolve()
    conditioning_name = str(cfg["dataset"]["saliency_patch_dir_name"])
    conditioning_dir = resolve_conditioning_dir(video_path, conditioning_name, args.conditioning_dir).resolve()
    prefer_plain_names = "unisal" in conditioning_name.lower()

    history_len = int(cfg["dataset"]["history_len"])
    pred_len = int(cfg["dataset"]["pred_len"])
    frame_stride = int(cfg["dataset"]["frame_stride"])
    feature_dim = int(cfg["dataset"]["saliency_patch_feature_dim"])
    token_count = int(cfg["dataset"]["saliency_patch_token_count"])
    cfg_scale = float(cfg["train"]["cfg_scale"])
    eta = float(cfg["diffusion"]["eval_scheduler"]["eta"])
    stim_w = int(cfg["dataset"]["stim_w"])
    stim_h = int(cfg["dataset"]["stim_h"])

    total_frames = get_video_frame_count(video_path)
    if args.max_frames is not None:
        total_frames = min(total_frames, int(args.max_frames))
    movie_w, movie_h = get_video_size(video_path)
    movie_fps = get_video_fps(video_path)

    output_dir = Path(args.output_dir).resolve() / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    current_history = torch.zeros((int(args.num_samples), 2, history_len), dtype=torch.float32)
    samples_per_window: list[torch.Tensor] = []

    for window_start in range(0, total_frames, pred_len):
        remaining = total_frames - window_start
        take = min(pred_len, remaining)
        # Each conditioning feature must represent the frame being predicted.
        # `current_history` contains preceding gaze coordinates only; it must
        # not shift the video/saliency timeline into a future window.
        patch_frame_indices = [
            min(total_frames - 1, window_start + offset)
            for offset in range(0, pred_len, frame_stride)
        ]
        conditioning = load_patch_sequence(
            conditioning_dir=conditioning_dir,
            frame_indices=patch_frame_indices,
            feature_dim=feature_dim,
            token_count=token_count,
            prefer_plain_names=prefer_plain_names,
        )
        window_pred = sample_window(
            model=model,
            scheduler=scheduler,
            conditioning=conditioning,
            history=current_history,
            pred_len=pred_len,
            num_samples=int(args.num_samples),
            cfg_scale=cfg_scale,
            eta=eta,
            seed=int(args.seed) + window_start,
            seed_step=int(args.seed_step),
            device=device,
        )
        window_keep = window_pred[:, :take, :]
        samples_per_window.append(window_keep)
        history_update = window_keep.permute(0, 2, 1)
        current_history = torch.cat([current_history, history_update], dim=2)[:, :, -history_len:]

    all_samples = torch.cat(samples_per_window, dim=1).numpy()
    metadata = {
        "video_path": str(video_path),
        "conditioning_dir": str(conditioning_dir),
        "num_samples": int(args.num_samples),
        "num_points": int(all_samples.shape[1]),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    for sample_idx in range(all_samples.shape[0]):
        normalized_xy = all_samples[sample_idx]
        resized_xy = unnormalize_points(normalized_xy, stim_w, stim_h)
        pixel_xy = resized_xy.copy()
        pixel_xy[:, 0] *= movie_w / stim_w
        pixel_xy[:, 1] *= movie_h / stim_h
        save_sample(output_dir, sample_idx, normalized_xy, pixel_xy)
        if not args.skip_video_overlay:
            render_overlay_video(
                video_path=video_path,
                output_path=output_dir / f"sample_{sample_idx:03d}" / "overlay.mp4",
                pixel_xy=pixel_xy,
                fps=movie_fps,
                trail=int(args.overlay_trail),
                radius=int(args.overlay_radius),
                thickness=int(args.overlay_thickness),
            )

    print(f"saved samples to {output_dir}")


if __name__ == "__main__":
    main()
