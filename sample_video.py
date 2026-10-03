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


def clip_normalized_history_to_movie(
    points: torch.Tensor, movie_w: int, movie_h: int
) -> torch.Tensor:
    """Clip [B, T, 2] normalized coordinates to valid movie-pixel locations.

    The autoregressive DIEM generator feeds clipped predictions back as history.
    Keeping that rule here prevents an off-frame prediction from becoming an
    unbounded conditioning history on a later window.
    """
    if points.shape[-1] != 2:
        raise ValueError(f"Expected points shaped [B,T,2], got {tuple(points.shape)}")
    clipped = points.clone()
    clipped[..., 0] = clipped[..., 0].clamp(-1.0, 1.0 - 2.0 / max(float(movie_w), 1.0))
    clipped[..., 1] = clipped[..., 1].clamp(-1.0, 1.0 - 2.0 / max(float(movie_h), 1.0))
    return clipped


def sample_window(
    model,
    scheduler,
    conditioning: torch.Tensor,
    history: torch.Tensor,
    pred_len: int,
    num_samples: int,
    cfg_scale: float,
    eta: float,
    bases: list[int],
    device: torch.device,
) -> torch.Tensor:
    conditioning = conditioning.to(device)
    if conditioning.dim() == 3:
        conditioning = conditioning.unsqueeze(0)
    conditioning = conditioning.expand(num_samples, *conditioning.shape[1:]).contiguous()
    history = history.to(device)
    if history.shape[0] == 1:
        history = history.expand(num_samples, -1, -1).contiguous()

    if len(bases) != num_samples:
        raise ValueError(f"Expected {num_samples} noise seeds, got {len(bases)}")
    noises = []
    for base in bases:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(base))
        noises.append(torch.randn((1, 2, pred_len), generator=generator, device=device))
    generated = torch.cat(noises, dim=0)

    with torch.inference_mode():
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


@torch.no_grad()
def sample_video(
    video_path: str | Path,
    *,
    conditioning_dir: str | Path | None = None,
    config: str | Path = DEFAULT_CONFIG,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    num_samples: int = 10,
    max_frames: int | None = None,
    seed: int = 12,
    seed_step: int = 1,
    device: str | torch.device | None = None,
    opts: list[str] | None = None,
    render_overlays: bool = True,
    overlay_trail: int = 20,
    overlay_radius: int = 6,
    overlay_thickness: int = -1,
) -> dict[str, object]:
    """Generate autoregressive scanpaths for an arbitrary video.

    This is the notebook-friendly counterpart of the CLI.  It uses the same
    rollout rule as :mod:`sample_diem_val`: every predicted window is clipped
    to the movie bounds and appended to the history used by the next window.
    General videos have no observed gaze history, so the first window starts
    with the model's zero-padded history.

    Returns paths and generated coordinates so notebooks do not have to parse
    the CSV files or emulate command-line arguments.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if seed_step == 0:
        raise ValueError("seed_step must be non-zero.")

    config_path = Path(config)
    checkpoint_path = Path(checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(f"Inference configuration not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    cfg = merge_opts_to_config(load_yaml_config(config_path), opts)
    seed_everything(int(seed))

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = instantiate_from_config(cfg["model"]).to(target_device)
    scheduler = instantiate_from_config(cfg["diffusion"]["eval_scheduler"])
    scheduler.set_timesteps(int(cfg["diffusion"]["eval_scheduler"]["num_inference_steps"]))
    load_checkpoint(checkpoint_path, model)
    model.eval()

    video_path = Path(video_path).resolve()
    conditioning_name = str(cfg["dataset"]["saliency_patch_dir_name"])
    conditioning_dir = resolve_conditioning_dir(video_path, conditioning_name, str(conditioning_dir) if conditioning_dir is not None else None).resolve()
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
    if max_frames is not None:
        total_frames = min(total_frames, int(max_frames))
    if total_frames <= 0:
        raise ValueError("The requested frame limit leaves no frames to sample.")
    movie_w, movie_h = get_video_size(video_path)
    movie_fps = get_video_fps(video_path)

    output_dir = Path(output_dir).resolve() / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    current_history = torch.zeros((int(num_samples), 2, history_len), dtype=torch.float32)
    samples_per_window: list[torch.Tensor] = []
    bases = [
        int(seed) + sample_idx * 1000 * int(seed_step)
        for sample_idx in range(int(num_samples))
    ]

    for rollout, window_start in enumerate(range(0, total_frames, pred_len)):
        window_end = min(total_frames, window_start + pred_len)
        take = window_end - window_start
        # Each conditioning feature must represent the frame being predicted.
        # `current_history` contains preceding gaze coordinates only; it must
        # not shift the video/saliency timeline into a future window.
        patch_frame_indices = list(range(window_start, window_end, frame_stride)) or [window_start]
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
            num_samples=int(num_samples),
            cfg_scale=cfg_scale,
            eta=eta,
            bases=[base + rollout * int(seed_step) for base in bases],
            device=target_device,
        )
        window_keep = window_pred[:, :take, :]
        samples_per_window.append(window_keep)
        history_update = clip_normalized_history_to_movie(window_keep, movie_w, movie_h).permute(0, 2, 1)
        current_history = torch.cat([current_history, history_update], dim=2)[:, :, -history_len:]

    all_samples = torch.cat(samples_per_window, dim=1).numpy()
    metadata = {
        "video_path": str(video_path),
        "conditioning_dir": str(conditioning_dir),
        "num_samples": int(num_samples),
        "num_points": int(all_samples.shape[1]),
        "history_len": history_len,
        "pred_len": pred_len,
        "frame_stride": frame_stride,
        "seed": int(seed),
        "seed_step": int(seed_step),
        "history_mode": "zero_padded_then_autoregressive_clipped_predictions",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    all_pixel_samples: list[np.ndarray] = []
    for sample_idx in range(all_samples.shape[0]):
        normalized_xy = all_samples[sample_idx]
        resized_xy = unnormalize_points(normalized_xy, stim_w, stim_h)
        pixel_xy = resized_xy.copy()
        pixel_xy[:, 0] *= movie_w / stim_w
        pixel_xy[:, 1] *= movie_h / stim_h
        pixel_xy[:, 0] = np.clip(pixel_xy[:, 0], 0.0, max(0.0, float(movie_w - 1)))
        pixel_xy[:, 1] = np.clip(pixel_xy[:, 1], 0.0, max(0.0, float(movie_h - 1)))
        all_pixel_samples.append(pixel_xy)
        save_sample(output_dir, sample_idx, normalized_xy, pixel_xy)
        if render_overlays:
            render_overlay_video(
                video_path=video_path,
                output_path=output_dir / f"sample_{sample_idx:03d}" / "overlay.mp4",
                pixel_xy=pixel_xy,
                fps=movie_fps,
                trail=int(overlay_trail),
                radius=int(overlay_radius),
                thickness=int(overlay_thickness),
            )

    return {
        "output_dir": output_dir,
        "normalized_xy": all_samples,
        "video_xy": np.stack(all_pixel_samples),
        "metadata": metadata,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    result = sample_video(
        args.video_path,
        conditioning_dir=args.conditioning_dir,
        config=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_frames=args.max_frames,
        seed=args.seed,
        seed_step=args.seed_step,
        device=args.device,
        opts=args.opts,
        render_overlays=not args.skip_video_overlay,
        overlay_trail=args.overlay_trail,
        overlay_radius=args.overlay_radius,
        overlay_thickness=args.overlay_thickness,
    )
    print(f"saved samples to {result['output_dir']}")


if __name__ == "__main__":
    main()
