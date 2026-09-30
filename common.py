import importlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def merge_opts_to_config(config: dict, opts: list[str] | None) -> dict:
    if not opts:
        return config
    if len(opts) % 2 != 0:
        raise ValueError("Override options must be passed as KEY VALUE pairs.")
    merged = json.loads(json.dumps(config))
    for key, raw_value in zip(opts[::2], opts[1::2]):
        target = merged
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = yaml.safe_load(raw_value)
    return merged


def instantiate_from_config(config: dict, **kwargs):
    if "target" not in config:
        raise KeyError("Config entry must include a `target` field.")
    module_name, cls_name = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    params = dict(config.get("params", {}))
    params.update(kwargs)
    return cls(**params)


def _ffprobe_video_stream(video_path: str | Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found: {video_path}")
    return streams[0]


def get_video_size(video_path: str | Path) -> tuple[int, int]:
    width = 0
    height = 0
    try:
        stream = _ffprobe_video_stream(video_path)
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
    except Exception:
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        except Exception:
            width = 0
            height = 0
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions: {video_path}")
    return width, height


def get_video_frame_count(video_path: str | Path) -> int:
    count = 0
    try:
        stream = _ffprobe_video_stream(video_path)
        count = int(stream.get("nb_frames") or 0)
    except Exception:
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(video_path))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception:
            count = 0
    if count <= 0:
        raise RuntimeError(f"Invalid frame count: {video_path}")
    return count


def get_video_fps(video_path: str | Path) -> float:
    fps = 0.0
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate",
            "-of",
            "json",
            str(video_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout)
        stream = payload.get("streams", [{}])[0]
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
        num_str, den_str = str(rate).split("/", 1)
        den = float(den_str)
        if den != 0.0:
            fps = float(num_str) / den
    except Exception:
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(video_path))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
        except Exception:
            fps = 0.0
    if fps <= 0.0:
        raise RuntimeError(f"Invalid FPS for video: {video_path}")
    return fps


def compute_offset(movie_w: int, movie_h: int, screen_w: int, screen_h: int) -> tuple[float, float]:
    return (screen_w - movie_w) / 2.0, (screen_h - movie_h) / 2.0


def normalize_saliency_patch_tensor(
    patches: torch.Tensor,
    has_time_dim: bool | None = None,
    target_feature_dim: int | None = None,
    target_num_patches: int | None = None,
) -> torch.Tensor:
    if not isinstance(patches, torch.Tensor):
        patches = torch.as_tensor(patches)
    patches = patches.to(torch.float32)
    if patches.ndim == 1:
        patches = patches.unsqueeze(-1)
    if patches.ndim < 2:
        raise ValueError(f"Unexpected conditioning tensor shape: {tuple(patches.shape)}")

    if has_time_dim is False:
        patches = _match_patch_token_count(patches, target_num_patches, has_time_dim=False)
        patches = patches if patches.ndim == 2 else patches.reshape(patches.shape[0], -1)
        return _match_patch_feature_dim(patches, target_feature_dim)

    if has_time_dim is True:
        patches = _match_patch_token_count(patches, target_num_patches, has_time_dim=True)
        if patches.ndim == 2:
            patches = patches.unsqueeze(0)
        elif patches.ndim > 3:
            patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        return _match_patch_feature_dim(patches, target_feature_dim)

    if patches.ndim >= 4:
        patches = _match_patch_token_count(patches, target_num_patches, has_time_dim=True)
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        return _match_patch_feature_dim(patches, target_feature_dim)

    if patches.ndim == 3:
        if patches.shape[0] <= 64 and patches.shape[1] >= 8:
            patches = _match_patch_token_count(patches, target_num_patches, has_time_dim=True)
            return _match_patch_feature_dim(patches, target_feature_dim)
        patches = _match_patch_token_count(patches, target_num_patches, has_time_dim=False)
        patches = patches.reshape(patches.shape[0], -1)
        return _match_patch_feature_dim(patches, target_feature_dim)

    return _match_patch_feature_dim(patches, target_feature_dim)


def _match_patch_token_count(patches: torch.Tensor, target_num_patches: int | None, has_time_dim: bool) -> torch.Tensor:
    if target_num_patches is None:
        return patches
    target_num_patches = int(target_num_patches)
    token_dim = 1 if has_time_dim else 0
    if patches.shape[token_dim] == target_num_patches:
        return patches
    if has_time_dim:
        trailing_shape = patches.shape[2:]
        flat = patches.reshape(patches.shape[0], patches.shape[1], -1).permute(0, 2, 1)
        flat = F.interpolate(flat, size=target_num_patches, mode="linear", align_corners=False)
        return flat.permute(0, 2, 1).reshape(patches.shape[0], target_num_patches, *trailing_shape)
    trailing_shape = patches.shape[1:]
    flat = patches.reshape(patches.shape[0], -1).T.unsqueeze(0)
    flat = F.interpolate(flat, size=target_num_patches, mode="linear", align_corners=False)
    return flat.squeeze(0).T.reshape(target_num_patches, *trailing_shape)


def _match_patch_feature_dim(patches: torch.Tensor, target_feature_dim: int | None) -> torch.Tensor:
    if target_feature_dim is None or patches.shape[-1] == int(target_feature_dim):
        return patches
    target_feature_dim = int(target_feature_dim)
    leading_shape = patches.shape[:-1]
    flat = patches.reshape(-1, 1, patches.shape[-1])
    flat = F.interpolate(flat, size=target_feature_dim, mode="linear", align_corners=False)
    return flat.reshape(*leading_shape, target_feature_dim)


def create_experiment_dir(output_root: str | Path, exp_name: str) -> Path:
    timestamp = time.strftime("%m%d%Y_%H%M%S")
    path = Path(output_root) / exp_name / timestamp
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    return path


def save_yaml_config(config: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload in {path}: expected a dictionary.")
    # Accept both the training checkpoint format used by this project and the
    # common bare/state_dict export formats used for inference-only bundles.
    state_dict = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported model state dictionary in {path}.")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # The 90/45 checkpoint was trained before the conditioning module was
        # renamed from dino_mlp to saliency_mlp.  Translate that known rename
        # explicitly, rather than suffix matching (which is ambiguous for
        # repeated Sequential layer names such as "0.weight").
        target_state = model.state_dict()
        migrated_state = {}
        for key, value in state_dict.items():
            migrated_key = key.removeprefix("module.")
            if migrated_key.startswith("dino_mlp."):
                migrated_key = "saliency_mlp." + migrated_key.removeprefix("dino_mlp.")
            migrated_state[migrated_key] = value
        model.load_state_dict(migrated_state)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return int(payload.get("epoch", -1)) + 1, int(payload.get("global_step", 0))


def unnormalize_points(points: np.ndarray | torch.Tensor, width: int, height: int) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    arr = np.asarray(points, dtype=np.float32).copy()
    arr[:, 0] = (arr[:, 0] + 1.0) * (width / 2.0)
    arr[:, 1] = (arr[:, 1] + 1.0) * (height / 2.0)
    return arr
