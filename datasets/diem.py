import json
import re
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from common import compute_offset, get_video_size, normalize_saliency_patch_tensor


@dataclass
class StimSub:
    stim: str
    sub: str


@dataclass
class EyeData:
    prefix: str
    m_eye_x_movie: list[float] = field(default_factory=list)
    m_eye_y_movie: list[float] = field(default_factory=list)
    m_dilation: list[float] = field(default_factory=list)


class DIEMDataset(Dataset):
    def __init__(
        self,
        root: str,
        stim_to_sub_pth: str | None,
        stim_w: int = 224,
        stim_h: int = 224,
        normalize: bool = True,
        should_preload: bool = False,
        history_len: int = 90,
        seq_len: int = 45,
        data_sample_stride: int = 135,
        frame_stride: int = 5,
        saliency_patch_dir_name: str = "saliency_unisal_latents_small",
        saliency_patch_feature_dim: int = 64,
        saliency_patch_token_count: int = 24,
        enable_saliency_patches_cache: bool = False,
        saliency_patches_cache_size: int = 128,
        patch_frames_aligned_to_prediction_window: bool = True,
        only_frame_stride_patches: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.data_root = self.root / "data"
        if not self.data_root.exists():
            self.data_root = self.root

        self.stim_w = int(stim_w)
        self.stim_h = int(stim_h)
        self.normalize = bool(normalize)
        self.should_preload = bool(should_preload)
        self.history_len = int(history_len)
        self.pred_len = int(seq_len)
        self.traj_len = self.history_len + self.pred_len
        self.data_sample_stride = int(data_sample_stride)
        self.frame_stride = int(frame_stride)
        self.saliency_patch_feature_dim = int(saliency_patch_feature_dim)
        self.saliency_patch_token_count = int(saliency_patch_token_count)
        self.patch_frames_aligned_to_prediction_window = bool(patch_frames_aligned_to_prediction_window)
        self.only_frame_stride_patches = bool(only_frame_stride_patches)
        self.saliency_patch_dir = self.root / str(saliency_patch_dir_name)
        self._prefer_plain_frame_names = "unisal" in str(saliency_patch_dir_name).lower()
        self._frame_file_index_offset = 1 if self._prefer_plain_frame_names else 0
        self._stride_patch_index_cache: dict[str, list[int]] = {}
        self._patch_cache = {} if enable_saliency_patches_cache else None
        self._patch_cache_size = int(saliency_patches_cache_size)

        self._load_all_subjects = stim_to_sub_pth is None
        if self._load_all_subjects:
            self.stim_to_sub_json: dict[str, list[str]] = {}
        else:
            with open(stim_to_sub_pth, "r", encoding="utf-8") as handle:
                stim_to_sub = json.load(handle)
            if not isinstance(stim_to_sub, dict):
                raise ValueError("stim_to_sub_pth must contain a JSON object of stimulus -> subjects.")
            self.stim_to_sub_json = {str(k): [str(x) for x in v] for k, v in stim_to_sub.items()}

        self.stim_video_paths: dict[str, Path] = {}
        self.stim_video_sizes: dict[str, tuple[int, int]] = {}
        self.eye_data_list: list[EyeData] = []
        self.stim_sub_list: list[StimSub] = []
        self._discover_and_load()
        self._valid_gaze_indices = [
            self._compute_valid_indices(eye, self.stim_sub_list[idx].stim) for idx, eye in enumerate(self.eye_data_list)
        ]
        self.sample_index = self._build_sample_index()
        if not self.sample_index:
            raise RuntimeError("No valid DIEM windows found.")

        if self.should_preload:
            self._preloaded = [self._get_item(i) for i in range(len(self.sample_index))]
        else:
            self._preloaded = None

    def _discover_and_load(self) -> None:
        stim_roots: dict[str, Path] = {}
        root_video_dir = self.data_root / "video"
        root_event_dir = self.data_root / "event_data"

        if root_video_dir.exists() and root_event_dir.exists():
            videos = sorted(root_video_dir.glob("*.mp4"))
            if videos:
                stim_name = videos[0].stem
                self.stim_video_paths[stim_name] = videos[0]
                stim_roots[stim_name] = self.data_root
        else:
            for stim_dir in sorted(self.data_root.iterdir()):
                if not stim_dir.is_dir():
                    continue
                video_dir = stim_dir / "video"
                event_dir = stim_dir / "event_data"
                videos = sorted(video_dir.glob("*.mp4"))
                if videos and event_dir.exists():
                    self.stim_video_paths[stim_dir.name] = videos[0]
                    stim_roots[stim_dir.name] = stim_dir

        if not stim_roots:
            raise FileNotFoundError(f"No DIEM stimuli found under {self.data_root}")

        for stim_name, stim_root in stim_roots.items():
            if not self._load_all_subjects and stim_name not in self.stim_to_sub_json:
                continue
            eye_tracks = self._load_root_dir(stim_root, stim_name)
            allowed = None if self._load_all_subjects else set(self.stim_to_sub_json[stim_name])
            if self._load_all_subjects:
                self.stim_to_sub_json[stim_name] = [eye.prefix for eye in eye_tracks]
            for eye in eye_tracks:
                if allowed is not None and eye.prefix not in allowed:
                    continue
                self.eye_data_list.append(eye)
                self.stim_sub_list.append(StimSub(stim=stim_name, sub=eye.prefix))

        if not self.stim_sub_list:
            raise RuntimeError("No DIEM subject data matched the requested split.")

    def _load_root_dir(self, root_dir: Path, stim_name: str) -> list[EyeData]:
        video_path = sorted((root_dir / "video").glob("*.mp4"))[0]
        movie_w, movie_h = get_video_size(video_path)
        self.stim_video_sizes[stim_name] = (movie_w, movie_h)
        offset_x, offset_y = compute_offset(movie_w, movie_h, 1280, 960)
        event_paths = sorted((root_dir / "event_data").glob("*.txt"))
        return [self._load_eye_file(path, offset_x, offset_y) for path in event_paths]

    def _detect_mode(self, tokens: list[str]) -> str:
        if len(tokens) >= 9:
            return "binocular"
        if len(tokens) >= 4:
            return "mono"
        raise ValueError("Could not detect eye-tracker file format.")

    def _load_eye_file(self, path: Path, offset_x: float, offset_y: float) -> EyeData:
        eye = EyeData(prefix=path.stem)
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        if not lines:
            return eye

        mode = self._detect_mode(lines[0].split())
        for line in lines:
            tokens = line.split()
            if mode == "binocular" and len(tokens) >= 9:
                _, lx, ly, ld, _le, rx, ry, rd, _re = tokens[:9]
                lx = float(lx)
                ly = float(ly)
                rx = float(rx)
                ry = float(ry)
                ld = float(ld)
                rd = float(rd)
                mx = (lx + rx) / 2.0
                my = (ly + ry) / 2.0
                invalid = lx == 0.0 and ly == 0.0 and rx == 0.0 and ry == 0.0
                eye.m_eye_x_movie.append(0.0 if invalid else mx - offset_x)
                eye.m_eye_y_movie.append(0.0 if invalid else my - offset_y)
                eye.m_dilation.append((ld + rd) / 2.0)
            elif mode == "mono" and len(tokens) >= 4:
                _, x, y, dilation = tokens[:4]
                x = float(x)
                y = float(y)
                dilation = float(dilation)
                invalid = x == 0.0 and y == 0.0
                eye.m_eye_x_movie.append(0.0 if invalid else x - offset_x)
                eye.m_eye_y_movie.append(0.0 if invalid else y - offset_y)
                eye.m_dilation.append(dilation)
        return eye

    def _compute_valid_indices(self, eye: EyeData, stim_name: str) -> np.ndarray:
        traj = np.stack([eye.m_eye_x_movie, eye.m_eye_y_movie], axis=1)
        finite_mask = ~np.isnan(traj).any(axis=1)
        dilation = np.asarray(eye.m_dilation, dtype=np.float32)
        zero_mask = (traj[:, 0] == 0.0) & (traj[:, 1] == 0.0) & (dilation == 0.0)
        movie_w, movie_h = self.stim_video_sizes[stim_name]
        in_bounds = (
            (traj[:, 0] >= 0.0)
            & (traj[:, 0] < float(movie_w))
            & (traj[:, 1] >= 0.0)
            & (traj[:, 1] < float(movie_h))
        )
        valid = finite_mask & in_bounds & (~zero_mask)
        return np.nonzero(valid)[0]

    def _build_sample_index(self) -> list[tuple[int, int]]:
        sample_index: list[tuple[int, int]] = []
        step = max(1, self.data_sample_stride + 1)
        for eye_idx, valid_indices in enumerate(self._valid_gaze_indices):
            max_start = len(valid_indices) - self.traj_len
            if max_start < 0:
                continue
            for start_idx in range(0, max_start + 1, step):
                sample_index.append((eye_idx, start_idx))
        return sample_index

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        if self._preloaded is not None:
            return self._preloaded[idx]
        return self._get_item(idx)

    def _get_item(self, idx: int):
        eye_idx, start_idx = self.sample_index[idx]
        stim_sub = self.stim_sub_list[eye_idx]
        traj = self.get_traj(eye_idx, start_idx)
        frame_indices = self.get_frame_indices(eye_idx, start_idx, self.traj_len)
        patch_frame_indices = self.get_patch_frame_indices(frame_indices)
        patches = self.get_saliency_patches(stim_sub.stim, patch_frame_indices)
        return traj, patches, stim_sub.stim, start_idx

    def get_traj(self, eye_idx: int, start_idx: int) -> torch.Tensor:
        eye = self.eye_data_list[eye_idx]
        stim_name = self.stim_sub_list[eye_idx].stim
        movie_w, movie_h = self.stim_video_sizes[stim_name]
        traj = np.stack([eye.m_eye_x_movie, eye.m_eye_y_movie], axis=1)[self._valid_gaze_indices[eye_idx]]
        traj = traj[start_idx : start_idx + self.traj_len].copy()
        traj[:, 0] *= self.stim_w / movie_w
        traj[:, 1] *= self.stim_h / movie_h
        if self.normalize:
            cx = self.stim_w / 2.0
            cy = self.stim_h / 2.0
            traj[:, 0] = (traj[:, 0] - cx) / cx
            traj[:, 1] = (traj[:, 1] - cy) / cy
        return torch.tensor(traj, dtype=torch.float32).permute(1, 0)

    def get_frame_indices(self, eye_idx: int, start_idx: int, length: int) -> list[int]:
        valid_idx = self._valid_gaze_indices[eye_idx]
        take_idx = start_idx + np.arange(0, length, self.frame_stride)
        if take_idx[-1] >= len(valid_idx):
            raise IndexError("Requested frame indices exceed available valid gaze points.")
        return valid_idx[take_idx].tolist()

    def get_patch_frame_indices(self, frame_indices: list[int]) -> list[int]:
        if not self.patch_frames_aligned_to_prediction_window:
            return frame_indices
        patch_start = (self.history_len + self.frame_stride - 1) // self.frame_stride
        return frame_indices[patch_start:]

    def get_saliency_patches(self, stim_name: str, frame_indices: list[int]) -> torch.Tensor:
        patches = [self._load_patch(stim_name, frame_idx) for frame_idx in frame_indices]
        return torch.stack(patches, dim=0)

    def _load_patch(self, stim_name: str, frame_idx: int) -> torch.Tensor:
        load_idx = self._resolve_patch_frame_idx(stim_name, int(frame_idx))
        cache_key = f"{stim_name}:{load_idx}"
        if self._patch_cache is not None and cache_key in self._patch_cache:
            return self._patch_cache[cache_key]

        base_dir = self.saliency_patch_dir / stim_name
        patch_path = self._resolve_frame_file(base_dir, load_idx)
        if patch_path.suffix.lower() == ".npy":
            patch = torch.from_numpy(np.load(patch_path))
        else:
            patch = torch.load(patch_path, map_location="cpu", weights_only=False)

        patch = normalize_saliency_patch_tensor(
            patch,
            has_time_dim=False,
            target_feature_dim=self.saliency_patch_feature_dim,
            target_num_patches=self.saliency_patch_token_count,
        )
        if self._patch_cache is not None:
            if len(self._patch_cache) >= self._patch_cache_size:
                first_key = next(iter(self._patch_cache))
                self._patch_cache.pop(first_key)
            self._patch_cache[cache_key] = patch
        return patch

    def _resolve_patch_frame_idx(self, stim_name: str, frame_idx: int) -> int:
        frame_idx = int(frame_idx) + self._frame_file_index_offset
        if not self.only_frame_stride_patches:
            return frame_idx
        available = self._get_available_patch_frames(stim_name)
        pos = bisect_left(available, frame_idx)
        if pos <= 0:
            return available[0]
        if pos >= len(available):
            return available[-1]
        left = available[pos - 1]
        right = available[pos]
        return left if abs(frame_idx - left) <= abs(right - frame_idx) else right

    def _get_available_patch_frames(self, stim_name: str) -> list[int]:
        cached = self._stride_patch_index_cache.get(stim_name)
        if cached is not None:
            return cached
        base_dir = self.saliency_patch_dir / stim_name
        pattern = re.compile(r"^(?:frame_)?(\d{6})\.(?:pt|npy)$")
        frames = []
        for path in base_dir.iterdir():
            if not path.is_file():
                continue
            match = pattern.match(path.name)
            if match:
                frames.append(int(match.group(1)))
        frames = sorted(set(frames))
        if not frames:
            raise FileNotFoundError(f"No conditioning frames found in {base_dir}")
        self._stride_patch_index_cache[stim_name] = frames
        return frames

    def _resolve_frame_file(self, base_dir: Path, frame_idx: int) -> Path:
        candidates = []
        for idx in dict.fromkeys([frame_idx, frame_idx + self._frame_file_index_offset, frame_idx - self._frame_file_index_offset]):
            if idx < 0:
                continue
            idx_str = f"{idx:06d}"
            if self._prefer_plain_frame_names:
                candidates.extend([base_dir / f"{idx_str}.pt", base_dir / f"{idx_str}.npy", base_dir / f"frame_{idx_str}.pt", base_dir / f"frame_{idx_str}.npy"])
            else:
                candidates.extend([base_dir / f"frame_{idx_str}.pt", base_dir / f"frame_{idx_str}.npy", base_dir / f"{idx_str}.pt", base_dir / f"{idx_str}.npy"])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not resolve conditioning frame {frame_idx} in {base_dir}")
