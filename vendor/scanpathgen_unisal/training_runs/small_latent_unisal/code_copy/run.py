from pathlib import Path
import os

import cv2
import torch
import unisal
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Global list to store captured pre-upsampling latent features for one video
pre_up_latents = []
latent_pbar = None
LOG_LATENT_SHAPES = True

# Directory to save pre-upsampling latent vectors
LATENT_SAVE_DIR = Path("pre_upsampling_latents")
LATENT_SAVE_DIR.mkdir(exist_ok=True, parents=True)

# Hook function to capture the latent right before first upsampling.
# This hook is attached to `model.upsampling_1` and reads its input tensor.
def hook_fn(module, input):
    # input is a tuple with one tensor: [batch, channels, h, w]
    latent = input[0]
    pre_up_latents.append(latent.detach().cpu().unsqueeze(1))
    if latent_pbar is not None:
        latent_pbar.update(1)


def get_processing_frame_order(n_frames, frame_modulo):
    """Return frame indices (1-based) in the same order run_inference processes."""
    order = []
    for offset in range(min(frame_modulo, n_frames)):
        order.extend(list(range(offset + 1, n_frames + 1, frame_modulo)))
    return order


def train(eval_sources=('DHF1K', 'SALICON', 'UCFSports', 'Hollywood'),
          **kwargs):
    """Run training and evaluation."""
    trainer = unisal.train.Trainer(**kwargs)
    trainer.fit()
    for source in eval_sources:
        trainer.score_model(source=source)
        trainer.export_scalars()
        trainer.writer.close()


def load_trainer(train_id=None):
    """Instantiate Trainer class from saved kwargs."""
    if train_id is None:
        train_id = 'pretrained_unisal'
    print(f"Train ID: {train_id}")
    train_dir = Path(os.environ["TRAIN_DIR"])
    train_dir = train_dir / train_id
    return unisal.train.Trainer.init_from_cfg_dir(train_dir)


def predictions_from_folder(folder_path, is_video, source=None, train_id=None, model_domain=None):
    """Generate predictions of files in a folder with a trained model."""
    folder_path = Path(folder_path).resolve()
    is_video = bool(is_video)

    trainer = load_trainer(train_id)

    # Prefer GPU for inference when available.
    trainer.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer.model.to(trainer.device)
    print(f"Using device: {trainer.device}")

    hook_handle = None
    # ONLY register hook for video datasets
    if is_video:
        # Ensure RNN exists and runs
        trainer.model.bypass_rnn = False
        hook_handle = trainer.model.upsampling_1.register_forward_pre_hook(hook_fn)

    # Clear previous video features
    global pre_up_latents, latent_pbar
    pre_up_latents = []
    latent_pbar = None
    if is_video and tqdm is not None:
        images_dir = folder_path / "images"
        n_frames = len(
            [p for p in images_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )
        latent_pbar = tqdm(total=n_frames, desc=f"Predicting {folder_path.name}", unit="frame")

    trainer.generate_predictions_from_path(
        folder_path, is_video, source=source, model_domain=model_domain)
    if latent_pbar is not None:
        latent_pbar.close()
        latent_pbar = None
    if hook_handle is not None:
        hook_handle.remove()

    # Save captured latent vectors to disk (one .pt file per frame)
    if is_video and pre_up_latents:
        # [batch, time, channels, h, w], batch is expected to be 1 for this path
        stacked_feats = torch.cat(pre_up_latents, dim=1)
        if LOG_LATENT_SHAPES:
            print(f"Stacked latent tensor shape: {tuple(stacked_feats.shape)}")
        video_name = folder_path.name
        save_dir = LATENT_SAVE_DIR / video_name
        save_dir.mkdir(exist_ok=True, parents=True)

        images_dir = folder_path / "images"
        frame_files = sorted(
            [p for p in images_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )
        source_name = source or "DHF1K"
        frame_modulo = 5 if source_name == "DHF1K" else 4
        processing_order = get_processing_frame_order(len(frame_files), frame_modulo)

        n_latents = stacked_feats.shape[1]
        if n_latents != len(processing_order):
            print(
                f"Warning: latent count ({n_latents}) != expected frame count "
                f"({len(processing_order)}). Saving by latent index."
            )
            for latent_idx in range(n_latents):
                latent = stacked_feats[0, latent_idx, ...].clone()
                save_path = save_dir / f"{latent_idx + 1:06d}.pt"
                if LOG_LATENT_SHAPES:
                    print(f"Saving latent shape {tuple(latent.shape)} to {save_path}")
                torch.save(latent, save_path)
        else:
            for latent_idx, frame_nr in enumerate(processing_order):
                latent = stacked_feats[0, latent_idx, ...].clone()
                frame_stem = frame_files[frame_nr - 1].stem
                save_path = save_dir / f"{frame_stem}.pt"
                if LOG_LATENT_SHAPES:
                    print(f"Saving latent shape {tuple(latent.shape)} to {save_path}")
                torch.save(latent, save_path)

        print(f"Saved {n_latents} pre-upsampling frame latents to {save_dir}")


def predict_examples(train_id=None):
    """Generate predictions for example datasets (videos only)."""
    print(Path(__file__).resolve().parent )
    for example_folder in (Path(__file__).resolve().parent / "examples").glob("*"):
        if not example_folder.is_dir():
            continue

        source = example_folder.name

        print(source)
        is_video = source not in ('SALICON', 'MIT1003')  # Only videos

        print(f"\nGenerating predictions for video folder: {str(source)}")

        print(is_video)
        if is_video:
            if not example_folder.is_dir():
                continue
            video_folders = [p for p in example_folder.glob('[!.]*') if p.is_dir()]
            iterator = video_folders
            if tqdm is not None:
                iterator = tqdm(video_folders, desc=f"Videos {source}", unit="video")
            for video_folder in iterator:   # ignore hidden files
                predictions_from_folder(
                    video_folder, is_video, train_id=train_id, source=source)

    print("All video pre-upsampling latents saved.")


def predict_custom_video(
    video_path,
    train_id=None,
    source="DHF1K",
    model_domain=None,
    output_root="custom_videos",
    frame_step=1,
):
    """Run saliency prediction + latent export on a custom video file.

    Creates:
      - <output_root>/<video_stem>/images/*.jpg
      - <output_root>/<video_stem>/saliency/*.jpg
      - pre_upsampling_latents/<video_stem>/*.pt
    """
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")

    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    video_name = video_path.stem
    work_dir = Path(output_root).resolve() / video_name
    images_dir = work_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames from video.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_idx = 0
    saved_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extract_pbar = None
    if tqdm is not None and total_frames > 0:
        extract_pbar = tqdm(total=total_frames, desc=f"Extracting {video_name}", unit="frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            saved_idx += 1
            out_file = images_dir / f"{saved_idx:06d}.jpg"
            cv2.imwrite(str(out_file), frame)
        frame_idx += 1
        if extract_pbar is not None:
            extract_pbar.update(1)
    cap.release()
    if extract_pbar is not None:
        extract_pbar.close()

    if saved_idx == 0:
        raise RuntimeError("No frames extracted from video.")

    print(f"Extracted {saved_idx} frames to {images_dir}")
    predictions_from_folder(
        work_dir, True, source=source, train_id=train_id, model_domain=model_domain
    )
    print(f"Done. Outputs saved under: {work_dir}")


if __name__ == "__main__":
    import fire

    fire.Fire()
