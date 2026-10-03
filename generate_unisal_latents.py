import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent
VENDORED_RUN_ROOT = REPO_ROOT / "vendor" / "scanpathgen_unisal" / "training_runs" / "small_latent_unisal"
VENDORED_CODE_ROOT = VENDORED_RUN_ROOT / "code_copy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate UNISAL saliency latents for a single video."
    )
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Directory where saliency_unisal_latents_small will be written. "
             "Defaults to a sibling folder next to the input video.",
    )
    parser.add_argument(
        "--train-id",
        type=str,
        default="small_latent_unisal",
        help="Vendored UNISAL training run to load.",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--source",
        type=str,
        default="DHF1K",
        help="UNISAL source domain. DHF1K matches the general-video latent workflow.",
    )
    parser.add_argument("--model-domain", type=str, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    return parser.parse_args()


def _load_vendored_run_module():
    if not VENDORED_CODE_ROOT.exists():
        raise FileNotFoundError(f"Vendored UNISAL code not found: {VENDORED_CODE_ROOT}")

    if str(VENDORED_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDORED_CODE_ROOT))

    run_path = VENDORED_CODE_ROOT / "run.py"
    spec = importlib.util.spec_from_file_location("vendored_unisal_run", run_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load vendored run module from {run_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_output_root(video_path: Path, override: str | None) -> Path:
    if override is not None:
        return Path(override).resolve()
    return (video_path.parent / "saliency_unisal_latents_small").resolve()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if int(args.frame_step) < 1:
        raise ValueError("--frame-step must be >= 1")

    run_module = _load_vendored_run_module()

    os.environ["TRAIN_DIR"] = str(VENDORED_RUN_ROOT.parent.resolve())
    output_root = _expected_output_root(video_path, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_module.LATENT_SAVE_DIR = output_root
    run_module.LATENT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    run_module.LOG_LATENT_SHAPES = False

    work_root = output_root.parent / "_saliency_unisal_workdir"
    work_root.mkdir(parents=True, exist_ok=True)

    run_module.predict_custom_video(
        video_path=str(video_path),
        train_id=args.train_id,
        source=args.source,
        model_domain=args.model_domain,
        output_root=str(work_root),
        frame_step=int(args.frame_step),
    )

    latent_dir = output_root / video_path.stem
    if not latent_dir.exists():
        raise RuntimeError(f"Latent export did not produce: {latent_dir}")

    pt_files = sorted(latent_dir.glob("*.pt"))
    if not pt_files:
        raise RuntimeError(f"No latent files were written to {latent_dir}")

    sample_latent = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    print(f"saved {len(pt_files)} latents to {latent_dir}")
    print(f"latent shape: {tuple(sample_latent.shape)}")

    if not args.keep_workdir and work_root.exists():
        this_workdir = work_root / video_path.stem
        if this_workdir.exists():
            shutil.rmtree(this_workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
