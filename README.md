## End-to-End Video Example

Use `full_video_pipeline.ipynb` as the complete example for processing one
video. It generates saliency latents from the input video, loads
`final_model/checkpoint_70.pth`, predicts a gaze trajectory, and writes an
`overlay.mp4` with the prediction drawn over the source video. Set
`INPUT_VIDEO` in the first notebook cell, then run the notebook top-to-bottom.


Checkpoint download: https://drive.google.com/drive/folders/1wlbaFsqxYYNagDdSrv44OEOjs5-vew4k?usp=sharing

## Minimal Environment Setup

Create and activate a Python 3.10 environment, then install the runtime
dependencies:

```bash
conda create -n inf_gaze python=3.10 -y
conda activate inf_gaze
pip install -r requirements.txt
```

For NVIDIA GPU inference, install the PyTorch build appropriate for your CUDA
setup before installing the remaining requirements. For example, CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Train

```bash
python train.py --config config/full_diem.yaml --root-dir artifacts
```

`--root-dir` must contain `datasets/DIEM`. In this checkout, the DIEM dataset
is located at `artifacts/datasets/DIEM`, so use `--root-dir artifacts` as shown
above. That dataset root must contain:

- `split_files/diem_final_video_train.json`
- `split_files/diem_final_video_val.json`
- `saliency_unisal_latents_small/<stim>/frame_XXXXXX.pt` or `XXXXXX.pt`
- DIEM gaze/video folders under the dataset root

## Sample One Video

```bash
python sample_video.py ^
  --config config/full_diem.yaml ^
  --checkpoint artifacts/experiments/unet_saliency_hires_original_diem/.../checkpoints/checkpoint_3000.pth ^
  --video-path path/to/video.mp4 ^
  --conditioning-dir path/to/saliency_unisal_latents_small/<video_stem>
```

If `--conditioning-dir` is omitted, the script looks for a sibling folder matching the config conditioning name.

Outputs are written under `artifacts/video_samples/<video_stem>/sample_XXX/`.
Each sample folder now includes:

- `scanpath.csv`
- `scanpath.json`
- `overlay.mp4`

`overlay.mp4` draws the generated gaze points over the original video. The runtime used to execute `sample_video.py` needs OpenCV (`cv2`) for overlay rendering.

## Generate UNISAL Conditioning

To create `saliency_unisal_latents_small/<video_stem>/*.pt` for a single input video:

```bash
python generate_unisal_latents.py ^
  --video-path path/to/video.mp4 ^
  --output-root path/to/saliency_unisal_latents_small
```

That writes:

- `path/to/saliency_unisal_latents_small/<video_stem>/000001.pt`
- `path/to/saliency_unisal_latents_small/<video_stem>/000002.pt`
- ...

Then use that folder with `sample_video.py --conditioning-dir`.

