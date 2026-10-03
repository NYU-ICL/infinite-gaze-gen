> To reproduce Table 1 exactly (eval/results/all_baselines_reproduce_final), follow the scripts and instructions in the [`commands`](commands) directory.

# Infinite Gaze Generation for Videos with Autoregressive Diffusion

**ECCV 2026**

Jenna Kang, Colin Groth, Tong Wu, Finley Torrens, Patsorn Sangkloy, Gordon Wetzstein, Qi Sun

[Paper](https://arxiv.org/abs/2603.24938) | Code

Official implementation of **Infinite Gaze Generation for Videos with Autoregressive Diffusion**.

We introduce a generative framework for **infinite-horizon raw gaze prediction** in videos of arbitrary length. The model uses autoregressive diffusion to generate continuous gaze trajectories with high-resolution temporal information, conditioned on saliency-aware visual features.

## End-to-End Inference

The easiest way to run the complete pipeline on a video is:

```text
full_video_pipeline.ipynb
```

The notebook:

1. Loads an input video.
2. Generates UNISAL saliency latents.
3. Loads the bundled 90/45 gaze-generation checkpoint.
4. Generates a gaze trajectory.
5. Renders the prediction over the original video.
6. Saves the result as `overlay.mp4`.

Set:

```python
INPUT_VIDEO = "path/to/video.mp4"
```

in the first notebook cell and run the notebook from top to bottom.

## Pretrained Checkpoint

The repository is configured for the included 90-history / 45-prediction
checkpoint and its paired inference configuration:

```text
final_model_90_45/
├── checkpoint_70.pth
└── inference_config.yaml
```

The full-video notebook writes temporary saliency latents and generated videos
to `artifacts/full_video_pipeline/` by default. Set
`INFINITE_GAZE_ARTIFACT_ROOT` to use an external artifact directory.
The same variable controls the default output directory for `sample_video.py`.

```text
artifacts/full_video_pipeline/
```

## Environment Setup

Create a Python 3.10 environment:

```bash
conda create -n inf_gaze python=3.10 -y
conda activate inf_gaze
pip install -r requirements.txt
```

For NVIDIA GPU inference, install the PyTorch build corresponding to your CUDA version before installing the remaining dependencies.

For example, for CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Generate UNISAL Conditioning

To generate saliency conditioning for a single video:

```bash
python generate_unisal_latents.py \
    --video-path path/to/video.mp4 \
    --output-root path/to/saliency_unisal_latents_small \
    --source DHF1K
```

`DHF1K` is the default source domain for general videos and produces the
conditioning format used by the sampling example.

This creates:

```text
saliency_unisal_latents_small/
└── <video_stem>/
    ├── 000001.pt
    ├── 000002.pt
    ├── 000003.pt
    └── ...
```

These latents can then be passed to the gaze-generation model.

## Sample a Video

Generate a gaze trajectory from an input video with:

```bash
python sample_video.py \
    --video-path path/to/video.mp4 \
    --conditioning-dir path/to/saliency_unisal_latents_small/<video_stem>
```

If `--conditioning-dir` is omitted, the script searches for a sibling directory corresponding to the conditioning name specified in the configuration.

By default, outputs are written to:

```text
artifacts/video_samples/<video_stem>/sample_XXX/
```

Each sample contains:

```text
scanpath.csv
scanpath.json
overlay.mp4
```

`overlay.mp4` visualizes the generated gaze trajectory over the source video.

OpenCV (`cv2`) is required for overlay rendering.

## Training

The bundled 90/45 YAML is inference-only. Training requires a separate
training configuration. The expected DIEM dataset location for training is:

```text
artifacts/datasets/DIEM/
```

The dataset root should contain:

```text
datasets/DIEM/
├── split_files/
│   ├── diem_final_video_train.json
│   └── diem_final_video_val.json
├── saliency_unisal_latents_small/
│   └── <stim>/
│       ├── frame_XXXXXX.pt
│       └── ...
└── <DIEM gaze and video data>
```

Saliency latent filenames may also use:

```text
XXXXXX.pt
```

instead of:

```text
frame_XXXXXX.pt
```

## Repository Structure

```text
.
├── config/
├── final_model_90_45/
│   ├── checkpoint_70.pth
│   └── inference_config.yaml
├── full_video_pipeline.ipynb
├── generate_unisal_latents.py
├── sample_video.py
├── train.py
└── requirements.txt
```

## Paper

**Infinite Gaze Generation for Videos with Autoregressive Diffusion**
Jenna Kang, Colin Groth, Tong Wu, Finley Torrens, Patsorn Sangkloy, Gordon Wetzstein, and Qi Sun.
ECCV 2026.

[arXiv:2603.24938](https://arxiv.org/abs/2603.24938)

## Citation

If you find this work useful, please cite:

```bibtex
@article{kang2026infinitegaze,
  title={Infinite Gaze Generation for Videos with Autoregressive Diffusion},
  author={Kang, Jenna and Groth, Colin and Wu, Tong and Torrens, Finley and Sangkloy, Patsorn and Wetzstein, Gordon and Sun, Qi},
  journal={arXiv preprint arXiv:2603.24938},
  year={2026}
}
```

## Acknowledgements

This repository contains the implementation associated with our ECCV 2026 work on autoregressive diffusion for long-horizon video gaze generation.
