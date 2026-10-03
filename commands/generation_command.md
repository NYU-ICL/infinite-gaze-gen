# Generation command

Run this command from the repository root to reproduce these DIEM scanpaths:

```powershell
python sample_diem_val.py `
  --dataset-root "path/to/diem" `
  --participants-manifest "baselines/diem_final_model_90_45_batched_reproduced/participants_by_generation.csv" `
  --output-dir "baselines/diem_final_model_90_45_batched_reproduced" `
  --config "final_model_90_45/inference_config.yaml" `
  --checkpoint "final_model_90_45/checkpoint_70.pth" `
  --num-predictions 10 `
  --chunk-seconds 3 `
  --fps 30 `
  --max-video-seconds 30 `
  --seed 12 `
  --seed-step 1 `
  --seed-reset-every 4 `
  --device cuda
```

`--seed-reset-ever` is to simulate the distributed generation that was used in the original paper.


`--manifest-only` is enabled by default.

## Required `--dataset-root` layout

Set `--dataset-root` to the DIEM artifact root, not directly to one video's
`video` directory..

For this multi-video generation, the expected layout is:

```text
<dataset-root>/
├── data/
│   └── <stimulus-name>/
│       ├── video/
│       │   └── <video-file>.mp4
│       └── event_data/
│           └── <participant-event-file>.txt
└── saliency_unisal_latents_small/
    └── <stimulus-name>/
        └── <frame-feature>.pt or <frame-feature>.npy
```

`<stimulus-name>` must exactly match the video directory name, the
participant-manifest `stimulus` value, and the saliency-feature directory name.
Each manifest participant must match an `event_data/*.txt` filename stem. The
loader reads the first `.mp4` in each `video` directory and requires the
matching per-frame saliency features under the dataset root (not under `data`).
