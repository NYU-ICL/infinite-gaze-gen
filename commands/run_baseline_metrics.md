# Running baseline metrics

Run the commands below from the repository root. The evaluator compares each
baseline's first 810 prediction points against DIEM ground truth after the
first 90 raw frames, then averages each video equally in the final summary.

## Evaluate the current baseline set

```powershell
python eval/run_baseline_metrics.py `
  --ground-truth-root "C:\Users\jk8659\NYU\research\Scanpath\diffeye\artifacts\datasets\DIEM\data" `
  --output-dir "eval/results/all_baselines_final" `
  --metrics dtw levenshtein temporal_correlation discrete_frechet `
  --model "deepgaze|simulated_raw|baselines/deepgaze3_scanpaths_repro" `
  --model "diffeye_video_modes|diffeye|baselines/diffeye_video_modes" `
  --model "gazeformer_isp|simulated_raw|baselines/gazeformer_isp_scanpaths/full_30s_rollout" `
  --model "hat|simulated_raw|baselines/hat/hat_scanpaths_autoreg30s" `
  --model "lstm_isp|simulated_raw|baselines/lstm_isp_scanpaths/full_30s_rollout" `
  --model "tppgaze_auto|simulated_raw|baselines/tppgaze_scanpaths_auto" `
  --model "unet_clip_history|unet|baselines/diem_final_model_90_45_batched_reproduced"
```

## Model kinds

`unet` reads each `scanpath.csv` and uses its first 810 points. Use it for the
DIEM UNet generator, whose first saved point corresponds to raw video frame 90.

`simulated_raw` reads `*_simraw.csv` and uses its first 810 points. Those files
must already begin at the evaluation start (raw frame 90); this evaluator does
not remove their first 90 points.

`diffeye` reads `prediction_*.csv` and evenly downsamples each full trajectory
to 810 points. It falls back to `scanpath.csv` when no `prediction_*.csv` files
exist.

## Output files

The output directory receives:

- `pairwise_comparisons.csv`: every prediction × GT-participant comparison.
- `per_gt_best_mean.csv`: mean and best prediction scores for each GT path.
- `per_video_best_mean.csv`: equal-weight video summaries.
- `final_metrics.csv` and `final_metrics.json`: the overall scores.
