import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from common import (
    create_experiment_dir,
    instantiate_from_config,
    load_checkpoint,
    load_yaml_config,
    merge_opts_to_config,
    save_checkpoint,
    save_yaml_config,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal DIEM-only UNet-saliency model.")
    parser.add_argument("--config", type=str, default="config/full_diem.yaml")
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def build_dataset_config(cfg: dict, split_cfg: dict, root_dir: Path) -> dict:
    dataset_cfg = {
        "target": split_cfg["target"],
        "params": dict(cfg["dataset"]),
    }
    dataset_cfg["params"].update(split_cfg.get("params", {}))
    dataset_cfg["params"]["seq_len"] = int(dataset_cfg["params"].pop("pred_len"))
    dataset_cfg["params"]["root"] = str((root_dir / str(dataset_cfg["params"]["root"])).resolve())
    dataset_cfg["params"]["stim_to_sub_pth"] = str(
        (Path(dataset_cfg["params"]["root"]) / str(dataset_cfg["params"]["stim_to_sub_pth"])).resolve()
    )
    return dataset_cfg


def main() -> None:
    args = parse_args()
    cfg = merge_opts_to_config(load_yaml_config(args.config), args.opts)
    root_dir = Path(args.root_dir).resolve()
    seed_everything(int(args.seed))

    exp_dir = create_experiment_dir(cfg["output_root"], cfg["experiment_name"])
    save_yaml_config(cfg, exp_dir / "config.yaml")

    train_dataset_cfg = build_dataset_config(cfg, cfg["train_dataset"], root_dir)
    train_dataset = instantiate_from_config(train_dataset_cfg)
    train_loader = instantiate_from_config(cfg["train_loader"], dataset=train_dataset)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = instantiate_from_config(cfg["model"]).to(device)
    optimizer = instantiate_from_config(cfg["optimizer"], params=model.parameters())
    criterion = instantiate_from_config(cfg["criterion"])
    train_scheduler = instantiate_from_config(cfg["diffusion"]["train_scheduler"])

    start_epoch = 0
    global_step = 0
    if args.checkpoint:
        start_epoch, global_step = load_checkpoint(args.checkpoint, model, optimizer, train_scheduler)

    history_len = int(cfg["dataset"]["history_len"])
    pred_len = int(cfg["dataset"]["pred_len"])
    num_epochs = int(cfg["train"]["num_epochs"])
    save_every = int(cfg["train"]["save_every"])
    log_every = max(1, int(cfg["train"]["log_every"]))
    cfg_on = bool(cfg["train"]["cfg_on"])
    cfg_drop_rate = float(cfg["train"]["cfg_drop_rate"])
    use_amp = bool(cfg["train"]["use_amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    losses_path = exp_dir / "losses.csv"
    with losses_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "mean_loss"])

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = []

        for batch_idx, (trajs, conditioning, _stim_names, _start_indices) in enumerate(train_loader, start=1):
            trajs = trajs.to(device)
            conditioning = conditioning.to(device, non_blocking=True)
            history = trajs[:, :, :history_len]
            target = trajs[:, :, history_len : history_len + pred_len]

            noise = torch.randn_like(target)
            timesteps = torch.randint(0, train_scheduler.config.num_train_timesteps, (trajs.shape[0],), device=device)
            noisy_target = train_scheduler.add_noise(target, noise, timesteps)
            model_input = torch.cat([history, noisy_target], dim=2) if history_len > 0 else noisy_target

            if cfg_on and torch.rand(1).item() < cfg_drop_rate:
                conditioning = torch.zeros_like(conditioning)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_noise, _ = model(model_input, timesteps, conditioning)
                pred_noise = pred_noise[:, :, -pred_len:]
                loss = criterion(pred_noise, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            epoch_losses.append(float(loss.item()))

            if batch_idx % log_every == 0:
                mean_so_far = float(np.mean(epoch_losses))
                print(f"epoch={epoch + 1} batch={batch_idx}/{len(train_loader)} loss={loss.item():.6f} mean={mean_so_far:.6f}")

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"epoch={epoch + 1}/{num_epochs} mean_loss={epoch_loss:.6f}")
        with losses_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([epoch + 1, epoch_loss])

        should_save = ((epoch + 1) % save_every == 0) or (epoch == num_epochs - 1)
        if should_save:
            save_checkpoint(exp_dir / "checkpoints" / f"checkpoint_{epoch + 1}.pth", model, optimizer, train_scheduler, epoch, global_step)


if __name__ == "__main__":
    main()
