from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from metrics import cal_metrics


NIMA_ROOT = Path("/root/Neural-IMage-Assessment")
if str(NIMA_ROOT) not in sys.path:
    sys.path.insert(0, str(NIMA_ROOT))

from model.model import NIMA  # noqa: E402


METRIC_NAMES = ["mse", "srcc", "plcc", "acc", "emd1", "emd2"]


class DistributionDataset(Dataset):
    def __init__(self, csv_file: Path, image_dir: Path, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = int(row.iloc[0])
        image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
        if self.transform:
            image = self.transform(image)

        dist = row.iloc[1:].to_numpy(dtype=np.float32)
        dist_sum = float(dist.sum())
        if dist_sum > 0:
            dist = dist / dist_sum

        return {
            "image_id": image_id,
            "image": image,
            "dos": torch.from_numpy(dist),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NIMA PyTorch baseline on the repeated PCCD 7:1:2 splits."
    )
    parser.add_argument("--nima-root", default=str(NIMA_ROOT))
    parser.add_argument(
        "--splits-root",
        default="/root/autodl-tmp/pccd_repeated_712_10runs/splits",
        help="Directory containing seed_00/train_pccd.csv, val_pccd.csv, test_pccd.csv.",
    )
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/nima_pccd_repeated_712_10runs",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--conv-lr", type=float, default=5e-4)
    parser.add_argument("--dense-lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-decay-rate", type=float, default=0.95)
    parser.add_argument("--lr-decay-freq", type=int, default=10)
    parser.add_argument(
        "--vgg16-weights",
        default=None,
        help="Optional local torchvision VGG16 state_dict path, e.g. vgg16-397923af.pth.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained VGG16. Use only for debugging.",
    )
    parser.add_argument(
        "--torch-home",
        default="/root/autodl-tmp/torch_cache",
        help="Where torchvision downloads/caches VGG16 weights.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return train_transform, eval_transform


def build_nima(args):
    if args.no_pretrained:
        base_model = models.vgg16(weights=None)
    elif args.vgg16_weights:
        base_model = models.vgg16(weights=None)
        state = torch.load(args.vgg16_weights, map_location="cpu")
        base_model.load_state_dict(state)
    else:
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            base_model = models.vgg16(weights=weights)
        except Exception as exc:
            raise RuntimeError(
                "Could not load/download torchvision VGG16 ImageNet weights. "
                "Upload vgg16-397923af.pth and pass --vgg16-weights /path/to/file, "
                "or rerun with network access."
            ) from exc

    model = NIMA(base_model)
    # The uploaded implementation uses nn.Softmax() without dim; keep the same
    # architecture but make the probability axis explicit for current PyTorch.
    model.classifier[-1] = nn.Softmax(dim=1)
    return model


def emd_loss(pred, target, r=2):
    pred_cdf = torch.cumsum(pred, dim=1)
    target_cdf = torch.cumsum(target, dim=1)
    diff = pred_cdf - target_cdf
    if r == 2:
        samplewise = torch.sqrt(torch.mean(diff.pow(2), dim=1))
    else:
        samplewise = torch.mean(diff.abs(), dim=1)
    return samplewise.mean()


def evaluate(model, loader, device, use_amp, desc):
    model.eval()
    preds, targets = [], []
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["dos"].to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp):
                pred = model(images)
                loss = emd_loss(pred, target)
            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
            losses.append(loss.item())

    pred_np = torch.cat(preds, dim=0).numpy()
    target_np = torch.cat(targets, dim=0).numpy()
    metrics = {
        name: float(value)
        for name, value in zip(METRIC_NAMES, cal_metrics(pred_np, target_np))
    }
    metrics["emd_loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def save_checkpoint(path: Path, model, epoch: int, best_srcc: float, val_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_srcc": best_srcc,
        "val_metrics": val_metrics,
        "selection": "val_srcc",
        "model": "NIMA_VGG16",
    }, tmp)
    os.replace(tmp, path)


def train_one_seed(args, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    seed_dir = Path(args.splits_root) / f"seed_{seed:02d}"
    image_dir = seed_dir / "images"
    train_csv = seed_dir / "train_pccd.csv"
    val_csv = seed_dir / "val_pccd.csv"
    test_csv = seed_dir / "test_pccd.csv"

    train_transform, eval_transform = build_transforms()
    train_set = DistributionDataset(train_csv, image_dir, train_transform)
    val_set = DistributionDataset(val_csv, image_dir, eval_transform)
    test_set = DistributionDataset(test_csv, image_dir, eval_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_nima(args).to(device)
    optimizer = optim.SGD([
        {"params": model.features.parameters(), "lr": args.conv_lr},
        {"params": model.classifier.parameters(), "lr": args.dense_lr},
    ], momentum=0.9, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_freq,
        gamma=args.lr_decay_rate,
    )

    run_dir = Path(args.output_root) / "runs" / f"seed_{seed:02d}"
    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    results_path = run_dir / "results.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "train_loss", "val_loss",
                "mse", "srcc", "plcc", "acc", "emd1", "emd2",
            ],
        )
        writer.writeheader()

    print(
        f"\n=== NIMA seed {seed} | "
        f"train={len(train_set)} val={len(val_set)} test={len(test_set)} ==="
    )

    best_srcc = -1.0
    best_epoch = -1
    best_val_metrics = None

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Train seed {seed} epoch {epoch}"):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["dos"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                pred = model(images)
                loss = emd_loss(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        scheduler.step()
        val_metrics = evaluate(
            model, val_loader, device, use_amp,
            desc=f"Val seed {seed} epoch {epoch}",
        )
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = val_metrics["emd_loss"]

        print(
            f"Seed {seed} | Epoch {epoch} | train_loss={train_loss:.4f} "
            f"| val_srcc={val_metrics['srcc']:.4f} "
            f"| val_plcc={val_metrics['plcc']:.4f} "
            f"| val_emd2={val_metrics['emd2']:.4f}"
        )

        with results_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch", "train_loss", "val_loss",
                    "mse", "srcc", "plcc", "acc", "emd1", "emd2",
                ],
            )
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{name: val_metrics[name] for name in METRIC_NAMES},
            })

        if val_metrics["srcc"] > best_srcc:
            best_srcc = val_metrics["srcc"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            save_checkpoint(ckpt_path, model, epoch, best_srcc, val_metrics)
            print(f"  New best val SRCC={best_srcc:.4f}; saved {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    test_metrics = evaluate(
        model, test_loader, device, use_amp, desc=f"Test seed {seed}"
    )

    final_path = run_dir / "final_test_results.yml"
    with final_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({
            "seed": seed,
            "best_epoch": best_epoch,
            "selected_by": "val_srcc",
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "checkpoint": str(ckpt_path),
        }, f, sort_keys=False)

    return {
        "seed": seed,
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
        "best_epoch": best_epoch,
        "run_dir": str(run_dir),
        **{name: test_metrics[name] for name in METRIC_NAMES},
    }


def write_summary(rows, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "summary.csv"
    fieldnames = [
        "seed", "train", "val", "test", "best_epoch", "run_dir",
        *METRIC_NAMES,
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    values = np.array([[row[name] for name in METRIC_NAMES] for row in rows])
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros_like(means)

    with (output_root / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# NIMA PCCD Repeated Random Split Results\n\n")
        f.write(f"Runs: {len(rows)}\n\n")
        f.write("Protocol: 7:1:2 random split x10, best checkpoint selected by validation SRCC.\n\n")
        f.write("| Metric | Mean | Std |\n")
        f.write("|---|---:|---:|\n")
        for name, mean, std in zip(METRIC_NAMES, means, stds):
            f.write(f"| {name.upper()} | {mean:.4f} | {std:.4f} |\n")


def main():
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    Path(args.torch_home).mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"NIMA root: {args.nima_root}")
    print(f"Splits root: {args.splits_root}")
    print(f"Output root: {args.output_root}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"TORCH_HOME: {os.environ['TORCH_HOME']}")

    if args.dry_run:
        for seed in args.seeds:
            seed_dir = Path(args.splits_root) / f"seed_{seed:02d}"
            print(seed, seed_dir, (seed_dir / "train_pccd.csv").exists())
        return

    rows = []
    for seed in args.seeds:
        row = train_one_seed(args, seed)
        rows.append(row)
        write_summary(rows, output_root)
        print(f"Seed {seed} final test: {row}")

    write_summary(rows, output_root)
    print(f"Summary written to: {output_root / 'summary.md'}")
    print(f"Raw CSV written to: {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
