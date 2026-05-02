from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import AVADataset
from model import AblationModel, ImprovedIAAModel


SCORE_BINS = np.arange(1, 11, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute PCCD classification metrics from existing checkpoints "
            "using the train-split raw-overall median as the threshold."
        )
    )
    parser.add_argument(
        "--pccd-root",
        default="/root/autodl-tmp/PCCD/PCCD",
        help="Raw PCCD root containing images/full and guru.json.",
    )
    parser.add_argument(
        "--splits-root",
        default="/root/autodl-tmp/pccd_repeated_712_10runs/splits",
        help="Root containing seed_00/train_pccd.csv, val_pccd.csv, test_pccd.csv.",
    )
    parser.add_argument(
        "--main-summary-csv",
        default="/root/autodl-tmp/pccd_repeated_712_10runs/summary.csv",
        help="Summary CSV for the main model runs.",
    )
    parser.add_argument(
        "--nima-summary-csv",
        default="/root/autodl-tmp/nima_pccd_repeated_712_10runs/summary.csv",
        help="Summary CSV for the NIMA runs.",
    )
    parser.add_argument(
        "--main-output-root",
        default="/root/autodl-tmp/pccd_repeated_712_10runs",
        help="Where to write the recomputed main-model classification summaries.",
    )
    parser.add_argument(
        "--nima-output-root",
        default="/root/autodl-tmp/nima_pccd_repeated_712_10runs",
        help="Where to write the recomputed NIMA classification summaries.",
    )
    parser.add_argument(
        "--comparison-root",
        default="/root/autodl-tmp",
        help="Where to write the two-model comparison summary.",
    )
    parser.add_argument(
        "--nima-root",
        default="/root/Neural-IMage-Assessment",
        help="Uploaded PyTorch NIMA project root.",
    )
    parser.add_argument(
        "--vgg16-weights",
        default="/root/autodl-tmp/torch_cache/hub/checkpoints/vgg16-397923af.pth",
        help="Local torchvision VGG16 weights for NIMA.",
    )
    parser.add_argument(
        "--main-batch-size",
        type=int,
        default=None,
        help="Optional override for main-model test batch size.",
    )
    parser.add_argument(
        "--main-num-workers",
        type=int,
        default=None,
        help="Optional override for main-model test num_workers.",
    )
    parser.add_argument(
        "--nima-batch-size",
        type=int,
        default=64,
        help="NIMA test batch size.",
    )
    parser.add_argument(
        "--nima-num-workers",
        type=int,
        default=8,
        help="NIMA test num_workers.",
    )
    return parser.parse_args()


def parse_overall(value):
    try:
        score = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def map_to_1_10(score, min_score, max_score):
    if max_score == min_score:
        return 5.5
    return 1.0 + (score - min_score) * 9.0 / (max_score - min_score)


def build_raw_lookup(pccd_root: Path):
    json_path = pccd_root / "guru.json"
    image_dir = pccd_root / "images" / "full"

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    parsed_scores = []
    for record in records:
        score = parse_overall(record.get("overall"))
        if score is not None:
            parsed_scores.append(score)

    if not parsed_scores:
        raise ValueError(f"No valid overall scores found in {json_path}")

    min_score = min(parsed_scores)
    max_score = max(parsed_scores)

    raw_lookup = {}
    converted = 0
    for record in records:
        overall = parse_overall(record.get("overall"))
        title = str(record.get("title", "")).strip()
        source_path = image_dir / title
        if overall is None or not title or not source_path.exists():
            continue
        converted += 1
        raw_lookup[converted] = float(overall)

    return raw_lookup, float(min_score), float(max_score)


def load_summary_rows(summary_csv: Path):
    df = pd.read_csv(summary_csv)
    rows = {}
    for _, row in df.iterrows():
        rows[int(row["seed"])] = row.to_dict()
    return rows


def compute_binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    pos = tp + fn
    neg = tn + fp
    tpr = tp / pos if pos else 0.0
    tnr = tn / neg if neg else 0.0
    balanced_acc = 0.5 * (tpr + tnr)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "balanced_acc": balanced_acc * 100.0,
        "f1": f1 * 100.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def seed_threshold_info(seed_dir: Path, raw_lookup, min_raw: float, max_raw: float):
    train_ids = pd.read_csv(seed_dir / "train_pccd.csv")["image_id"].astype(int).tolist()
    test_ids = pd.read_csv(seed_dir / "test_pccd.csv")["image_id"].astype(int).tolist()

    train_raw = np.array([raw_lookup[i] for i in train_ids], dtype=np.float32)
    test_raw = np.array([raw_lookup[i] for i in test_ids], dtype=np.float32)

    threshold_raw = float(np.median(train_raw))
    threshold_mapped = float(map_to_1_10(threshold_raw, min_raw, max_raw))

    return {
        "threshold_raw": threshold_raw,
        "threshold_mapped": threshold_mapped,
        "train_pos_ratio": float(np.mean(train_raw >= threshold_raw)),
        "test_pos_ratio": float(np.mean(test_raw >= threshold_raw)),
        "test_size": int(len(test_raw)),
    }


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer_main_model(config_path: Path, ckpt_path: Path, batch_size=None, num_workers=None):
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if batch_size is not None:
        config["batch_size"] = batch_size
    if num_workers is not None:
        config["num_workers"] = num_workers

    dataset = AVADataset(config, "test")
    loader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 8),
        pin_memory=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    model = AblationModel(config).to(device) if config.get("ablation_mode") else ImprovedIAAModel(config).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    image_ids = []
    pred_means = []

    with torch.no_grad():
        for samples in tqdm(loader, desc=f"Main test {config_path.stem}"):
            image_ids.extend(int(x) for x in samples["image_id"].tolist())
            samples["image"] = samples["image"].to(device, non_blocking=True)
            samples["dos"] = samples["dos"].to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                output = model(samples)

            pred_means.append(output.cpu().numpy() @ SCORE_BINS)

    free_cuda()
    return np.array(image_ids, dtype=np.int64), np.concatenate(pred_means, axis=0)


class NIMADistributionDataset(Dataset):
    def __init__(self, csv_file: Path, image_dir: Path, transform):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = int(row.iloc[0])
        image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
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


def load_nima_class(nima_root: Path):
    module_path = nima_root / "model" / "model.py"
    spec = importlib.util.spec_from_file_location("nima_model_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load NIMA model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NIMA


def build_nima_model(nima_root: Path, vgg16_weights: Path):
    NIMA = load_nima_class(nima_root)
    base_model = tv_models.vgg16(weights=None)
    base_state = torch.load(vgg16_weights, map_location="cpu")
    base_model.load_state_dict(base_state)
    model = NIMA(base_model)
    model.classifier[-1] = nn.Softmax(dim=1)
    return model


def nima_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def infer_nima_model(seed_dir: Path, ckpt_path: Path, nima_root: Path, vgg16_weights: Path, batch_size: int, num_workers: int):
    dataset = NIMADistributionDataset(
        csv_file=seed_dir / "test_pccd.csv",
        image_dir=seed_dir / "images",
        transform=nima_eval_transform(),
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    model = build_nima_model(nima_root, vgg16_weights).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image_ids = []
    pred_means = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"NIMA test {seed_dir.name}"):
            image_ids.extend(int(x) for x in batch["image_id"].tolist())
            images = batch["image"].to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp):
                output = model(images)
            pred_means.append(output.cpu().numpy() @ SCORE_BINS)

    free_cuda()
    return np.array(image_ids, dtype=np.int64), np.concatenate(pred_means, axis=0)


def evaluate_seed(image_ids, pred_means, threshold_raw, threshold_mapped, raw_lookup):
    true_raw = np.array([raw_lookup[int(i)] for i in image_ids], dtype=np.float32)
    y_true = (true_raw >= threshold_raw).astype(np.int64)
    y_pred = (pred_means >= threshold_mapped).astype(np.int64)

    metrics = compute_binary_metrics(y_true, y_pred)
    metrics["pred_pos_ratio"] = float(np.mean(y_pred))
    metrics["pred_score_mean"] = float(np.mean(pred_means))
    metrics["pred_score_min"] = float(np.min(pred_means))
    metrics["pred_score_max"] = float(np.max(pred_means))
    return metrics


def write_model_summary(rows, output_root: Path, title: str):
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "summary_classification_train_median_raw.csv"
    fieldnames = [
        "seed",
        "threshold_raw",
        "threshold_mapped",
        "train_pos_ratio",
        "test_pos_ratio",
        "pred_pos_ratio",
        "balanced_acc",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "pred_score_mean",
        "pred_score_min",
        "pred_score_max",
        "checkpoint",
    ]

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    balanced_accs = np.array([row["balanced_acc"] for row in rows], dtype=np.float64)
    f1s = np.array([row["f1"] for row in rows], dtype=np.float64)
    bacc_mean = float(balanced_accs.mean())
    bacc_std = float(balanced_accs.std(ddof=1)) if len(rows) > 1 else 0.0
    f1_mean = float(f1s.mean())
    f1_std = float(f1s.std(ddof=1)) if len(rows) > 1 else 0.0

    summary_md = output_root / "summary_classification_train_median_raw.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Runs: {len(rows)}\n\n")
        f.write("Protocol: threshold = median(raw_overall of training split); positive = raw_overall >= threshold.\n\n")
        f.write("| Metric | Mean | Std |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| Balanced Accuracy | {bacc_mean:.4f} | {bacc_std:.4f} |\n")
        f.write(f"| F1 | {f1_mean:.4f} | {f1_std:.4f} |\n\n")
        f.write("| Seed | Threshold(raw) | Threshold(mapped) | Train Pos% | Test Pos% | Pred Pos% | BAcc | F1 |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['seed']} | {row['threshold_raw']:.4f} | {row['threshold_mapped']:.4f} | "
                f"{row['train_pos_ratio'] * 100:.2f} | {row['test_pos_ratio'] * 100:.2f} | "
                f"{row['pred_pos_ratio'] * 100:.2f} | {row['balanced_acc']:.4f} | {row['f1']:.4f} |\n"
            )

    return {
        "balanced_acc_mean": bacc_mean,
        "balanced_acc_std": bacc_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "summary_csv": summary_csv,
        "summary_md": summary_md,
    }


def write_comparison(main_stats, nima_stats, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "pccd_train_median_raw_classification_comparison.csv"
    md_path = output_root / "pccd_train_median_raw_classification_comparison.md"

    rows = [
        {
            "method": "Ours",
            "balanced_acc_mean": main_stats["balanced_acc_mean"],
            "balanced_acc_std": main_stats["balanced_acc_std"],
            "f1_mean": main_stats["f1_mean"],
            "f1_std": main_stats["f1_std"],
        },
        {
            "method": "NIMA",
            "balanced_acc_mean": nima_stats["balanced_acc_mean"],
            "balanced_acc_std": nima_stats["balanced_acc_std"],
            "f1_mean": nima_stats["f1_mean"],
            "f1_std": nima_stats["f1_std"],
        },
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "balanced_acc_mean",
                "balanced_acc_std",
                "f1_mean",
                "f1_std",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# PCCD Classification Comparison (Train Raw Median Threshold)\n\n")
        f.write("| Method | Balanced Accuracy | F1 |\n")
        f.write("|---|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | "
                f"{row['balanced_acc_mean']:.4f} ± {row['balanced_acc_std']:.4f} | "
                f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f} |\n"
            )

    return csv_path, md_path


def main():
    args = parse_args()

    pccd_root = Path(args.pccd_root)
    splits_root = Path(args.splits_root)
    main_summary = Path(args.main_summary_csv)
    nima_summary = Path(args.nima_summary_csv)
    main_output_root = Path(args.main_output_root)
    nima_output_root = Path(args.nima_output_root)
    comparison_root = Path(args.comparison_root)
    nima_root = Path(args.nima_root)
    vgg16_weights = Path(args.vgg16_weights)

    raw_lookup, min_raw, max_raw = build_raw_lookup(pccd_root)
    main_runs = load_summary_rows(main_summary)
    nima_runs = load_summary_rows(nima_summary)

    common_seeds = sorted(set(main_runs) & set(nima_runs))
    if not common_seeds:
        raise ValueError("No overlapping seeds found between the main and NIMA summary CSV files.")

    main_rows = []
    nima_rows = []

    for seed in common_seeds:
        seed_dir = splits_root / f"seed_{seed:02d}"
        threshold_info = seed_threshold_info(seed_dir, raw_lookup, min_raw, max_raw)

        main_run_dir = Path(main_runs[seed]["run_dir"])
        main_ckpt = main_run_dir / "checkpoints" / "best_model.pt"
        main_config = main_run_dir / f"seed_{seed:02d}.yml"
        main_ids, main_pred_means = infer_main_model(
            config_path=main_config,
            ckpt_path=main_ckpt,
            batch_size=args.main_batch_size,
            num_workers=args.main_num_workers,
        )
        main_metrics = evaluate_seed(
            image_ids=main_ids,
            pred_means=main_pred_means,
            threshold_raw=threshold_info["threshold_raw"],
            threshold_mapped=threshold_info["threshold_mapped"],
            raw_lookup=raw_lookup,
        )
        main_rows.append({
            "seed": seed,
            **threshold_info,
            **main_metrics,
            "checkpoint": str(main_ckpt),
        })

        nima_run_dir = Path(nima_runs[seed]["run_dir"])
        nima_ckpt = nima_run_dir / "checkpoints" / "best_model.pt"
        nima_ids, nima_pred_means = infer_nima_model(
            seed_dir=seed_dir,
            ckpt_path=nima_ckpt,
            nima_root=nima_root,
            vgg16_weights=vgg16_weights,
            batch_size=args.nima_batch_size,
            num_workers=args.nima_num_workers,
        )
        nima_metrics = evaluate_seed(
            image_ids=nima_ids,
            pred_means=nima_pred_means,
            threshold_raw=threshold_info["threshold_raw"],
            threshold_mapped=threshold_info["threshold_mapped"],
            raw_lookup=raw_lookup,
        )
        nima_rows.append({
            "seed": seed,
            **threshold_info,
            **nima_metrics,
            "checkpoint": str(nima_ckpt),
        })

    main_stats = write_model_summary(
        rows=main_rows,
        output_root=main_output_root,
        title="Main Model PCCD Classification Results (Train Raw Median Threshold)",
    )
    nima_stats = write_model_summary(
        rows=nima_rows,
        output_root=nima_output_root,
        title="NIMA PCCD Classification Results (Train Raw Median Threshold)",
    )
    comparison_csv, comparison_md = write_comparison(main_stats, nima_stats, comparison_root)

    print(f"Main summary: {main_stats['summary_md']}")
    print(f"NIMA summary: {nima_stats['summary_md']}")
    print(f"Comparison CSV: {comparison_csv}")
    print(f"Comparison MD: {comparison_md}")


if __name__ == "__main__":
    main()
