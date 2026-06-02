from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
METRIC_NAMES = ["mse", "srcc", "plcc", "acc", "emd1", "emd2"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated random-split PCCD experiments."
    )
    parser.add_argument("--base-config", default="config.yml")
    parser.add_argument(
        "--data-root",
        default="/root/autodl-tmp/PCCD/PCCD_AVA_Format",
        help="Converted PCCD root containing images, AVA_Comments_Full.pkl, and CSVs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Experiment output root. Defaults to /root/autodl-tmp/pccd_repeated_712_<timestamp>.",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def link_or_copy(source: Path, target: Path):
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source, target_is_directory=source.is_dir())


def load_all_rows(data_root: Path) -> pd.DataFrame:
    train_csv = data_root / "train_pccd.csv"
    test_csv = data_root / "test_pccd.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Expected train_pccd.csv and test_pccd.csv under {data_root}"
        )
    rows = pd.concat(
        [pd.read_csv(train_csv), pd.read_csv(test_csv)],
        ignore_index=True,
    )
    return rows.sort_values("image_id").reset_index(drop=True)


def write_split(rows: pd.DataFrame, split_dir: Path, data_root: Path, seed: int, ratios):
    train_ratio, val_ratio, test_ratio = ratios
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    n_total = len(rows)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))

    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    split_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(data_root / "images", split_dir / "images")
    link_or_copy(data_root / "AVA_Comments_Full.pkl", split_dir / "AVA_Comments_Full.pkl")

    rows.iloc[train_idx].to_csv(split_dir / "train_pccd.csv", index=False)
    rows.iloc[val_idx].to_csv(split_dir / "val_pccd.csv", index=False)
    rows.iloc[test_idx].to_csv(split_dir / "test_pccd.csv", index=False)

    return {
        "seed": seed,
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
    }


def parse_final_metrics(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"No final metrics found in {path}")
    values = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", lines[-1])]
    if len(values) < 6:
        raise ValueError(f"Could not parse six metrics from {path}: {lines[-1]}")
    return dict(zip(METRIC_NAMES, values[-6:]))


def latest_child(path: Path) -> Path:
    children = [item for item in path.iterdir() if item.is_dir()]
    if not children:
        raise FileNotFoundError(f"No run directory found under {path}")
    return max(children, key=lambda item: item.stat().st_mtime)


def write_summary(rows, output_root: Path):
    summary_csv = output_root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "seed", "train", "val", "test", "run_dir",
            *METRIC_NAMES,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    metric_matrix = np.array([[row[name] for name in METRIC_NAMES] for row in rows])
    means = metric_matrix.mean(axis=0)
    stds = metric_matrix.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros_like(means)

    summary_md = output_root / "summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# PCCD Repeated Random Split Results\n\n")
        f.write(f"Runs: {len(rows)}\n\n")
        f.write("| Metric | Mean | Std |\n")
        f.write("|---|---:|---:|\n")
        for name, mean, std in zip(METRIC_NAMES, means, stds):
            f.write(f"| {name.upper()} | {mean:.4f} | {std:.4f} |\n")


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_root = Path(
        args.output_root or f"/root/autodl-tmp/pccd_repeated_712_{timestamp}"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with open(ROOT / args.base_config, encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    all_rows = load_all_rows(data_root)
    split_rows = []
    result_rows = []

    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Total samples: {len(all_rows)}")
    print(f"Seeds: {args.seeds}")
    print(
        "Ratios: "
        f"{args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}"
    )
    print(f"Epochs per split: {args.epochs}")

    for seed in args.seeds:
        split_dir = output_root / "splits" / f"seed_{seed:02d}"
        split_info = write_split(
            all_rows,
            split_dir,
            data_root,
            seed,
            ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        )
        split_rows.append(split_info)

        run_root = output_root / "runs" / f"seed_{seed:02d}"
        run_root.mkdir(parents=True, exist_ok=True)
        config = dict(base_config)
        config.update({
            "dataset": "pccd",
            "ava_dataset_dir": str(split_dir),
            "seed": seed,
            "epochs": args.epochs,
            "use_val_split": True,
            "selection_split": "val",
            "save_root": str(run_root),
            "save_latest": False,
            "save_best": True,
            "save_every_epoch": False,
            "checkpoint_trainable_only": True,
            "best_include_optimizer": False,
            "resume": "",
        })
        if args.num_workers is not None:
            config["num_workers"] = args.num_workers

        config_path = output_root / "configs" / f"seed_{seed:02d}.yml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

        print(
            f"\n=== Seed {seed} | "
            f"train={split_info['train']} val={split_info['val']} test={split_info['test']} ==="
        )
        print(f"Config: {config_path}")
        if args.dry_run:
            continue

        subprocess.run(
            [sys.executable, "main.py", "--config", str(config_path)],
            cwd=ROOT,
            check=True,
        )

        run_dir = latest_child(run_root)
        metrics = parse_final_metrics(run_dir / "final_test_results.txt")
        result_row = {
            **split_info,
            "run_dir": str(run_dir),
            **metrics,
        }
        result_rows.append(result_row)
        write_summary(result_rows, output_root)
        print(f"Seed {seed} final test: {metrics}")

    if not args.dry_run:
        write_summary(result_rows, output_root)
        print(f"\nSummary written to: {output_root / 'summary.md'}")
        print(f"Raw CSV written to: {output_root / 'summary.csv'}")
    else:
        pd.DataFrame(split_rows).to_csv(output_root / "dry_run_splits.csv", index=False)
        print(f"\nDry-run split summary written to: {output_root / 'dry_run_splits.csv'}")


if __name__ == "__main__":
    main()
