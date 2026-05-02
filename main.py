"""
训练主流程（改进版）

升级内容：
  1. AdamW + 分组学习率（backbone 用 0.1× lr，head 用 1× lr）
  2. 线性预热 + 余弦衰减调度器
  3. 梯度累积（accum_steps）
  4. 混合精度（torch.amp）
  5. 使用 CombinedLoss（EMD + MSE + Rank）
  6. 保存 best model（按 SRCC）
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from model import ImprovedIAAModel, AblationModel
import os
import math
import random
import numpy as np
from datetime import datetime
import yaml
from dataset import AVADataset
from loss import CombinedLoss, EMDLoss
from metrics import cal_metrics
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import ImageFile
import warnings
from shutil import copy

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings(action="ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# 分组学习率
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, base_lr, weight_decay=1e-4):
    """
    两组参数：
      - CLIP backbone        → 0.1 × base_lr
      - BERT（已冻结，不传） → 不参与优化
      - 其余（head, adapter, fusion, attr）→ base_lr
    """
    clip_ids = set()
    if hasattr(model, "visual_encoder") and hasattr(model.visual_encoder, "clip"):
        clip_ids = {id(p) for p in model.visual_encoder.clip.parameters()}

    clip_params  = [p for p in model.parameters()
                   if p.requires_grad and id(p) in clip_ids]
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in clip_ids]

    return AdamW(
        [
            {"params": clip_params,  "lr": base_lr * 0.1},
            {"params": other_params, "lr": base_lr},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, total_steps, warmup_ratio=0.05):
    """线性预热 → 余弦衰减"""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _trainable_model_state_dict(model):
    """只保存可训练参数；冻结骨干在初始化时从本地预训练权重重新加载。"""
    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def _model_state_dict_for_checkpoint(model, trainable_only=True):
    if trainable_only:
        return _trainable_model_state_dict(model)
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }


def _save_checkpoint(path, checkpoint):
    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def _build_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    best_srcc,
    metrics,
    trainable_only=True,
    include_optimizer=True,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": _model_state_dict_for_checkpoint(
            model, trainable_only=trainable_only
        ),
        "best_srcc": best_srcc,
        "metrics": metrics,
        "checkpoint_trainable_only": trainable_only,
    }
    if include_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    return checkpoint


def evaluate_model(model, loader, device, use_amp, desc):
    model.eval()
    y_pred, y_true = [], []

    with torch.no_grad():
        for samples in tqdm(loader, desc=desc):
            samples["image"] = samples["image"].to(device, non_blocking=True)
            samples["dos"] = samples["dos"].to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                output = model(samples)

            y_pred.append(output.cpu())
            y_true.append(samples["dos"].cpu())

    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_true = torch.cat(y_true, dim=0).numpy()
    return cal_metrics(y_pred, y_true)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="Path to YAML config file.")
    parser.add_argument("--resume", default=None, help="Override resume checkpoint path in the config.")
    parser.add_argument(
        "--ablation-mode",
        default=None,
        choices=["baseline", "text_only", "concat", "direct_ca", "icif", "icif_ha"],
        help="Override ablation_mode in the config.",
    )
    args = parser.parse_args()

    # ── 配置 ────────────────────────────────────────────────────────────────
    with open(args.config, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.ablation_mode is not None:
        config["ablation_mode"] = args.ablation_mode
    if args.resume is not None:
        config["resume"] = args.resume

    # ── 随机种子 ────────────────────────────────────────────────────────────
    seed = config.get("seed", 42)
    random.seed(seed);  np.random.seed(seed);  torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    accum_steps  = config.get("accum_steps", 4)
    base_lr      = config["lr"]
    lambda_mean  = config.get("lambda_mean", 0.2)
    lambda_rank  = config.get("lambda_rank", 0.1)

    # ── 保存目录 ─────────────────────────────────────────────────────────────
    save_root = config.get("save_root", "./save")
    os.makedirs(save_root, exist_ok=True)
    ablation_mode = config.get("ablation_mode", None)
    _suffix      = f"_abl-{ablation_mode}" if ablation_mode else ""
    save_name    = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + _suffix
    save_dir     = os.path.join(save_root, save_name)
    pt_save_dir  = os.path.join(save_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(pt_save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, "results.txt")
    save_latest = config.get("save_latest", True)
    save_best = config.get("save_best", True)
    save_every_epoch = config.get("save_every_epoch", False)
    checkpoint_trainable_only = config.get("checkpoint_trainable_only", True)
    best_include_optimizer = config.get("best_include_optimizer", False)

    writer = SummaryWriter(log_dir=save_dir)
    copy(args.config, os.path.join(save_dir, os.path.basename(args.config)))
    with open(os.path.join(save_dir, "effective_config.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    use_val_split = config.get("use_val_split", False)
    selection_split = config.get("selection_split", "val" if use_val_split else "test")
    if selection_split not in ["val", "test"]:
        raise ValueError(f"Unsupported selection_split: {selection_split}")

    with open(results_path, "w") as f:
        f.write(
            f"epoch, time_elapsed, split={selection_split}, "
            "[mse, srcc, plcc, acc, emd1, emd2]\n"
        )

    # ── 数据集 ───────────────────────────────────────────────────────────────
    train_dataset = AVADataset(config, "train")
    train_loader  = DataLoader(
        dataset=train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=config.get("num_workers", 8),
        pin_memory=True,
    )
    val_loader = None
    if use_val_split:
        val_dataset = AVADataset(config, "val")
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=config["batch_size"],
            shuffle=False, num_workers=config.get("num_workers", 8),
            pin_memory=True,
        )
    test_dataset  = AVADataset(config, "test")
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config.get("num_workers", 8),
        pin_memory=True,
    )
    selection_loader = val_loader if selection_split == "val" else test_loader
    if selection_loader is None:
        raise ValueError("selection_split='val' requires use_val_split=True")

    # ── 模型 ────────────────────────────────────────────────────────────────
    if config.get("ablation_mode"):
        model = AblationModel(config).to(device)
        print(f"[Ablation] mode = {config['ablation_mode']}")
    else:
        model = ImprovedIAAModel(config).to(device)

    # ── 优化器 / 调度器 ───────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps     = steps_per_epoch * config["epochs"]
    optimizer       = build_optimizer(model, base_lr)
    scheduler       = build_scheduler(optimizer, total_steps, warmup_ratio=0.05)
    scaler          = GradScaler(enabled=use_amp)

    # ── 损失函数 ─────────────────────────────────────────────────────────────
    criterion = CombinedLoss(
        dist_r=2,
        lambda_mean=lambda_mean,
        lambda_rank=lambda_rank,
    ).to(device)
    emd_eval = EMDLoss(dist_r=2)   # 纯 EMD，用于评估日志

    # ── 恢复断点 ─────────────────────────────────────────────────────────────
    start_epoch = 0
    best_srcc   = -1.0
    if config.get("resume"):
        ckpt = torch.load(config["resume"], map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError as e:
                print(f"[Resume] skip optimizer state: {e}")
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except ValueError as e:
                print(f"[Resume] skip scheduler state: {e}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_srcc   = ckpt.get("best_srcc", -1.0)
        print(f"[Resume] from epoch {start_epoch}, best_srcc={best_srcc:.4f}")

    opt_step    = 0
    start_time  = datetime.now()

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_losses = []

        for batch_idx, samples in enumerate(train_loader):
            samples["image"] = samples["image"].to(device, non_blocking=True)
            samples["dos"]   = samples["dos"].to(device, non_blocking=True)
            # caption 是字符串列表，无需 .to(device)

            with autocast("cuda", enabled=use_amp):
                output = model(samples)
                loss   = criterion(output, samples["dos"]) / accum_steps

            scaler.scale(loss).backward()

            # 梯度累积
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                opt_step += 1

            batch_losses.append(loss.item() * accum_steps)

            if batch_idx % 50 == 0:
                lr_now = optimizer.param_groups[-1]["lr"]
                print(f"Epoch {epoch} | Iter {batch_idx}/{len(train_loader)} "
                      f"| Loss={loss.item()*accum_steps:.4f} | LR={lr_now:.2e}")

        avg_loss = sum(batch_losses) / len(batch_losses)
        writer.add_scalar("loss/train", avg_loss, epoch)
        print(f"Epoch {epoch} | AvgLoss={avg_loss:.4f}")

        # ── 验证/测试集选择评估 ─────────────────────────────────────────────
        metrics = evaluate_model(
            model, selection_loader, device, use_amp,
            desc=f"Eval {selection_split} epoch {epoch}",
        )
        # metrics = [mse, srcc, plcc, acc, emd1, emd2]
        mse, srcc, plcc, acc, emd1, emd2 = metrics

        elapsed = datetime.now() - start_time
        log_str = f"{elapsed} [{mse:.4f}, {srcc:.4f}, {plcc:.4f}, {acc:.2f}%, {emd1:.4f}, {emd2:.4f}]"
        print(f"Epoch {epoch} | {selection_split} | {log_str}")

        with open(results_path, "a") as f:
            f.write(f"{epoch}, {log_str}\n")

        writer.add_scalar(f"srcc/{selection_split}",  srcc, epoch)
        writer.add_scalar(f"plcc/{selection_split}",  plcc, epoch)
        writer.add_scalar(f"mse/{selection_split}",   mse,  epoch)
        writer.add_scalar(f"acc/{selection_split}",   acc,  epoch)

        # ── 保存 checkpoint ───────────────────────────────────────────────────
        is_best = srcc > best_srcc
        if is_best:
            best_srcc = srcc

        ckpt_data = _build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_srcc=best_srcc,
            metrics=metrics,
            trainable_only=checkpoint_trainable_only,
            include_optimizer=True,
        )

        # latest.pt 每个 epoch 覆盖，避免 checkpoint_0/1/2... 堆满磁盘。
        if save_latest:
            _save_checkpoint(os.path.join(pt_save_dir, "latest.pt"), ckpt_data)

        # 只有显式开启时才保留每个 epoch 的历史 checkpoint。
        if save_every_epoch:
            _save_checkpoint(
                os.path.join(pt_save_dir, f"checkpoint_{epoch}.pt"),
                ckpt_data,
            )

        # 保存最优模型；默认不带 optimizer，供评估/推理或后续微调加载。
        if save_best and is_best:
            best_ckpt_data = _build_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_srcc=best_srcc,
                metrics=metrics,
                trainable_only=checkpoint_trainable_only,
                include_optimizer=best_include_optimizer,
            )
            _save_checkpoint(os.path.join(pt_save_dir, "best_model.pt"), best_ckpt_data)
            print(f"  ★ New best SRCC={best_srcc:.4f} → saved best_model.pt")

    if use_val_split:
        best_path = os.path.join(pt_save_dir, "best_model.pt")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        test_metrics = evaluate_model(
            model, test_loader, device, use_amp, desc="Final test"
        )
        mse, srcc, plcc, acc, emd1, emd2 = test_metrics
        test_log = (
            f"[{mse:.4f}, {srcc:.4f}, {plcc:.4f}, "
            f"{acc:.2f}%, {emd1:.4f}, {emd2:.4f}]"
        )
        print(f"Final test selected_by={selection_split} | {test_log}")
        with open(os.path.join(save_dir, "final_test_results.txt"), "w") as f:
            f.write("selected_by, [mse, srcc, plcc, acc, emd1, emd2]\n")
            f.write(f"{selection_split}, {test_log}\n")
        writer.add_scalar("srcc/final_test", srcc, config["epochs"])
        writer.add_scalar("plcc/final_test", plcc, config["epochs"])
        writer.add_scalar("mse/final_test", mse, config["epochs"])
        writer.add_scalar("acc/final_test", acc, config["epochs"])

    writer.close()


if __name__ == "__main__":
    main()
