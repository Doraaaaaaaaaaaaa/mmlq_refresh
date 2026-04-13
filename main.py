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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from model import ImprovedIAAModel
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
    三组参数：
      - Swin backbone        → 0.1 × base_lr
      - BERT（已冻结，不传） → 不参与优化
      - 其余（head, adapter, fusion, attr）→ base_lr
    """
    swin_ids = {id(p) for p in model.visual_encoder.swin.parameters()}

    swin_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) in swin_ids]
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in swin_ids]

    return AdamW(
        [
            {"params": swin_params,  "lr": base_lr * 0.1},
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


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 随机种子 ────────────────────────────────────────────────────────────
    seed = 42
    random.seed(seed);  np.random.seed(seed);  torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    # ── 配置 ────────────────────────────────────────────────────────────────
    with open("config.yml", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    accum_steps  = config.get("accum_steps", 4)
    base_lr      = config["lr"]
    lambda_mean  = config.get("lambda_mean", 0.2)
    lambda_rank  = config.get("lambda_rank", 0.1)

    # ── 保存目录 ─────────────────────────────────────────────────────────────
    os.makedirs("./save", exist_ok=True)
    save_name    = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir     = os.path.join("./save", save_name)
    pt_save_dir  = os.path.join(save_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(pt_save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, "results.txt")

    writer = SummaryWriter(log_dir=save_dir)
    copy("config.yml", save_dir)
    with open(results_path, "w") as f:
        f.write("epoch, time_elapsed, [mse, srcc, plcc, acc, emd1, emd2]\n")

    # ── 数据集 ───────────────────────────────────────────────────────────────
    train_dataset = AVADataset(config, "train")
    test_dataset  = AVADataset(config, "test")
    train_loader  = DataLoader(
        dataset=train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=config.get("num_workers", 8),
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config.get("num_workers", 8),
        pin_memory=True,
    )

    # ── 模型 ────────────────────────────────────────────────────────────────
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
        ckpt = torch.load(config["resume"], map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
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

        # ── 评估 ─────────────────────────────────────────────────────────────
        model.eval()
        y_pred, y_true = [], []

        with torch.no_grad():
            for samples in tqdm(test_loader, desc=f"Eval epoch {epoch}"):
                samples["image"] = samples["image"].to(device, non_blocking=True)
                samples["dos"]   = samples["dos"].to(device, non_blocking=True)

                with autocast("cuda", enabled=use_amp):
                    output = model(samples)

                y_pred.append(output.cpu())
                y_true.append(samples["dos"].cpu())

        y_pred = torch.cat(y_pred, dim=0).numpy()
        y_true = torch.cat(y_true, dim=0).numpy()
        metrics = cal_metrics(y_pred, y_true)
        # metrics = [mse, srcc, plcc, acc, emd1, emd2]
        mse, srcc, plcc, acc, emd1, emd2 = metrics

        elapsed = datetime.now() - start_time
        log_str = f"{elapsed} [{mse:.4f}, {srcc:.4f}, {plcc:.4f}, {acc:.2f}%, {emd1:.4f}, {emd2:.4f}]"
        print(f"Epoch {epoch} | {log_str}")

        with open(results_path, "a") as f:
            f.write(f"{epoch}, {log_str}\n")

        writer.add_scalar("srcc/test",  srcc, epoch)
        writer.add_scalar("plcc/test",  plcc, epoch)
        writer.add_scalar("mse/test",   mse,  epoch)
        writer.add_scalar("acc/test",   acc,  epoch)

        # ── 保存 checkpoint ───────────────────────────────────────────────────
        ckpt_data = {
            "epoch":              epoch,
            "model_state_dict":   model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_srcc":          best_srcc,
            "metrics":            metrics,
        }
        # 每个 epoch 保存
        torch.save(ckpt_data, os.path.join(pt_save_dir, f"checkpoint_{epoch}.pt"))

        # 保存最优模型
        if srcc > best_srcc:
            best_srcc = srcc
            ckpt_data["best_srcc"] = best_srcc
            torch.save(ckpt_data, os.path.join(pt_save_dir, "best_model.pt"))
            print(f"  ★ New best SRCC={best_srcc:.4f} → saved best_model.pt")


if __name__ == "__main__":
    main()
