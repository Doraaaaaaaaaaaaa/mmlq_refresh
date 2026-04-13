"""
复合损失函数
  1. EMD Loss（主损失，dist_r=2）
  2. MSE 均值损失（辅助，weight=lambda_mean）
  3. Pairwise Ranking Loss（辅助，weight=lambda_rank）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMDLoss(nn.Module):
    """Earth Mover's Distance loss，支持 r=1 或 r=2"""
    def __init__(self, dist_r=2):
        super().__init__()
        self.dist_r = dist_r

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        cdf_target   = torch.cumsum(p_target,   dim=1)
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        if self.dist_r == 2:
            samplewise = torch.sqrt(torch.mean(cdf_diff.pow(2), dim=1))
        else:
            samplewise = torch.mean(cdf_diff.abs(), dim=1)
        return samplewise.mean()


def pairwise_rank_loss(pred_mean, gt_mean, margin=0.2):
    """
    Batch-wise pairwise margin ranking loss。
    对所有满足 gt[i] > gt[j] 的对，惩罚 pred[i] - pred[j] < margin。
    """
    diff_pred = pred_mean.unsqueeze(1) - pred_mean.unsqueeze(0)   # (B, B)
    diff_gt   = gt_mean.unsqueeze(1)   - gt_mean.unsqueeze(0)     # (B, B)
    mask      = (diff_gt > 0).float()
    rank_loss = F.relu(margin - diff_pred) * mask
    n_pairs   = mask.sum().clamp(min=1)
    return rank_loss.sum() / n_pairs


class CombinedLoss(nn.Module):
    """
    总损失 = EMD + lambda_mean * MSE(均值) + lambda_rank * RankLoss
    bins: 评分区间数（AVA 为 10）
    """
    def __init__(self, dist_r=2, lambda_mean=0.2, lambda_rank=0.1, bins=10):
        super().__init__()
        self.emd         = EMDLoss(dist_r=dist_r)
        self.lambda_mean = lambda_mean
        self.lambda_rank = lambda_rank
        # 用于计算均值分数的权重向量（1~bins）
        self.register_buffer(
            "bins_t",
            torch.arange(1, bins + 1, dtype=torch.float32)
        )

    def forward(self, pred, target):
        """
        pred   : (B, bins) — Softmax 输出的概率分布
        target : (B, bins) — 归一化的 ground-truth 分布
        """
        loss_emd = self.emd(target, pred)

        pred_mean = (pred   * self.bins_t).sum(dim=1)   # (B,)
        gt_mean   = (target * self.bins_t).sum(dim=1)   # (B,)

        loss_mean = F.mse_loss(pred_mean, gt_mean)
        loss_rank = pairwise_rank_loss(pred_mean, gt_mean)

        return (loss_emd
                + self.lambda_mean * loss_mean
                + self.lambda_rank * loss_rank)
