"""
compute_complexity.py — LQ-ICIF 模型复杂度测量
输出：逐模块参数量、总参数/可训练参数、GMACs（供 Table IV 填写）

依赖：
    pip install thop torchinfo
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import yaml

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("[Warning] thop 未安装，运行: pip install thop")

try:
    from torchinfo import summary as torchinfo_summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False
    print("[Warning] torchinfo 未安装，运行: pip install torchinfo")

from model import ImprovedIAAModel


# ─────────────────────────────────────────────────────────────────────────────
# Profiling 包装器
# 将 TextEncoder 内部的 tokenizer 调用绕过，接受预分词的 tensor 输入，
# 使 thop / torchinfo 可以正常 trace 整个前向图。
# ─────────────────────────────────────────────────────────────────────────────

class _ProfilingWrapper(nn.Module):
    def __init__(self, model: ImprovedIAAModel):
        super().__init__()
        self.m = model

    def forward(self, image, input_ids, attention_mask):
        """
        image          : (B, 3, 224, 224)  float
        input_ids      : (B, seq_len)      long
        attention_mask : (B, seq_len)      long
        """
        B = image.shape[0]

        # Stage 1: 视觉
        V_list = self.m.visual_encoder(image)

        # Stage 1: 文本 —— 直接调 BERT，绕过内部 tokenizer
        # 注意：此处故意不用 torch.no_grad()，确保 profiling hook 能正常计数
        bert_out = self.m.text_encoder.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = bert_out.hidden_states          # 13 × (B, seq, 768)
        T_list = []
        for i, layer_idx in enumerate(self.m.text_encoder.LAYER_INDICES):
            h = hidden_states[layer_idx]
            T_list.append(h + self.m.text_encoder.adapters[i](h))

        pad_mask = (attention_mask == 0)                # (B, seq)

        # Stage 2: 弱交互式融合
        all_VQ, all_TQ = [], []
        for l in range(4):
            VQ = self.m.visual_queries[l].expand(B, -1, -1)
            TQ = self.m.text_queries[l].expand(B, -1, -1)
            VQ, TQ = self.m.fusion_layers[l](
                VQ, TQ, V_list[l], T_list[l], text_pad_mask=pad_mask
            )
            all_VQ.append(VQ)
            all_TQ.append(TQ)

        # Stage 3: 层级聚合 + 属性推理
        VQ_global = self.m.visual_hier_agg(all_VQ)
        TQ_global = self.m.text_hier_agg(all_TQ)
        F_cat = torch.cat([VQ_global.unsqueeze(1),
                           TQ_global.unsqueeze(1)], dim=1)
        F_hat = self.m.attribute_module(image, F_cat)

        # Stage 4: 预测头
        fv   = F_hat[:, 0, :]
        ft   = F_hat[:, 1, :]
        feat = self.m.pred_proj(torch.cat([fv, ft], dim=-1))
        return self.m.softmax(self.m.pred_head(feat))


# ─────────────────────────────────────────────────────────────────────────────
# 参数量统计
# ─────────────────────────────────────────────────────────────────────────────

def _param_stats(module):
    """返回 (total, trainable) 参数量"""
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_param_breakdown(model: ImprovedIAAModel):
    # 可学习 Query 是 nn.ParameterList，单独统计
    query_params = sum(p.numel() for p in model.visual_queries) \
                 + sum(p.numel() for p in model.text_queries)

    rows = [
        ("VisualEncoder  (CLIP ViT-L/14, frozen)", _param_stats(model.visual_encoder)),
        ("TextEncoder    (BERT-base, frozen + Adapters)", _param_stats(model.text_encoder)),
        ("Learnable Queries (×4 layers, ×2 modalities)", (query_params, query_params)),
        ("Fusion Layers  (WeakInteractionFusion ×4)", _param_stats(model.fusion_layers)),
        ("Visual HierAgg (HierarchicalAttentionAggregator)", _param_stats(model.visual_hier_agg)),
        ("Text HierAgg   (HierarchicalAttentionAggregator)", _param_stats(model.text_hier_agg)),
        ("Attribute Module (PARN + CrossAttn)", _param_stats(model.attribute_module)),
        ("Pred Head      (Linear × 2)", (
            sum(p.numel() for p in model.pred_proj.parameters()) +
            sum(p.numel() for p in model.pred_head.parameters()),
            sum(p.numel() for p in model.pred_proj.parameters() if p.requires_grad) +
            sum(p.numel() for p in model.pred_head.parameters() if p.requires_grad),
        )),
    ]

    print(f"\n{'Module':<52} {'Total(M)':>9} {'Train(M)':>9} {'Frozen(M)':>9}")
    print("─" * 82)
    for name, (tot, trn) in rows:
        frz = tot - trn
        print(f"{name:<52} {tot/1e6:>9.2f} {trn/1e6:>9.2f} {frz/1e6:>9.2f}")

    print("─" * 82)
    t_all, tr_all = _param_stats(model)
    # 加上 query params（ParameterList 会被 model.parameters() 覆盖，不需重加）
    frz_all = t_all - tr_all
    print(f"{'TOTAL':<52} {t_all/1e6:>9.2f} {tr_all/1e6:>9.2f} {frz_all/1e6:>9.2f}")
    print()
    return t_all, tr_all


# ─────────────────────────────────────────────────────────────────────────────
# GMACs 测量
# ─────────────────────────────────────────────────────────────────────────────

def measure_macs(model: ImprovedIAAModel, device: str, text_len: int = 77):
    """
    text_len=77: CLIP 标准文本长度，与 CLIP-DQA V2 Table IV 保持一致，
                 便于横向对比。若需用实际平均评论长度可改为 256。
    """
    wrapper = _ProfilingWrapper(model).to(device).eval()

    dummy_image = torch.randn(1, 3, 224, 224, device=device)
    dummy_ids   = torch.randint(0, 30522, (1, text_len), device=device)
    dummy_mask  = torch.ones(1, text_len, dtype=torch.long, device=device)

    gmacs_thop = None
    gmacs_ti   = None

    if HAS_THOP:
        try:
            macs, _ = thop_profile(
                wrapper,
                inputs=(dummy_image, dummy_ids, dummy_mask),
                verbose=False,
            )
            gmacs_thop = macs / 1e9
            print(f"  [thop]      GMACs = {gmacs_thop:.2f}  (text_len={text_len})")
        except Exception as e:
            print(f"  [thop] 失败: {e}")

    if HAS_TORCHINFO:
        try:
            result = torchinfo_summary(
                wrapper,
                input_data=(dummy_image, dummy_ids, dummy_mask),
                verbose=0,
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
            )
            gmacs_ti = result.total_mult_adds / 1e9
            print(f"  [torchinfo] GMACs = {gmacs_ti:.2f}  (text_len={text_len})")
        except Exception as e:
            print(f"  [torchinfo] 失败: {e}")

    if gmacs_thop is None and gmacs_ti is None:
        print("  [!] 两种工具均失败，请检查安装。")

    return gmacs_thop, gmacs_ti


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 82)
    print("  LQ-ICIF Model Complexity Analysis  (for Table IV)")
    print("=" * 82)
    print(f"  Device: {device}")

    model = ImprovedIAAModel(config).to(device).eval()

    # 1. 参数量
    print("\n── 1. Parameter Breakdown ──────────────────────────────────────────────────")
    total_params, trainable_params = print_param_breakdown(model)

    # 2. GMACs
    print("── 2. Computational Cost (GMACs) ───────────────────────────────────────────")
    gmacs_thop, gmacs_ti = measure_macs(model, device, text_len=77)

    # 3. 汇总（直接用于填表）
    gmacs = gmacs_ti if gmacs_ti is not None else gmacs_thop
    print()
    print("── 3. 论文 Table IV 填表数据 ────────────────────────────────────────────────")
    print(f"  Trainable Params : {trainable_params / 1e6:.1f} M")
    print(f"  Total Params     : {total_params / 1e6:.1f} M")
    if gmacs is not None:
        print(f"  GMACs            : {gmacs:.2f}")
    else:
        print("  GMACs            : 见上方输出")
    print()
    print("  注：对比方法（NIMA / AMM-Net / MMLQ 等）的参数量和 GMACs")
    print("      建议通过各自官方代码同样用此方式测量，保证测量条件一致。")
    print("=" * 82)


if __name__ == "__main__":
    main()