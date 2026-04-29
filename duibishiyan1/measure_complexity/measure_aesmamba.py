from __future__ import annotations

import argparse
from contextlib import contextmanager
import sys
import types

import torch

from common import BASELINES, count_params, maybe_profile_macs, prepend_sys_path, print_result, pushd


AESMAMBA_ROOT = BASELINES / "AesMamba-main" / "AesMamba-main"


def install_mamba_ssm_stub_if_missing() -> bool:
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    def selective_scan_stub(xs, *args, **kwargs):
        return torch.zeros_like(xs)

    mamba_pkg = types.ModuleType("mamba_ssm")
    ops_pkg = types.ModuleType("mamba_ssm.ops")
    scan_mod = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    scan_mod.selective_scan_fn = selective_scan_stub
    scan_mod.selective_scan_ref = selective_scan_stub

    sys.modules["mamba_ssm"] = mamba_pkg
    sys.modules["mamba_ssm.ops"] = ops_pkg
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = scan_mod
    return True


def install_mmcv_stub_if_missing() -> bool:
    try:
        from mmcv.cnn import ConvModule  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    class ConvModule(torch.nn.Sequential):
        def __init__(self, in_channels, out_channels, kernel_size=1, norm_cfg=None, act_cfg=dict(type="ReLU"), **kwargs):
            layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)]
            if norm_cfg is not None:
                layers.append(torch.nn.BatchNorm2d(out_channels))
            if act_cfg is not None:
                layers.append(torch.nn.ReLU(inplace=True))
            super().__init__(*layers)

    mmcv_pkg = types.ModuleType("mmcv")
    cnn_pkg = types.ModuleType("mmcv.cnn")
    cnn_pkg.ConvModule = ConvModule
    sys.modules["mmcv"] = mmcv_pkg
    sys.modules["mmcv.cnn"] = cnn_pkg
    return True


def install_bert_random_init_patch() -> None:
    from transformers import BertConfig, BertModel, BertTokenizer

    original_model_from_pretrained = BertModel.from_pretrained
    original_tokenizer_from_pretrained = BertTokenizer.from_pretrained

    class SimpleBatch(dict):
        def to(self, device):
            return SimpleBatch({key: value.to(device) for key, value in self.items()})

    class SimpleTokenizer:
        def __call__(self, text, padding=True, truncation=True, return_tensors="pt"):
            batch_size = len(text) if isinstance(text, list) else 1
            seq_len = 8
            return SimpleBatch({
                "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
                "token_type_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            })

    def model_from_pretrained(name, *args, **kwargs):
        try:
            return original_model_from_pretrained(name, *args, **kwargs)
        except Exception:
            return BertModel(BertConfig())

    def tokenizer_from_pretrained(name, *args, **kwargs):
        try:
            return original_tokenizer_from_pretrained(name, *args, **kwargs)
        except Exception:
            return SimpleTokenizer()

    BertModel.from_pretrained = model_from_pretrained
    BertTokenizer.from_pretrained = tokenizer_from_pretrained


def append_note(note: str | None, extra: str | None) -> str | None:
    if not extra:
        return note
    if not note:
        return extra
    return f"{note}; {extra}"


@contextmanager
def allow_missing_vmamba_checkpoint():
    original_load = torch.load
    missing = {"used": False}

    def patched_load(f, *args, **kwargs):
        if isinstance(f, str) and f.endswith((".pth", ".pt")):
            try:
                return original_load(f, *args, **kwargs)
            except FileNotFoundError:
                if "vmamba" in f or "vssm" in f:
                    missing["used"] = True
                    return {"model": {}}
                raise
        return original_load(f, *args, **kwargs)

    torch.load = patched_load
    try:
        yield missing
    finally:
        torch.load = original_load


def measure_viaa() -> None:
    workdir = AESMAMBA_ROOT / "AesMamba_v"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with prepend_sys_path(workdir), pushd(workdir):
        stubbed_mamba = install_mamba_ssm_stub_if_missing()
        from models.AesMamba import AesMamba_v

        with allow_missing_vmamba_checkpoint() as missing_ckpt:
            model = AesMamba_v(type="vmamba_tiny", dataset="AVA").to(device).eval()
        image = torch.randn(1, 3, 224, 224, device=device)
        total, trainable = count_params(model)
        gmacs, note = maybe_profile_macs(model, (image,))
        note = append_note(note, "mamba_ssm missing: selective_scan op was stubbed, so GMACs exclude it" if stubbed_mamba else None)
        note = append_note(note, "VMamba checkpoint missing: backbone randomly initialized for complexity measurement" if missing_ckpt["used"] else None)
        print_result("AesMamba-V", total, trainable, gmacs, note)


def measure_miaa() -> None:
    workdir = AESMAMBA_ROOT / "AesMamba_m"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with prepend_sys_path(workdir), pushd(workdir):
        stubbed_mamba = install_mamba_ssm_stub_if_missing()
        stubbed_mmcv = install_mmcv_stub_if_missing()
        install_bert_random_init_patch()
        from models.AesMamba_p import AesMamba_p

        with allow_missing_vmamba_checkpoint() as missing_ckpt:
            model = AesMamba_p("vmamba_tiny", "AVA", device).to(device).eval()
        image = torch.randn(1, 3, 224, 224, device=device)
        text = ["a well composed photo"]
        total, trainable = count_params(model)
        gmacs, note = maybe_profile_macs(model, (image, text))
        note = append_note(note, "mamba_ssm missing: selective_scan op was stubbed, so GMACs exclude it" if stubbed_mamba else None)
        note = append_note(note, "mmcv missing: ConvModule compatibility stub installed" if stubbed_mmcv else None)
        note = append_note(note, "BERT checkpoint missing: text backbone randomly initialized for complexity measurement")
        note = append_note(note, "VMamba checkpoint missing: backbone randomly initialized for complexity measurement" if missing_ckpt["used"] else None)
        print_result("AesMamba-M", total, trainable, gmacs, note)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["viaa", "miaa"], default="viaa")
    args = parser.parse_args()
    if args.variant == "viaa":
        measure_viaa()
    else:
        measure_miaa()


if __name__ == "__main__":
    main()
