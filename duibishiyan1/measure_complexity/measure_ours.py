from __future__ import annotations

import argparse
import types

import torch
import yaml

from common import ROOT, count_params, maybe_profile_macs, prepend_sys_path, print_result


def patch_frozen_backbones_for_adaptation_profile(model, config, device):
    """Keep trainable adapters/fusion modules while replacing frozen encoders."""
    batch_size = 1
    seq_len = config["max_caption_length"]

    def clip_forward(self, image, return_layers=(6, 12, 18, 24)):
        bsz = image.shape[0]
        return [
            torch.randn(bsz, 257, 1024, device=image.device)
            for _ in return_layers
        ]

    class DummyBertOutput:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states

    class DummyBert(torch.nn.Module):
        def forward(self, **tokens):
            input_ids = tokens["input_ids"]
            bsz, length = input_ids.shape
            hidden_states = tuple(
                torch.randn(bsz, length, 768, device=input_ids.device)
                for _ in range(13)
            )
            return DummyBertOutput(hidden_states)

    def parn_extract_attributes(self, image):
        bsz = image.shape[0]
        return (
            torch.randn(bsz, 2048, device=image.device),
            torch.randn(bsz, 11, device=image.device),
        )

    model.visual_encoder.clip.forward = types.MethodType(clip_forward, model.visual_encoder.clip)
    model.text_encoder.bert = DummyBert().to(device).eval()
    model.attribute_module.parn.extract_attributes = types.MethodType(
        parn_extract_attributes, model.attribute_module.parn
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "adaptation"], default="full")
    args = parser.parse_args()

    with prepend_sys_path(ROOT):
        from model import ImprovedIAAModel

    with open(ROOT / "config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImprovedIAAModel(config).to(device).eval()
    if args.mode == "adaptation":
        patch_frozen_backbones_for_adaptation_profile(model, config, device)

    image = torch.randn(1, 3, 224, 224, device=device)
    samples = {"image": image, "caption": ["a well composed photo"]}

    total, trainable = count_params(model)
    gmacs, note = maybe_profile_macs(model, (samples,))
    method = "LQ-ICIF (Ours)" if args.mode == "full" else "LQ-ICIF (Ours, adaptation)"
    if args.mode == "adaptation":
        note = "frozen CLIP/BERT/PARN backbone computations excluded" if note is None else f"{note}; frozen CLIP/BERT/PARN backbone computations excluded"
    print_result(method, total, trainable, gmacs, note)


if __name__ == "__main__":
    main()
