from __future__ import annotations

import torch
import yaml
import os

from common import BASELINES, count_params, maybe_profile_macs, prepend_sys_path, print_result, pushd


CHARM_DIR = BASELINES / "Charm-main" / "Charm-main"


def install_random_init_transformers_patch() -> None:
    """Build HF backbones from config so complexity can be measured offline."""
    from transformers import (
        AutoConfig,
        AutoModel,
        Dinov2Config,
        Dinov2Model,
        SwinConfig,
        SwinForImageClassification,
        SwinModel,
        ViTConfig,
        ViTModel,
    )

    def dinov2_small_config() -> Dinov2Config:
        return Dinov2Config(
            image_size=224,
            patch_size=14,
            num_channels=3,
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
        )

    def config_for_name(name: str):
        try:
            return AutoConfig.from_pretrained(name, local_files_only=True)
        except Exception:
            if "dinov2-small" in name:
                return dinov2_small_config()
            if "swin" in name:
                return SwinConfig(image_size=224, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24])
            return ViTConfig(image_size=224, patch_size=16, hidden_size=384, num_hidden_layers=12, num_attention_heads=6, intermediate_size=1536)

    def auto_from_pretrained(name: str, *args, **kwargs):
        cfg = config_for_name(name)
        if isinstance(cfg, Dinov2Config):
            return Dinov2Model(cfg)
        return AutoModel.from_config(cfg)

    def vit_from_pretrained(name: str, *args, **kwargs):
        return ViTModel(config_for_name(name))

    def swin_from_pretrained(name: str, *args, **kwargs):
        return SwinModel(config_for_name(name))

    def swin_cls_from_pretrained(name: str, *args, **kwargs):
        return SwinForImageClassification(config_for_name(name))

    AutoModel.from_pretrained = auto_from_pretrained
    ViTModel.from_pretrained = vit_from_pretrained
    SwinModel.from_pretrained = swin_from_pretrained
    SwinForImageClassification.from_pretrained = swin_cls_from_pretrained


def main() -> None:
    import ml_collections

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with prepend_sys_path(CHARM_DIR), pushd(CHARM_DIR):
        install_random_init_transformers_patch()
        from model import Model
        from utils import prepare_config

        with open("config.yaml", "r", encoding="utf-8") as f:
            config = ml_collections.ConfigDict(yaml.safe_load(f))

        config.data.dataset = "ava"
        config.data.patch_selection = "original"
        config.model.num_classes = 10
        if "flexiViT" not in config.model:
            config.model.flexiViT = {"enable": False}
        config = prepare_config(config)

        model = Model(config).to(device).eval()
        image = torch.randn(1, 3, 224, 224, device=device)
        pos_embeds = torch.zeros(1, 1, dtype=torch.long, device=device)
        masks = torch.zeros(1, 1, dtype=torch.long, device=device)

        total, trainable = count_params(model)
        gmacs, note = maybe_profile_macs(model, (image, pos_embeds, masks))
        print_result("Charm", total, trainable, gmacs, note)


if __name__ == "__main__":
    main()
