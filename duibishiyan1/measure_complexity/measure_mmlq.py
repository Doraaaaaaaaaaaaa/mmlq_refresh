from __future__ import annotations

import argparse
import importlib.util
import sys
import types

import torch
import yaml

from common import BASELINES, ROOT, count_params, maybe_profile_macs, prepend_sys_path, print_result


MMLQ_DIR = BASELINES / "mmlq_iaa-master"
LAVIS_DIR = ROOT.parent / "LAVIS"


def install_moviepy_editor_compat() -> None:
    """LAVIS imports moviepy.editor, which was removed in MoviePy 2.x."""
    try:
        import moviepy.editor  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    try:
        from moviepy import VideoFileClip
    except Exception:
        return

    import sys
    import types

    editor = types.ModuleType("moviepy.editor")
    editor.VideoFileClip = VideoFileClip
    sys.modules["moviepy.editor"] = editor


def load_module(name: str, path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_lavis_minimal_modules() -> None:
    """Load only the LAVIS modules used by MMLQ, avoiding full LAVIS imports."""
    lavis_pkg = types.ModuleType("lavis")
    lavis_pkg.__path__ = [str(LAVIS_DIR / "lavis")]
    common_pkg = types.ModuleType("lavis.common")
    common_pkg.__path__ = [str(LAVIS_DIR / "lavis" / "common")]
    models_pkg = types.ModuleType("lavis.models")
    models_pkg.__path__ = [str(LAVIS_DIR / "lavis" / "models")]

    sys.modules["lavis"] = lavis_pkg
    sys.modules["lavis.common"] = common_pkg
    sys.modules["lavis.models"] = models_pkg

    load_module("lavis.common.dist_utils", LAVIS_DIR / "lavis" / "common" / "dist_utils.py")
    load_module("lavis.models.eva_vit", LAVIS_DIR / "lavis" / "models" / "eva_vit.py")
    load_module("lavis.models.clip_vit", LAVIS_DIR / "lavis" / "models" / "clip_vit.py")


def patch_frozen_backbones_for_adaptation_profile(model, config):
    """Keep MMLQ query/fusion layers while excluding frozen encoders."""
    def vision_forward(self, images):
        bsz = images.shape[0]
        return torch.randn(bsz, 257, config["vision_feats_size"], device=images.device)

    def caption_forward(self, captions):
        bsz = len(captions)
        seq_len = config["max_caption_length"]
        device = next(model.parameters()).device
        feats = torch.randn(bsz, seq_len, config["caption_feats_size"], device=device)
        attn_mask = torch.zeros(bsz, 1, 1, seq_len, device=device)
        return feats, attn_mask

    model.vision_encoder.encoder.forward = types.MethodType(
        vision_forward, model.vision_encoder.encoder
    )
    model.caption_encoder.forward = types.MethodType(
        caption_forward, model.caption_encoder
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "adaptation"], default="full")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The author-provided MMLQ code hardcodes token tensors to cuda. "
            "Run this adapter in a CUDA environment."
        )

    with prepend_sys_path(LAVIS_DIR), prepend_sys_path(MMLQ_DIR):
        install_moviepy_editor_compat()
        install_lavis_minimal_modules()
        from model import MultiModalQueryNetwork

        with open(MMLQ_DIR / "config.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model = MultiModalQueryNetwork(config).cuda().eval().float()
        if args.mode == "adaptation":
            patch_frozen_backbones_for_adaptation_profile(model, config)
        samples = {
            "image": torch.randn(1, 3, 224, 224, device="cuda"),
            "caption": ["a well composed photo"],
        }

        total, trainable = count_params(model)
        gmacs, note = maybe_profile_macs(model, (samples,))
        method = "MMLQ" if args.mode == "full" else "MMLQ (adaptation)"
        if args.mode == "adaptation":
            note = "frozen vision/text backbone computations excluded" if note is None else f"{note}; frozen vision/text backbone computations excluded"
        print_result(method, total, trainable, gmacs, note)


if __name__ == "__main__":
    main()
