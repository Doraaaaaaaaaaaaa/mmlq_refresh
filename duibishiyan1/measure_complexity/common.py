from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "duibishiyan1"


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@contextlib.contextmanager
def prepend_sys_path(path: Path):
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def count_params(model: Any) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_millions(value: int) -> str:
    return f"{value / 1e6:.3f}M"


def maybe_profile_macs(
    model: Any,
    inputs: tuple[Any, ...],
    custom_ops: dict[Any, Callable] | None = None,
) -> tuple[float | None, str | None]:
    try:
        from thop import profile
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"thop not available: {exc}"

    try:
        macs, _ = profile(model, inputs=inputs, custom_ops=custom_ops, verbose=False)
        return macs / 1e9, None
    except Exception as exc:  # pragma: no cover - model dependent
        return None, f"thop failed: {exc}"


def print_result(
    name: str,
    total: int,
    trainable: int,
    gmacs: float | None,
    note: str | None = None,
) -> None:
    result = {
        "method": name,
        "total_params": total,
        "total_params_m": round(total / 1e6, 3),
        "trainable_params": trainable,
        "trainable_params_m": round(trainable / 1e6, 3),
        "gmacs": None if gmacs is None else round(gmacs, 3),
        "note": note,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
