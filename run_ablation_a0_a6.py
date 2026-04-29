from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

ABLATIONS = [
    ("A0", "baseline", "visual-only baseline"),
    ("A1", "text_only", "text-only baseline"),
    ("A2", "concat", "simple image-text concat"),
    ("A3", "direct_ca", "MMLQ-style direct learnable-query cross-attention"),
    ("A4", "icif", "ICIF without hierarchical attention"),
    ("A5", "icif_ha", "ICIF with hierarchical attention, without attributes"),
    ("A6", None, "full model with attribute reasoning"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only selected IDs or modes, for example: --only A0 A3 A6",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = None
    if args.only:
        selected = {item.lower() for item in args.only}

    for ablation_id, mode, desc in ABLATIONS:
        if selected and ablation_id.lower() not in selected and (mode or "full").lower() not in selected:
            continue

        cmd = [sys.executable, "main.py", "--config", args.config]
        if mode is not None:
            cmd += ["--ablation-mode", mode]

        print(f"\n=== {ablation_id}: {desc} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"{ablation_id} failed with return code {result.returncode}")
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
