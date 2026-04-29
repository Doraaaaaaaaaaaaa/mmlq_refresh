from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


COMMANDS = [
    ("LQ-ICIF (Ours)", [sys.executable, str(HERE / "measure_ours.py")]),
    ("MMLQ", [sys.executable, str(HERE / "measure_mmlq.py")]),
    ("TANet", [sys.executable, str(HERE / "measure_tanet.py")]),
    ("Charm", [sys.executable, str(HERE / "measure_charm.py")]),
    ("AesMamba-V", [sys.executable, str(HERE / "measure_aesmamba.py"), "--variant", "viaa"]),
    ("AesMamba-M", [sys.executable, str(HERE / "measure_aesmamba.py"), "--variant", "miaa"]),
]


def main() -> None:
    results = []
    for name, cmd in COMMANDS:
        print(f"\n=== {name} ===", flush=True)
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(proc.stdout)
        results.append({
            "method": name,
            "returncode": proc.returncode,
            "output": proc.stdout.strip(),
        })

    out_path = HERE / "run_all_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved raw results to {out_path}")


if __name__ == "__main__":
    main()
