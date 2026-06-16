from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_full_ft_d3_bn2 import CKPT_ROOT, load_state_dict  # noqa: E402


BN_TASK_PATTERN = re.compile(r"(^bn0|\.bnF|\.bnS)\.(\d)\.")


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    path = ROOT / path_text
    if path.exists():
        return path
    path = CKPT_ROOT / path_text
    if path.exists():
        return path
    raise FileNotFoundError(path_text)


def replace_task(key: str, task_id: int) -> str:
    return BN_TASK_PATTERN.sub(lambda match: f"{match.group(1)}.{task_id}.", key)


def is_task_bn_key(key: str, task_id: int) -> bool:
    match = BN_TASK_PATTERN.search(key)
    return bool(match and int(match.group(2)) == task_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--source-task-id", type=int, required=True)
    parser.add_argument("--target-task-id", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve(args.checkpoint)
    state = load_state_dict(checkpoint_path, device)
    out = {key: value.clone() for key, value in state.items()}

    copied = []
    for key in state:
        if not is_task_bn_key(key, args.target_task_id):
            continue
        source_key = replace_task(key, args.source_task_id)
        out[key] = state[source_key].clone()
        copied.append({"target": key, "source": source_key})

    out_dir = ROOT / "runs" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.checkpoint_name
    torch.save(out, out_path)

    meta = {
        "checkpoint": str(checkpoint_path),
        "source_task_id": args.source_task_id,
        "target_task_id": args.target_task_id,
        "num_bn_items_copied": len(copied),
        "saved": str(out_path),
    }
    (out_dir / f"{out_path.stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
