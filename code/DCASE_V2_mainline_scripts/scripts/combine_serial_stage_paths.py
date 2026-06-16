from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_full_ft_d3_bn2 import (  # noqa: E402
    CKPT_ROOT,
    D2_TEST_CSV,
    D2_TEST_WAV,
    D3_TEST_CSV,
    D3_TEST_WAV,
    build_model,
    eval_wavlevel,
    load_state_dict,
)


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


def bn_task_id(key: str) -> int | None:
    match = BN_TASK_PATTERN.search(key)
    return int(match.group(2)) if match else None


def is_task_bn_key(key: str, task_id: int) -> bool:
    return bn_task_id(key) == task_id


def weighted_sum(values: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    return sum(weight * value.float() for weight, value in zip(weights, values))


def combine_running_var(
    anchor: dict[str, torch.Tensor],
    paths: list[dict[str, torch.Tensor]],
    target_key: str,
    source_task_id: int,
    target_task_id: int,
    anchor_weight: float,
    path_weight: float,
) -> torch.Tensor:
    anchor_var_key = replace_task(target_key, source_task_id)
    anchor_mean_key = anchor_var_key.replace("running_var", "running_mean")
    path_var_key = replace_task(target_key, target_task_id)
    path_mean_key = path_var_key.replace("running_var", "running_mean")

    weights = [anchor_weight] + [path_weight / len(paths)] * len(paths)
    means = [anchor[anchor_mean_key].float()] + [state[path_mean_key].float() for state in paths]
    variances = [anchor[anchor_var_key].float()] + [state[path_var_key].float() for state in paths]
    mean_center = weighted_sum(means, weights)
    second_moment = weighted_sum([var + mean.pow(2) for var, mean in zip(variances, means)], weights)
    return torch.clamp(second_moment - mean_center.pow(2), min=1e-8)


def combine_stage(
    anchor: dict[str, torch.Tensor],
    paths: list[dict[str, torch.Tensor]],
    source_task_id: int,
    target_task_id: int,
    beta: float,
) -> dict[str, torch.Tensor]:
    anchor_weight = 1.0 - beta
    path_weight = beta
    out = {key: value.clone() for key, value in anchor.items()}

    for key, anchor_tensor in anchor.items():
        task_id = bn_task_id(key)

        if task_id == target_task_id:
            old_key = replace_task(key, source_task_id)
            if key.endswith(".num_batches_tracked"):
                out[key] = paths[0][key].clone()
            elif key.endswith(".running_var"):
                out[key] = combine_running_var(
                    anchor,
                    paths,
                    key,
                    source_task_id,
                    target_task_id,
                    anchor_weight,
                    path_weight,
                ).to(dtype=anchor_tensor.dtype)
            elif torch.is_floating_point(anchor_tensor):
                values = [anchor[old_key].float()] + [state[key].float() for state in paths]
                weights = [anchor_weight] + [path_weight / len(paths)] * len(paths)
                out[key] = weighted_sum(values, weights).to(dtype=anchor_tensor.dtype)
            else:
                out[key] = paths[0][key].clone()
            continue

        if task_id is not None:
            out[key] = anchor_tensor.clone()
            continue

        if torch.is_floating_point(anchor_tensor):
            mean_new = sum(state[key].float() for state in paths) / len(paths)
            out[key] = (anchor_tensor.float() + beta * (mean_new - anchor_tensor.float())).to(dtype=anchor_tensor.dtype)
        else:
            out[key] = anchor_tensor.clone()

    return out


def evaluate_state(name: str, state: dict[str, torch.Tensor], task_id: int, device: torch.device, out_dir: Path) -> dict:
    model = build_model().to(device)
    model.load_state_dict(state)
    model.eval()
    result = {
        "name": name,
        "task_id": task_id,
        "D2_wav": eval_wavlevel(model, D2_TEST_WAV, D2_TEST_CSV, device, task_id=task_id, name=f"{name}:D2", no_progress=True),
        "D3_wav": eval_wavlevel(model, D3_TEST_WAV, D3_TEST_CSV, device, task_id=task_id, name=f"{name}:D3", no_progress=True),
    }
    result["avg_D2_D3_wav_official_acc"] = (
        result["D2_wav"]["official_domain_acc"] + result["D3_wav"]["official_domain_acc"]
    ) / 2.0
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary(result: dict, out_dir: Path) -> None:
    row = {
        "name": result["name"],
        "task_id": result["task_id"],
        "D2_wav_official_acc": result["D2_wav"]["official_domain_acc"] * 100,
        "D3_wav_official_acc": result["D3_wav"]["official_domain_acc"] * 100,
        "avg_D2_D3_wav_official_acc": result["avg_D2_D3_wav_official_acc"] * 100,
        "D2_wav_sample_acc": result["D2_wav"]["sample_acc"] * 100,
        "D3_wav_sample_acc": result["D3_wav"]["sample_acc"] * 100,
    }
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(
        "{name:28s} task={task_id} D2={D2_wav_official_acc:6.2f}% "
        "D3={D3_wav_official_acc:6.2f}% avg={avg_D2_D3_wav_official_acc:6.2f}%".format(**row),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--anchor-checkpoint", required=True)
    parser.add_argument("--path-checkpoints", nargs="+", required=True)
    parser.add_argument("--source-task-id", type=int, required=True)
    parser.add_argument("--target-task-id", type=int, required=True)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "runs" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_path = resolve(args.anchor_checkpoint)
    path_paths = [resolve(path_text) for path_text in args.path_checkpoints]
    anchor = load_state_dict(anchor_path, device)
    paths = [load_state_dict(path, device) for path in path_paths]

    combined = combine_stage(anchor, paths, args.source_task_id, args.target_task_id, args.beta)
    checkpoint_path = out_dir / args.checkpoint_name
    torch.save(combined, checkpoint_path)

    meta = {
        "anchor_checkpoint": str(anchor_path),
        "path_checkpoints": [str(path) for path in path_paths],
        "source_task_id": args.source_task_id,
        "target_task_id": args.target_task_id,
        "beta": args.beta,
        "formula": {
            "shared_weights": "anchor + beta * (mean(paths) - anchor)",
            "target_bn": "source BN branch from anchor is mapped to target BN branch, then combined with target BN branches from paths",
            "target_bn_weights": f"{1.0 - args.beta:.6g} * anchor_BN{args.source_task_id + 1} + {args.beta:.6g} * mean(path_BN{args.target_task_id + 1})",
            "running_var": "mixture variance: E[var + mean^2] - E[mean]^2",
            "other_bn_branches": "copied from anchor checkpoint",
        },
        "saved": str(checkpoint_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    result = evaluate_state(args.eval_name, combined, args.target_task_id, device, out_dir)
    write_summary(result, out_dir)
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)
    print(f"Saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
