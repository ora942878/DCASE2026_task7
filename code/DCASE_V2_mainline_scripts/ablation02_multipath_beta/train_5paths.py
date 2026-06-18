from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MAINLINE_ROOT = SCRIPT_DIR.parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_full_ft_d3_bn2.py"
DEFAULT_SEEDS = [101, 202, 303, 404, 505]


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train five D3 BN2 fine-tuning paths for beta ablation.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prefix", default="rand_d3_ccsg")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--run-name", default="rand_d3_ccsg5_40ep_last_balanced_bs64_ckpt2_bn2")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--final-checkpoint", choices=["best", "last"], default="last")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        method = f"{args.prefix}_s{seed}"
        final_metrics = MAINLINE_ROOT / "runs" / args.run_name / method / "final_metrics.json"
        if final_metrics.exists() and not args.resume:
            print(f"SKIP existing result: {method}", flush=True)
            continue

        command = [
            args.python,
            str(TRAIN_SCRIPT),
            "--method",
            method,
            "--run-name",
            args.run_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--eta-min",
            str(args.eta_min),
            "--num-workers",
            str(args.num_workers),
            "--task-id",
            str(args.task_id),
            "--class-weight",
            args.class_weight,
            "--seed",
            str(seed),
            "--final-checkpoint",
            args.final_checkpoint,
        ]
        if args.resume:
            command.append("--resume")
        if args.no_progress:
            command.append("--no-progress")
        run_command(command)


if __name__ == "__main__":
    main()
