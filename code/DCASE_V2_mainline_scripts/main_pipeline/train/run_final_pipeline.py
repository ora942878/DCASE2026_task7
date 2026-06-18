from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
MAINLINE_ROOT = PIPELINE_ROOT.parent
TOOLS_ROOT = PIPELINE_ROOT / "tools"
DEFAULT_SEEDS = [101, 202, 303, 404, 505]


def beta_tag(beta: float) -> str:
    return f"beta{int(round(beta * 100)):03d}"


def view_names(prefix: str, seeds: list[int]) -> list[str]:
    return [f"{prefix}_s{seed}" for seed in seeds]


def run_command(command: list[str], dry_run: bool) -> None:
    text = " ".join(command)
    print(text, flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def build_views(args: argparse.Namespace, domain: str, prefix: str) -> None:
    command = [
        args.python,
        str(TOOLS_ROOT / "build_random_domain_views.py"),
        "--domain",
        domain,
        "--prefix",
        prefix,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--p-concat",
        str(args.p_concat),
        "--p-shift",
        str(args.p_shift),
        "--p-gain",
        str(args.p_gain),
        "--aug-ratio",
        str(args.aug_ratio),
        "--minority-power",
        str(args.minority_power),
    ]
    if args.overwrite_views:
        command.append("--overwrite")
    run_command(command, args.dry_run)


def copy_bn(args: argparse.Namespace, checkpoint: str, out_name: str, checkpoint_name: str, source_task_id: int, target_task_id: int) -> str:
    command = [
        args.python,
        str(TOOLS_ROOT / "copy_bn_branch.py"),
        "--checkpoint",
        checkpoint,
        "--out-name",
        out_name,
        "--checkpoint-name",
        checkpoint_name,
        "--source-task-id",
        str(source_task_id),
        "--target-task-id",
        str(target_task_id),
    ]
    run_command(command, args.dry_run)
    return f"runs/{out_name}/{checkpoint_name}"


def train_paths(
    args: argparse.Namespace,
    domain: str,
    task_id: int,
    run_name: str,
    prefix: str,
    init_checkpoint: str,
) -> list[str]:
    checkpoints = []
    for seed, method in zip(args.seeds, view_names(prefix, args.seeds)):
        command = [
            args.python,
            str(TOOLS_ROOT / "train_stage_path.py"),
            "--method",
            method,
            "--domain",
            domain,
            "--run-name",
            run_name,
            "--init-checkpoint",
            init_checkpoint,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--lr",
            str(args.lr),
            "--eta-min",
            str(args.eta_min),
            "--task-id",
            str(task_id),
            "--class-weight",
            args.class_weight,
            "--final-checkpoint",
            args.final_checkpoint,
            "--seed",
            str(seed),
        ]
        if args.resume:
            command.append("--resume")
        if args.no_progress:
            command.append("--no-progress")
        run_command(command, args.dry_run)
        checkpoints.append(f"runs/{run_name}/{method}/checkpoint_{domain}_fullft_bn{task_id + 1}_last.pth")
    return checkpoints


def combine_paths(
    args: argparse.Namespace,
    out_name: str,
    anchor_checkpoint: str,
    path_checkpoints: list[str],
    source_task_id: int,
    target_task_id: int,
    checkpoint_name: str,
    eval_name: str,
) -> str:
    command = [
        args.python,
        str(TOOLS_ROOT / "combine_paths_with_beta.py"),
        "--out-name",
        out_name,
        "--anchor-checkpoint",
        anchor_checkpoint,
        "--path-checkpoints",
        *path_checkpoints,
        "--source-task-id",
        str(source_task_id),
        "--target-task-id",
        str(target_task_id),
        "--beta",
        str(args.beta),
        "--checkpoint-name",
        checkpoint_name,
        "--eval-name",
        eval_name,
    ]
    run_command(command, args.dry_run)
    return f"runs/{out_name}/{checkpoint_name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final D1 -> D2 -> D3 five-view mean-update pipeline.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--d2-prefix", default="final_d2_ccsg")
    parser.add_argument("--d3-prefix", default="final_d3_ccsg")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--p-concat", type=float, default=0.5)
    parser.add_argument("--p-shift", type=float, default=0.5)
    parser.add_argument("--p-gain", type=float, default=0.5)
    parser.add_argument("--aug-ratio", type=float, default=1.0)
    parser.add_argument("--minority-power", type=float, default=0.5)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--final-checkpoint", choices=["best", "last"], default="last")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-views", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tag = beta_tag(args.beta)
    d2_run = f"final_d1bn1_to_d2_5paths_{tag}"
    d3_run = f"final_d2method_to_d3_5paths_{tag}"
    init_run = f"final_stage_inits_{tag}_mean5"
    d2_combine_run = f"final_checkpoint2_{tag}_mean5"
    d3_combine_run = f"final_checkpoint3_{tag}_mean5"

    print("D2 views:", ", ".join(view_names(args.d2_prefix, args.seeds)), flush=True)
    print("D3 views:", ", ".join(view_names(args.d3_prefix, args.seeds)), flush=True)

    build_views(args, "D2", args.d2_prefix)
    d2_init = copy_bn(args, "checkpoint_D1.pth", init_run, "checkpoint_D1_bn1_to_bn2_init.pth", 0, 1)
    d2_paths = train_paths(args, "D2", 1, d2_run, args.d2_prefix, d2_init)
    d2_method = combine_paths(
        args,
        d2_combine_run,
        "checkpoint_D1.pth",
        d2_paths,
        0,
        1,
        f"checkpoint_D2_method_{tag}_mean5.pth",
        f"checkpoint2_method_{tag}_mean5",
    )

    build_views(args, "D3", args.d3_prefix)
    d3_init = copy_bn(args, d2_method, init_run, "checkpoint_D2_method_bn2_to_bn3_init.pth", 1, 2)
    d3_paths = train_paths(args, "D3", 2, d3_run, args.d3_prefix, d3_init)
    combine_paths(
        args,
        d3_combine_run,
        d2_method,
        d3_paths,
        1,
        2,
        f"checkpoint_D3_method_{tag}_mean5.pth",
        f"checkpoint3_method_{tag}_mean5",
    )


if __name__ == "__main__":
    main()
