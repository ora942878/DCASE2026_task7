from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MAINLINE_ROOT = SCRIPT_DIR.parent
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


DEFAULT_SEEDS = [101, 202, 303, 404, 505]
DEFAULT_BETAS = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]


def load_path_states(
    run_name: str,
    methods: list[str],
    checkpoint: str,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    states = []
    for method in methods:
        path = MAINLINE_ROOT / "runs" / run_name / method / f"checkpoint_D3_fullft_bn2_{checkpoint}.pth"
        if not path.exists():
            raise FileNotFoundError(path)
        states.append(load_state_dict(path, device))
    return states


def mean_delta(old: dict[str, torch.Tensor], states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    delta = {}
    for key, old_value in old.items():
        if not torch.is_floating_point(old_value):
            continue
        delta[key] = sum(state[key].float() - old_value.float() for state in states) / len(states)
    return delta


def add_delta(old: dict[str, torch.Tensor], delta: dict[str, torch.Tensor], beta: float) -> dict[str, torch.Tensor]:
    out = {key: value.clone() for key, value in old.items()}
    for key, value in old.items():
        if key in delta:
            out[key] = (value.float() + beta * delta[key]).to(dtype=value.dtype)
    return out


def average_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in states[0].items():
        if torch.is_floating_point(value):
            out[key] = (sum(state[key].float() for state in states) / len(states)).to(dtype=value.dtype)
        else:
            out[key] = value.clone()
    return out


def evaluate_state(
    name: str,
    state: dict[str, torch.Tensor],
    device: torch.device,
    out_dir: Path,
    save_models: bool,
) -> dict:
    model = build_model().to(device)
    model.load_state_dict(state)
    model.eval()
    d2 = eval_wavlevel(model, D2_TEST_WAV, D2_TEST_CSV, device, task_id=1, name=f"{name}:D2", no_progress=True)
    d3 = eval_wavlevel(model, D3_TEST_WAV, D3_TEST_CSV, device, task_id=1, name=f"{name}:D3", no_progress=True)
    result = {
        "name": name,
        "D2_wav": d2,
        "D3_wav": d3,
        "avg_D2_D3_wav_official_acc": (d2["official_domain_acc"] + d3["official_domain_acc"]) / 2.0,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if save_models:
        torch.save(state, out_dir / f"{name}.pth")
    return result


def write_summary(results: list[dict], out_dir: Path) -> None:
    rows = []
    for result in results:
        rows.append(
            {
                "name": result["name"],
                "D2_wav_official_acc": result["D2_wav"]["official_domain_acc"] * 100,
                "D3_wav_official_acc": result["D3_wav"]["official_domain_acc"] * 100,
                "avg_D2_D3_wav_official_acc": result["avg_D2_D3_wav_official_acc"] * 100,
                "D2_wav_sample_acc": result["D2_wav"]["sample_acc"] * 100,
                "D3_wav_sample_acc": result["D3_wav"]["sample_acc"] * 100,
            }
        )
    rows.sort(key=lambda row: row["avg_D2_D3_wav_official_acc"], reverse=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            "{name:24s} D2={D2_wav_official_acc:6.2f}% "
            "D3={D3_wav_official_acc:6.2f}% "
            "avg={avg_D2_D3_wav_official_acc:6.2f}%".format(**row),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep beta for five-path D3 BN2 checkpoint updates.")
    parser.add_argument("--prefix", default="rand_d3_ccsg")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--run-name", default="rand_d3_ccsg5_40ep_last_balanced_bs64_ckpt2_bn2")
    parser.add_argument("--out-name", default="rand_d3_ccsg5_beta_scan_40ep_last_balanced_bs64")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="last")
    parser.add_argument("--betas", nargs="+", type=float, default=DEFAULT_BETAS)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    methods = [f"{args.prefix}_s{seed}" for seed in args.seeds]
    out_dir = MAINLINE_ROOT / "runs" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    old = load_state_dict(CKPT_ROOT / "checkpoint_D2.pth", device)
    states = load_path_states(args.run_name, methods, args.checkpoint, device)
    delta = mean_delta(old, states)

    combos: list[tuple[str, dict[str, torch.Tensor]]] = []
    if args.include_baselines:
        combos.append(("old_checkpoint_D2", old))
        combos.append(("mean_ft", average_states(states)))
    for beta in args.betas:
        combos.append((f"delta_beta_{int(round(beta * 1000)):03d}", add_delta(old, delta, beta)))

    meta = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "methods": methods,
        "betas": args.betas,
        "formula": "checkpoint_D2 + beta * mean(path_i - checkpoint_D2)",
        "task_id": 1,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    results = []
    for name, state in combos:
        print(f"Evaluating {name}", flush=True)
        results.append(evaluate_state(name, state, device, out_dir, args.save_models))
    write_summary(results, out_dir)
    print(f"Saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
