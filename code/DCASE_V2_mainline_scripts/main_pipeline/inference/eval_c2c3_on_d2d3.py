from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
MAINLINE_ROOT = PIPELINE_ROOT.parent
CODE_ROOT = MAINLINE_ROOT.parent
RELEASE_ROOT = CODE_ROOT.parent
INFERENCE_ROOT = MAINLINE_ROOT / "ablation03_inference_window"
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from eval_final_inference_ablation import (  # noqa: E402
    D2_TEST_CSV,
    D2_TEST_WAV,
    D3_TEST_CSV,
    D3_TEST_WAV,
    build_model,
    eval_domain,
    load_compatible_state,
    load_state_dict,
    parse_center_sets,
)


def resolve_checkpoint(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidates = [
        MAINLINE_ROOT / path_text,
        MAINLINE_ROOT / "runs" / path_text,
        RELEASE_ROOT / "checkpoints" / "official" / path_text,
        RELEASE_ROOT / "checkpoints" / "ours" / path_text,
        RELEASE_ROOT / "checkpoints" / path_text,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path_text)


def write_summary(rows: list[dict], out_dir: Path) -> None:
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def model_specs(args: argparse.Namespace) -> list[dict]:
    if args.checkpoint:
        return [
            {
                "label": args.label,
                "checkpoint": args.checkpoint,
                "fixed_task_id": args.task_id,
            }
        ]
    return [
        {
            "label": "C2",
            "checkpoint": args.c2_checkpoint,
            "fixed_task_id": args.c2_task_id,
        },
        {
            "label": "C3",
            "checkpoint": args.c3_checkpoint,
            "fixed_task_id": args.c3_task_id,
        },
    ]


def task_ids_for_mode(spec: dict, mode: str) -> dict[str, int]:
    if mode == "fixed-model":
        return {"D2": spec["fixed_task_id"], "D3": spec["fixed_task_id"]}
    raise ValueError(f"Unsupported eval mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick final-inference evaluation on D2 and D3 dev-test.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--label", default="checkpoint")
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--c2-checkpoint", default="Gao_SHNU_task7_1_D2_dictionary.pth")
    parser.add_argument("--c3-checkpoint", default="Gao_SHNU_task7_1_D3_dictionary.pth")
    parser.add_argument("--c2-task-id", type=int, default=1)
    parser.add_argument("--c3-task-id", type=int, default=2)
    parser.add_argument("--eval-mode", choices=["fixed-model"], default="fixed-model")
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--center-set", default="quint5:0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--full-clip", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-name", default="quick_eval_c2c3_d2d3")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = MAINLINE_ROOT / "runs" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    center_name, centers = parse_center_sets([args.center_set])[0]
    config = (
        {
            "tag": "full_clip",
            "kind": "full_clip",
            "window_sec": None,
            "center_name": "full",
            "centers": None,
            "mode": "full_clip",
        }
        if args.full_clip
        else {
            "tag": f"w{args.window_sec:g}_{center_name}_max_conf",
            "kind": "window",
            "window_sec": args.window_sec,
            "center_name": center_name,
            "centers": centers,
            "mode": "max_conf",
        }
    )

    results = {
        "device": str(device),
        "inference_config": config,
        "eval_mode": args.eval_mode,
        "models": {},
    }
    rows = []
    eval_modes = [args.eval_mode]
    for spec in model_specs(args):
        label = spec["label"]
        checkpoint_text = spec["checkpoint"]
        checkpoint_path = resolve_checkpoint(checkpoint_text)
        model = build_model().to(device)
        load_compatible_state(model, load_state_dict(checkpoint_path, device))
        model.eval()

        results["models"][label] = {
            "checkpoint": str(checkpoint_path),
            "fixed_task_id": spec["fixed_task_id"],
        }
        for mode in eval_modes:
            task_ids = task_ids_for_mode(spec, mode)
            model_results = {}
            for domain, wav_dir, csv_path in [
                ("D2", D2_TEST_WAV, D2_TEST_CSV),
                ("D3", D3_TEST_WAV, D3_TEST_CSV),
            ]:
                task_id = task_ids[domain]
                result = eval_domain(model, wav_dir, csv_path, device, task_id, config, args.batch_size, f"{label}:{mode}:{domain}")
                model_results[domain] = result
                rows.append(
                    {
                        "model": label,
                        "checkpoint": str(checkpoint_path),
                        "eval_mode": mode,
                        "domain": domain,
                        "task_id": task_id,
                        "inference": config["tag"],
                        "wav_official_acc": result["official_domain_acc"] * 100,
                        "wav_sample_acc": result["sample_acc"] * 100,
                    }
                )

            avg = (model_results["D2"]["official_domain_acc"] + model_results["D3"]["official_domain_acc"]) / 2.0
            model_results["avg_D2_D3_wav_official_acc"] = avg
            rows.append(
                {
                    "model": label,
                    "checkpoint": str(checkpoint_path),
                    "eval_mode": mode,
                    "domain": "D2_D3_avg",
                    "task_id": f"D2:{task_ids['D2']},D3:{task_ids['D3']}",
                    "inference": config["tag"],
                    "wav_official_acc": avg * 100,
                    "wav_sample_acc": "",
                }
            )
            results["models"][label][mode] = model_results

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(rows, out_dir)
    for row in rows:
        if row["domain"] == "D2_D3_avg":
            print(f"{row['model']} {row['eval_mode']} {row['domain']}: {row['wav_official_acc']:.2f}%", flush=True)
            continue
        sample = row["wav_sample_acc"]
        print(
            f"{row['model']} {row['eval_mode']} {row['domain']}: task={row['task_id']} "
            f"official={row['wav_official_acc']:.2f}% sample={sample:.2f}%",
            flush=True,
        )
    print(f"Saved: {out_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
