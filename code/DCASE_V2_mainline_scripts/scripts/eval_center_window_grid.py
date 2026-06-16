from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm


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
    compute_macro_recall,
    forward_logits,
    load_state_dict,
)
from configs.CFG_PATH import CFG  # noqa: E402


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


def read_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(path)
    if sr != CFG.sample_rate:
        raise ValueError(f"Sample-rate mismatch: {path}, got {sr}")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32)


def parse_center_sets(items: list[str]) -> list[tuple[str, list[float]]]:
    out = []
    for item in items:
        name, values = item.split(":", 1)
        centers = [float(value) for value in values.split(",") if value.strip()]
        if not centers:
            raise ValueError(f"Empty center set: {item}")
        out.append((name, centers))
    return out


def centered_windows(x: np.ndarray, window_samples: int, centers: list[float]) -> tuple[np.ndarray, list[int]]:
    length = len(x)
    windows = []
    starts = []
    seen = set()
    if length >= window_samples:
        max_start = length - window_samples
        for center in centers:
            center_sample = int(round(center * max(length - 1, 0)))
            start = int(round(center_sample - window_samples / 2))
            start = min(max(start, 0), max_start)
            if start in seen:
                continue
            seen.add(start)
            starts.append(start)
            windows.append(x[start : start + window_samples])
    else:
        padded = np.zeros(window_samples, dtype=np.float32)
        offset = max(0, (window_samples - length) // 2)
        padded[offset : offset + length] = x
        starts = [0]
        windows = [padded]
    return np.stack(windows).astype(np.float32), starts


def vote_from_logits(logits: torch.Tensor) -> int:
    preds = logits.argmax(dim=1)
    counts = torch.bincount(preds, minlength=logits.size(1))
    winners = torch.nonzero(counts == counts.max(), as_tuple=False).flatten()
    if winners.numel() == 1:
        return int(winners.item())
    mean_logits = logits.mean(dim=0)
    return int(winners[mean_logits[winners].argmax()].item())


def predict_from_logits(logits: torch.Tensor, mode: str) -> int:
    if mode == "mean_logits":
        return int(logits.mean(dim=0).argmax().item())
    if mode == "mean_probs":
        return int(F.softmax(logits, dim=1).mean(dim=0).argmax().item())
    if mode == "max_conf":
        probs = F.softmax(logits, dim=1)
        idx = int(probs.max(dim=1).values.argmax().item())
        return int(probs[idx].argmax().item())
    if mode == "vote":
        return vote_from_logits(logits)
    raise ValueError(f"Unknown mode: {mode}")


@torch.no_grad()
def predict_one(
    model,
    x: np.ndarray,
    device: torch.device,
    task_id: int,
    window_sec: float,
    centers: list[float],
    batch_size: int,
    mode: str,
) -> tuple[int, dict]:
    window_samples = max(1, int(round(window_sec * CFG.sample_rate)))
    windows, starts = centered_windows(x, window_samples, centers)
    logits_list = []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).float().to(device)
        logits_list.append(forward_logits(model, batch, task_id).detach().cpu())
    logits = torch.cat(logits_list, dim=0)
    return predict_from_logits(logits, mode), {"num_windows": len(starts), "starts": starts}


def eval_domain(
    model,
    wav_dir: Path,
    csv_path: Path,
    device: torch.device,
    task_id: int,
    window_sec: float,
    center_name: str,
    centers: list[float],
    batch_size: int,
    mode: str,
    name: str,
) -> dict:
    df = pd.read_csv(csv_path)
    y_true: list[int] = []
    y_pred: list[int] = []
    num_windows: list[int] = []
    model.eval()
    for _, row in tqdm(df.iterrows(), total=len(df), desc=name, leave=False):
        wav_path = wav_dir / str(row["filename"])
        label = CFG.dict_class_labels[str(row["class"])]
        pred, extra = predict_one(
            model,
            read_wav(wav_path),
            device,
            task_id,
            window_sec,
            centers,
            batch_size,
            mode,
        )
        y_true.append(label)
        y_pred.append(pred)
        num_windows.append(extra["num_windows"])

    out = compute_macro_recall(y_true, y_pred)
    out["window_sec"] = window_sec
    out["center_name"] = center_name
    out["centers"] = centers
    out["mode"] = mode
    out["avg_num_windows"] = float(np.mean(num_windows))
    out["max_num_windows"] = int(np.max(num_windows))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--window-sec", nargs="+", type=float, default=[2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
    parser.add_argument(
        "--center-sets",
        nargs="+",
        default=[
            "center:0.5",
            "tri3:0.25,0.5,0.75",
            "quint5:0.1,0.3,0.5,0.7,0.9",
            "dense7:0.05,0.2,0.35,0.5,0.65,0.8,0.95",
        ],
    )
    parser.add_argument("--modes", nargs="+", default=["mean_logits", "mean_probs", "max_conf"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "runs" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve(args.checkpoint)

    model = build_model().to(device)
    model.load_state_dict(load_state_dict(checkpoint_path, device))
    model.eval()

    center_sets = parse_center_sets(args.center_sets)
    rows = []
    meta = {
        "checkpoint": str(checkpoint_path),
        "task_id": args.task_id,
        "window_sec": args.window_sec,
        "center_sets": {name: centers for name, centers in center_sets},
        "modes": args.modes,
        "batch_size": args.batch_size,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for window_sec in args.window_sec:
        for center_name, centers in center_sets:
            for mode in args.modes:
                tag = f"w{window_sec:g}_{center_name}_{mode}"
                d2 = eval_domain(
                    model,
                    D2_TEST_WAV,
                    D2_TEST_CSV,
                    device,
                    args.task_id,
                    window_sec,
                    center_name,
                    centers,
                    args.batch_size,
                    mode,
                    f"D2:{tag}",
                )
                d3 = eval_domain(
                    model,
                    D3_TEST_WAV,
                    D3_TEST_CSV,
                    device,
                    args.task_id,
                    window_sec,
                    center_name,
                    centers,
                    args.batch_size,
                    mode,
                    f"D3:{tag}",
                )
                result = {
                    "tag": tag,
                    "window_sec": window_sec,
                    "center_name": center_name,
                    "centers": centers,
                    "mode": mode,
                    "D2_wav": d2,
                    "D3_wav": d3,
                    "avg_D2_D3_wav_official_acc": (d2["official_domain_acc"] + d3["official_domain_acc"]) / 2.0,
                }
                (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                row = {
                    "tag": tag,
                    "window_sec": window_sec,
                    "center_name": center_name,
                    "mode": mode,
                    "D2_wav_official_acc": d2["official_domain_acc"] * 100,
                    "D3_wav_official_acc": d3["official_domain_acc"] * 100,
                    "avg_D2_D3_wav_official_acc": result["avg_D2_D3_wav_official_acc"] * 100,
                    "D2_wav_sample_acc": d2["sample_acc"] * 100,
                    "D3_wav_sample_acc": d3["sample_acc"] * 100,
                    "D2_avg_windows": d2["avg_num_windows"],
                    "D3_avg_windows": d3["avg_num_windows"],
                    "D2_max_windows": d2["max_num_windows"],
                    "D3_max_windows": d3["max_num_windows"],
                }
                rows.append(row)
                rows_sorted = sorted(rows, key=lambda item: item["avg_D2_D3_wav_official_acc"], reverse=True)
                with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows_sorted)
                print(
                    "{tag:28s} D2={D2_wav_official_acc:6.2f}% "
                    "D3={D3_wav_official_acc:6.2f}% avg={avg_D2_D3_wav_official_acc:6.2f}%".format(**row),
                    flush=True,
                )

    print(f"Saved: {out_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
