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
from sklearn import metrics
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CODE_ROOT = ROOT.parent
RELEASE_ROOT = CODE_ROOT.parent
PROJECT_ROOT = CODE_ROOT / "base"
FINAL_EVAL_ROOT = CODE_ROOT / "final_eval"
if str(FINAL_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FINAL_EVAL_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from configs.CFG_PATH import CFG  # noqa: E402
from domain_net import MCnn14  # noqa: E402


RAW_ROOT = RELEASE_ROOT / "data"
CHECKPOINT_ROOT = RELEASE_ROOT / "checkpoints"
D2_TEST_WAV = RAW_ROOT / "D2" / "d2-dev-test"
D2_TEST_CSV = RAW_ROOT / "D2" / "metadata" / "d2-dev-test.csv"
D3_TEST_WAV = RAW_ROOT / "D3" / "d3-dev-test"
D3_TEST_CSV = RAW_ROOT / "D3" / "metadata" / "d3-dev-test.csv"


DEFAULT_CENTER_SETS = [
    "quint5:0.1,0.3,0.5,0.7,0.9",
    "dense7:0.05,0.2,0.35,0.5,0.65,0.8,0.95",
]


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    path = ROOT / path_text
    if path.exists():
        return path
    for base in (CHECKPOINT_ROOT / "ours", CHECKPOINT_ROOT / "official", CHECKPOINT_ROOT):
        path = base / path_text
        if path.exists():
            return path
    raise FileNotFoundError(path_text)


def build_model() -> MCnn14:
    return MCnn14(
        sample_rate=CFG.sample_rate,
        window_size=CFG.window_size,
        hop_size=CFG.hop_size,
        mel_bins=CFG.mel_bins,
        fmin=CFG.fmin,
        fmax=CFG.fmax,
        classes_num=CFG.classes_num_DIL,
        nb_tasks=CFG.NUM_TASKS,
    )


def load_state_dict(path: Path, device: torch.device) -> dict:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint format: {path}")


def load_compatible_state(model: torch.nn.Module, state_dict: dict) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    bad_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith("spectrogram_extractor.")
    ]
    bad_unexpected = [
        key for key in incompatible.unexpected_keys
        if not key.startswith("spectrogram_extractor.stft.")
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Checkpoint is not compatible with the local eval model. "
            f"Missing keys: {bad_missing}; unexpected keys: {bad_unexpected}"
        )


def forward_logits(model: torch.nn.Module, x: torch.Tensor, task_id: int) -> torch.Tensor:
    out = model(x, task_id)
    return out["clipwise_output"] if isinstance(out, dict) else out


def compute_macro_recall(y_true: list[int], y_pred: list[int]) -> dict:
    labels = list(range(CFG.classes_num_DIL))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = cm.sum(axis=1)
    class_recall = np.divide(
        cm.diagonal(),
        row_totals,
        out=np.zeros_like(row_totals, dtype=float),
        where=row_totals > 0,
    )
    present = row_totals > 0
    domain_acc = float(class_recall[present].mean()) if present.any() else 0.0
    sample_acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if y_true else 0.0
    return {
        "official_domain_acc": domain_acc,
        "sample_acc": sample_acc,
        "class_recall": class_recall.tolist(),
        "class_total": row_totals.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def read_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(path)
    if sr != CFG.sample_rate:
        raise ValueError(f"Sample-rate mismatch: {path}, got {sr}")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32)


def pad_to_min_clip(x: np.ndarray) -> np.ndarray:
    if len(x) >= CFG.clip_samples:
        return x.astype(np.float32)
    return np.pad(x, (0, CFG.clip_samples - len(x))).astype(np.float32)


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


def max_conf_prediction(logits: torch.Tensor) -> int:
    probs = F.softmax(logits, dim=1)
    best_window = int(probs.max(dim=1).values.argmax().item())
    return int(probs[best_window].argmax().item())


@torch.no_grad()
def predict_full_clip(model, x: np.ndarray, device: torch.device, task_id: int) -> tuple[int, dict]:
    x = pad_to_min_clip(x)
    batch = torch.from_numpy(x).float().unsqueeze(0).to(device)
    logits = forward_logits(model, batch, task_id).detach().cpu()
    return int(logits.argmax(dim=1).item()), {"num_windows": 1, "starts": [0]}


@torch.no_grad()
def predict_windows(
    model,
    x: np.ndarray,
    device: torch.device,
    task_id: int,
    window_sec: float,
    centers: list[float],
    batch_size: int,
) -> tuple[int, dict]:
    window_samples = max(1, int(round(window_sec * CFG.sample_rate)))
    windows, starts = centered_windows(x, window_samples, centers)
    logits_list = []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).float().to(device)
        logits_list.append(forward_logits(model, batch, task_id).detach().cpu())
    logits = torch.cat(logits_list, dim=0)
    return max_conf_prediction(logits), {"num_windows": len(starts), "starts": starts}


def eval_domain(
    model,
    wav_dir: Path,
    csv_path: Path,
    device: torch.device,
    task_id: int,
    config: dict,
    batch_size: int,
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
        x = read_wav(wav_path)

        if config["kind"] == "full_clip":
            pred, extra = predict_full_clip(model, x, device, task_id)
        elif config["kind"] == "window":
            pred, extra = predict_windows(
                model,
                x,
                device,
                task_id,
                config["window_sec"],
                config["centers"],
                batch_size,
            )
        else:
            raise ValueError(f"Unknown config kind: {config['kind']}")

        y_true.append(label)
        y_pred.append(pred)
        num_windows.append(extra["num_windows"])

    out = compute_macro_recall(y_true, y_pred)
    out["avg_num_windows"] = float(np.mean(num_windows))
    out["max_num_windows"] = int(np.max(num_windows))
    return out


def build_configs(include_full_clip: bool, window_sec: list[float], center_sets: list[tuple[str, list[float]]]) -> list[dict]:
    configs = []
    if include_full_clip:
        configs.append(
            {
                "tag": "full_clip",
                "kind": "full_clip",
                "window_sec": None,
                "center_name": "full",
                "centers": None,
                "mode": "full_clip",
            }
        )

    for seconds in window_sec:
        for center_name, centers in center_sets:
            configs.append(
                {
                    "tag": f"w{seconds:g}_{center_name}_max_conf",
                    "kind": "window",
                    "window_sec": seconds,
                    "center_name": center_name,
                    "centers": centers,
                    "mode": "max_conf",
                }
            )
    return configs


def write_summary(rows: list[dict], out_dir: Path) -> None:
    rows_sorted = sorted(rows, key=lambda item: item["avg_D2_D3_wav_official_acc"], reverse=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused final-inference ablation: full clip plus 3/4/5s x 5/7 windows.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--window-sec", nargs="+", type=float, default=[3.0, 4.0, 5.0])
    parser.add_argument("--center-sets", nargs="+", default=DEFAULT_CENTER_SETS)
    parser.add_argument("--include-full-clip", action="store_true", default=True)
    parser.add_argument("--no-full-clip", action="store_false", dest="include_full_clip")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "runs" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve(args.checkpoint)

    model = build_model().to(device)
    load_compatible_state(model, load_state_dict(checkpoint_path, device))
    model.eval()

    center_sets = parse_center_sets(args.center_sets)
    configs = build_configs(args.include_full_clip, args.window_sec, center_sets)
    meta = {
        "checkpoint": str(checkpoint_path),
        "task_id": args.task_id,
        "window_sec": args.window_sec,
        "center_sets": {name: centers for name, centers in center_sets},
        "include_full_clip": args.include_full_clip,
        "mode": "max_conf for windowed configs",
        "batch_size": args.batch_size,
        "configs": configs,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rows = []
    for config in configs:
        tag = config["tag"]
        d2 = eval_domain(model, D2_TEST_WAV, D2_TEST_CSV, device, args.task_id, config, args.batch_size, f"D2:{tag}")
        d3 = eval_domain(model, D3_TEST_WAV, D3_TEST_CSV, device, args.task_id, config, args.batch_size, f"D3:{tag}")
        result = {
            "tag": tag,
            "kind": config["kind"],
            "window_sec": config["window_sec"],
            "center_name": config["center_name"],
            "centers": config["centers"],
            "mode": config["mode"],
            "task_id": args.task_id,
            "D2_wav": d2,
            "D3_wav": d3,
            "avg_D2_D3_wav_official_acc": (d2["official_domain_acc"] + d3["official_domain_acc"]) / 2.0,
        }
        (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        row = {
            "tag": tag,
            "kind": config["kind"],
            "window_sec": "" if config["window_sec"] is None else config["window_sec"],
            "center_name": config["center_name"],
            "mode": config["mode"],
            "task_id": args.task_id,
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
        write_summary(rows, out_dir)
        print(
            "{tag:28s} task={task_id} D2={D2_wav_official_acc:6.2f}% "
            "D3={D3_wav_official_acc:6.2f}% avg={avg_D2_D3_wav_official_acc:6.2f}%".format(**row),
            flush=True,
        )

    print(f"Saved: {out_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
