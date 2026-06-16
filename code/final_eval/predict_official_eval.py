from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from domain_net import MCnn14


SUBMISSION_LABEL = "Gao_SHNU_task7_1"
FINAL_EVAL_DIR = Path(__file__).resolve().parent
CODE_ROOT = FINAL_EVAL_DIR.parent
RELEASE_ROOT = CODE_ROOT.parent
DATA_ROOT = RELEASE_ROOT / "data"
CHECKPOINT_ROOT = RELEASE_ROOT / "checkpoints"
SUBMISSION_DIR = FINAL_EVAL_DIR / "submission" / SUBMISSION_LABEL
DEFAULT_AUDIO_DIR = DATA_ROOT / "eval"
DEFAULT_CHECKPOINT = CHECKPOINT_ROOT / "ours" / f"{SUBMISSION_LABEL}_D3_dictionary.pth"
DEFAULT_OUTPUT_CSV = SUBMISSION_DIR / f"{SUBMISSION_LABEL}.output.csv"
DEFAULT_MANIFEST = SUBMISSION_DIR / f"{SUBMISSION_LABEL}_manifest.json"
DEFAULT_PREVIEW_CSV = SUBMISSION_DIR / f"{SUBMISSION_LABEL}_preview.csv"

SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000
CLASSES_NUM = 10
NUM_TASKS = 3
FINAL_TASK_ID = 2
FINAL_WINDOW_SECONDS = 3.0
FINAL_CENTERS = (0.1, 0.3, 0.5, 0.7, 0.9)
OFFICIAL_SEPARATOR = "    "

LABEL_TO_INDEX = {
    "alarm": 0,
    "baby_cry": 1,
    "dog_bark": 2,
    "engine": 3,
    "fire": 4,
    "footsteps": 5,
    "knocking": 6,
    "telephone_ringing": 7,
    "piano": 8,
    "speech": 9,
}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


def build_official_model() -> MCnn14:
    return MCnn14(
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        mel_bins=MEL_BINS,
        fmin=FMIN,
        fmax=FMAX,
        classes_num=CLASSES_NUM,
        nb_tasks=NUM_TASKS,
    )


def extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported checkpoint format.")


def load_model(checkpoint_path: Path, device: torch.device) -> MCnn14:
    model = build_official_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
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
            "Checkpoint is not compatible with the final eval model. "
            f"Missing keys: {bad_missing}; unexpected keys: {bad_unexpected}"
        )
    model.eval()
    return model


def read_audio_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sr}: {path}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def centered_windows(
    audio: np.ndarray,
    window_seconds: float = FINAL_WINDOW_SECONDS,
    centers: Iterable[float] = FINAL_CENTERS,
) -> tuple[np.ndarray, list[int]]:
    window_samples = max(1, int(round(window_seconds * SAMPLE_RATE)))
    length = len(audio)
    windows: list[np.ndarray] = []
    starts: list[int] = []
    seen_starts: set[int] = set()

    if length < window_samples:
        padded = np.zeros(window_samples, dtype=np.float32)
        offset = max(0, (window_samples - length) // 2)
        padded[offset : offset + length] = audio
        return np.stack([padded]), [0]

    max_start = length - window_samples
    for center in centers:
        center_sample = int(round(float(center) * max(length - 1, 0)))
        start = int(round(center_sample - window_samples / 2))
        start = min(max(start, 0), max_start)
        if start in seen_starts:
            continue
        seen_starts.add(start)
        starts.append(start)
        windows.append(audio[start : start + window_samples])

    return np.stack(windows).astype(np.float32), starts


@torch.no_grad()
def predict_windows(
    model: MCnn14,
    windows: np.ndarray,
    device: torch.device,
    task_id: int,
    batch_size: int,
) -> torch.Tensor:
    logits = []
    for index in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[index : index + batch_size]).float().to(device)
        logits.append(model(batch, task_id).detach().cpu())
    return torch.cat(logits, dim=0)


def max_confidence_prediction(logits: torch.Tensor) -> int:
    probs = F.softmax(logits, dim=1)
    best_window = int(probs.max(dim=1).values.argmax().item())
    return int(probs[best_window].argmax().item())


def predict_audio(
    model: MCnn14,
    audio_path: Path,
    device: torch.device,
    task_id: int,
    window_seconds: float,
    centers: tuple[float, ...],
    batch_size: int,
) -> tuple[str, dict]:
    audio = read_audio_mono(audio_path)
    windows, starts = centered_windows(audio, window_seconds=window_seconds, centers=centers)
    logits = predict_windows(model, windows, device, task_id=task_id, batch_size=batch_size)
    label_index = max_confidence_prediction(logits)
    return INDEX_TO_LABEL[label_index], {
        "samples": int(len(audio)),
        "num_windows": int(len(windows)),
        "starts": starts,
    }


def parse_centers(text: str) -> tuple[float, ...]:
    centers = tuple(float(item) for item in text.split(",") if item.strip())
    if not centers:
        raise ValueError("At least one center is required.")
    return centers


def write_outputs(
    predictions: list[dict],
    output_csv: Path,
    preview_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        for item in predictions:
            handle.write(f"{item['filename']}{OFFICIAL_SEPARATOR}{item['label']}\n")

    with preview_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "label", "samples", "num_windows", "starts"])
        writer.writeheader()
        for item in predictions:
            writer.writerow(item)


def write_prediction_row(output_handle, preview_writer: csv.DictWriter, item: dict) -> None:
    output_handle.write(f"{item['filename']}{OFFICIAL_SEPARATOR}{item['label']}\n")
    output_handle.flush()
    preview_writer.writerow(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official Task7 eval-set inference with checkpoint3 and 3s quint5 max-confidence windows.")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preview-csv", type=Path, default=DEFAULT_PREVIEW_CSV)
    parser.add_argument("--task-id", type=int, default=FINAL_TASK_ID)
    parser.add_argument("--window-sec", type=float, default=FINAL_WINDOW_SECONDS)
    parser.add_argument("--centers", default=",".join(str(item) for item in FINAL_CENTERS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto selection.")
    parser.add_argument("--dry-run", action="store_true", help="Only enumerate wav files and write alarm labels.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    audio_dir = args.audio_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_csv = args.output_csv.resolve()
    manifest_path = args.manifest.resolve()
    preview_csv = args.preview_csv.resolve()
    centers = parse_centers(args.centers)

    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing eval wav directory: {audio_dir}")
    audio_paths = sorted(audio_dir.glob("*.wav"))
    if not audio_paths:
        raise FileNotFoundError(f"No wav files found in: {audio_dir}")
    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("--max-files must be positive.")
        audio_paths = audio_paths[: args.max_files]
    if not args.dry_run and not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    if args.dry_run:
        predictions = [
            {"filename": path.name, "label": "alarm", "samples": "", "num_windows": "", "starts": ""}
            for path in audio_paths
        ]
        write_outputs(predictions, output_csv, preview_csv)
        device_text = "dry-run"
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = load_model(checkpoint_path, device)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        partial_output = output_csv.with_suffix(output_csv.suffix + ".partial")
        partial_preview = preview_csv.with_suffix(preview_csv.suffix + ".partial")
        with partial_output.open("w", encoding="utf-8", newline="") as output_handle, partial_preview.open(
            "w", encoding="utf-8", newline=""
        ) as preview_handle:
            preview_writer = csv.DictWriter(preview_handle, fieldnames=["filename", "label", "samples", "num_windows", "starts"])
            preview_writer.writeheader()
            for index, audio_path in enumerate(audio_paths, start=1):
                label, extra = predict_audio(
                    model,
                    audio_path,
                    device,
                    task_id=args.task_id,
                    window_seconds=args.window_sec,
                    centers=centers,
                    batch_size=args.batch_size,
                )
                item = {
                    "filename": audio_path.name,
                    "label": label,
                    "samples": extra["samples"],
                    "num_windows": extra["num_windows"],
                    "starts": " ".join(str(value) for value in extra["starts"]),
                }
                write_prediction_row(output_handle, preview_writer, item)
                preview_handle.flush()
                if index == 1 or index % 100 == 0 or index == len(audio_paths):
                    print(f"[{index:04d}/{len(audio_paths):04d}] {audio_path.name} -> {label}", flush=True)
        partial_output.replace(output_csv)
        partial_preview.replace(preview_csv)
        device_text = str(device)

    manifest = {
        "audio_dir": str(audio_dir),
        "num_wavs": len(audio_paths),
        "checkpoint": str(checkpoint_path) if not args.dry_run else str(checkpoint_path),
        "task_id": args.task_id,
        "window_sec": args.window_sec,
        "centers": centers,
        "aggregation": "max softmax confidence over windows",
        "output_csv": str(output_csv),
        "preview_csv": str(preview_csv),
        "device": device_text,
        "format": "No header. Each row is: filename.wav<4 spaces>class_label",
        "class_labels": list(LABEL_TO_INDEX.keys()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved official output: {output_csv}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved preview: {preview_csv}")


if __name__ == "__main__":
    main()
