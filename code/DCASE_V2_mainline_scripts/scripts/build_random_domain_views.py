from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
AUG_ROOT = ROOT / "processed_aug"
NO_AUG_ROOT = AUG_ROOT / "no_aug"
SR = 32000
CLIP = SR * 4


def load_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(path)
    if sr != SR:
        raise ValueError(f"Sample-rate mismatch: {path}, got {sr}")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32)


def save_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    if len(x) != CLIP:
        raise ValueError(f"Length mismatch for {path}: {len(x)}")
    sf.write(path, x, SR)


def label_from_path(path: Path) -> str:
    return path.stem.split("-")[-1]


def collect_clean(src_dir: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(src_dir.glob("*.wav")):
        by_class[label_from_path(path)].append(path)
    if not by_class:
        raise FileNotFoundError(f"No wav files in {src_dir}")
    return dict(by_class)


def valid_piece(x: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    idx = np.flatnonzero(np.abs(x) > threshold)
    if len(idx) == 0:
        return x.copy()
    start = max(0, int(idx[0]) - int(0.03 * SR))
    end = min(len(x), int(idx[-1]) + int(0.03 * SR))
    return x[start:end].copy()


def collect_pieces(by_class: dict[str, list[Path]]) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for label, paths in by_class.items():
        pieces = []
        for path in paths:
            piece = valid_piece(load_wav(path))
            if len(piece) > 0:
                pieces.append(piece)
        out[label] = pieces
    return out


def make_keepgap_concat(pieces: list[np.ndarray], rng: random.Random, max_gap_ratio: float = 0.18) -> np.ndarray:
    out = np.zeros(0, dtype=np.float32)
    total_gap = 0
    max_gap = int(CLIP * max_gap_ratio)
    tries = 0
    while len(out) < CLIP and tries < 100:
        tries += 1
        remaining = CLIP - len(out)
        if len(out) > 0 and total_gap < max_gap and remaining > int(0.25 * SR):
            gap = rng.randint(int(0.04 * SR), int(0.18 * SR))
            gap = min(gap, remaining, max_gap - total_gap)
            if gap > 0:
                out = np.concatenate([out, np.zeros(gap, dtype=np.float32)])
                total_gap += gap
                remaining = CLIP - len(out)
        piece = rng.choice(pieces)
        if len(piece) > remaining:
            piece = piece[:remaining]
        out = np.concatenate([out, piece])
    if len(out) < CLIP:
        out = np.pad(out, (0, CLIP - len(out)))
    return out[:CLIP].astype(np.float32)


def temporal_shift(x: np.ndarray, rng: random.Random, max_shift_ratio: float = 0.18) -> np.ndarray:
    max_shift = int(CLIP * max_shift_ratio)
    shift = rng.randint(-max_shift, max_shift)
    out = np.zeros_like(x)
    if shift > 0:
        out[shift:] = x[:-shift]
    elif shift < 0:
        out[:shift] = x[-shift:]
    else:
        out[:] = x
    return out


def apply_gain(x: np.ndarray, rng: random.Random, max_db: float = 3.0) -> np.ndarray:
    db = rng.uniform(-max_db, max_db)
    y = x * (10.0 ** (db / 20.0))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def class_sampling_weights(by_class: dict[str, list[Path]], minority_power: float) -> tuple[list[str], np.ndarray]:
    labels = sorted(label for label, paths in by_class.items() if len(paths) > 0)
    counts = np.asarray([len(by_class[label]) for label in labels], dtype=np.float64)
    weights = (1.0 / counts) ** minority_power
    weights /= weights.sum()
    return labels, weights


def copy_clean_split(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob("*.wav")):
        shutil.copy2(path, dst / path.name)


def build_view(
    domain: str,
    seed: int,
    name: str,
    p_concat: float,
    p_shift: float,
    p_gain: float,
    aug_ratio: float,
    minority_power: float,
    overwrite: bool,
) -> dict:
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    out_root = AUG_ROOT / name
    train_out = out_root / f"{domain}-train-chunk-4"
    test_out = out_root / f"{domain}-test-chunk-4"
    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    if train_out.exists() and any(train_out.glob("*.wav")):
        return {"name": name, "domain": domain, "seed": seed, "status": "exists"}

    src_train = NO_AUG_ROOT / f"{domain}-train-chunk-4"
    src_test = NO_AUG_ROOT / f"{domain}-test-chunk-4"
    copy_clean_split(src_train, train_out)
    copy_clean_split(src_test, test_out)

    by_class = collect_clean(src_train)
    pieces_by_class = collect_pieces(by_class)
    labels, weights = class_sampling_weights(by_class, minority_power)
    clean_n = sum(len(v) for v in by_class.values())
    aug_n = int(round(clean_n * aug_ratio))

    type_counts = defaultdict(int)
    class_counts = defaultdict(int)
    manifest = []
    for i in tqdm(range(aug_n), desc=f"build:{name}"):
        label = str(rng_np.choice(labels, p=weights))
        use_concat = rng.random() < p_concat
        if use_concat:
            x = make_keepgap_concat(pieces_by_class[label], rng)
            type_counts["concat"] += 1
        else:
            x = load_wav(rng.choice(by_class[label]))
            type_counts["clean_resample"] += 1
        use_shift = rng.random() < p_shift
        use_gain = rng.random() < p_gain
        if use_shift:
            x = temporal_shift(x, rng)
            type_counts["shift"] += 1
        if use_gain:
            x = apply_gain(x, rng)
            type_counts["gain"] += 1
        out_name = f"aug-{label}-{i:05d}-{label}.wav"
        save_wav(train_out / out_name, x)
        class_counts[label] += 1
        manifest.append({"file": out_name, "label": label, "concat": int(use_concat), "shift": int(use_shift), "gain": int(use_gain)})

    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "concat", "shift", "gain"])
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "name": name,
        "domain": domain,
        "seed": seed,
        "status": "built",
        "clean_n": clean_n,
        "aug_n": aug_n,
        "final_train_n": clean_n + aug_n,
        "p_concat": p_concat,
        "p_shift": p_shift,
        "p_gain": p_gain,
        "aug_ratio": aug_ratio,
        "minority_power": minority_power,
        "type_counts": dict(type_counts),
        "aug_class_counts": dict(sorted(class_counts.items())),
    }
    (out_root / "view_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["D2", "D3"], required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--p-concat", type=float, default=0.5)
    parser.add_argument("--p-shift", type=float, default=0.5)
    parser.add_argument("--p-gain", type=float, default=0.5)
    parser.add_argument("--aug-ratio", type=float, default=1.0)
    parser.add_argument("--minority-power", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        name = f"{args.prefix}_s{seed}"
        results.append(
            build_view(args.domain, seed, name, args.p_concat, args.p_shift, args.p_gain, args.aug_ratio, args.minority_power, args.overwrite)
        )
    out = ROOT / "runs" / f"{args.prefix}_view_build_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
