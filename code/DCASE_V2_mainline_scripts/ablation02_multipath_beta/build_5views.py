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
MAINLINE_ROOT = SCRIPT_DIR.parent
CODE_ROOT = MAINLINE_ROOT.parent
RELEASE_ROOT = CODE_ROOT.parent

SR = 32000
CLIP = SR * 4
DEFAULT_SOURCE_ROOT = RELEASE_ROOT / "data" / "processed_data"
DEFAULT_OUT_ROOT = RELEASE_ROOT / "data" / "processed_aug"
DEFAULT_RUN_ROOT = MAINLINE_ROOT / "runs" / "ablation02_multipath_beta"
DEFAULT_SEEDS = [101, 202, 303, 404, 505]


def load_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(path)
    if sr != SR:
        raise ValueError(f"Sample-rate mismatch: {path}, got {sr}, expected {SR}")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32)


def save_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    if len(x) != CLIP:
        raise ValueError(f"Length mismatch for {path}: {len(x)}, expected {CLIP}")
    sf.write(path, np.clip(x, -1.0, 1.0), SR)


def label_from_path(path: Path) -> str:
    return path.stem.split("-")[-1]


def valid_piece(x: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    idx = np.flatnonzero(np.abs(x) > threshold)
    if len(idx) == 0:
        return x.copy()
    pad = int(0.03 * SR)
    start = max(0, int(idx[0]) - pad)
    end = min(len(x), int(idx[-1]) + pad)
    return x[start:end].copy()


def collect_clean(src_dir: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(src_dir.glob("*.wav")):
        by_class[label_from_path(path)].append(path)
    if not by_class:
        raise FileNotFoundError(f"No wav files in {src_dir}")
    return dict(by_class)


def collect_pieces(by_class: dict[str, list[Path]]) -> dict[str, list[np.ndarray]]:
    pieces_by_class: dict[str, list[np.ndarray]] = {}
    for label, paths in by_class.items():
        pieces = []
        for path in paths:
            piece = valid_piece(load_wav(path))
            if len(piece) > 0:
                pieces.append(piece)
        pieces_by_class[label] = pieces
    return pieces_by_class


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
    return out.astype(np.float32)


def apply_gain(x: np.ndarray, rng: random.Random, max_db: float = 3.0) -> np.ndarray:
    db = rng.uniform(-max_db, max_db)
    return np.clip(x * (10.0 ** (db / 20.0)), -1.0, 1.0).astype(np.float32)


def class_sampling_weights(by_class: dict[str, list[Path]], minority_power: float) -> tuple[list[str], np.ndarray]:
    labels = sorted(label for label, paths in by_class.items() if paths)
    counts = np.asarray([len(by_class[label]) for label in labels], dtype=np.float64)
    weights = (1.0 / counts) ** minority_power
    weights /= weights.sum()
    return labels, weights


def copy_clean_split(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob("*.wav")):
        shutil.copy2(path, dst / path.name)


def build_view(args: argparse.Namespace, seed: int) -> dict:
    domain = args.domain.upper()
    name = f"{args.prefix}_s{seed}"
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    source_root = Path(args.source_root)
    out_root = Path(args.out_root) / name
    train_out = out_root / f"{domain}-train-chunk-4"
    test_out = out_root / f"{domain}-test-chunk-4"

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    if train_out.exists() and any(train_out.glob("*.wav")) and not args.overwrite:
        return {"name": name, "domain": domain, "seed": seed, "status": "exists", "out_root": str(out_root)}

    src_train = source_root / f"{domain}-train-chunk-4"
    src_test = source_root / f"{domain}-test-chunk-4"
    copy_clean_split(src_train, train_out)
    copy_clean_split(src_test, test_out)

    by_class = collect_clean(src_train)
    pieces_by_class = collect_pieces(by_class)
    labels, weights = class_sampling_weights(by_class, args.minority_power)
    clean_n = sum(len(paths) for paths in by_class.values())
    aug_n = int(round(clean_n * args.aug_ratio))

    op_counts: dict[str, int] = defaultdict(int)
    class_counts: dict[str, int] = defaultdict(int)
    manifest = []
    for index in tqdm(range(aug_n), desc=f"build:{name}"):
        label = str(rng_np.choice(labels, p=weights))
        use_concat = rng.random() < args.p_concat
        if use_concat:
            x = make_keepgap_concat(pieces_by_class[label], rng)
            op_counts["concat"] += 1
        else:
            x = load_wav(rng.choice(by_class[label]))
            op_counts["clean_resample"] += 1

        use_shift = rng.random() < args.p_shift
        use_gain = rng.random() < args.p_gain
        if use_shift:
            x = temporal_shift(x, rng, args.max_shift_ratio)
            op_counts["shift"] += 1
        if use_gain:
            x = apply_gain(x, rng, args.max_gain_db)
            op_counts["gain"] += 1

        out_name = f"aug-ccsg-{index:05d}-{label}.wav"
        save_wav(train_out / out_name, x)
        class_counts[label] += 1
        manifest.append(
            {
                "file": out_name,
                "label": label,
                "source_policy": "same_class_concat_shift_gain",
                "concat": int(use_concat),
                "shift": int(use_shift),
                "gain": int(use_gain),
            }
        )

    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "source_policy", "concat", "shift", "gain"])
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "name": name,
        "domain": domain,
        "seed": seed,
        "source_root": str(source_root),
        "out_root": str(out_root),
        "clean_n": clean_n,
        "aug_n": aug_n,
        "final_train_n": clean_n + aug_n,
        "p_concat": args.p_concat,
        "p_shift": args.p_shift,
        "p_gain": args.p_gain,
        "aug_ratio": args.aug_ratio,
        "minority_power": args.minority_power,
        "op_counts": dict(op_counts),
        "aug_class_counts": dict(sorted(class_counts.items())),
        "status": "built",
    }
    (out_root / "view_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build five stochastic D3 views for beta ablation.")
    parser.add_argument("--domain", choices=["D2", "D3"], default="D3")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--prefix", default="rand_d3_ccsg")
    parser.add_argument("--p-concat", type=float, default=0.5)
    parser.add_argument("--p-shift", type=float, default=0.5)
    parser.add_argument("--p-gain", type=float, default=0.5)
    parser.add_argument("--aug-ratio", type=float, default=1.0)
    parser.add_argument("--minority-power", type=float, default=0.5)
    parser.add_argument("--max-shift-ratio", type=float, default=0.18)
    parser.add_argument("--max-gain-db", type=float, default=3.0)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = [build_view(args, seed) for seed in args.seeds]
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    out = run_root / f"{args.prefix}_view_build_summary.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out), "views": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
