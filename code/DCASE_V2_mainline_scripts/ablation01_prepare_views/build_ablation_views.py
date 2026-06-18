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
CLIP_SECONDS = 4
CLIP = SR * CLIP_SECONDS

DEFAULT_SOURCE_ROOT = RELEASE_ROOT / "data" / "processed_data"
DEFAULT_RAW_DATA_ROOT = RELEASE_ROOT / "data"
DEFAULT_OUT_ROOT = RELEASE_ROOT / "data" / "processed_aug" / "ablation_d23"
DEFAULT_RUN_ROOT = MAINLINE_ROOT / "runs" / "ablation_d23"
DEFAULT_VARIANTS = [
    "plain",
    "self_concat",
    "same_class_concat",
    "self_concat_shift_gain",
    "same_class_concat_shift_gain",
]
ALL_VARIANTS = DEFAULT_VARIANTS + [
    "current_full",
    "shift_gain",
    "time_shift",
    "gain",
    "full_without_concat",
    "class_concat_only",
]


def load_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(path)
    if sr != SR:
        raise ValueError(f"Sample-rate mismatch: {path}, got {sr}, expected {SR}")
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if len(x) < CLIP:
        x = np.pad(x, (0, CLIP - len(x)))
    return x[:CLIP]


def save_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    if len(x) != CLIP:
        raise ValueError(f"Length mismatch for {path}: {len(x)}, expected {CLIP}")
    sf.write(path, np.clip(x, -1.0, 1.0), SR)


def label_from_path(path: Path) -> str:
    return path.stem.split("-")[-1]


def read_label_csv(csv_path: Path) -> dict[str, str]:
    label_dict = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "filename" not in (reader.fieldnames or []) or "class" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain filename and class columns.")
        for row in reader:
            label_dict[Path(row["filename"]).name] = str(row["class"]).strip().lower()
    return label_dict


def load_raw_wav(path: Path) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise ImportError("librosa is required only when auto-building chunk-4 data from raw wav files.") from exc

    x, _ = librosa.load(str(path), sr=SR, mono=True)
    return x.astype(np.float32)


def chunk_one_split(wav_dir: Path, csv_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.glob("*.wav")):
        return
    if not wav_dir.exists():
        raise FileNotFoundError(f"Missing raw wav directory: {wav_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metadata csv: {csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    label_dict = read_label_csv(csv_path)
    min_len = CLIP // 2

    for wav_path in tqdm(sorted(wav_dir.glob("*.wav")), desc=f"chunk:{out_dir.name}"):
        if wav_path.name not in label_dict:
            continue
        x = load_raw_wav(wav_path)
        if len(x) == 0:
            continue
        num_segments = (len(x) - CLIP + CLIP - 1) // CLIP + 1
        for index in range(num_segments):
            segment = x[index * CLIP : (index + 1) * CLIP]
            if num_segments > 1 and len(segment) < min_len:
                break
            if len(segment) < CLIP:
                segment = np.pad(segment, (0, CLIP - len(segment)))
            label = label_dict[wav_path.name]
            save_wav(out_dir / f"{wav_path.stem}-{index:02d}-{label}.wav", segment)


def build_missing_chunks(source_root: Path, raw_data_root: Path, domains: list[str]) -> None:
    jobs = {
        "D2": [
            ("train", raw_data_root / "D2" / "d2-dev-train", raw_data_root / "D2" / "metadata" / "d2-dev-train.csv"),
            ("test", raw_data_root / "D2" / "d2-dev-test", raw_data_root / "D2" / "metadata" / "d2-dev-test.csv"),
        ],
        "D3": [
            ("train", raw_data_root / "D3" / "d3-dev-train", raw_data_root / "D3" / "metadata" / "d3-dev-train.csv"),
            ("test", raw_data_root / "D3" / "d3-dev-test", raw_data_root / "D3" / "metadata" / "d3-dev-test.csv"),
        ],
    }
    for domain in domains:
        for split, wav_dir, csv_path in jobs[domain]:
            chunk_one_split(wav_dir, csv_path, source_root / f"{domain}-{split}-chunk-4")


def valid_piece(x: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    idx = np.flatnonzero(np.abs(x) > threshold)
    if len(idx) == 0:
        return x.copy()
    pad = int(0.03 * SR)
    start = max(0, int(idx[0]) - pad)
    end = min(len(x), int(idx[-1]) + pad)
    return x[start:end].copy()


def temporal_shift(x: np.ndarray, rng: random.Random, max_shift_ratio: float) -> np.ndarray:
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


def apply_gain(x: np.ndarray, rng: random.Random, max_db: float) -> np.ndarray:
    db = rng.uniform(-max_db, max_db)
    return np.clip(x * (10.0 ** (db / 20.0)), -1.0, 1.0).astype(np.float32)


def append_with_gap(
    chunks: list[np.ndarray],
    piece: np.ndarray,
    rng: random.Random,
    total_len: int,
    add_gap: bool,
    min_gap: float = 0.04,
    max_gap: float = 0.18,
) -> tuple[int, bool]:
    if add_gap and total_len < CLIP:
        gap = min(rng.randint(int(min_gap * SR), int(max_gap * SR)), CLIP - total_len)
        if gap > 0:
            chunks.append(np.zeros(gap, dtype=np.float32))
            total_len += gap

    remaining = CLIP - total_len
    if remaining <= 0:
        return total_len, False
    chunks.append(piece[:remaining].astype(np.float32))
    return total_len + min(len(piece), remaining), True


def finish_clip(chunks: list[np.ndarray]) -> np.ndarray:
    out = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if len(out) < CLIP:
        out = np.pad(out, (0, CLIP - len(out)))
    return out[:CLIP].astype(np.float32)


def make_self_concat(source: Path, rng: random.Random) -> np.ndarray:
    piece = valid_piece(load_wav(source))
    if len(piece) == 0:
        piece = load_wav(source)

    chunks: list[np.ndarray] = []
    total_len = 0
    appended_any = False
    tries = 0
    while total_len < CLIP and tries < 100:
        tries += 1
        part = piece
        if len(piece) > CLIP - total_len:
            start = rng.randint(0, max(0, len(piece) - (CLIP - total_len)))
            part = piece[start : start + (CLIP - total_len)]
        total_len, appended = append_with_gap(
            chunks, part, rng, total_len, appended_any, min_gap=0.03, max_gap=0.12
        )
        appended_any = appended_any or appended
        if not appended:
            break
    return finish_clip(chunks)


def make_same_class_concat(source: Path, same_class_paths: list[Path], rng: random.Random) -> tuple[np.ndarray, int]:
    source_piece = valid_piece(load_wav(source))
    if len(source_piece) == 0:
        source_piece = load_wav(source)

    other_paths = [path for path in same_class_paths if path != source]
    rng.shuffle(other_paths)

    chunks: list[np.ndarray] = []
    total_len = 0
    appended_any = False
    for path in other_paths:
        piece = valid_piece(load_wav(path))
        if len(piece) == 0:
            continue
        total_len, appended = append_with_gap(chunks, piece, rng, total_len, appended_any)
        appended_any = appended_any or appended
        if total_len >= CLIP:
            return finish_clip(chunks), 0

    self_fill_count = 0
    while total_len < CLIP and self_fill_count < 100:
        total_len, appended = append_with_gap(chunks, source_piece, rng, total_len, appended_any)
        appended_any = appended_any or appended
        if not appended:
            break
        self_fill_count += 1
    return finish_clip(chunks), self_fill_count


def make_class_pool_concat(pieces: list[np.ndarray], rng: random.Random) -> np.ndarray:
    chunks: list[np.ndarray] = []
    total_len = 0
    appended_any = False
    while total_len < CLIP:
        piece = rng.choice(pieces)
        total_len, appended = append_with_gap(chunks, piece, rng, total_len, appended_any)
        appended_any = appended_any or appended
        if not appended:
            break
    return finish_clip(chunks)


def collect_split(source_root: Path, domains: list[str], split: str) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for domain in domains:
        chunk_dir = source_root / f"{domain}-{split}-chunk-4"
        if not chunk_dir.exists():
            raise FileNotFoundError(f"Missing chunk directory: {chunk_dir}")
        files.extend((domain, path) for path in sorted(chunk_dir.glob("*.wav")))
    if not files:
        raise FileNotFoundError(f"No wav files found under {source_root} for {domains} {split}")
    return files


def copy_clean(files: list[tuple[str, Path]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for domain, path in files:
        dst = out_dir / f"{domain}_{path.name}"
        shutil.copy2(path, dst)
        out_paths.append(dst)
    return out_paths


def group_by_class(paths: list[Path]) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_class[label_from_path(path)].append(path)
    return dict(by_class)


def collect_pieces(by_class: dict[str, list[Path]]) -> dict[str, list[np.ndarray]]:
    pieces_by_class: dict[str, list[np.ndarray]] = {}
    for label, paths in by_class.items():
        pieces = [valid_piece(load_wav(path)) for path in paths]
        pieces_by_class[label] = [piece for piece in pieces if len(piece) > 0]
    return pieces_by_class


def class_sampling_weights(by_class: dict[str, list[Path]], minority_power: float) -> tuple[list[str], np.ndarray]:
    labels = sorted(label for label, paths in by_class.items() if paths)
    counts = np.asarray([len(by_class[label]) for label in labels], dtype=np.float64)
    weights = (1.0 / counts) ** minority_power
    weights /= weights.sum()
    return labels, weights


def build_augmented_audio(
    variant: str,
    label: str,
    by_class: dict[str, list[Path]],
    pieces_by_class: dict[str, list[np.ndarray]],
    rng: random.Random,
    p_concat: float,
    p_shift: float,
    p_gain: float,
    max_shift_ratio: float,
    max_gain_db: float,
) -> tuple[np.ndarray, dict[str, int | str]]:
    source = rng.choice(by_class[label])
    ops: dict[str, int | str] = {
        "source": source.name,
        "concat": 0,
        "self_concat": 0,
        "self_fill": 0,
        "shift": 0,
        "gain": 0,
    }

    if variant == "self_concat":
        x = make_self_concat(source, rng)
        ops["self_concat"] = 1
    elif variant == "same_class_concat":
        x, self_fill = make_same_class_concat(source, by_class[label], rng)
        ops["concat"] = 1
        ops["self_fill"] = self_fill
    elif variant == "self_concat_shift_gain":
        x = make_self_concat(source, rng)
        ops["self_concat"] = 1
        if rng.random() < p_shift:
            x = temporal_shift(x, rng, max_shift_ratio)
            ops["shift"] = 1
        if rng.random() < p_gain:
            x = apply_gain(x, rng, max_gain_db)
            ops["gain"] = 1
    elif variant in {"same_class_concat_shift_gain", "current_full"}:
        if rng.random() < p_concat:
            x, self_fill = make_same_class_concat(source, by_class[label], rng)
            ops["concat"] = 1
            ops["self_fill"] = self_fill
        else:
            x = load_wav(source)
        if rng.random() < p_shift:
            x = temporal_shift(x, rng, max_shift_ratio)
            ops["shift"] = 1
        if rng.random() < p_gain:
            x = apply_gain(x, rng, max_gain_db)
            ops["gain"] = 1
    elif variant == "shift_gain":
        x = apply_gain(temporal_shift(load_wav(source), rng, max_shift_ratio), rng, max_gain_db)
        ops["shift"] = 1
        ops["gain"] = 1
    elif variant == "time_shift":
        x = temporal_shift(load_wav(source), rng, max_shift_ratio)
        ops["shift"] = 1
    elif variant == "gain":
        x = apply_gain(load_wav(source), rng, max_gain_db)
        ops["gain"] = 1
    elif variant == "full_without_concat":
        x = load_wav(source)
        if rng.random() < p_shift:
            x = temporal_shift(x, rng, max_shift_ratio)
            ops["shift"] = 1
        if rng.random() < p_gain:
            x = apply_gain(x, rng, max_gain_db)
            ops["gain"] = 1
    elif variant == "class_concat_only":
        x = make_class_pool_concat(pieces_by_class[label], rng)
        ops["concat"] = 1
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return x.astype(np.float32), ops


def build_view(args: argparse.Namespace, variant: str) -> dict:
    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    domains = [domain.upper() for domain in args.domains]
    view_name = args.name_template.format(variant=variant, seed=args.seed)
    out_root = Path(args.out_root) / view_name
    train_out = out_root / "mixed-train-chunk-4"
    test_out = out_root / "mixed-test-chunk-4"

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    if train_out.exists() and any(train_out.glob("*.wav")) and not args.overwrite:
        return {"name": view_name, "variant": variant, "status": "exists", "out_root": str(out_root)}

    source_root = Path(args.source_root)
    if args.auto_chunk:
        build_missing_chunks(source_root, Path(args.raw_data_root), domains)
    clean_train = copy_clean(collect_split(source_root, domains, "train"), train_out)
    clean_test = copy_clean(collect_split(source_root, domains, "test"), test_out)
    by_class = group_by_class(clean_train)
    pieces_by_class = collect_pieces(by_class)
    labels, weights = class_sampling_weights(by_class, args.minority_power)

    aug_n = 0 if variant == "plain" else int(round(len(clean_train) * args.aug_ratio))
    manifest = []
    op_counts: dict[str, int] = defaultdict(int)
    class_counts: dict[str, int] = defaultdict(int)

    for index in tqdm(range(aug_n), desc=f"build:{view_name}"):
        label = str(rng_np.choice(labels, p=weights))
        x, ops = build_augmented_audio(
            variant,
            label,
            by_class,
            pieces_by_class,
            rng,
            args.p_concat,
            args.p_shift,
            args.p_gain,
            args.max_shift_ratio,
            args.max_gain_db,
        )
        out_name = f"aug-{variant}-{index:05d}-{label}.wav"
        save_wav(train_out / out_name, x)

        manifest.append({"file": out_name, "label": label, "variant": variant, **ops})
        class_counts[label] += 1
        for key in ["concat", "self_concat", "self_fill", "shift", "gain"]:
            op_counts[key] += int(ops[key])

    if manifest:
        with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["file", "label", "variant", "source", "concat", "self_concat", "self_fill", "shift", "gain"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)

    summary = {
        "name": view_name,
        "variant": variant,
        "domains": domains,
        "seed": args.seed,
        "source_root": str(source_root),
        "out_root": str(out_root),
        "clean_train_n": len(clean_train),
        "clean_test_n": len(clean_test),
        "aug_n": aug_n,
        "final_train_n": len(clean_train) + aug_n,
        "aug_ratio": args.aug_ratio,
        "minority_power": args.minority_power,
        "p_concat": args.p_concat,
        "p_shift": args.p_shift,
        "p_gain": args.p_gain,
        "op_counts": dict(op_counts),
        "aug_class_counts": dict(sorted(class_counts.items())),
        "status": "built",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "view_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build D2+D3 augmentation ablation views used by the Task 7 report."
    )
    parser.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=DEFAULT_VARIANTS)
    parser.add_argument("--domains", nargs="+", default=["D2", "D3"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--aug-ratio", type=float, default=0.5)
    parser.add_argument("--minority-power", type=float, default=0.5)
    parser.add_argument("--p-concat", type=float, default=0.5)
    parser.add_argument("--p-shift", type=float, default=0.5)
    parser.add_argument("--p-gain", type=float, default=0.5)
    parser.add_argument("--max-shift-ratio", type=float, default=0.18)
    parser.add_argument("--max-gain-db", type=float, default=3.0)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--raw-data-root", default=str(DEFAULT_RAW_DATA_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--name-template", default="d23_ab_{variant}_s{seed}")
    parser.add_argument("--no-auto-chunk", dest="auto_chunk", action="store_false")
    parser.set_defaults(auto_chunk=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = [build_view(args, variant) for variant in args.variants]
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / f"build_ablation_views_s{args.seed}.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "views": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
