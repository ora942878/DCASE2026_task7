from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
MAINLINE_ROOT = SCRIPT_DIR.parent
CODE_ROOT = MAINLINE_ROOT.parent
RELEASE_ROOT = CODE_ROOT.parent
BASE_ROOT = CODE_ROOT / "base"
if str(BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_ROOT))

from configs.CFG_PATH import CFG  # noqa: E402
from models.domain_net import MCnn14  # noqa: E402


RAW_ROOT = RELEASE_ROOT / "data"
PROCESSED_AUG_ROOT = RAW_ROOT / "processed_aug"
RUN_ROOT = MAINLINE_ROOT / "runs"
CHECKPOINT_ROOT = RELEASE_ROOT / "checkpoints"
D2_TEST_WAV = RAW_ROOT / "D2" / "d2-dev-test"
D2_TEST_CSV = RAW_ROOT / "D2" / "metadata" / "d2-dev-test.csv"
D3_TEST_WAV = RAW_ROOT / "D3" / "d3-dev-test"
D3_TEST_CSV = RAW_ROOT / "D3" / "metadata" / "d3-dev-test.csv"


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


def forward_logits(model: nn.Module, x: torch.Tensor, task_id: int) -> torch.Tensor:
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
    return {
        "official_domain_acc": float(class_recall[present].mean()) if present.any() else 0.0,
        "sample_acc": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if y_true else 0.0,
        "class_recall": class_recall.tolist(),
        "class_total": row_totals.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def load_state_dict(path: Path, device: torch.device) -> dict:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint format: {path}")


def resolve_checkpoint(path_text: str) -> Path:
    path = Path(path_text)
    candidates = (
        path,
        CHECKPOINT_ROOT / path_text,
        CHECKPOINT_ROOT / "official" / path_text,
        CHECKPOINT_ROOT / "ours" / path_text,
        RUN_ROOT / path_text,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path_text)


DEFAULT_VIEW_ROOT = PROCESSED_AUG_ROOT / "ablation_d23"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def label_from_path(path: Path) -> str:
    return path.stem.split("-")[-1]


class ChunkDataset(Dataset):
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = Path(chunk_dir)
        self.files = sorted(self.chunk_dir.glob("*.wav"))
        if not self.files:
            raise FileNotFoundError(f"No wav files found in {self.chunk_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        label_name = label_from_path(path)
        y = CFG.dict_class_labels[label_name]
        x, sr = sf.read(path)
        if sr != CFG.sample_rate:
            raise ValueError(f"Sample rate mismatch: {path}, got {sr}")
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = x.astype(np.float32)
        if len(x) != CFG.clip_samples:
            raise ValueError(f"Chunk length mismatch: {path}, got {len(x)}")
        return torch.from_numpy(x), torch.tensor(y).long(), str(path)


class WavDataset(Dataset):
    def __init__(self, wav_dir: Path, csv_path: Path):
        self.wav_dir = Path(wav_dir)
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        path = self.wav_dir / str(row["filename"])
        label_name = str(row["class"])
        y = CFG.dict_class_labels[label_name]
        x, sr = sf.read(path)
        if sr != CFG.sample_rate:
            raise ValueError(f"Sample rate mismatch: {path}, got {sr}")
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = x.astype(np.float32)
        if len(x) < CFG.clip_samples:
            x = np.pad(x, (0, CFG.clip_samples - len(x)))
        return torch.from_numpy(x), torch.tensor(y).long(), str(path)


def collate_wav_pad(batch):
    xs, ys, paths = zip(*batch)
    max_len = max(int(x.numel()) for x in xs)
    padded = []
    for x in xs:
        if x.numel() < max_len:
            x = torch.nn.functional.pad(x, (0, max_len - x.numel()))
        padded.append(x)
    return torch.stack(padded, dim=0), torch.stack(ys, dim=0), list(paths)


def make_class_weights(dataset: ChunkDataset, device: torch.device) -> torch.Tensor:
    counts = np.zeros(CFG.classes_num_DIL, dtype=np.float64)
    for path in dataset.files:
        counts[CFG.dict_class_labels[label_from_path(path)]] += 1
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task_id: int,
    optimizer: optim.Optimizer | None = None,
    desc: str = "",
    no_progress: bool = False,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y, _paths in tqdm(loader, desc=desc, leave=False, disable=no_progress):
        x = x.float().to(device)
        y = y.long().to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = forward_logits(model, x, task_id)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
        pred = logits.argmax(dim=1)
        bs = x.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
    out = compute_macro_recall(y_true, y_pred)
    out["loss"] = total_loss / max(total, 1)
    return out


@torch.no_grad()
def eval_wavlevel(
    model: nn.Module,
    wav_dir: Path,
    csv_path: Path,
    device: torch.device,
    task_id: int,
    name: str,
    batch_size: int,
    num_workers: int,
    no_progress: bool = False,
) -> dict:
    loader = DataLoader(
        WavDataset(wav_dir, csv_path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_wav_pad,
    )
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y, _paths in tqdm(loader, desc=f"wav_eval:{name}", leave=False, disable=no_progress):
        x = x.float().to(device)
        y = y.long().to(device)
        logits = forward_logits(model, x, task_id)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    return compute_macro_recall(y_true, y_pred)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_metric: float, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
        },
        path,
    )


def append_history(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def eval_competition_metric(model, device: torch.device, task_id: int, args) -> dict:
    d2 = eval_wavlevel(
        model,
        D2_TEST_WAV,
        D2_TEST_CSV,
        device,
        task_id,
        "D2",
        args.eval_batch_size,
        args.eval_workers,
        args.no_progress,
    )
    d3 = eval_wavlevel(
        model,
        D3_TEST_WAV,
        D3_TEST_CSV,
        device,
        task_id,
        "D3",
        args.eval_batch_size,
        args.eval_workers,
        args.no_progress,
    )
    return {
        "D2_wav": d2,
        "D3_wav": d3,
        "avg_D2_D3_wav_official_acc": (d2["official_domain_acc"] + d3["official_domain_acc"]) / 2.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one D2+D3 mixed augmentation ablation model.")
    parser.add_argument(
        "--view-name",
        default="d23_ab_same_class_concat_shift_gain_s3407",
        help="Name built by build_ablation_views.py.",
    )
    parser.add_argument("--run-name", default="aug_ablation_40ep")
    parser.add_argument("--init-checkpoint", default=str(CHECKPOINT_ROOT / "official" / "checkpoint_D1.pth"))
    parser.add_argument("--task-id", type=int, default=0, help="Default is BN1, matching official checkpoint_D1.pth.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--best-metric", choices=["wav_avg", "chunk_val"], default="wav_avg")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--view-root", default=str(DEFAULT_VIEW_ROOT))
    parser.add_argument("--run-root", default=str(RUN_ROOT / "ablation_d23"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    view_dir = Path(args.view_root) / args.view_name
    train_dir = view_dir / "mixed-train-chunk-4"
    val_dir = view_dir / "mixed-test-chunk-4"
    run_dir = Path(args.run_root) / args.run_name / args.view_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set = ChunkDataset(train_dir)
    val_set = ChunkDataset(val_dir)
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        **loader_kwargs,
    )

    init_checkpoint = resolve_checkpoint(args.init_checkpoint)
    model = build_model().to(device)
    model.load_state_dict(load_state_dict(init_checkpoint, device))
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(train_set, device) if args.class_weight == "balanced" else None)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    start_epoch = 1
    best_metric = -1.0
    last_path = run_dir / "last.pth"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt["best_metric"])

    meta = {
        "view_name": args.view_name,
        "run_dir": str(run_dir),
        "view_dir": str(view_dir),
        "init_checkpoint": str(init_checkpoint),
        "task_id": args.task_id,
        "best_metric": args.best_metric,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "class_weight": args.class_weight,
        "train_n": len(train_set),
        "val_n": len(val_set),
        "device": str(device),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)

    latest_comp = None
    for epoch in range(start_epoch, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, device, args.task_id, optimizer, f"train:{epoch:03d}", args.no_progress)
        val_m = run_epoch(model, val_loader, criterion, device, args.task_id, None, f"chunk_val:{epoch:03d}", args.no_progress)
        scheduler.step()

        d2_acc = ""
        d3_acc = ""
        wav_avg = ""
        if epoch == args.epochs or epoch % args.eval_every == 0:
            latest_comp = eval_competition_metric(model, device, args.task_id, args)
            d2_acc = latest_comp["D2_wav"]["official_domain_acc"]
            d3_acc = latest_comp["D3_wav"]["official_domain_acc"]
            wav_avg = latest_comp["avg_D2_D3_wav_official_acc"]

        current = float(wav_avg if args.best_metric == "wav_avg" and wav_avg != "" else val_m["official_domain_acc"])
        if current > best_metric:
            best_metric = current
            save_checkpoint(run_dir / "best.pth", model, optimizer, scheduler, epoch, best_metric, args)
            if latest_comp is not None:
                (run_dir / "best_metrics.json").write_text(json.dumps(latest_comp, indent=2), encoding="utf-8")

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_metric, args)
        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_domain_acc": train_m["official_domain_acc"],
            "val_loss": val_m["loss"],
            "val_domain_acc": val_m["official_domain_acc"],
            "D2_wav_acc": d2_acc,
            "D3_wav_acc": d3_acc,
            "avg_D2_D3_wav_acc": wav_avg,
            "lr": scheduler.get_last_lr()[0],
            "best_metric": best_metric,
        }
        append_history(run_dir / "history.csv", row)
        print(
            f"epoch={epoch:03d} train={train_m['official_domain_acc']:.4f} "
            f"chunk_val={val_m['official_domain_acc']:.4f} wav_avg={wav_avg} best={best_metric:.4f}",
            flush=True,
        )

    best_ckpt = torch.load(run_dir / "best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_comp = eval_competition_metric(model, device, args.task_id, args)
    final_out = {
        **meta,
        "best_epoch": int(best_ckpt["epoch"]),
        "best_metric_value": float(best_ckpt["best_metric"]),
        **final_comp,
    }
    (run_dir / "final_metrics.json").write_text(json.dumps(final_out, indent=2), encoding="utf-8")
    print(json.dumps(final_out, indent=2), flush=True)


if __name__ == "__main__":
    main()
