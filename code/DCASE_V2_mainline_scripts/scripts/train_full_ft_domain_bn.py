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
PROJECT_ROOT = CODE_ROOT / "DCASE_CODE_V2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.CFG_PATH import CFG  # noqa: E402
from models.domain_net import MCnn14  # noqa: E402

ROOT = MAINLINE_ROOT
RAW_ROOT = RELEASE_ROOT / "data"
AUG_ROOT = ROOT / "processed_aug"
CKPT_ROOT = RELEASE_ROOT / "checkpoints" / "official"
RUN_ROOT = ROOT / "runs"

TESTS = {
    "D2": (
        RAW_ROOT / "D2" / "d2-dev-test",
        RAW_ROOT / "D2" / "metadata" / "d2-dev-test.csv",
        1,
    ),
    "D3": (
        RAW_ROOT / "D3" / "d3-dev-test",
        RAW_ROOT / "D3" / "metadata" / "d3-dev-test.csv",
        2,
    ),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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
        label_name = path.stem.split("-")[-1]
        y = CFG.dict_class_labels[label_name]
        x, sr = sf.read(path)
        if sr != CFG.sample_rate:
            raise ValueError(f"Sample rate mismatch: {path}, got {sr}")
        if x.ndim == 2:
            x = x.mean(axis=1)
        if len(x) != CFG.clip_samples:
            raise ValueError(f"Chunk length mismatch: {path}, got {len(x)}")
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(y).long(), str(path)


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


def collate_wav_one(batch):
    x, y, path = batch[0]
    return x.unsqueeze(0), y.unsqueeze(0), [path]


def compute_macro_recall(y_true: list[int], y_pred: list[int]) -> dict:
    labels = list(range(CFG.classes_num_DIL))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = cm.sum(axis=1)
    class_recall = np.divide(cm.diagonal(), row_totals, out=np.zeros_like(row_totals, dtype=float), where=row_totals > 0)
    present = row_totals > 0
    return {
        "official_domain_acc": float(class_recall[present].mean()) if present.any() else 0.0,
        "sample_acc": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if y_true else 0.0,
        "class_recall": class_recall.tolist(),
        "class_total": row_totals.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def make_class_weights(dataset: ChunkDataset, device: torch.device) -> torch.Tensor:
    counts = np.zeros(CFG.classes_num_DIL, dtype=np.float64)
    for path in dataset.files:
        counts[CFG.dict_class_labels[path.stem.split("-")[-1]]] += 1
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def load_state_dict(path: Path, device: torch.device) -> dict:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint format: {path}")


def resolve_checkpoint(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    ckpt_path = CKPT_ROOT / path_text
    if ckpt_path.exists():
        return ckpt_path
    run_path = RUN_ROOT / path_text
    if run_path.exists():
        return run_path
    raise FileNotFoundError(path_text)


def forward_logits(model: nn.Module, x: torch.Tensor, task_id: int) -> torch.Tensor:
    return model(x, task_id)


def run_epoch(model, loader, criterion, device, task_id, optimizer=None, desc="", no_progress=False):
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
def eval_wavlevel(model, wav_dir: Path, csv_path: Path, device: torch.device, task_id: int, name: str, no_progress: bool = False) -> dict:
    loader = DataLoader(
        WavDataset(wav_dir, csv_path),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_wav_one,
    )
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y, _paths in tqdm(loader, desc=f"wav_eval:{name}", leave=False, disable=no_progress):
        x = x.float().to(device)
        y = y.long().to(device)
        logits = forward_logits(model, x, task_id)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--domain", choices=["D2", "D3"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--final-checkpoint", choices=["best", "last"], default="last")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RUN_ROOT / args.run_name / args.method
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dir = AUG_ROOT / args.method / f"{args.domain}-train-chunk-4"
    val_dir = AUG_ROOT / args.method / f"{args.domain}-test-chunk-4"
    train_set = ChunkDataset(train_dir)
    val_set = ChunkDataset(val_dir)
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, **loader_kwargs)

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

    print("device:", device)
    print("run_dir:", run_dir)
    print("init_checkpoint:", init_checkpoint)
    print("domain:", args.domain, "task_id:", args.task_id)
    print("train_dir:", train_dir, "n=", len(train_set))
    print("val_dir:", val_dir, "n=", len(val_set))

    for epoch in range(start_epoch, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, device, args.task_id, optimizer, desc=f"train:{epoch:03d}", no_progress=args.no_progress)
        val_m = run_epoch(model, val_loader, criterion, device, args.task_id, optimizer=None, desc=f"chunk_val:{epoch:03d}", no_progress=args.no_progress)
        scheduler.step()
        current = float(val_m["official_domain_acc"])
        if current > best_metric:
            best_metric = current
            save_checkpoint(run_dir / "best.pth", model, optimizer, scheduler, epoch, best_metric, args)
            save_checkpoint(run_dir / f"checkpoint_{args.domain}_fullft_bn{args.task_id + 1}_best.pth", model, optimizer, scheduler, epoch, best_metric, args)
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_metric, args)
        append_history(
            run_dir / "history.csv",
            {
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_domain_acc": train_m["official_domain_acc"],
                "val_loss": val_m["loss"],
                "val_domain_acc": val_m["official_domain_acc"],
                "lr": scheduler.get_last_lr()[0],
                "best_metric": best_metric,
            },
        )
        print(f"epoch={epoch:03d} train={train_m['official_domain_acc']:.4f} val={val_m['official_domain_acc']:.4f} best={best_metric:.4f}")

    chosen = run_dir / ("best.pth" if args.final_checkpoint == "best" else "last.pth")
    ckpt = torch.load(chosen, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    final_path = run_dir / f"checkpoint_{args.domain}_fullft_bn{args.task_id + 1}_last.pth"
    save_checkpoint(final_path, model, optimizer, scheduler, int(ckpt["epoch"]), float(ckpt["best_metric"]), args)

    metrics_out = {
        "method": args.method,
        "domain": args.domain,
        "run_dir": str(run_dir),
        "init_checkpoint": str(init_checkpoint),
        "task_id": args.task_id,
        "final_checkpoint": args.final_checkpoint,
        "final_epoch": int(ckpt["epoch"]),
        "best_chunk_domain_acc": float(ckpt["best_metric"]),
    }
    for eval_domain, (wav_dir, csv_path, eval_task_id) in TESTS.items():
        if wav_dir.exists() and csv_path.exists():
            metrics_out[f"{eval_domain}_wav"] = eval_wavlevel(
                model, wav_dir, csv_path, device, eval_task_id, name=f"{args.method}:{eval_domain}", no_progress=args.no_progress
            )
    if "D2_wav" in metrics_out and "D3_wav" in metrics_out:
        metrics_out["avg_D2_D3_wav_official_acc"] = (
            metrics_out["D2_wav"]["official_domain_acc"] + metrics_out["D3_wav"]["official_domain_acc"]
        ) / 2.0
    (run_dir / "final_metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    print(json.dumps(metrics_out, indent=2), flush=True)


if __name__ == "__main__":
    main()
