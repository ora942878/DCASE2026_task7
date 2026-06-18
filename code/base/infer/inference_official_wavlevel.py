from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.CFG_PATH import CFG, PATH  # noqa: E402
from models.domain_net import MCnn14  # noqa: E402


def infer_num_tasks_from_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    match = re.search(r"D(\d+)", checkpoint_path.stem)
    if match is None:
        raise ValueError(f"Cannot infer num_tasks from checkpoint name: {checkpoint_path}")
    return int(match.group(1))


def pad_sequence(x, min_len):
    """Official-style test handling: pad short audio, keep long audio unchanged."""
    if len(x) < min_len:
        return np.concatenate((x, np.zeros(min_len - len(x), dtype=x.dtype)))
    return x


class WavLevelDataset(Dataset):
    def __init__(self, audio_dir, csv_path):
        self.audio_dir = Path(audio_dir)
        self.csv_path = Path(csv_path)

        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Missing audio directory: {self.audio_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing csv file: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if "filename" not in self.df.columns or "class" not in self.df.columns:
            raise ValueError(f"Expected columns filename,class in {self.csv_path}. Got {list(self.df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = self.audio_dir / str(row["filename"])
        class_name = str(row["class"])

        if class_name not in CFG.dict_class_labels:
            raise ValueError(f"Unknown class '{class_name}' in {self.csv_path}")
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing wav file: {wav_path}")

        x, sr = sf.read(wav_path)
        if sr != CFG.sample_rate:
            raise ValueError(f"Sample rate mismatch: {wav_path}, file_sr={sr}, expected={CFG.sample_rate}")
        if x.ndim == 2:
            x = x.mean(axis=1)

        x = pad_sequence(x.astype(np.float32), CFG.clip_samples)
        y = CFG.dict_class_labels[class_name]

        return torch.from_numpy(x).float(), torch.tensor(y).long(), str(wav_path)


def collate_batch_size_one(batch):
    if len(batch) != 1:
        raise ValueError("Official wav-level evaluation uses batch_size=1 because test wavs are variable-length.")
    x, y, path = batch[0]
    return x.unsqueeze(0), y.unsqueeze(0), [path]


def build_model():
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


def load_checkpoint_state(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")


def forward_with_entropy_task_selection(model, x, seen_tasks):
    probs_list = []
    logits_list = []

    for task_id in range(seen_tasks):
        logits_t = model(x, task_id)
        probs_t = torch.softmax(logits_t, dim=1)
        logits_list.append(logits_t)
        probs_list.append(probs_t)

    probs_stack = torch.stack(probs_list, dim=0)
    logits_stack = torch.stack(logits_list, dim=0)
    entropy = -torch.sum(probs_stack * torch.log(probs_stack + sys.float_info.min), dim=2)
    best_task = torch.argmin(entropy, dim=0)

    batch_indices = torch.arange(x.size(0), device=x.device)
    final_logits = logits_stack[best_task, batch_indices, :]
    return final_logits, best_task


def compute_official_domain_accuracy(y_true, y_pred):
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
    official_domain_acc = float(class_recall[present].mean()) if present.any() else 0.0
    sample_acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) else 0.0
    return official_domain_acc, sample_acc, class_recall, row_totals, cm


@torch.no_grad()
def eval_wavlevel_by_checkpoint(
    audio_dir,
    csv_path,
    checkpoint_path,
    device,
    fixed_task_id=None,
):
    audio_dir = Path(audio_dir)
    csv_path = Path(csv_path)
    checkpoint_path = Path(checkpoint_path)

    model = build_model().to(device)
    model.load_state_dict(load_checkpoint_state(checkpoint_path, device))
    model.eval()

    dataset = WavLevelDataset(audio_dir=audio_dir, csv_path=csv_path)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=CFG.PIN_MEMORY,
        drop_last=False,
        collate_fn=collate_batch_size_one,
    )

    num_tasks = infer_num_tasks_from_checkpoint(checkpoint_path)
    task_counter = torch.zeros(num_tasks, dtype=torch.long)
    y_true = []
    y_pred = []

    for x, y, _paths in tqdm(loader, desc=f"WavEval {checkpoint_path.name}"):
        x = x.float().to(device)
        y = y.long().to(device)

        if fixed_task_id is not None:
            logits = model(x, fixed_task_id)
            best_task = torch.full((x.size(0),), fixed_task_id, device=device)
        else:
            logits, best_task = forward_with_entropy_task_selection(
                model=model,
                x=x,
                seen_tasks=num_tasks,
            )

        pred = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

        for task_id in best_task.cpu():
            task_counter[int(task_id.item())] += 1

    official_domain_acc, sample_acc, class_recall, row_totals, cm = compute_official_domain_accuracy(
        y_true=y_true,
        y_pred=y_pred,
    )

    id_to_class = {v: k for k, v in CFG.dict_class_labels.items()}

    print()
    print("Official wav-level evaluation finished")
    print("audio_dir:", audio_dir)
    print("csv_path:", csv_path)
    print("checkpoint:", checkpoint_path)
    print("num_tasks:", num_tasks)
    print(f"sample_acc: {sample_acc:.4f} ({sample_acc * 100:.2f}%)")
    print(f"official_domain_acc: {official_domain_acc:.4f} ({official_domain_acc * 100:.2f}%)")
    print("class-wise recall used by official metric:")

    for class_id in range(CFG.classes_num_DIL):
        class_name = id_to_class.get(class_id, str(class_id))
        total = int(row_totals[class_id])
        correct = int(cm[class_id, class_id])
        if total == 0:
            print(f"  {class_id:2d} {class_name:20s}: ignored (0 samples)")
        else:
            print(f"  {class_id:2d} {class_name:20s}: {correct:4d} / {total:4d} = {class_recall[class_id]:.4f}")

    print("selected task counts:")
    for task_id, count in enumerate(task_counter.tolist()):
        print(f"  task {task_id}: {count}")

    return {
        "sample_acc": sample_acc,
        "official_domain_acc": official_domain_acc,
        "official_domain_acc_pct": official_domain_acc * 100,
        "class_recall": class_recall.tolist(),
        "class_total": row_totals.tolist(),
        "confusion_matrix": cm.tolist(),
        "task_counter": task_counter.tolist(),
        "total": len(y_true),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cases = [
        ("D2_after_D2", PATH.TEST_D2, PATH.TEST_D2_CSV, Path(PATH.OFFICIAL_CHECKPOINT_ROOT) / "checkpoint_D2.pth"),
        ("D2_after_D3", PATH.TEST_D2, PATH.TEST_D2_CSV, Path(PATH.OFFICIAL_CHECKPOINT_ROOT) / "checkpoint_D3.pth"),
        ("D3_after_D3", PATH.TEST_D3, PATH.TEST_D3_CSV, Path(PATH.OFFICIAL_CHECKPOINT_ROOT) / "checkpoint_D3.pth"),
    ]

    results = []
    for name, audio_dir, csv_path, checkpoint_path in cases:
        print("=" * 80)
        print(name)
        result = eval_wavlevel_by_checkpoint(
            audio_dir=audio_dir,
            csv_path=csv_path,
            checkpoint_path=checkpoint_path,
            device=device,
            fixed_task_id=None,
        )
        results.append((name, result))

    print("=" * 80)
    print("SUMMARY")
    for name, result in results:
        print(
            f"{name}: official_domain_acc={result['official_domain_acc_pct']:.2f}%, "
            f"sample_acc={result['sample_acc'] * 100:.2f}%"
        )

    if len(results) >= 3:
        avg_after_d3 = (
            results[1][1]["official_domain_acc_pct"]
            + results[2][1]["official_domain_acc_pct"]
        ) / 2
        print(f"Avg_after_D3: {avg_after_d3:.2f}%")


if __name__ == "__main__":
    main()
