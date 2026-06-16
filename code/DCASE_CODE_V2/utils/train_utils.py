import random
import os

from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.CFG_PATH import PATH, CFG
from models.domain_net import MCnn14
from utils.Chunked_Audio_Dataset import ChunkedAudioDataset
from utils.train_baseline_utils import (
    train_one_epoch,
    eval_one_epoch,
    load_checkpoint_as_state_dict,
)
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

def get_chunk_dir(domain_id, split):
    domain_id = int(domain_id)
    split = split.lower() # 强制切换成小写

    if domain_id == 2 and split == "train":
        return Path(PATH.TRAIN_D2_CHUNK_4)
    if domain_id == 2 and split == "test":
        return Path(PATH.TEST_D2_CHUNK_4)
    if domain_id == 3 and split == "train":
        return Path(PATH.TRAIN_D3_CHUNK_4)
    if domain_id == 3 and split == "test":
        return Path(PATH.TEST_D3_CHUNK_4)

    raise ValueError(f"Unsupported domain/split: D{domain_id}, {split}")

def build_loader(domain_id, split, batch_size, shuffle):
    data_dir = get_chunk_dir(domain_id, split)

    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data folder: {data_dir}")

    dataset = ChunkedAudioDataset(data_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
        drop_last=False,
    )


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model, lr, weight_decay):
    return optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

def build_cosine_scheduler(optimizer):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG.epochs,
        eta_min=0.001,
    )
    return scheduler

def get_ckpt_path(domain_id, kind):
    return Path(PATH.SAVE_DIR) / f"D{domain_id}{kind}.pth"


def save_checkpoint(path, model, optimizer, epoch, best_acc):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        },
        path,
    )


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt["epoch"], ckpt["best_acc"]


def resume_if_needed(domain_id, model, optimizer, resume):
    if not resume:
        return 1, 0.0

    path = get_ckpt_path(domain_id, "last")
    epoch, best_acc = load_checkpoint(path, model, optimizer)

    return epoch + 1, best_acc


def save_last_and_best(domain_id, model, optimizer, epoch, acc, best_acc):
    last_path = get_ckpt_path(domain_id, "last")
    save_checkpoint(last_path, model, optimizer, epoch, best_acc)

    if acc > best_acc:
        best_acc = acc
        best_path = get_ckpt_path(domain_id, "best")
        save_checkpoint(best_path, model, optimizer, epoch, best_acc)

    return best_acc